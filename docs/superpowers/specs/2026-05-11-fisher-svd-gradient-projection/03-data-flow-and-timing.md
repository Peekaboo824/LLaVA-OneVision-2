# 03 — 数据流与执行时序

## 1. 离线阶段时序（一次性）

```
T0  build_fisher_data.py                       (~ 1 分钟)
    │
    ├── 加载 5 个 HF 数据集 (走 HF_DATASETS_CACHE)
    │     MATH / MBPP / LogiQA / HaluEval / Dolly
    │
    ├── 按 seed=42 采样
    │     {math:550, mbpp:300, logiqa:450, halueval:348, dolly:400}
    │
    ├── 字段映射 → messages = [user, assistant]
    │     (此步不 tokenize, 不做 label mask)
    │
    └── 写 fisher_anchor_dataset.jsonl  (~ 数 MB, 2048 行)
           │
           ▼
T1  compute_fisher.py  --model Qwen3-4B-Instruct     (~ 1-2 小时, 单 GPU)
    │
    ├── 加载 HF 模型 (bf16 ≈ 8 GB) + processor
    ├── 冻结非目标参数 (只 7 类层 × 36 层 = 252 个参数 requires_grad=True)
    ├── 初始化 accum: dict[name → zeros_like(param, fp32)]
    │
    ├── for sample in jsonl:                         # 2048 次
    │     full   = apply_chat_template(messages)
    │     prompt = apply_chat_template([user], add_generation_prompt=True)
    │     labels[:, :len(prompt)] = -100
    │     loss = model(input_ids, labels=labels).loss
    │     loss.backward()
    │     for p in targets:  accum[name] += p.grad.float() ** 2
    │     model.zero_grad(set_to_none=True)
    │
    └── torch.save(accum, fisher_dict_qwen3_4b.pt)   (fp32, ~ 数 GB)
           │
           ▼
T2  extract_subspace.py  --fisher ... --model ...    (~ 10-30 分钟 GPU)
    │
    ├── 加载 fisher_dict + HF state_dict
    │
    ├── for name_hf in fisher_dict:                  # 252 次
    │     F_log   = log10(F + 1e-12)
    │     F_norm  = (F_log - min) / (max - min)      # 逐层全元素 min-max
    │     W_tilde = F_norm * W
    │     U,S,Vh  = svd(W_tilde.float())
    │     k       = min(arg(Σσ²/Σσ² ≥ 0.95), 0.5·in_dim)
    │     V_k     = Vh[:k].T.contiguous().bfloat16()
    │
    ├── 按 HF→Megatron 映射 + 融合规则组装 entry
    │     - fused_qkv: 聚合 q/k/v 三层 → 1 entry (含 split_sizes=[4096,1024,1024])
    │     - plain:     o → 1 entry
    │     - fused_gate_up: 聚合 gate/up → 1 entry (split_sizes=[9728,9728])
    │     - plain:     down → 1 entry
    │     每层 4 entry × 36 层 = 144 entry
    │
    ├── 校验: V_k.shape[0] 与 in_dim 对齐
    └── torch.save(subspace_dict, subspace_dict_qwen3_4b.pt)   (bf16, 预计 1-2 GB)
```

离线产物链路上**唯一交付物**：`subspace_dict_qwen3_4b.pt`。
其他 `.jsonl` / `.pt` 是中间产物，训练不依赖。

---

## 2. 在线训练时序（每次 Stage 2 启动）

```
shell: stage_2_instruct_llava_ov_4b.sh
    │   export TEXT_SUBSPACE_PATH=/path/to/subspace_dict.pt   (可选)
    ▼
torchrun → train.py
    │
    ├── parse_train_args()
    │     args.text_subspace_path = "..."   (若 env 未设则 None)
    │
    ├── build_model_trainer(args)
    │     → MegatronTrainer (不 aware subspace 逻辑, 零侵入)
    │
    └── MegatronTrainer.train()
            │
            └── pretrain()   ← fork 版 (training_utils.py)
                │
                ├── initialize_aiak_megatron()
                │
                ├── setup_model_and_optimizer()   ← 核心注入点
                │     │
                │     ├── model = get_model(model_provider)
                │     │     for each chunk:
                │     │       chunk = Float16Module(DDP(GPTModel))
                │     │
                │     ├── optimizer = get_megatron_optimizer(...)
                │     │     → DistributedOptimizer 实例, main_grad 机制就绪
                │     │
                │     ├── if args.load: load_checkpoint(model, optimizer, ...)
                │     │     → 参数/优化器状态从 release-format ckpt 恢复
                │     │
                │     └── if args.text_subspace_path:                    ◄─┐
                │           attach_text_subspace_projection(             │ │ 新增
                │               model, optimizer, args.text_subspace_path │ │
                │           )                                           ◄─┘
                │                │
                │                ├── _assert_parallel_layout(args)
                │                │     TP=PP=CP=1, use_distributed_optimizer=True
                │                │
                │                ├── subspace_dict = torch.load(path, map_cpu)
                │                │     校验 _meta 存在
                │                │
                │                ├── targets = _collect_targets(model, subspace_dict)
                │                │     for chunk in model:
                │                │       bare = unwrap_model(chunk)  # 穿透 DDP/Float16
                │                │       for name, param in bare.named_parameters():
                │                │         if name in subspace_dict:
                │                │           _validate_shape(...)
                │                │           targets.append((name, param, entry))
                │                │
                │                ├── for _, _, entry in targets:
                │                │     entry["V_k"*].cuda()    # bf16 V_k 上 GPU
                │                │
                │                ├── _wrap_optimizer_step(optimizer, targets)
                │                │     原 step 保存 → patched_step 覆盖
                │                │     optimizer._text_subspace_wrapped = True
                │                │
                │                └── rank0 log:
                │                      "Matched 144/144 entries, V_k on GPU: X MB"
                │
                ├── build_train_valid_test_data_iterators(...)
                │
                └── train(forward_step_func, model, optimizer, ...):   # training loop
                     │
                     for iter in range(train_iters):
                       │
                       ├── forward_backward_step(model, data_iterator)
                       │     微批次循环:
                       │       forward → loss
                       │       backward → 梯度累积到 param.main_grad (fp32)
                       │
                       ├── DP all-reduce (在 DistOpt.step 内部前段完成)
                       │     此后 param.main_grad 是全局平均梯度
                       │
                       ├── optimizer.step()   ← patched_step
                       │     │
                       │     ├── _apply_projection(targets):       ← 新增
                       │     │     for (name, param, entry) in targets:
                       │     │       if param.main_grad is None: continue
                       │     │       _project_one(param.main_grad, entry)
                       │     │         plain:
                       │     │           G ← G − (G @ V) @ V.T     (in-place sub_)
                       │     │         fused:
                       │     │           G_chunks = G.split(split_sizes, dim=0)  # views
                       │     │           for G_i, V_i in zip(...):
                       │     │             G_i ← G_i − (G_i @ V_i) @ V_i.T
                       │     │
                       │     └── original_step(*args, **kwargs)   ← Megatron 原有
                       │           根据 main_grad 更新参数 (AdamW, bf16 master)
                       │
                       ├── opt_param_scheduler.step()
                       ├── 梯度清零 (Megatron 内部)
                       │
                       └── (log / save_checkpoint / eval)
```

---

## 3. 关键时序不变量

| 时序点 | 不变量 | 违反后果 |
|---|---|---|
| `attach_text_subspace_projection` 进入 | `args.text_subspace_path is not None` | 由 `training_utils.py` 的 `if` 保证 |
| attach 进入时 | 并行布局 TP=PP=CP=1 | `NotImplementedError` |
| attach 进入时 | 模型已构建 + ckpt 已加载 + 优化器已建 | 在 `setup_model_and_optimizer` 末尾调用保证 |
| `_collect_targets` 返回 | `len(targets) > 0` | `RuntimeError` 指示 dict/模型不匹配 |
| `_validate_shape` 内 | `V_k.shape[0] == param.shape[1]` | `AssertionError` 附参数名 |
| `_validate_shape` 内（融合层） | `sum(split_sizes) == param.shape[0]` | `AssertionError` 附参数名 |
| 每次 `patched_step` 进入 | `optimizer.main_grad` 已完成 DP all-reduce | 由 DistOpt 自身保证 |
| `_project_one` 返回 | `param.main_grad` 已 in-place 更新 | `G.split` 返回 view, `sub_` 直接落回 |
| `patched_step` 调用链 | `original_step` 必然被调用 | `try/except` 不吞 step 内部异常 |
| 多次调用 `attach` | 幂等 | `_text_subspace_wrapped` 哨兵 |

---

## 4. 投影在 DP 中的一致性证明（简）

设 DP world = N，rank_i 本地梯度为 `G_i`：

1. DistOpt.step 入口前：`main_grad_i = (1/N) Σ_j G_j` （已 all-reduced，每张卡相同）
2. `_apply_projection` 在 `original_step` 之前：每 rank 读到相同的 `main_grad`，用相同的 `V_k`
3. 投影结果：`main_grad' = main_grad − (main_grad @ V_k) @ V_k.T`，每 rank 结果仍相同
4. `original_step` 用相同梯度更新相同初值的参数 → 参数保持跨 rank 一致

∴ 加投影后 DP 一致性不变。

**反证**：若在 `forward_backward` 内、DP all-reduce 前（即在 `param.grad` 或未 reduce 的 `main_grad` 上）投影，由于各 rank 的 `G_i` 不同但 `V_k` 相同，投影后的 `(1/N) Σ_j (G_j − (G_j V)V^T) = (1/N) Σ G_j − ((1/N) Σ G_j V)V^T`，线性性保持，也一致。**但 Megatron 的 main_grad 更新在 TE backward 回调里，不经过 `.grad` 路径**，所以 PyTorch `register_hook` 无法在 reduce 前触发 → 选择 reduce 后的 step-pre-hook 更可靠。

---

## 5. trainable-modules 与投影作用域的交互

Stage 2 默认：`--trainable-modules language_model adapter vision_model`。

| 模块类 | requires_grad | 在 subspace_dict 中 | 行为 |
|---|---|---|---|
| LLM `decoder.layers.*.self_attention.linear_qkv.weight` | True | ✅ | 投影 |
| LLM `decoder.layers.*.self_attention.linear_proj.weight` | True | ✅ | 投影 |
| LLM `decoder.layers.*.mlp.linear_fc1.weight` | True | ✅ | 投影 |
| LLM `decoder.layers.*.mlp.linear_fc2.weight` | True | ✅ | 投影 |
| LLM `decoder.layers.*.*.layer_norm.weight` | True | ❌ | 自由更新 |
| LLM `decoder.embedding.word_embeddings.weight` | True | ❌ | 自由更新 |
| LLM `output_layer.weight` (如未 tied) | True | ❌ | 自由更新 |
| vision_model.* | True | ❌ | 自由更新 |
| adapter.* | True | ❌ | 自由更新 |

**tied embedding**：Qwen3-4B `tie_word_embeddings=true`，LM head 与 embedding 共享权重；两者都不在字典中，无影响。

---

## 6. 失败情境的时序

**情境 A：路径不存在**
```
setup_model_and_optimizer → load_checkpoint OK
  → attach_text_subspace_projection
    → _load_subspace → FileNotFoundError
  → 异常冒泡至 torchrun → 进程退出
```
不进入 training loop；日志显示完整路径方便排查。

**情境 B：模型版本与 dict 不匹配（参数名变化）**
```
attach → _collect_targets → 0 targets → RuntimeError
  ("No parameters matched. First 5 dict keys: [...]. Model uses different naming?")
```

**情境 C：shape 不对齐（dict 用 Qwen3-8B 生成, 跑 4B 训练）**
```
attach → _validate_shape → AssertionError
  ("decoder.layers.0.self_attention.linear_qkv.weight: V_q.shape[0]=4096 != param.in=2560")
```

**情境 D：中途某 step main_grad 为 None**
```
patched_step → _apply_projection
  → 该 entry continue, 其他正常投影
  → original_step 正常执行
```
（不应发生，但容错）

**情境 E：NaN / Inf 在投影后出现**
```
patched_step → _apply_projection 后 main_grad 含 NaN
  → original_step → Megatron check_for_nan_in_loss_and_grad 触发
  → 若 --ignore-forward-steps 配置允许, 跳过此 step; 否则训练终止
```
（由 Megatron 原有机制处理，不在本模块责任内）
