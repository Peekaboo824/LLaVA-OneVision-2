# 02 — 训练集成（LLaVA-OneVision-2/aiak_training_llm/）

## 0. 改动落点

| 文件 | 改动 | 行级位置（参考） |
|---|---|---|
| `aiak_training_llm/train/arguments.py` | 新增 `--text-subspace-path` 参数 | `extra-training` 组内追加（约 357 行附近） |
| `aiak_training_llm/train/text_subspace_projection.py` | **新增** | 整个文件 |
| `aiak_training_llm/train/training_utils.py` | `setup_model_and_optimizer()` 末尾追加条件 attach | 见下 §3 |
| `examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_4b.sh` | 环境变量透传 | TRAINING_ARGS 区域 |

**未改**：`aiak_megatron/` 任何文件、`sft_llavaov_1_5_vl.py`、`megatron_trainer.py`、其他 trainer/provider。

---

## 1. `arguments.py` 新增参数

在 `extra-training` 参数组内追加：

```python
group.add_argument('--text-subspace-path', type=str, default=None,
                   help='Path to precomputed Fisher-weighted SVD subspace dict '
                        '(.pt produced by extract_subspace.py). When set, gradient '
                        'updates on LLM Q/K/V/O/Gate/Up/Down linear layers are '
                        'projected onto the orthogonal complement of the text-important '
                        'subspace. Requires TP=PP=CP=1, --use-distributed-optimizer, --bf16.')
```

不与任何现有参数互斥。`default=None` → 未传时整条链路不激活。

---

## 2. `text_subspace_projection.py` 新增模块

### 2.1 公共 API（唯一外部入口）

```python
def attach_text_subspace_projection(
    model: list,            # Megatron get_model() 返回的 model_chunks 列表
    optimizer,              # MegatronOptimizer 实例
    subspace_path: str,
) -> None:
    """
    Load subspace dict, match params, wrap optimizer.step to inject
    main_grad projection before each real step.

    Idempotent: re-calling on the same optimizer is a no-op.
    """
```

### 2.2 内部函数职责（SRP 拆解）

| 函数 | 职责 | 失败行为 |
|---|---|---|
| `_assert_parallel_layout()` | TP=PP=CP=1，`--use-distributed-optimizer` 已开 | `NotImplementedError` |
| `_load_subspace(path)` | `torch.load(map_location='cpu')` + `_meta` 字段校验 | `FileNotFoundError` / `RuntimeError` |
| `_collect_targets(model, subspace_dict)` | 穿透 wrapper 取 named_parameters，按字典键匹配 | 0 匹配 → `RuntimeError` |
| `_move_entry_to_cuda(entry)` | 把 entry 中的 tensor `.cuda()` | 直传错误 |
| `_apply_projection(registry)` | 对每个 (param, entry) 投影其 main_grad | 跳过 main_grad is None |
| `_wrap_optimizer_step(optimizer, registry)` | monkey-patch `optimizer.step` | 幂等哨兵 `_text_subspace_wrapped` |

### 2.3 关键实现细节

**穿透 wrapper**：

```python
from megatron.training.utils import unwrap_model

def _collect_targets(model_chunks, subspace_dict):
    targets = []  # list of (name, param, entry)
    unmatched_keys = set(k for k in subspace_dict if not k.startswith("_"))
    for chunk in model_chunks:
        bare = unwrap_model(chunk)
        for name, param in bare.named_parameters():
            if name in subspace_dict:
                entry = subspace_dict[name]
                _validate_shape(name, param, entry)
                targets.append((name, param, entry))
                unmatched_keys.discard(name)
    if not targets:
        raise RuntimeError(
            f"No parameters matched in subspace dict. "
            f"First 5 dict keys: {list(subspace_dict.keys())[:5]}. "
            f"Model uses different naming?"
        )
    return targets, unmatched_keys
```

**形状校验**：

```python
def _validate_shape(name, param, entry):
    # param.shape = [out, in]; V_k.shape = [in, k]
    in_dim = param.shape[1]
    if entry["type"] == "plain":
        assert entry["V_k"].shape[0] == in_dim, \
            f"{name}: V_k.shape[0]={entry['V_k'].shape[0]} != param.in={in_dim}"
    else:
        for V_i in entry["V_k_list"]:
            assert V_i.shape[0] == in_dim, \
                f"{name}: fused V_k.shape[0]={V_i.shape[0]} != param.in={in_dim}"
        # split_sizes 沿 out 维, 求和应等于 param.shape[0]
        assert sum(entry["split_sizes"]) == param.shape[0], \
            f"{name}: sum(split_sizes)={sum(entry['split_sizes'])} != param.out={param.shape[0]}"
```

**投影核**（in-place sub_ 到 main_grad）：

```python
@torch.no_grad()
def _project_one(main_grad, entry):
    """In-place project main_grad. main_grad: [out, in], typically fp32."""
    if entry["type"] == "plain":
        V = entry["V_k"]                     # [in, k], bf16 on GPU
        Vf = V.to(main_grad.dtype)
        # (G @ V) → [out, k]; @ V.T → [out, in]
        main_grad.sub_(torch.mm(torch.mm(main_grad, Vf), Vf.t()))
    else:
        # fused_qkv / fused_gate_up
        chunks = main_grad.split(entry["split_sizes"], dim=0)
        # chunks 是 view; 对 view 做 sub_ 直接落回原 main_grad
        for G_i, V_i in zip(chunks, entry["V_k_list"]):
            Vf = V_i.to(main_grad.dtype)
            G_i.sub_(torch.mm(torch.mm(G_i, Vf), Vf.t()))

def _apply_projection(registry):
    for _, param, entry in registry:
        if param.main_grad is None:
            continue   # 该参数本 step 没有梯度（罕见，通常 trainable_modules 控制后不会发生）
        _project_one(param.main_grad, entry)
```

**optimizer.step wrap**（幂等）：

```python
def _wrap_optimizer_step(optimizer, registry):
    if getattr(optimizer, "_text_subspace_wrapped", False):
        return
    original_step = optimizer.step

    def patched_step(*args, **kwargs):
        _apply_projection(registry)
        return original_step(*args, **kwargs)

    optimizer.step = patched_step
    optimizer._text_subspace_wrapped = True
    optimizer._text_subspace_registry = registry   # 防 GC + 调试可见
```

### 2.4 attach 主流程伪代码

```python
def attach_text_subspace_projection(model, optimizer, subspace_path):
    args = get_args()
    _assert_parallel_layout(args)

    subspace_dict = _load_subspace(subspace_path)   # CPU
    targets, unmatched = _collect_targets(model, subspace_dict)

    # V_k 一次性搬到 GPU; 替换 entry 内的 tensor
    total_bytes = 0
    for _, _, entry in targets:
        _move_entry_to_cuda(entry)
        total_bytes += _entry_bytes(entry)

    _wrap_optimizer_step(optimizer, targets)

    if torch.distributed.get_rank() == 0:
        print(f"[TextSubspace] Matched {len(targets)} entries, "
              f"unmatched dict keys: {len(unmatched)}")
        print(f"[TextSubspace] V_k on GPU: {total_bytes / 1024**2:.1f} MB")
        if unmatched:
            print(f"[TextSubspace] First 5 unmatched: {list(unmatched)[:5]}")
```

---

## 3. `training_utils.py` 改动

在 `setup_model_and_optimizer()` 的末尾、`return` 之前追加：

```python
# (现有逻辑结束, 紧邻 return 之前)
if getattr(args, "text_subspace_path", None):
    from .text_subspace_projection import attach_text_subspace_projection
    attach_text_subspace_projection(
        model=model,
        optimizer=optimizer,
        subspace_path=args.text_subspace_path,
    )

return model, ema, optimizer, opt_param_scheduler
```

**为什么在这里**：
- 已完成 `get_model()` → 模型完全构建（含 Float16Module/DDP 包装器）
- 已完成 `load_checkpoint()` → 参数恢复到训练初值
- 优化器已构建（`optimizer.param_groups` 已固化）
- 还未进入 training loop → 第一次 `optimizer.step` 前完成 wrap

**为什么 import 写在函数体内**：
- 零侵入：未传 `--text-subspace-path` 时不触发 import
- 避免循环依赖风险

---

## 4. `stage_2_instruct_llava_ov_4b.sh` 改动

在 `TRAINING_ARGS=(...)` 之前或之后追加：

```bash
# === Text subspace projection (optional) ===
TEXT_SUBSPACE_PATH="${TEXT_SUBSPACE_PATH:-}"
TEXT_SUBSPACE_ARGS=()
if [ -n "$TEXT_SUBSPACE_PATH" ]; then
    TEXT_SUBSPACE_ARGS+=(--text-subspace-path "$TEXT_SUBSPACE_PATH")
    echo "[stage_2] text-subspace projection enabled: $TEXT_SUBSPACE_PATH"
fi
```

在 torchrun 命令中追加 `"${TEXT_SUBSPACE_ARGS[@]}"`：

```bash
PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    ${IMG_ARGS:+${IMG_ARGS[@]}} \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${TEXT_SUBSPACE_ARGS[@]}" \
    2>&1 | tee "$logfile"
```

启用方式：

```bash
export TEXT_SUBSPACE_PATH=/path/to/subspace_dict_qwen3_4b.pt
bash examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_4b.sh 1 1 32768 1 224 3500
```

不导出该变量时，`TEXT_SUBSPACE_ARGS` 为空数组，行为与原脚本完全一致。

---

## 5. 与现有训练栈的交互检查

| 现有特性 | 兼容性 | 备注 |
|---|---|---|
| `--use-distributed-optimizer` | **依赖** | 投影作用于 `main_grad`，须 DistOpt 路径 |
| `--bf16` | **依赖** | V_k 以 bf16 存盘；`main_grad` 为 fp32 自动 cast |
| `--trainable-modules language_model adapter vision_model` | 兼容 | 仅 LLM 内 7 类层匹配字典；vision/adapter 自由更新 |
| `--recompute-granularity full` | 兼容 | 投影与 recompute 互不影响 |
| `--attention-backend flash` | 兼容 | 投影发生在 backward 完成后 |
| `--ckpt-format torch` / 加载 release 格式 ckpt | 兼容 | attach 在 `load_checkpoint` 之后 |
| 多机 DP | 兼容 | 所有 rank 加载同一 V_k；投影发生在 reduce 后，DP 一致 |
| `--moe-use-upcycling` | 不验证 | 当前模型非 MoE，不在范围 |
| TP > 1 / PP > 1 / CP > 1 | **不支持** | attach 入口 `NotImplementedError` |
| LoRA / PEFT | 不验证 | 当前栈无 LoRA 路径，不在范围 |

## 6. 显存与性能预估

**显存增量**（attach 完成后稳定）：
- V_k bf16，按 spec §01.3.8，估 **1–2 GB**（实际以 `[Summary]` 为准）
- 投影时临时 `(G @ V)` 中间张量：`[out, k]` fp32，最大 `9728 * 4864 * 4 ≈ 180 MB`，按层串行执行 → 峰值仅 ~200 MB

**每 step 投影开销**：
- 144 entry × 2 次 GEMM ≈ 数十毫秒（A100 上 < 50 ms）
- 相比 32k seq-len 的 forward+backward（秒级）可忽略
