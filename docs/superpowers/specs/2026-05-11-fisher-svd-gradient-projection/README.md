# Fisher 加权 SVD 梯度投影 — 设计规格

> 解决 LLaVA-OV-1.5（Qwen3-4B-Instruct backbone）经视觉指令微调后的纯文本灾难性遗忘。
> 在 LLM 7 类目标线性层（Q/K/V/O/Gate/Up/Down）的梯度上施加正交投影，将更新限制在「文本重要子空间」的正交补内。

## 目录

- [00 — 顶层架构与契约](./00-overview.md)
- [01 — 预处理流水线（MLLM-Pure-Text-Preservation/）](./01-preprocessing.md)
- [02 — 训练集成（LLaVA-OneVision-2/aiak_training_llm/）](./02-training-integration.md)
- [03 — 数据流与执行时序](./03-data-flow-and-timing.md)

## 状态

| 项 | 决定 |
|---|---|
| Hook 注入位置 | 改本仓库 fork 版 `setup_model_and_optimizer`，attach 在 `load_checkpoint` 之后、首次 `optimizer.step` 之前 |
| Fisher 加权归一化 | 逐层 min-max → `[0,1]`，min/max 跨整层全元素 |
| 融合层 `split_sizes` | 存在 `subspace_dict` 的融合层 entry 内 |
| 投影作用范围 | 仅 `subspace_dict` 中出现的参数（按 Megatron 全名匹配） |
| Fisher 数据集 | MATH 550 / MBPP 300 / LogiQA 450 / HaluEval 348 / Dolly 400 = 2048 |
| 数据集加载方式 | HF Hub + `HF_DATASETS_CACHE=/vepfs-mlp2/.../datasets_cache/` |
| 旧脚本处理 | 共存、不删除、不调用 |
| SVD 阈值 / k 上限 | τ = 0.95 且 `k ≤ 0.5·in_dim` |
| V_k 存盘精度 | bf16（与 `--bf16` 训练对齐） |
| V_k 训练时驻留位置 | GPU（attach 时一次性 `.cuda()`） |
| DP 一致性 | 所有 rank 加载同一 V_k；投影作用于 DP all-reduce 后的 `main_grad` |

## 关键约束

- TP = PP = CP = 1（不支持张量/流水/上下文并行）
- 必须 `--use-distributed-optimizer` + `--bf16`
- 零侵入：未传 `--text-subspace-path` 时，整条链路不激活
- 不改 vendored `aiak_megatron/`，所有改动落在 `aiak_training_llm/`

## 目标模型规格摘要（Qwen3-4B-Instruct）

```
num_layers       = 36
hidden_size      = 2560
ffn_hidden_size  = 9728
num_attn_heads   = 32   (q_dim = 32 * 128 = 4096)
num_kv_heads     = 8    (k_dim = v_dim = 8 * 128 = 1024)
head_dim         = 128
```

linear_qkv 沿 out 维 split_sizes = `[4096, 1024, 1024]`
linear_fc1 沿 out 维 split_sizes = `[9728, 9728]`
