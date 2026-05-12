# 00 — 顶层架构与契约

## 1. 两仓两阶段架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  MLLM-Pure-Text-Preservation/   (预处理 - 离线, 单 GPU)               │
│                                                                       │
│  build_fisher_data.py  ──► fisher_anchor_dataset.jsonl                │
│  compute_fisher.py     ──► fisher_dict_qwen3_4b.pt                    │
│  extract_subspace.py   ──► subspace_dict_qwen3_4b.pt   ◄── 单一契约   │
└────────────────────────────────────────────────────────────┬─────────┘
                                                             │ .pt
┌────────────────────────────────────────────────────────────▼─────────┐
│  LLaVA-OneVision-2/  (训练 - Megatron, 多 GPU DP)                     │
│                                                                       │
│  stage_2_instruct_llava_ov_4b.sh                                      │
│     env: TEXT_SUBSPACE_PATH=...                                       │
│  arguments.py            : + --text-subspace-path                      │
│  training_utils.py       : setup_model_and_optimizer 内 attach        │
│  text_subspace_projection.py (NEW)                                    │
│     attach_text_subspace_projection(model, optimizer, path)           │
│     - assert TP=PP=CP=1                                               │
│     - load subspace_dict, V_k → GPU (bf16)                            │
│     - wrap optimizer.step: 投影 main_grad, 再调原 step                │
└──────────────────────────────────────────────────────────────────────┘
```

## 2. `subspace_dict_qwen3_4b.pt` 契约

顶层是 dict，键为 **Megatron 裸模型参数全名**（无 wrapper 前缀），值是三类 entry 之一，外加 `_meta`。

### 2.1 entry 类型

**plain**（对应 `linear_proj`、`linear_fc2`）
```python
{
  "type": "plain",
  "V_k": Tensor[in, k],   # bf16, 行正交: V_k.T @ V_k ≈ I_k
}
```

**fused_qkv**（对应 `linear_qkv`，沿 out 维拼接 Q/K/V 三块）
```python
{
  "type": "fused_qkv",
  "V_k_list": [V_q, V_k, V_v],   # 各 [in, k_i], bf16
  "split_sizes": [q_dim, k_dim, v_dim],   # 沿 out 维; 对 4B: [4096, 1024, 1024]
}
```

**fused_gate_up**（对应 `linear_fc1`，沿 out 维拼接 Gate/Up 两块）
```python
{
  "type": "fused_gate_up",
  "V_k_list": [V_gate, V_up],
  "split_sizes": [ffn, ffn],   # 对 4B: [9728, 9728]
}
```

### 2.2 `_meta` 字段

```python
{
  "_meta": {
    "model": "Qwen3-4B-Instruct",
    "tau": 0.95,
    "k_cap_ratio": 0.5,
    "norm": "per-layer-minmax",
    "fisher_anchor_size": 2048,
    "fisher_anchor_recipe": {"math":550,"mbpp":300,"logiqa":450,"halueval":348,"dolly":400},
    "num_layers": 36,
    "dtype": "bfloat16",
  }
}
```

### 2.3 命名规范

Megatron 参数全名形如：
```
decoder.layers.{L}.self_attention.linear_qkv.weight   # L ∈ [0, 35]
decoder.layers.{L}.self_attention.linear_proj.weight
decoder.layers.{L}.mlp.linear_fc1.weight
decoder.layers.{L}.mlp.linear_fc2.weight
```

每层 4 个 entry × 36 层 = **144 个 entry**（不含 `_meta`）。

## 3. 投影数学

对每个目标参数的梯度 `G ∈ R^{out × in}`：

```
plain:          G ← G − (G @ V_k) @ V_kᵀ                where V_k ∈ R^{in × k}

fused_qkv:      G = concat([G_q, G_k, G_v], dim=0)       (按 split_sizes 切)
                G_i ← G_i − (G_i @ V_i) @ V_iᵀ           for i in {q, k, v}
                G ← concat([G_q, G_k, G_v], dim=0)

fused_gate_up:  类似 fused_qkv, 2 块
```

**关键**：`G.split(split_sizes, dim=0)` 返回 view，对 view 做 `sub_` 直接落回原 `main_grad`。

**精度**：`V_k` 以 bf16 存盘 → GPU 上 cast 到与 `main_grad` 一致的 dtype（Megatron `--bf16 --use-distributed-optimizer` 下 `main_grad` 为 fp32）。

## 4. 零侵入语义

- 未传 `--text-subspace-path` → `text_subspace_projection.py` 不被 import，`attach_*` 不被调用，零开销。
- 传了 `--text-subspace-path` 但路径无效 → 在 attach 入口立即 `FileNotFoundError`，不进训练 loop。

## 5. 仓库改动一览（diff 边界）

| 仓库 / 文件 | 改动类型 | 责任 |
|---|---|---|
| `MLLM-Pure-Text-Preservation/build_fisher_data.py` | 新增 | 数据集采样 + chat-template + JSONL |
| `MLLM-Pure-Text-Preservation/compute_fisher_qwen3_4b.py` | 新增 | HF 模型 forward+backward，累积 g² |
| `MLLM-Pure-Text-Preservation/extract_subspace_qwen3_4b.py` | 新增 | Fisher 加权 + SVD + 融合 + Megatron 命名 |
| `LLaVA-OneVision-2/aiak_training_llm/train/arguments.py` | 加 1 参数 | `--text-subspace-path` |
| `LLaVA-OneVision-2/aiak_training_llm/train/text_subspace_projection.py` | 新增 | attach + 投影实现 |
| `LLaVA-OneVision-2/aiak_training_llm/train/training_utils.py` | 改 `setup_model_and_optimizer` 末尾 | 条件性调用 `attach_text_subspace_projection` |
| `LLaVA-OneVision-2/examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_4b.sh` | 加环境变量透传 | `TEXT_SUBSPACE_PATH` → `--text-subspace-path` |

未改：`aiak_megatron/`（vendored）、`sft_llavaov_1_5_vl.py`、`megatron_trainer.py`、其他模型 trainer。
