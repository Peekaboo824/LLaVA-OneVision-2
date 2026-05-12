# 01 — 预处理流水线（MLLM-Pure-Text-Preservation/）

## 0. 概述

三个新脚本，按顺序运行一次即可产出 `subspace_dict_qwen3_4b.pt`。
旧脚本（`build_fisher_dataset.py` / 旧版 `compute_fisher.py` / `extract_baseline_subspace.py` / 旧版 `extract_subspace.py` 等）保留共存，不调用、不修改。

> **命名说明**：为避免与旧版同名脚本冲突，新脚本命名为 `compute_fisher_qwen3_4b.py` 与 `extract_subspace_qwen3_4b.py`；`build_fisher_data.py` 为原创名，无冲突。

```
build_fisher_data.py             →  fisher_anchor_dataset.jsonl   (~ 2048 行)
                                      │
compute_fisher_qwen3_4b.py       →  fisher_dict_qwen3_4b.pt       (~ 数 GB fp32)
                                      │
extract_subspace_qwen3_4b.py     →  subspace_dict_qwen3_4b.pt     (目标 ≈ 380MB bf16)
```

所有脚本：
- 入口为 `if __name__ == "__main__":` + `argparse`
- 默认走 `HF_DATASETS_CACHE=/vepfs-mlp2/c20250505/240906016/jjy/datasets_cache`
- 仅单 GPU；脚本内部不做分布式。

---

## 1. `build_fisher_data.py`

### 1.1 职责

从 5 个 HF 数据集采样 2048 条单轮对话，按 Qwen3 对话模板组装 `messages`，写出 JSONL。
**不**做 tokenize，**不**做 label mask（留到 `compute_fisher.py`）。

### 1.2 数据集与字段映射

| source | HF id | split / config | 数量 | user 字段构造 | assistant 字段构造 |
|---|---|---|---|---|---|
| `math` | `lighteval/MATH` | `train` (all subjects 合并) | 550 | `problem` | `solution` |
| `mbpp` | `google-research-datasets/mbpp` | `sanitized`, `train` | 300 | `prompt + "\n\nTests:\n" + "\n".join(test_list)` | `code` |
| `logiqa` | `lucasmccabe/logiqa` | `train` | 450 | `context + "\n\nQuestion: " + query + "\n\nOptions:\n" + 枚举 options` | `options[correct_option]` |
| `halueval` | `pminervini/HaluEval` | `qa`, `data` (该数据集仅 1 split) | 348 | `"Knowledge: " + knowledge + "\n\nQuestion: " + question` | `right_answer` |
| `dolly` | `databricks/databricks-dolly-15k` | `train` | 400 | `instruction + ("\n\n" + context if context else "")` | `response` |

**采样规则**：固定 `--seed 42`，每集做 `random.sample(indices, k)`。若数据集长度不足 k，打印 warning 并取全集（不报错）。

### 1.3 输出格式

`fisher_anchor_dataset.jsonl`，每行：
```json
{"source": "math", "messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

### 1.4 CLI

```bash
python build_fisher_data.py \
    --output fisher_anchor_dataset.jsonl \
    --seed 42 \
    --cache-dir /vepfs-mlp2/c20250505/240906016/jjy/datasets_cache
```

可选 override：`--counts math=550 mbpp=300 logiqa=450 halueval=348 dolly=400`。

### 1.5 日志要求

- 每个数据集加载完打印「loaded N samples → sampled K samples」
- 末尾打印「✅ wrote 2048 lines to <output>」
- 最后给出 source 分布表

---

## 2. `compute_fisher.py`

### 2.1 职责

加载 HF 格式 Qwen3-4B-Instruct（bf16），对 `fisher_anchor_dataset.jsonl` 每条样本：
1. 拼 chat-template，构造 `input_ids` 与 `labels`（**prompt 部分 mask 为 -100**）
2. `loss = model(input_ids, labels=labels).loss`
3. `loss.backward()`
4. 累加 `param.grad ** 2` 到 `accum[full_name]`（fp32 累积）
5. `model.zero_grad(set_to_none=True)`

**不归一化**（不除以 N，不归一到 [0,1]）—— 归一化留到 `extract_subspace.py`。

### 2.2 Label Masking 实现

```python
# 给定 messages = [{"role":"user",...}, {"role":"assistant",...}]
full = processor.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=False)
prompt = processor.apply_chat_template(messages[:1], tokenize=False,
                                       add_generation_prompt=True)

enc_full   = processor(full,   return_tensors="pt", add_special_tokens=False)
enc_prompt = processor(prompt, return_tensors="pt", add_special_tokens=False)

input_ids = enc_full.input_ids.to(device)
labels    = input_ids.clone()
prompt_len = enc_prompt.input_ids.shape[1]
labels[:, :prompt_len] = -100      # 关键: 仅对 assistant 部分计算 loss
```

**前缀对齐校验**：assert `enc_full.input_ids[0, :prompt_len].tolist() == enc_prompt.input_ids[0].tolist()`。
不一致则跳过该条 + warning（极少见，多由 special token 差异引起）。

### 2.3 目标参数选择

模型加载后，遍历 `model.named_modules()`，挑出名字以下面任一后缀结尾的 `nn.Linear`：

```
self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj
mlp.gate_proj,    mlp.up_proj,      mlp.down_proj
```

记录其 `weight` 的完整参数名（如 `model.layers.0.self_attn.q_proj.weight`）。
对非目标参数：`param.requires_grad = False`（节省 backward 显存）。
对目标参数：保留 `requires_grad=True`。

### 2.4 长度处理

- `--max-length 4096` 截断（forward 节省显存；长尾不影响 Fisher 估计趋势）
- `batch_size = 1`（不同样本长度差异大，packing 不值得；保留实现简单）

### 2.5 输出

`fisher_dict_qwen3_4b.pt`：
```python
{
  "model.layers.0.self_attn.q_proj.weight": Tensor[out, in],   # fp32, g² 累积
  "model.layers.0.self_attn.k_proj.weight": ...,
  ...
  "_meta": {
    "model_path": "...",
    "num_samples_processed": 2048,
    "num_samples_skipped": K,
    "max_length": 4096,
    "target_module_suffixes": ["q_proj","k_proj","v_proj","o_proj",
                               "gate_proj","up_proj","down_proj"],
  }
}
```

### 2.6 CLI

```bash
python compute_fisher_qwen3_4b.py \
    --model-path /path/to/Qwen3-4B-Instruct \
    --dataset-path fisher_anchor_dataset.jsonl \
    --output fisher_dict_qwen3_4b.pt \
    --max-length 4096 \
    --batch-size 1
```

### 2.7 显存/时间预估

Qwen3-4B bf16 ≈ 8GB；fp32 grad ≈ 16GB；总 ≈ 25-30GB。A100-80GB 单卡 OK。
2048 样本 × 平均 1-2s/样本 ≈ 1-2 小时。

---

## 3. `extract_subspace.py`

### 3.1 职责

1. 加载 `fisher_dict_qwen3_4b.pt` + 同一份 HF 模型权重
2. 逐 HF 线性层做 Fisher 加权 → SVD → 取 V_k
3. 按 HF→Megatron 命名映射 + 融合规则组装 entry
4. 写出 `subspace_dict_qwen3_4b.pt`

### 3.2 逐层处理（fp32 计算，bf16 输出）

对每个 HF 层名 `name_hf` ∈ fisher_dict（排除 `_meta`）：

```python
F_hf = fisher_dict[name_hf].float()        # [out, in]
W_hf = state_dict[name_hf].float()         # [out, in]

# Step 1: log
F_log = torch.log10(F_hf + 1e-12)

# Step 2: 全层 min-max → [0, 1]
F_min, F_max = F_log.min(), F_log.max()
F_norm = (F_log - F_min) / (F_max - F_min + 1e-12)

# Step 3: 加权
W_tilde = F_norm * W_hf                    # [out, in]

# Step 4: SVD (右奇异向量), 不需要 full_matrices
U, S, Vh = torch.linalg.svd(W_tilde, full_matrices=False)
# Vh: [min(out,in), in]

# Step 5: 选 k
energy = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
k_thresh = int((energy >= 0.95).nonzero()[0].item()) + 1
k_cap    = max(1, W_hf.shape[1] // 2)
k        = min(k_thresh, k_cap)

# Step 6: V_k
V_k = Vh[:k, :].T.contiguous().to(torch.bfloat16)   # [in, k]
```

**纯 CPU 还是 GPU？** SVD 移到 GPU 可显著加速（Qwen3-4B 36 层 × 7 类 = 252 次 SVD）。脚本支持 `--device cuda` 默认开启。

### 3.3 HF → Megatron 命名映射

| HF 模式 | Megatron 模式 | 类型 |
|---|---|---|
| `model.layers.{L}.self_attn.q_proj.weight` | 聚合到 `decoder.layers.{L}.self_attention.linear_qkv.weight` | fused_qkv: V_q |
| `model.layers.{L}.self_attn.k_proj.weight` | ↑ | fused_qkv: V_k |
| `model.layers.{L}.self_attn.v_proj.weight` | ↑ | fused_qkv: V_v |
| `model.layers.{L}.self_attn.o_proj.weight` | `decoder.layers.{L}.self_attention.linear_proj.weight` | plain |
| `model.layers.{L}.mlp.gate_proj.weight` | 聚合到 `decoder.layers.{L}.mlp.linear_fc1.weight` | fused_gate_up: V_gate |
| `model.layers.{L}.mlp.up_proj.weight` | ↑ | fused_gate_up: V_up |
| `model.layers.{L}.mlp.down_proj.weight` | `decoder.layers.{L}.mlp.linear_fc2.weight` | plain |

### 3.4 融合层 split_sizes 计算

从 HF config 读：

```python
hidden        = cfg.hidden_size                    # 2560
ffn           = cfg.intermediate_size              # 9728
n_heads       = cfg.num_attention_heads            # 32
n_kv          = cfg.num_key_value_heads            # 8
head_dim      = cfg.head_dim                       # 128

q_dim = n_heads * head_dim                         # 4096
k_dim = n_kv * head_dim                            # 1024
v_dim = n_kv * head_dim                            # 1024
linear_qkv_split_sizes  = [q_dim, k_dim, v_dim]    # [4096, 1024, 1024]
linear_fc1_split_sizes  = [ffn,   ffn]             # [9728, 9728]
```

### 3.5 输出格式

按 `00-overview.md` §2 给出的契约写入。

**写入时校验**：
- 每个 fused entry：`V_k_list[i].shape[0] == hidden`（in 维 = hidden_size）
- plain `linear_proj`：`V_k.shape[0] == q_dim`（o_proj 的 in 维 = n_heads·head_dim = q_dim，不是 hidden）
- plain `linear_fc2`：`V_k.shape[0] == ffn`（down_proj 的 in 维 = ffn）

### 3.6 CLI

```bash
python extract_subspace_qwen3_4b.py \
    --fisher-dict fisher_dict_qwen3_4b.pt \
    --model-path /path/to/Qwen3-4B-Instruct \
    --output subspace_dict_qwen3_4b.pt \
    --tau 0.95 \
    --k-cap-ratio 0.5 \
    --device cuda
```

### 3.7 日志

逐层一行：`[L=0] q k_thresh=512 k_cap=1280 k=512 | k k_thresh=... | v k_thresh=... | o k_thresh=...`

末尾汇总：
```
[Summary] num_layers=36, num_entries=144
[Summary] V_k total bytes (bf16) = 380.2 MB
[Summary] _meta written
```

### 3.8 显存预估校验

每层 4 entry 总 V_k bytes（bf16）：

- `linear_qkv` 三块：`hidden * (k_q + k_k + k_v) * 2`
- `linear_proj`：`hidden * k_o * 2`
- `linear_fc1` 两块：`hidden * (k_gate + k_up) * 2`
- `linear_fc2`：`ffn * k_down * 2`

若所有 k 都打到 `0.5·in_dim` 上限：
- qkv: `2560 * (1280 + 1280 + 1280) * 2 = 19.7 MB/层`
- proj: `2560 * 1280 * 2 = 6.55 MB/层`
- fc1: `2560 * (4864 + 4864) * 2 = 49.8 MB/层`
- fc2: `9728 * 4864 * 2 = 94.6 MB/层`
- **每层 ≈ 170 MB → 36 层 ≈ 6.1 GB**

实际多数层 k 远小于上限（τ=0.95 通常给出 k ≈ 0.3-0.6 × in），预计落在 1-2 GB 区间。
**380 MB 预算**仅在 k 普遍偏小或仅作用部分层时可达；写脚本时不要硬卡 380 MB，跑完看实际即可。

> 实际显存以脚本输出的 `[Summary] V_k total bytes` 为准；若超过 4 GB，则训练时考虑 `--k-cap-ratio 0.3` 重跑或选择性投影。
