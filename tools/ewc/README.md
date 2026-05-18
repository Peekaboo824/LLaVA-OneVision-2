# EWC tooling for stage-2 SFT

Builds the diagonal Fisher and anchor weights that
`QwenSFTTrainer.compute_loss` consumes when `--ewc_lambda > 0`.

Pipeline (run from this directory):

```bash
# 1. Sample 2048 pure-text dialogues across 5 HF datasets.
python build_fisher_data.py --output fisher_anchor_dataset.jsonl

# 2. Accumulate diagonal Fisher (g²) on the ORIGINAL Qwen3 LLM.
python compute_fisher_qwen3_4b.py \
    --model-path /path/to/Qwen3-4B-Instruct-2507 \
    --dataset-path fisher_anchor_dataset.jsonl \
    --output fisher_dict_qwen3_4b.pt

# 3. Extract matching anchor weights θ*_A.
python extract_anchor_qwen3_4b.py \
    --model-path /path/to/Qwen3-4B-Instruct-2507 \
    --fisher-path fisher_dict_qwen3_4b.pt \
    --output anchor_dict_qwen3_4b.pt
```

Targets covered by Fisher / anchor:

- 7 per-layer linear weights: `self_attn.{q,k,v,o}_proj` + `mlp.{gate,up,down}_proj`
- top-level `lm_head.weight`

Note: Qwen3-4B has `tie_word_embeddings=true`, so `lm_head.weight` and
`model.embed_tokens.weight` are the same tensor — the `lm_head` entry
captures gradient signal from both the output projection and the
embedding-lookup path.

Outputs are NOT mean-normalised; tune `ewc_lambda` (start ≈ 1e-3) so
that `loss_ewc / loss_lm` lands in [0.01, 0.1] during stage-2 training.
