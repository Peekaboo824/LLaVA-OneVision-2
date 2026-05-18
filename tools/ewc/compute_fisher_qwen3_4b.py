"""Accumulate diagonal Fisher (g² sums) on Qwen3-4B-Instruct over the anchor set.

For each sample in ``fisher_anchor_dataset.jsonl``:
  1. apply chat template to build (input_ids, labels) with prompt tokens masked to -100
  2. loss = model(input_ids, labels=labels).loss
  3. loss.backward()
  4. accum[name] += param.grad ** 2  (fp32)
  5. zero_grad

Only the 7 target linear-layer weights (per transformer layer) plus the
top-level ``lm_head`` weight have requires_grad=True; all other parameters are
frozen to keep backward memory in check.

Note: Qwen3-4B has tie_word_embeddings=true, so ``lm_head.weight`` and
``model.embed_tokens.weight`` are the SAME tensor in memory. The accumulated
g² on ``lm_head.weight`` therefore also captures gradient signal flowing
through the embedding-lookup path. This is by design.

The output is **not** normalized — normalization is the job of
``extract_subspace_qwen3_4b.py``.
"""

import argparse
import json
import os
import sys
import time

import torch

DEFAULT_CACHE_DIR = "/vepfs-mlp2/c20250505/240906016/jjy/datasets_cache"

TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

# Top-level modules to include in addition to per-layer suffix matches.
# Matched by exact module name (not suffix) to avoid false positives.
TARGET_EXACT_NAMES = (
    "lm_head",
)


def _is_target_module_name(name):
    if name in TARGET_EXACT_NAMES:
        return True
    return any(name.endswith(suf) for suf in TARGET_SUFFIXES)


def _select_targets(model):
    """Freeze non-target params; return list of (full_param_name, param)."""
    targets = []
    target_module_names = set()
    for mod_name, mod in model.named_modules():
        if _is_target_module_name(mod_name) and isinstance(mod, torch.nn.Linear):
            target_module_names.add(mod_name)
    for name, param in model.named_parameters():
        # weight params live as e.g. "model.layers.0.self_attn.q_proj.weight"
        mod_name = name.rsplit(".", 1)[0]
        if mod_name in target_module_names and name.endswith(".weight"):
            param.requires_grad = True
            targets.append((name, param))
        else:
            param.requires_grad = False
    return targets


def _build_inputs(processor, messages, max_length, device):
    """Build (input_ids, labels) for one sample.

    Returns (input_ids, labels) or (None, None) if prefix alignment fails.
    """
    full = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    prompt = processor.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True,
    )
    enc_full = processor(full, return_tensors="pt", add_special_tokens=False)
    enc_prompt = processor(prompt, return_tensors="pt", add_special_tokens=False)

    full_ids = enc_full.input_ids
    prompt_ids = enc_prompt.input_ids
    prompt_len = prompt_ids.shape[1]

    # Prefix alignment check
    if full_ids.shape[1] < prompt_len:
        return None, None
    if full_ids[0, :prompt_len].tolist() != prompt_ids[0].tolist():
        return None, None

    # Truncate to max_length, but keep prompt as long as possible
    if full_ids.shape[1] > max_length:
        full_ids = full_ids[:, :max_length]
    if full_ids.shape[1] <= prompt_len:
        # Entire window is prompt; nothing to learn from
        return None, None

    input_ids = full_ids.to(device)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    return input_ids, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, required=True,
                        help="HF model dir, e.g. Qwen3-4B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="fisher_anchor_dataset.jsonl",
                        help="JSONL produced by build_fisher_data.py")
    parser.add_argument("--output", type=str, default="fisher_dict_qwen3_4b.pt",
                        help="Output .pt path (g² sums, fp32)")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Currently only 1 is supported (variable lengths)")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    if args.batch_size != 1:
        raise NotImplementedError("Only batch_size=1 is supported")

    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    device = torch.device(args.device)
    print(f"[compute_fisher] loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()  # disable dropout; we still call backward()

    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        processor = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if processor.pad_token is None:
        processor.pad_token = processor.eos_token

    targets = _select_targets(model)
    print(f"[compute_fisher] selected {len(targets)} target parameters")
    sample_target_names = [n for n, _ in targets[:3]]
    print(f"[compute_fisher] e.g. {sample_target_names}")

    accum = {
        name: torch.zeros_like(param, dtype=torch.float32, device=device)
        for name, param in targets
    }

    # Stream samples
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"[compute_fisher] loaded {len(samples)} samples from {args.dataset_path}")

    processed = 0
    skipped = 0
    t0 = time.time()
    for idx, sample in enumerate(samples):
        messages = sample["messages"]
        input_ids, labels = _build_inputs(processor, messages, args.max_length, device)
        if input_ids is None:
            skipped += 1
            continue

        # Forward + backward
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        if not torch.isfinite(loss):
            skipped += 1
            model.zero_grad(set_to_none=True)
            continue

        loss.backward()

        with torch.no_grad():
            for name, param in targets:
                if param.grad is None:
                    continue
                accum[name].add_(param.grad.float().pow_(2))

        model.zero_grad(set_to_none=True)
        processed += 1

        if (idx + 1) % 50 == 0:
            dt = time.time() - t0
            rate = (idx + 1) / max(dt, 1e-6)
            eta = (len(samples) - idx - 1) / max(rate, 1e-6)
            print(f"[compute_fisher] {idx + 1}/{len(samples)} "
                  f"processed={processed} skipped={skipped} "
                  f"elapsed={dt / 60:.1f}m eta={eta / 60:.1f}m")

    save_dict = {name: t.detach().to("cpu") for name, t in accum.items()}
    save_dict["_meta"] = {
        "model_path": args.model_path,
        "num_samples_processed": processed,
        "num_samples_skipped": skipped,
        "max_length": args.max_length,
        "target_module_suffixes": list(TARGET_SUFFIXES),
        "target_module_exact_names": list(TARGET_EXACT_NAMES),
    }
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(save_dict, args.output)

    total = time.time() - t0
    print(f"[compute_fisher] processed={processed} skipped={skipped} "
          f"total_time={total / 60:.1f}m")
    print(f"✅ saved Fisher dict to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
