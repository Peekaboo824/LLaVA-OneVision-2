"""Extract the EWC anchor (theta*_A) from Qwen3-4B-Instruct.

Given a Fisher .pt produced by ``compute_fisher_qwen3_4b.py``, this script:
  1. opens the Fisher dict and collects every key except ``_meta``
  2. loads the source model's ``state_dict`` (lazily via ``safetensors``-aware
     HF loader; the model itself is **not** instantiated)
  3. copies each matching tensor into a new dict, in fp32, on CPU
  4. saves the result as a ``.pt`` whose key set equals the Fisher key set

The resulting anchor is intended to be consumed by ``QwenSFTTrainer`` together
with the matching Fisher when ``--ewc_lambda > 0``.

Why fp32: we add ``lambda * F * (theta - theta*)^2`` to the loss every step;
keeping the anchor in fp32 avoids bf16 round-off when the difference is small.
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, required=True,
                        help="HF model dir, e.g. Qwen3-4B-Instruct-2507")
    parser.add_argument("--fisher-path", type=str, required=True,
                        help="Fisher .pt produced by compute_fisher_qwen3_4b.py")
    parser.add_argument("--output", type=str, default="anchor_dict_qwen3_4b.pt",
                        help="Output .pt path (fp32 anchor weights, CPU)")
    args = parser.parse_args()

    print(f"[extract_anchor] loading Fisher from {args.fisher_path}")
    fisher = torch.load(args.fisher_path, map_location="cpu")
    fisher_keys = [k for k in fisher.keys() if k != "_meta"]
    print(f"[extract_anchor] Fisher contains {len(fisher_keys)} target tensors")
    if not fisher_keys:
        print("[extract_anchor] ERROR: Fisher dict has no tensors; aborting")
        sys.exit(1)

    # Load model state_dict only — we don't need to run the model.
    # AutoModelForCausalLM is the simplest way to get correctly tied weights
    # (lm_head.weight is the same tensor as model.embed_tokens.weight when
    # tie_word_embeddings=True). Loading on meta would save memory but make
    # tie-detection unreliable, so we load to CPU bf16 first then upcast.
    print(f"[extract_anchor] loading model state from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    state = dict(model.state_dict())
    del model

    missing = [k for k in fisher_keys if k not in state]
    extra = [k for k in state.keys() if k in fisher_keys]  # for stats only
    if missing:
        print(f"[extract_anchor] ERROR: {len(missing)} Fisher keys missing in model "
              f"state_dict, e.g. {missing[:5]}")
        sys.exit(2)

    anchor = {}
    for k in fisher_keys:
        # Detach + clone in fp32 on CPU so the resulting file is self-contained.
        anchor[k] = state[k].detach().to(dtype=torch.float32, device="cpu").clone()

    anchor["_meta"] = {
        "model_path": args.model_path,
        "fisher_path": args.fisher_path,
        "num_tensors": len(fisher_keys),
        "source_meta": fisher.get("_meta", {}),
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(anchor, args.output)
    print(f"✅ saved {len(fisher_keys)} anchor tensors to {args.output}")

    # Quick stats so the user can sanity-check
    total_numel = sum(t.numel() for k, t in anchor.items() if k != "_meta")
    total_bytes = total_numel * 4  # fp32
    print(f"[extract_anchor] total params: {total_numel:,} "
          f"(~{total_bytes / 1024**3:.2f} GB fp32)")


if __name__ == "__main__":
    sys.exit(main())
