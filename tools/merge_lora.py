#!/usr/bin/env python3
"""Merge LLaVA-OneVision-1.5 LoRA adapter into a full base model.

This script handles the specific shape of artefacts produced by
ds/src/train/train_sft.py:
  - adapter_config.json + adapter_model.safetensors    (PEFT LoRA)
  - non_lora_state_dict.bin                            (extra trainable params
                                                        kept in fp32/bf16 by
                                                        get_peft_state_non_lora_maybe_zero_3,
                                                        typically the
                                                        non-frozen merger
                                                        weights)

The repo-provided ds/src/merge_lora_weights.py hardcodes the
Qwen2VLForConditionalGeneration class for the base model, which is wrong for
LLaVAOneVision1_5; ds/merge_model.py only handles a fresh from-scratch build.
Neither fits this scenario, hence this dedicated helper.

Pipeline:
  1. Load the base LLaVAOneVision1_5 model in bf16 on CPU.
  2. Restore the trained merger weights from non_lora_state_dict.bin
     (the model's non-LoRA trainable parameters at the end of training).
  3. Wrap with PeftModel.from_pretrained and call merge_and_unload(); LoRA
     deltas get baked into the matching base weights, the adapter modules
     are removed.
  4. Save the unwrapped HF model + tokenizer + processor to the output dir,
     mirroring the layout of e.g. LLaVA-OneVision-1.5-4B-780k-Instruct-mlp-routing-60.

Usage:
  python tools/merge_lora.py \\
      --adapter_path checkpoints/stage_2_sft_lora_r32a64_4b \\
      --output_path  LLaVA-OneVision-1.5-4B-780k-Instruct-lora-r32a64
  # --base_model_path defaults to whatever adapter_config.json was trained against.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoTokenizer


def resolve_base_from_adapter(adapter_path: Path) -> str:
    cfg = json.loads((adapter_path / "adapter_config.json").read_text())
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise SystemExit(
            f"adapter_config.json at {adapter_path} has no base_model_name_or_path"
        )
    return base


def load_non_lora_state(path: Path) -> dict[str, torch.Tensor]:
    """Load and normalise the keys of non_lora_state_dict.bin.

    Keys in this file are saved through `get_peft_state_non_lora_maybe_zero_3`
    while the model is wrapped in PeftModel, so they look like
    'base_model.model.<...>'. Two more wrinkles to handle:

    1. The 'base_model.model.' prefix has to be stripped so the dict can be
       applied to a raw (un-Peft) base model.
    2. For modules that PEFT *also* attached a LoRA adapter to, PEFT renames
       the original weight 'foo.weight' to 'foo.base_layer.weight' inside
       the PeftModel. The same module on the raw base model is still called
       'foo.weight', so we have to drop the '.base_layer' infix as well.

    Concrete example (vision_lora=True, merger.mlp.0/2 are LoRA-wrapped):
      base_model.model.model.visual.merger.mlp.0.base_layer.weight
        -> model.visual.merger.mlp.0.weight
    """
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in raw:
        raw = raw["state_dict"]
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        nk = k
        if nk.startswith("base_model.model."):
            nk = nk[len("base_model.model.") :]
        elif nk.startswith("base_model."):
            nk = nk[len("base_model.") :]
        # Drop '.base_layer' infix introduced by PEFT for modules it wrapped.
        nk = nk.replace(".base_layer.", ".")
        cleaned[nk] = v
    return cleaned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter_path", required=True, help="dir with adapter_config.json + adapter_model.safetensors + non_lora_state_dict.bin")
    ap.add_argument("--output_path", required=True, help="destination directory for the merged HF model")
    ap.add_argument("--base_model_path", default=None, help="override the base model in adapter_config.json")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cpu", help="cpu | cuda:N | auto")
    ap.add_argument("--skip_non_lora", action="store_true", help="don't apply non_lora_state_dict.bin (debug)")
    args = ap.parse_args()

    adapter_path = Path(args.adapter_path).resolve()
    output_path = Path(args.output_path).resolve()
    if not adapter_path.is_dir():
        raise SystemExit(f"adapter_path is not a directory: {adapter_path}")
    if output_path.exists() and any(output_path.iterdir()):
        raise SystemExit(f"output_path already exists and is non-empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    base_model_path = args.base_model_path or resolve_base_from_adapter(adapter_path)
    print(f"[merge_lora] adapter:    {adapter_path}")
    print(f"[merge_lora] base model: {base_model_path}")
    print(f"[merge_lora] output:     {output_path}")
    print(f"[merge_lora] dtype:      {args.dtype}, device: {args.device}")

    # Make the LLaVAOneVision1_5 code (it ships in the base model dir as
    # modeling_llavaonevision1_5.py + configuration_llavaonevision1_5.py)
    # importable. trust_remote_code below would do this too, but being
    # explicit avoids surprises with auto_map resolution.
    sys.path.insert(0, str(Path(base_model_path)))

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    from_pretrained_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
    if args.device == "auto":
        from_pretrained_kwargs["device_map"] = "auto"
    elif args.device != "cpu":
        from_pretrained_kwargs["device_map"] = {"": args.device}

    print("[merge_lora] step 1/4 — loading base model (this loads ~9 GB on CPU)...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(base_model_path, **from_pretrained_kwargs)
    model.eval()

    if not args.skip_non_lora:
        non_lora_file = adapter_path / "non_lora_state_dict.bin"
        if non_lora_file.exists():
            print(f"[merge_lora] step 2/4 — restoring non-LoRA trainable weights from {non_lora_file.name}...")
            non_lora = load_non_lora_state(non_lora_file)
            missing_in_model: list[str] = []
            applied = 0
            with torch.no_grad():
                state = model.state_dict()
                for k, v in non_lora.items():
                    if k in state:
                        # Match dtype/device of the target parameter.
                        target = state[k]
                        state[k] = v.to(dtype=target.dtype, device=target.device)
                        applied += 1
                    else:
                        missing_in_model.append(k)
                model.load_state_dict(state, strict=False)
            print(f"[merge_lora]   applied {applied} / {len(non_lora)} non-LoRA tensors")
            if missing_in_model:
                print(f"[merge_lora]   warning: {len(missing_in_model)} non-LoRA keys not found in base model "
                      f"(first 5: {missing_in_model[:5]})")
        else:
            print("[merge_lora] step 2/4 — no non_lora_state_dict.bin, skipping")
    else:
        print("[merge_lora] step 2/4 — --skip_non_lora set, skipping")

    print("[merge_lora] step 3/4 — attaching adapter and merging LoRA weights into base...")
    model = PeftModel.from_pretrained(model, str(adapter_path), torch_dtype=torch_dtype)
    model = model.merge_and_unload()  # adapter is gone after this, model is plain base class
    print(f"[merge_lora]   merged model type: {type(model).__name__}")

    print(f"[merge_lora] step 4/4 — saving to {output_path}...")
    model.save_pretrained(str(output_path), safe_serialization=True)

    # Bring along everything an end user expects in the model directory:
    # tokenizer, processor, modelling/config code, chat template, etc.
    # Pull from the base model dir so the output looks like the reference
    # LLaVA-OneVision-1.5-4B-780k-Instruct-mlp-routing-60 layout.
    print("[merge_lora]   saving tokenizer + processor...")
    try:
        tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tok.save_pretrained(str(output_path))
    except Exception as e:
        print(f"[merge_lora]   tokenizer save failed: {e}")
    try:
        proc = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        proc.save_pretrained(str(output_path))
    except Exception as e:
        print(f"[merge_lora]   processor save failed (often non-fatal): {e}")

    # Copy modelling code + extra config files that AutoModel needs to
    # round-trip via trust_remote_code. save_pretrained already wrote
    # config.json and the weights, but auto_map points at .py files
    # that live in the base model dir.
    print("[merge_lora]   copying trust_remote_code .py + auxiliary configs...")
    import shutil
    aux_files = [
        "modeling_llavaonevision1_5.py",
        "configuration_llavaonevision1_5.py",
        "chat_template.jinja",
        "generation_config.json",
        "video_preprocessor_config.json",
        "preprocessor_config.json",
    ]
    base_dir = Path(base_model_path)
    for fname in aux_files:
        src = base_dir / fname
        dst = output_path / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"[merge_lora]     copied {fname}")

    print()
    print(f"[merge_lora] DONE. Merged model at: {output_path}")
    print("[merge_lora] Verify with:")
    print(f"  python -c \"from transformers import AutoModel; AutoModel.from_pretrained('{output_path}', trust_remote_code=True)\"")


if __name__ == "__main__":
    main()
