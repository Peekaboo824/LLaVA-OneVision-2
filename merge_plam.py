"""
PlaM: Training-free model merging via linear interpolation of late-layer attention projections.

Strategy:
- Layers 0-20: keep model_ft weights unchanged
- Layers 21-35: merge q_proj, k_proj, v_proj, o_proj via W_merged = 0.2*W_base + 0.9*W_ft
- Vision encoder, projector, MLP, LayerNorm: never touched
"""

import os
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MODEL_BASE_PATH = "/vepfs-mlp2/c20250505/240906016/jjy/LLaVA/checkpoints/Qwen3-4B-Instruct-2507"
MODEL_FT_PATH = "/vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-1.5/LLaVA-OneVision-1.5-4B-780K-Instruct-1"
OUTPUT_DIR = "/vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-2/LLaVA-OneVision-1.5-4B-stage-PlaM"

MERGE_LAYER_START = 21
MERGE_LAYER_END = 35
ALPHA_BASE = 0.2
ALPHA_FT = 0.9
TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}


def load_sharded_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    shard_files = set(index["weight_map"].values())
    state_dict = {}
    for shard in sorted(shard_files):
        shard_path = os.path.join(model_path, shard)
        print(f"  Loading {shard}")
        state_dict.update(load_file(shard_path, device="cpu"))
    return state_dict


def should_merge(key: str) -> bool:
    """Check if a key corresponds to a late-layer attention projection weight."""
    if not key.startswith("model.layers."):
        return False
    parts = key.split(".")
    # model.layers.{idx}.self_attn.{module}.weight
    if len(parts) < 5:
        return False
    try:
        layer_idx = int(parts[2])
    except ValueError:
        return False
    if layer_idx < MERGE_LAYER_START or layer_idx > MERGE_LAYER_END:
        return False
    if parts[3] != "self_attn":
        return False
    module_name = parts[4]
    if module_name not in TARGET_MODULES:
        return False
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading base model state_dict...")
    sd_base = load_sharded_state_dict(MODEL_BASE_PATH)

    print("Loading fine-tuned model state_dict...")
    sd_ft = load_sharded_state_dict(MODEL_FT_PATH)

    merged_count = 0
    for key in sd_ft:
        if should_merge(key):
            if key not in sd_base:
                print(f"  WARNING: {key} not found in base model, skipping merge")
                continue
            w_base = sd_base[key].to(torch.float32)
            w_ft = sd_ft[key].to(torch.float32)
            w_merged = ALPHA_BASE * w_base + ALPHA_FT * w_ft
            sd_ft[key] = w_merged.to(torch.bfloat16)
            merged_count += 1

    print(f"\nMerged {merged_count} tensors (layers {MERGE_LAYER_START}-{MERGE_LAYER_END}, {TARGET_MODULES})")

    # Save merged weights as safetensors shards matching original layout
    print("Saving merged state_dict...")
    with open(os.path.join(MODEL_FT_PATH, "model.safetensors.index.json")) as f:
        ft_index = json.load(f)

    shard_to_keys: dict[str, list[str]] = {}
    for k, shard in ft_index["weight_map"].items():
        shard_to_keys.setdefault(shard, []).append(k)

    new_metadata = {"metadata": ft_index.get("metadata", {}), "weight_map": {}}
    for shard, keys in sorted(shard_to_keys.items()):
        shard_dict = {k: sd_ft[k] for k in keys if k in sd_ft}
        out_path = os.path.join(OUTPUT_DIR, shard)
        save_file(shard_dict, out_path)
        for k in keys:
            new_metadata["weight_map"][k] = shard
        print(f"  Saved {shard} ({len(shard_dict)} tensors)")

    with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(new_metadata, f, indent=2)

    # Copy config files from ft model
    config_files = [
        "config.json", "configuration_llavaonevision1_5.py", "modeling_llavaonevision1_5.py",
        "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json",
        "preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja",
    ]
    for fname in config_files:
        src = os.path.join(MODEL_FT_PATH, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUTPUT_DIR, fname))

    print(f"\nDone! Merged model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
