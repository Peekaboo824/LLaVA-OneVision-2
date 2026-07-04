"""
Shared-channel MLP merge based on routing mask.

Strategy:
- Load shared neuron mask from mlp_routing_masks_ranked60.pt
- For shared channels (gate_proj row, up_proj row, down_proj col):
    merged = 0.8 * stage1_558k + 0.2 * routing60
- All other weights: keep routing60 unchanged
"""

import os
import json
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MASK_PATH = "/vepfs-mlp2/c20250505/240906016/jjy/visualization/mlp_routing_masks_ranked60.pt"
MODEL_A_DIR = "/vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-2/LLaVA-OneVision-1.5-4B-780k-Instruct-mlp-routing-60"
MODEL_B_DIR = "/vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-1.5/LLaVA-OneVision-1.5-4B-stage-1-558k"
OUTPUT_DIR = "/vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-2/LLaVA-OneVision-1.5-4B-780k-Instruct-mlp-routing-60-merge-5-5"

ALPHA_STAGE1 = 0.5
ALPHA_ROUTING60 = 0.5
NUM_LAYERS = 36


def load_shared_masks():
    mask_dict = torch.load(MASK_PATH, map_location="cpu", weights_only=False)
    shared_masks = {}
    for idx in range(NUM_LAYERS):
        key = f"model.language_model.layers.{idx}.mlp.down_proj"
        shared_masks[idx] = mask_dict["shared"][key].bool()
    return shared_masks


def main():
    shared_masks = load_shared_masks()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(MODEL_A_DIR, "model.safetensors.index.json")) as f:
        index_a = json.load(f)

    shard_files = sorted(set(index_a["weight_map"].values()))
    print(f"Shards: {shard_files}")
    print(f"Merge ratio: {ALPHA_STAGE1:.0%} stage1 + {ALPHA_ROUTING60:.0%} routing60 on shared channels\n")

    new_weight_map = {}

    for shard_name in shard_files:
        print(f"Processing {shard_name}...")
        path_a = os.path.join(MODEL_A_DIR, shard_name)
        path_b = os.path.join(MODEL_B_DIR, shard_name)

        tensors_a = {}
        with safe_open(path_a, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors_a[key] = f.get_tensor(key)

        tensors_b = {}
        with safe_open(path_b, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors_b[key] = f.get_tensor(key)

        merged_tensors = {}
        for key in tensors_a:
            new_weight_map[key] = shard_name
            tensor_a = tensors_a[key]

            if ".mlp." not in key or not key.endswith(".weight"):
                merged_tensors[key] = tensor_a
                continue

            parts = key.split(".")
            try:
                layer_idx = int(parts[2])
            except (IndexError, ValueError):
                merged_tensors[key] = tensor_a
                continue

            if layer_idx >= NUM_LAYERS:
                merged_tensors[key] = tensor_a
                continue

            proj_type = parts[4]  # gate_proj / up_proj / down_proj
            if proj_type not in ("gate_proj", "up_proj", "down_proj"):
                merged_tensors[key] = tensor_a
                continue

            mask = shared_masks[layer_idx]
            tensor_b = tensors_b.get(key)
            if tensor_b is None:
                print(f"  WARNING: {key} not in stage-1, keeping routing60")
                merged_tensors[key] = tensor_a
                continue

            merged = tensor_a.clone().float()
            b_float = tensor_b.float()

            if proj_type in ("gate_proj", "up_proj"):
                # shape [I, hidden], merge rows
                merged[mask] = ALPHA_STAGE1 * b_float[mask] + ALPHA_ROUTING60 * merged[mask]
            else:
                # down_proj shape [hidden, I], merge columns
                merged[:, mask] = ALPHA_STAGE1 * b_float[:, mask] + ALPHA_ROUTING60 * merged[:, mask]

            merged_tensors[key] = merged.to(tensor_a.dtype)
            n_shared = mask.sum().item()
            print(f"  {key}: merged {n_shared}/{mask.numel()} shared channels")

        out_path = os.path.join(OUTPUT_DIR, shard_name)
        save_file(merged_tensors, out_path)
        print(f"  Saved {out_path}\n")

    with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": index_a.get("metadata", {}), "weight_map": new_weight_map}, f, indent=2)

    for fname in os.listdir(MODEL_A_DIR):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        src = os.path.join(MODEL_A_DIR, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(OUTPUT_DIR, fname))

    print(f"Done! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
