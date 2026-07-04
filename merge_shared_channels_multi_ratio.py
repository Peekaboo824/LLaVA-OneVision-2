"""
Shared-channel MLP merge with multiple ratios.

For shared channels (gate_proj row, up_proj row, down_proj col):
    merged = alpha_new * model_new + alpha_base * model_base
All other weights: keep model_new unchanged.

Model new (A): shared-full-update trained model
Model base (B): stage-1-558k (pre-training baseline)

Generates 5 outputs with ratios new:base = 1:9, 3:7, 5:5, 7:3, 9:1
"""

import os
import json
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MASK_PATH = "/vepfs-mlp2/c20250505/240906016/jjy/visualization/mlp_routing_masks_ranked60.pt"
MODEL_A_DIR = "/vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-2/LLaVA-OneVision-1.5-4B-780k-Instruct-rank-60-shared-full-update"
MODEL_B_DIR = "/vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-1.5/LLaVA-OneVision-1.5-4B-stage-1-558k"

MERGE_RATIOS = [
    (0.1, 0.9, "merge-shared-1-9"),
    (0.3, 0.7, "merge-shared-3-7"),
    (0.5, 0.5, "merge-shared-5-5"),
    (0.7, 0.3, "merge-shared-7-3"),
    (0.9, 0.1, "merge-shared-9-1"),
]

OUTPUT_BASE = "/vepfs-mlp2/c20250505/240906016/jjy/LLaVA-OneVision-2"
NUM_LAYERS = 36


def load_shared_masks():
    mask_dict = torch.load(MASK_PATH, map_location="cpu", weights_only=False)
    shared_masks = {}
    for idx in range(NUM_LAYERS):
        key = f"model.language_model.layers.{idx}.mlp.down_proj"
        shared_masks[idx] = mask_dict["shared"][key].bool()
    return shared_masks


def merge_one_ratio(alpha_a, alpha_b, output_dir, shared_masks):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(MODEL_A_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    print(f"\n{'='*60}")
    print(f"Merge ratio: {alpha_a:.0%} model_A + {alpha_b:.0%} model_B")
    print(f"Model A: {MODEL_A_DIR}")
    print(f"Model B: {MODEL_B_DIR}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    new_weight_map = {}

    for shard_name in shard_files:
        print(f"  Processing {shard_name}...")
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
                print(f"    WARNING: {key} not in model_B, keeping model_A")
                merged_tensors[key] = tensor_a
                continue

            merged = tensor_a.clone().float()
            b_float = tensor_b.float()

            if proj_type in ("gate_proj", "up_proj"):
                # shape [9728, 2560], mask on dim=0
                merged[mask] = alpha_a * merged[mask] + alpha_b * b_float[mask]
            else:
                # down_proj shape [2560, 9728], mask on dim=1
                merged[:, mask] = alpha_a * merged[:, mask] + alpha_b * b_float[:, mask]

            merged_tensors[key] = merged.to(tensor_a.dtype)
            n_shared = mask.sum().item()
            print(f"    {key}: merged {n_shared}/{mask.numel()} shared channels")

        out_path = os.path.join(output_dir, shard_name)
        save_file(merged_tensors, out_path)
        print(f"    Saved {out_path}")

    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": index.get("metadata", {}), "weight_map": new_weight_map}, f, indent=2)

    for fname in os.listdir(MODEL_A_DIR):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        src = os.path.join(MODEL_A_DIR, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    print(f"  Done: {output_dir}\n")


def main():
    print("Loading shared masks...")
    shared_masks = load_shared_masks()
    n_shared_total = sum(m.sum().item() for m in shared_masks.values())
    print(f"Total shared channels across {NUM_LAYERS} layers: {n_shared_total}")

    for alpha_a, alpha_b, suffix in MERGE_RATIOS:
        output_dir = os.path.join(
            OUTPUT_BASE,
            f"LLaVA-OneVision-1.5-4B-780k-Instruct-rank-60-shared-full-update-{suffix}",
        )
        merge_one_ratio(alpha_a, alpha_b, output_dir, shared_masks)

    print("All merges complete!")


if __name__ == "__main__":
    main()
