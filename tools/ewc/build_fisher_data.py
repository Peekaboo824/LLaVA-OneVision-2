"""Build the Fisher anchor dataset for text-subspace extraction.

Samples 2048 single-turn dialogues from 5 HF datasets, formats them as
``messages = [{role: user, ...}, {role: assistant, ...}]`` and writes a JSONL.

This script does NOT tokenize or build label masks; that happens in
``compute_fisher.py``.
"""

import argparse
import json
import os
import random
import sys
from collections import OrderedDict

DEFAULT_CACHE_DIR = "/vepfs-mlp2/c20250505/240906016/jjy/datasets_cache"

DEFAULT_COUNTS = OrderedDict([
    ("math", 550),
    ("mbpp", 300),
    ("logiqa", 450),
    ("halueval", 348),
    ("dolly", 400),
])


def _parse_counts(overrides):
    """Parse ``--counts math=550 mbpp=300 ...`` into a dict."""
    counts = OrderedDict(DEFAULT_COUNTS)
    if not overrides:
        return counts
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"--counts expects key=value, got '{kv}'")
        key, value = kv.split("=", 1)
        if key not in counts:
            raise ValueError(f"unknown dataset name '{key}' in --counts")
        counts[key] = int(value)
    return counts


def _take(loaded, k, source):
    """Random-sample ``k`` indices from a HF dataset.

    Returns a list of int indices. If len(loaded) < k, warns and returns all.
    """
    n = len(loaded)
    if n < k:
        print(f"[WARN] {source}: dataset has {n} samples but k={k}; taking all")
        return list(range(n))
    return random.sample(range(n), k)


def _build_math(cache_dir, k):
    from datasets import load_dataset
    math_configs = [
        'algebra', 'counting_and_probability', 'geometry', 
        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
    ]
    out = []
    num_configs = len(math_configs)
    for i, config in enumerate(math_configs):
        # 计算当前 config 应该分到的样本数 (带余数分配逻辑)
        take_n = (k // num_configs) + (1 if i < k % num_configs else 0)
        if take_n == 0:
            continue
        try:
            # 加载特定的子数据集
            ds = load_dataset("EleutherAI/hendrycks_math", config, split="train", cache_dir=cache_dir)
            # 确定实际抽取的索引 (调用你原有的 _take 辅助函数)
            # 注意：这里传给 _take 的 k 应该是当前子类的 take_n
            idxs = _take(ds, take_n, f"math_{config}")
            print(f"[math-{config}] loaded {len(ds)} samples → sampled {len(idxs)} samples")

            for idx in idxs:
                ex = ds[idx]
                out.append({
                    "source": f"math_{config}",
                    "messages": [
                        {"role": "user", "content": ex["problem"]},
                        {"role": "assistant", "content": ex["solution"]},
                    ],
                })
        except Exception as e:
            print(f"Error loading math config {config}: {e}")
            
    return out


def _build_mbpp(cache_dir, k):
    from datasets import load_dataset
    ds = load_dataset(
        "Muennighoff/mbpp", "sanitized",
        split="test", cache_dir=cache_dir,
    )
    print(f"[mbpp] loaded {len(ds)} samples → sampled {min(k, len(ds))} samples")
    idxs = _take(ds, k, "mbpp")
    out = []
    for i in idxs:
        ex = ds[i]
        tests = "\n".join(ex["test_list"])
        user = f"{ex['prompt']}\n\nTests:\n{tests}"
        out.append({
            "source": "mbpp",
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": ex["code"]},
            ],
        })
    return out


def _build_logiqa(cache_dir, k):
    from datasets import load_dataset
    ds = load_dataset(
        "lucasmccabe/logiqa", split="train", cache_dir=cache_dir,
    )
    print(f"[logiqa] loaded {len(ds)} samples → sampled {min(k, len(ds))} samples")
    idxs = _take(ds, k, "logiqa")
    out = []
    for i in idxs:
        ex = ds[i]
        options = ex["options"]
        opt_text = "\n".join(f"{chr(ord('A') + j)}. {o}" for j, o in enumerate(options))
        user = (
            f"{ex['context']}\n\n"
            f"Question: {ex['query']}\n\n"
            f"Options:\n{opt_text}"
        )
        assistant = options[ex["correct_option"]]
        out.append({
            "source": "logiqa",
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        })
    return out


def _build_halueval(cache_dir, k):
    from datasets import load_dataset
    # pminervini/HaluEval qa config only has a 'data' split.
    ds = load_dataset(
        "pminervini/HaluEval", "qa", split="data", cache_dir=cache_dir,
    )
    print(f"[halueval] loaded {len(ds)} samples → sampled {min(k, len(ds))} samples")
    idxs = _take(ds, k, "halueval")
    out = []
    for i in idxs:
        ex = ds[i]
        user = f"Knowledge: {ex['knowledge']}\n\nQuestion: {ex['question']}"
        out.append({
            "source": "halueval",
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": ex["right_answer"]},
            ],
        })
    return out


def _build_dolly(cache_dir, k):
    from datasets import load_dataset
    ds = load_dataset(
        "databricks/databricks-dolly-15k", split="train", cache_dir=cache_dir,
    )
    print(f"[dolly] loaded {len(ds)} samples → sampled {min(k, len(ds))} samples")
    idxs = _take(ds, k, "dolly")
    out = []
    for i in idxs:
        ex = ds[i]
        ctx = ex.get("context", "") or ""
        user = ex["instruction"] + (f"\n\n{ctx}" if ctx else "")
        out.append({
            "source": "dolly",
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": ex["response"]},
            ],
        })
    return out


_BUILDERS = {
    "math": _build_math,
    "mbpp": _build_mbpp,
    "logiqa": _build_logiqa,
    "halueval": _build_halueval,
    "dolly": _build_dolly,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="fisher_anchor_dataset.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for per-source sampling (default: 42)")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="HF datasets cache dir (also exported as HF_DATASETS_CACHE)")
    parser.add_argument("--counts", nargs="*", default=None,
                        help="Override per-source counts, e.g. math=550 mbpp=300 ...")
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os.makedirs(args.cache_dir, exist_ok=True)
    random.seed(args.seed)

    counts = _parse_counts(args.counts)
    print(f"[build_fisher_data] seed={args.seed} cache_dir={args.cache_dir}")
    print(f"[build_fisher_data] target counts: {dict(counts)} "
          f"(total={sum(counts.values())})")

    all_rows = []
    for source, k in counts.items():
        builder = _BUILDERS[source]
        rows = builder(args.cache_dir, k)
        all_rows.extend(rows)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ wrote {len(all_rows)} lines to {args.output}")
    dist = {}
    for row in all_rows:
        dist[row["source"]] = dist.get(row["source"], 0) + 1
    print("[build_fisher_data] source distribution:")
    for src, cnt in dist.items():
        print(f"    {src:<10s} {cnt}")


if __name__ == "__main__":
    sys.exit(main())
