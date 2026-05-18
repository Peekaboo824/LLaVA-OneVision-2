"""Normalise a Fisher .pt produced by compute_fisher_qwen3_4b.py.

EWC's F_i is the expectation of the squared gradient over the data
distribution. ``compute_fisher_qwen3_4b.py`` accumulates the raw sum of g²
over ``num_samples_processed`` examples, so the standard expectation is

    F_i = (1 / N) * sum_n g_n_i**2

This script reads the raw Fisher dict, divides every tensor by N (read from
``_meta['num_samples_processed']``), and writes a new file. The relative
structure across tensors is preserved exactly — only the global scale
changes — so the qualitative EWC behaviour is identical; the only practical
consequence is that the trainer's ``--ewc_lambda`` is now interpreted in
per-sample units (multiply your old λ by N to match the previous penalty
strength).
"""

import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True,
                        help="Raw Fisher .pt (sum of g²)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path. Default: <input>_normalized.pt")
    args = parser.parse_args()

    in_path = os.path.abspath(args.input)
    if args.output is None:
        root, ext = os.path.splitext(in_path)
        args.output = root + "_normalized" + ext

    print(f"[normalize_fisher] loading {in_path}")
    d = torch.load(in_path, map_location="cpu")

    meta = d.get("_meta", {})
    n = meta.get("num_samples_processed")
    if not isinstance(n, int) or n <= 0:
        print(f"[normalize_fisher] ERROR: _meta['num_samples_processed']={n!r} "
              "is not a positive int; cannot normalise.")
        sys.exit(1)
    if meta.get("normalized") is True:
        print(f"[normalize_fisher] ERROR: input is already marked normalized "
              "in _meta; aborting to avoid double division.")
        sys.exit(2)

    keys = [k for k in d.keys() if k != "_meta"]
    print(f"[normalize_fisher] dividing {len(keys)} tensors by N={n}")
    inv_n = 1.0 / float(n)

    out = {}
    for k in keys:
        out[k] = d[k].to(dtype=torch.float32).mul(inv_n).contiguous()

    new_meta = dict(meta)
    new_meta["normalized"] = True
    new_meta["normalization_divisor"] = n
    new_meta["source_fisher_path"] = in_path
    out["_meta"] = new_meta

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(out, args.output)
    print(f"✅ saved normalised Fisher to {args.output}")

    # quick before/after stats on one key for a sanity hint
    probe = keys[0]
    before_mean = d[probe].float().mean().item()
    after_mean = out[probe].mean().item()
    print(f"[normalize_fisher] sanity: {probe} mean {before_mean:.3e} → "
          f"{after_mean:.3e} (ratio {after_mean / before_mean:.3e}, "
          f"expected {inv_n:.3e})")


if __name__ == "__main__":
    sys.exit(main())
