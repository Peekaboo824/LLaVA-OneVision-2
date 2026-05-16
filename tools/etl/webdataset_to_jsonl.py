#!/usr/bin/env python3
"""Convert LLaVA-NeXT-780k WebDataset (.tar shards) → ds/ JSONL + image folder.

Input layout (per shard):
    pretrain-{N}.tar
      Each sample is keyed by either `image_{i}` (multimodal) or `text_{i}`
      (text-only) and groups together a JSON descriptor plus, for multimodal
      samples, one raw image file:

        # multimodal sample
        image_{i}.json                                # text annotations
        image_{i}.{img_i}_<orig_name>.{jpg,png,...}   # raw image bytes

        # text-only sample
        text_{i}.json                                 # text-only annotations

JSON schema inside tar:
    Multimodal:
      {"texts": [{"content": "<image>\\n...", "role": "user"|"assistant"}, ...],
       "media": "image",
       "name":  ["<i>_<orig_name>.<ext>"]}
    Text-only:
      {"texts": [{"content": "...", "role": "user"|"assistant"}, ...],
       "media": "text",
       "name":  null}

Output layout:
    OUT/images/pretrain-{N}/<original_in_tar_filename>   # only for multimodal
    OUT/annotations.jsonl                                # merged ds/ samples
    OUT/.shards_done/pretrain-{N}.jsonl                  # per-shard partials
    OUT/.shards_done/pretrain-{N}.done                   # idempotency marker

Each line in annotations.jsonl follows ds/SupervisedDataset schema. Multimodal
lines include an "image" field, text-only lines omit it (sft_dataset.py:115-119
+ 161-162 already handles the text-only branch):

    # multimodal
    {"image": "images/pretrain-{N}/<filename>",
     "conversations": [{"from": "human", "value": "<image>\\n..."}, ...]}

    # text-only
    {"conversations": [{"from": "human", "value": "..."}, ...]}

Run:
    python tools/etl/webdataset_to_jsonl.py \\
        --wds-dir /vepfs.../Datasets/LLaVA-NeXT-780k-webdataset \\
        --out-dir /vepfs.../Datasets/LLaVA-NeXT-780k-unpacked \\
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# WebDataset role → LLaVA conversation role
ROLE_MAP = {"user": "human", "assistant": "gpt"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def parse_sample_key(member_name: str) -> str:
    """Tar member like 'image_42.json' or 'image_42.0_xxx.jpg' → 'image_42'."""
    return member_name.split(".", 1)[0]


def process_shard(args_tuple):
    """Process one .tar shard. Pure-function for ProcessPoolExecutor.

    Returns (shard_name, n_ok, n_skipped, error_or_None, n_multimodal, n_text_only).
    """
    tar_path_s, out_dir_s, force = args_tuple
    tar_path = Path(tar_path_s)
    out_dir = Path(out_dir_s)
    shard_name = tar_path.stem  # e.g. 'pretrain-0'

    done_dir = out_dir / ".shards_done"
    done_marker = done_dir / f"{shard_name}.done"
    part_jsonl = done_dir / f"{shard_name}.jsonl"
    done_dir.mkdir(parents=True, exist_ok=True)

    if done_marker.exists() and not force:
        return (shard_name, 0, 0, "SKIPPED_ALREADY_DONE", 0, 0)

    image_subdir = out_dir / "images" / shard_name
    image_subdir.mkdir(parents=True, exist_ok=True)

    json_by_key: dict[str, dict | None] = {}
    image_by_key: dict[str, str] = {}

    # Stream mode 'r|' — sequential, low memory; extractfile() works on the
    # current member only, which matches our one-pass extraction pattern.
    try:
        with tarfile.open(tar_path, mode="r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = member.name
                lower = name.lower()
                key = parse_sample_key(name)

                if lower.endswith(".json"):
                    try:
                        raw = tf.extractfile(member).read().decode("utf-8")
                        json_by_key[key] = json.loads(raw)
                    except Exception:
                        json_by_key[key] = None
                elif lower.endswith(IMAGE_EXTS):
                    basename = Path(name).name
                    target = image_subdir / basename
                    if not target.exists():
                        with tf.extractfile(member) as src, open(target, "wb") as dst:
                            # 1 MiB chunks — images are usually < 1 MB but be safe.
                            while True:
                                buf = src.read(1 << 20)
                                if not buf:
                                    break
                                dst.write(buf)
                    image_by_key[key] = basename
                # else: silently ignore unknown members
    except Exception as e:
        return (shard_name, 0, 0, f"EXTRACT_FAIL: {e!r}", 0, 0)

    # Compose ds/ samples
    #
    # Two valid sample shapes exist in the source data:
    #   - multimodal:  sample key 'image_{i}', media='image', has one image file
    #   - text-only:   sample key 'text_{i}',  media='text',  no image, no
    #                  <image> placeholder in any turn
    #
    # A sample is skipped only when the source data is malformed (broken JSON,
    # missing required image for a media='image' sample, odd turn count,
    # role/turn-alternation violation, or an unexpected <image> token inside a
    # text-only sample).
    ok = 0
    skipped = 0
    n_multimodal = 0
    n_text_only = 0
    with open(part_jsonl, "w", encoding="utf-8") as fout:
        for key, payload in json_by_key.items():
            if payload is None:
                skipped += 1
                continue

            texts = payload.get("texts")
            if not isinstance(texts, list) or len(texts) == 0 or len(texts) % 2 != 0:
                skipped += 1
                continue

            conversations = []
            bad = False
            for i, turn in enumerate(texts):
                mapped = ROLE_MAP.get(turn.get("role"))
                expected = "human" if i % 2 == 0 else "gpt"
                if mapped != expected:
                    bad = True
                    break
                conversations.append({
                    "from": mapped,
                    "value": turn.get("content", ""),
                })
            if bad:
                skipped += 1
                continue

            media = payload.get("media")
            img_basename = image_by_key.get(key)
            has_image_token = any(
                "<image>" in c.get("value", "") for c in conversations
            )

            if media == "image":
                # Multimodal sample: image MUST be on disk.
                if img_basename is None:
                    skipped += 1
                    continue
                sample = {
                    "image": f"images/{shard_name}/{img_basename}",
                    "conversations": conversations,
                }
                n_multimodal += 1
            elif media == "text":
                # Text-only sample: must NOT carry an <image> placeholder.
                if has_image_token:
                    skipped += 1
                    continue
                sample = {"conversations": conversations}
                n_text_only += 1
            else:
                # Unknown media tag — fall back: if an image is paired use it,
                # else treat as text-only when no <image> token is present.
                if img_basename is not None:
                    sample = {
                        "image": f"images/{shard_name}/{img_basename}",
                        "conversations": conversations,
                    }
                    n_multimodal += 1
                elif not has_image_token:
                    sample = {"conversations": conversations}
                    n_text_only += 1
                else:
                    skipped += 1
                    continue

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            ok += 1

    done_marker.touch()
    return (shard_name, ok, skipped, None, n_multimodal, n_text_only)


def merge_partials(out_dir: Path, shard_names: list[str]) -> int:
    """Concatenate per-shard partials into annotations.jsonl. Returns total samples."""
    final_jsonl = out_dir / "annotations.jsonl"
    n_total = 0
    with open(final_jsonl, "w", encoding="utf-8") as fout:
        for shard_name in shard_names:
            part = out_dir / ".shards_done" / f"{shard_name}.jsonl"
            if not part.exists():
                continue
            with open(part, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    n_total += 1
    return n_total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wds-dir", required=True, help="dir containing pretrain-*.tar")
    ap.add_argument("--out-dir", required=True, help="destination (images/ + annotations.jsonl)")
    ap.add_argument("--workers", type=int, default=8, help="parallel shard processes")
    ap.add_argument("--force", action="store_true", help="reprocess shards even if .done exists")
    ap.add_argument("--limit", type=int, default=0, help="process only first N shards (debug)")
    ap.add_argument("--merge-only", action="store_true",
                    help="skip extraction, just merge existing per-shard jsonl partials")
    args = ap.parse_args()

    wds_dir = Path(args.wds_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(wds_dir.glob("pretrain-*.tar"))
    if args.limit:
        shards = shards[: args.limit]
    if not shards:
        sys.exit(f"No pretrain-*.tar under {wds_dir}")

    shard_names = [p.stem for p in shards]

    if args.merge_only:
        n = merge_partials(out_dir, shard_names)
        print(f"merge-only: {n} samples → {out_dir / 'annotations.jsonl'}")
        return

    print(f"Found {len(shards)} shards; workers={args.workers}; out={out_dir}", flush=True)
    t0 = time.time()
    jobs = [(str(p), str(out_dir), args.force) for p in shards]

    n_ok_total = 0
    n_skip_total = 0
    n_mm_total = 0
    n_text_total = 0
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            shard_path = futs[fut]
            try:
                shard_name, n_ok, n_skip, err, n_mm, n_text = fut.result()
            except Exception as e:
                shard_name = Path(shard_path).stem
                n_ok, n_skip, err, n_mm, n_text = 0, 0, f"WORKER_EXC: {e!r}", 0, 0
            n_ok_total += n_ok
            n_skip_total += n_skip
            n_mm_total += n_mm
            n_text_total += n_text
            if err and err != "SKIPPED_ALREADY_DONE":
                n_err += 1
            elapsed = time.time() - t0
            print(
                f"[{i}/{len(shards)}] {shard_name}: ok={n_ok} (mm={n_mm} text={n_text}) "
                f"skip={n_skip} err={err} "
                f"(total ok={n_ok_total} mm={n_mm_total} text={n_text_total} "
                f"skip={n_skip_total} | {elapsed:.1f}s)",
                flush=True,
            )

    total = merge_partials(out_dir, shard_names)
    print(
        f"\nDone. shards={len(shards)} errors={n_err} "
        f"samples_ok={n_ok_total} (multimodal={n_mm_total} text_only={n_text_total}) "
        f"samples_skipped={n_skip_total} "
        f"merged_lines={total}",
        flush=True,
    )
    print(f"annotations: {out_dir / 'annotations.jsonl'}")
    print(f"images_root: {out_dir / 'images'}")


if __name__ == "__main__":
    main()
