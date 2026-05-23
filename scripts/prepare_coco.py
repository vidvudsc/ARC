#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm

from arc.data_vision import normalize_caption


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_hf(dataset: str, split: str, config: str | None, streaming: bool) -> Iterator[dict]:
    kwargs = {"split": split, "streaming": streaming}
    if config:
        ds = load_dataset(dataset, config, **kwargs)
    else:
        ds = load_dataset(dataset, **kwargs)
    yield from ds


def caption_values(row: dict) -> list[str]:
    for key in ("captions", "caption", "sentences"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [normalize_caption(value)]
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    out.append(normalize_caption(item))
                elif isinstance(item, dict):
                    text = item.get("raw") or item.get("caption") or item.get("text")
                    if isinstance(text, str) and text.strip():
                        out.append(normalize_caption(text))
            if out:
                return out
    return []


def image_ref(row: dict, *, image_out_dir: Path | None = None, index: int = 0) -> str | None:
    for key in ("image", "image_path", "path", "file_name", "filename", "coco_url", "url"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if image_out_dir is not None and isinstance(value, Image.Image):
            image_out_dir.mkdir(parents=True, exist_ok=True)
            path = image_out_dir / f"coco_{index:09d}.jpg"
            value.convert("RGB").save(path, quality=95)
            return str(path)
    return None


def keep_caption(caption: str, min_tokens: int, max_tokens: int) -> bool:
    tokens = caption.split()
    return min_tokens <= len(tokens) <= max_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare COCO-style captions as Arc image-caption JSONL")
    parser.add_argument("--source_jsonl", default=None)
    parser.add_argument("--dataset", default="jxie/coco_captions")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/vl/coco_captions.jsonl")
    parser.add_argument("--image_out_dir", default=None, help="Optional directory for datasets that yield PIL images")
    parser.add_argument("--max_examples", type=int, default=300_000)
    parser.add_argument("--min_tokens", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--no_dedup", action="store_false", dest="dedup")
    parser.set_defaults(dedup=True)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    rows = iter_jsonl(args.source_jsonl) if args.source_jsonl else iter_hf(
        args.dataset, args.split, args.config, args.streaming
    )
    out = Path(args.out)
    image_out_dir = Path(args.image_out_dir) if args.image_out_dir else None
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = kept = dropped = 0
    seen_pairs: set[tuple[str, str]] = set()
    with out.open("w") as handle:
        for row in tqdm(rows, total=args.max_examples if args.source_jsonl else None, dynamic_ncols=True):
            if kept >= args.max_examples:
                break
            seen += 1
            image = image_ref(dict(row), image_out_dir=image_out_dir, index=seen)
            captions = caption_values(dict(row))
            if image is None or not captions:
                dropped += 1
                continue
            for caption in captions:
                if kept >= args.max_examples:
                    break
                if not keep_caption(caption, args.min_tokens, args.max_tokens):
                    dropped += 1
                    continue
                pair = (image, caption)
                if args.dedup and pair in seen_pairs:
                    dropped += 1
                    continue
                seen_pairs.add(pair)
                handle.write(json.dumps({"source": "coco", "image": image, "caption": caption}) + "\n")
                kept += 1
    summary = {"seen": seen, "kept": kept, "dropped": dropped, "out": str(out)}
    (out.with_suffix(out.suffix + ".summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
