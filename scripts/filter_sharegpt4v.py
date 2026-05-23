#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tqdm.auto import tqdm

from arc.data_vision import extract_caption, extract_image_ref, is_caption_candidate, load_vision_filter


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter ShareGPT4V-style caption data into Arc image-caption JSONL")
    parser.add_argument("--source_jsonl", default=None)
    parser.add_argument("--dataset", default="Lin-Chen/ShareGPT4V")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--filter", default="configs/sharegpt4v_filter.json")
    parser.add_argument("--out", default="data/vl/sharegpt4v_filtered.jsonl")
    parser.add_argument("--max_examples", type=int, default=700_000)
    parser.add_argument("--strict", action="store_true", help="Use strict_max_tokens instead of max_tokens")
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    flt = load_vision_filter(args.filter)
    rows = iter_jsonl(args.source_jsonl) if args.source_jsonl else iter_hf(
        args.dataset, args.split, args.config, args.streaming
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = kept = dropped = 0
    with out.open("w") as handle:
        for row in tqdm(rows, total=args.max_examples if args.source_jsonl else None, dynamic_ncols=True):
            if kept >= args.max_examples:
                break
            seen += 1
            caption = extract_caption(dict(row), skip_conversations=True)
            image = extract_image_ref(dict(row))
            if caption is None or image is None:
                dropped += 1
                continue
            if not is_caption_candidate(caption, flt, strict=args.strict):
                dropped += 1
                continue
            handle.write(
                json.dumps(
                    {
                        "source": "sharegpt4v",
                        "image": image,
                        "caption": caption,
                        "caption_tokens_est": len(caption.split()),
                    }
                )
                + "\n"
            )
            kept += 1
    summary = {"seen": seen, "kept": kept, "dropped": dropped, "out": str(out)}
    (out.with_suffix(out.suffix + ".summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
