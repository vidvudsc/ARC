#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

from arc.data import read_mixture
from arc.tokenizer import train_byte_bpe


def iter_jsonl_texts(path: str | Path, samples: int) -> Iterator[str]:
    yielded = 0
    while yielded < samples:
        with Path(path).open() as handle:
            for line in handle:
                if yielded >= samples:
                    return
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row.get("text", "")).strip()
                if len(text.split()) < 5:
                    continue
                yielded += 1
                yield text


def iter_hf_texts(mixture_path: str, samples: int, seed: int) -> Iterator[str]:
    specs = read_mixture(mixture_path)
    rng = random.Random(seed)
    weights = [spec.weight for spec in specs]
    streams = []
    for spec in specs:
        kwargs = {"split": spec.split, "streaming": True}
        if spec.config and spec.config != "default":
            ds = load_dataset(spec.dataset, spec.config, **kwargs)
        else:
            ds = load_dataset(spec.dataset, **kwargs)
        streams.append(iter(ds.shuffle(buffer_size=10_000, seed=seed)))
    yielded = 0
    while yielded < samples:
        idx = rng.choices(range(len(streams)), weights=weights, k=1)[0]
        try:
            row = next(streams[idx])
        except StopIteration:
            continue
        text = str(row.get("text", "")).strip()
        if len(text.split()) < 20:
            continue
        yielded += 1
        yield text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Arc byte-level BPE tokenizer")
    parser.add_argument("--mixture", default="configs/stage1_text_mix.jsonl")
    parser.add_argument("--source_jsonl", default=None, help="Optional local JSONL with a text field; bypasses HF streaming")
    parser.add_argument("--out_dir", default="tokenizer_16k")
    parser.add_argument("--samples", type=int, default=2_000_000)
    parser.add_argument("--vocab_size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    texts = iter_jsonl_texts(args.source_jsonl, samples=args.samples) if args.source_jsonl else iter_hf_texts(
        args.mixture, samples=args.samples, seed=args.seed
    )
    train_byte_bpe(
        texts,
        out_dir=args.out_dir,
        vocab_size=args.vocab_size,
        limit=args.samples,
    )


if __name__ == "__main__":
    main()
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
