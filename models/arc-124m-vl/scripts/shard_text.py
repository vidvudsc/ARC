#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from arc.data import read_mixture
from arc.tokenizer import ArcTokenizer


def iter_jsonl_rows(path: str | Path) -> Iterator[tuple[str, str]]:
    path = Path(path)
    while True:
        yielded = False
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row.get("text", "")).strip()
                source = str(row.get("source", "jsonl"))
                if text:
                    yielded = True
                    yield source, text
        if not yielded:
            raise ValueError(f"No usable text rows found in {path}")


def iter_hf_rows(mixture_path: str, seed: int) -> Iterator[tuple[str, str]]:
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
        streams.append((spec.name, iter(ds.shuffle(buffer_size=10_000, seed=seed))))
    while True:
        idx = rng.choices(range(len(streams)), weights=weights, k=1)[0]
        name, stream = streams[idx]
        try:
            row = next(stream)
        except StopIteration:
            continue
        text = str(row.get("text", "")).strip()
        if text:
            yield name, text


class ShardWriter:
    def __init__(self, out_dir: Path, shard_tokens: int):
        self.out_dir = out_dir
        self.shard_tokens = shard_tokens
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.buffer: list[int] = []
        self.index = 0
        self.total = 0
        self.manifest = []

    def add(self, ids: list[int]) -> None:
        self.buffer.extend(ids)
        while len(self.buffer) >= self.shard_tokens:
            self._flush_exact(self.shard_tokens)

    def close(self) -> None:
        if self.buffer:
            self._flush_exact(len(self.buffer))
        manifest = {
            "total_tokens": self.total,
            "shards": self.manifest,
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _flush_exact(self, count: int) -> None:
        ids = self.buffer[:count]
        del self.buffer[:count]
        arr = np.asarray(ids, dtype=np.uint16)
        path = self.out_dir / f"tokens_{self.index:06d}.bin"
        arr.tofile(path)
        self.manifest.append({"path": path.name, "tokens": int(arr.size)})
        self.index += 1
        self.total += int(arr.size)


def write_split_summary(out_dir: Path, train_writer: ShardWriter, val_writer: ShardWriter | None) -> None:
    summary = {
        "train_tokens": train_writer.total,
        "val_tokens": val_writer.total if val_writer is not None else 0,
        "total_tokens": train_writer.total + (val_writer.total if val_writer is not None else 0),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def log_jsonl(handle: TextIO | None, payload: dict[str, object]) -> None:
    if handle is not None:
        handle.write(json.dumps(payload) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Arc uint16 packed token shards")
    parser.add_argument("--mixture", default="configs/stage1_text_mix.jsonl")
    parser.add_argument("--source_jsonl", default=None, help="Optional local JSONL with a text field; bypasses HF streaming")
    parser.add_argument("--tokenizer_dir", default="tokenizer_32k")
    parser.add_argument("--out_dir", default="data/tokens_stage1")
    parser.add_argument("--target_tokens", type=int, default=10_000_000)
    parser.add_argument("--val_tokens", type=int, default=0)
    parser.add_argument("--shard_tokens", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log_jsonl", default=None)
    args = parser.parse_args()

    tok = ArcTokenizer.from_dir(args.tokenizer_dir)
    out_dir = Path(args.out_dir)
    if args.val_tokens > 0:
        train_writer = ShardWriter(out_dir / "train", shard_tokens=args.shard_tokens)
        val_writer = ShardWriter(out_dir / "val", shard_tokens=args.shard_tokens)
    else:
        train_writer = ShardWriter(out_dir, shard_tokens=args.shard_tokens)
        val_writer = None
    rows = iter_jsonl_rows(args.source_jsonl) if args.source_jsonl else iter_hf_rows(args.mixture, seed=args.seed)
    total_target = args.target_tokens + args.val_tokens
    pbar = tqdm(total=total_target, unit="tok", dynamic_ncols=True)
    log_handle = Path(args.log_jsonl).open("w") if args.log_jsonl else None
    try:
        while train_writer.total + len(train_writer.buffer) < args.target_tokens:
            source, text = next(rows)
            ids = tok.encode(text, add_bos=True, add_eos=True)
            if max(ids, default=0) >= 65536:
                raise ValueError("Token id does not fit in uint16")
            train_writer.add(ids)
            pbar.update(len(ids))
            log_jsonl(log_handle, {"split": "train", "source": source, "tokens": len(ids)})
        if val_writer is not None:
            while val_writer.total + len(val_writer.buffer) < args.val_tokens:
                source, text = next(rows)
                ids = tok.encode(text, add_bos=True, add_eos=True)
                if max(ids, default=0) >= 65536:
                    raise ValueError("Token id does not fit in uint16")
                val_writer.add(ids)
                pbar.update(len(ids))
                log_jsonl(log_handle, {"split": "val", "source": source, "tokens": len(ids)})
    finally:
        train_writer.close()
        if val_writer is not None:
            val_writer.close()
        write_split_summary(out_dir, train_writer, val_writer)
        if log_handle is not None:
            log_handle.close()
        pbar.close()
    if val_writer is None:
        print(f"Wrote {train_writer.total:,} train tokens to {args.out_dir}")
    else:
        print(f"Wrote {train_writer.total:,} train tokens and {val_writer.total:,} val tokens to {args.out_dir}")


if __name__ == "__main__":
    main()
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
