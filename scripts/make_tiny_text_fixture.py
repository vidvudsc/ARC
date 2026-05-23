#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SENTENCES = [
    "Arc is a small model trained to continue text and describe simple images.",
    "A clear caption names visible objects, colors, actions, and the scene.",
    "The training data should be clean, direct, and easy for a small model to learn.",
    "A dog can sit on a couch, run through grass, or sleep beside a blanket.",
    "Educational text helps the model learn simple explanations and common facts.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny uint16 token fixture for trainer smoke tests")
    parser.add_argument("--out_dir", default="data/fixtures/tokens")
    parser.add_argument("--tokens", type=int, default=20000)
    parser.add_argument("--vocab_size", type=int, default=32768)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    values = []
    seed_tokens = [ord(ch) % min(args.vocab_size, 32000) for ch in " ".join(SENTENCES)]
    while len(values) < args.tokens:
        values.extend(seed_tokens)
        values.append(2)
    arr = np.asarray(values[: args.tokens], dtype=np.uint16)
    arr.tofile(out / "tokens_000000.bin")
    print(f"Wrote {arr.size} tokens to {out / 'tokens_000000.bin'}")


if __name__ == "__main__":
    main()

