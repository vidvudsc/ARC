#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SENTENCES = [
    "Arc is a small base model trained to continue clear text and describe simple images.",
    "A useful caption names visible objects, colors, actions, and the surrounding scene.",
    "Educational text helps a compact model learn direct explanations and common facts.",
    "The image shows a brown dog sitting on a couch next to a folded blanket.",
    "Clean training data matters because a small model has very little capacity to waste.",
    "A person is walking through a city street while carrying a red umbrella.",
    "The recipe explains how to mix flour, water, salt, and yeast before baking bread.",
    "A train moves across a bridge with green hills and cloudy sky in the background.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny JSONL text corpus for tokenizer/sharder smoke tests")
    parser.add_argument("--out", default="data/fixtures/text_corpus.jsonl")
    parser.add_argument("--rows", type=int, default=200)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for idx in range(args.rows):
            text = " ".join(SENTENCES[(idx + offset) % len(SENTENCES)] for offset in range(4))
            handle.write(json.dumps({"source": "tiny_fixture", "text": text}) + "\n")
    print(f"Wrote {args.rows} rows to {out}")


if __name__ == "__main__":
    main()
