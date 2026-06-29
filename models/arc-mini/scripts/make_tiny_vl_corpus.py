#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


COCO_ROWS = [
    {
        "image": "images/dog_couch.jpg",
        "captions": [
            "A dog sits on a couch.",
            "A brown dog rests beside a blanket.",
            "A pet is sitting in a living room.",
        ],
    },
    {
        "image": "images/city_umbrella.jpg",
        "captions": [
            "A person carries an umbrella on a street.",
            "A red umbrella stands out on a rainy sidewalk.",
        ],
    },
]


FLICKR30K_ROWS = [
    {
        "image": "images/dog_couch.jpg",
        "sentences": [
            "A brown dog relaxes on a couch in a living room.",
            "A dog is resting near a blanket on the sofa.",
        ],
    },
    {
        "image": "images/city_umbrella.jpg",
        "sentences": [
            "A person with a red umbrella walks down a wet street.",
            "Someone carries an umbrella through the city.",
        ],
    },
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_fixture_images(out: Path) -> None:
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("dog_couch.jpg", (122, 86, 62), "dog"),
        ("city_umbrella.jpg", (48, 88, 132), "umbrella"),
    ]
    for name, color, label in specs:
        image = Image.new("RGB", (256, 256), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((32, 150, 224, 210), fill=(60, 60, 60))
        draw.text((40, 40), label, fill=(255, 255, 255))
        image.save(image_dir / name, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny image-caption JSONL fixtures")
    parser.add_argument("--out_dir", default="data/fixtures/vl")
    args = parser.parse_args()
    out = Path(args.out_dir)
    write_fixture_images(out)
    write_jsonl(out / "coco_raw.jsonl", COCO_ROWS)
    write_jsonl(out / "flickr30k_raw.jsonl", FLICKR30K_ROWS)
    print(f"Wrote VL fixtures to {out}")


if __name__ == "__main__":
    main()
