#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Iterator

from PIL import Image
from tqdm.auto import tqdm


def iter_manifest(paths: list[str]) -> Iterator[dict]:
    for manifest in paths:
        with Path(manifest).open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("image") and row.get("caption"):
                        yield row


def resolve_image(image_ref: str, image_root: str | None) -> Path:
    path = Path(image_ref)
    if path.is_absolute() or image_root is None:
        return path
    return Path(image_root) / path


def encode_jpeg(path: Path, image_size: int, quality: int) -> bytes:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard Arc image-caption manifests into resized tar shards")
    parser.add_argument("--manifest", action="append", required=True, help="Image-caption JSONL; can be passed multiple times")
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--out_dir", default="data/vl/shards")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--samples_per_shard", type=int, default=5000)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_missing", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_index = 0
    shard_count = 0
    total = 0
    skipped = 0
    tar: tarfile.TarFile | None = None
    manifest = []

    def open_shard() -> tarfile.TarFile:
        nonlocal shard_index, shard_count
        shard_count = 0
        path = out / f"vl_{shard_index:06d}.tar"
        shard_index += 1
        return tarfile.open(path, "w")

    try:
        tar = open_shard()
        for row in tqdm(iter_manifest(args.manifest), total=args.max_samples, dynamic_ncols=True):
            if args.max_samples is not None and total >= args.max_samples:
                break
            image_path = resolve_image(str(row["image"]), args.image_root)
            if not image_path.exists():
                if args.skip_missing:
                    skipped += 1
                    continue
                raise FileNotFoundError(f"Missing image: {image_path}")
            if shard_count >= args.samples_per_shard:
                tar.close()
                tar = open_shard()
            key = f"{total:09d}"
            jpg = encode_jpeg(image_path, image_size=args.image_size, quality=args.jpeg_quality)
            meta = {
                "source": row.get("source", "unknown"),
                "image": str(row["image"]),
                "caption": str(row["caption"]),
            }
            add_bytes(tar, f"{key}.jpg", jpg)
            add_bytes(tar, f"{key}.json", json.dumps(meta).encode("utf-8"))
            total += 1
            shard_count += 1
        if tar is not None:
            tar.close()
    finally:
        if tar is not None:
            tar.close()

    for path in sorted(out.glob("vl_*.tar")):
        manifest.append({"path": path.name, "bytes": path.stat().st_size})
    summary = {
        "samples": total,
        "skipped": skipped,
        "shards": manifest,
        "image_size": args.image_size,
        "jpeg_quality": args.jpeg_quality,
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
