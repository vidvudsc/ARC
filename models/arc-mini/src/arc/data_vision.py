from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .data import DatasetSpec, read_mixture
from .tokenizer import ArcTokenizer


DEFAULT_CAPTION_PREFIXES = [
    "",
    "A photo of ",
    "The image shows ",
    "In this scene, ",
    "This is ",
]


@dataclass(frozen=True)
class VisionFilter:
    min_tokens: int
    max_tokens: int
    strict_max_tokens: int
    drop_substrings_case_insensitive: list[str]


def load_vision_filter(path: str | Path) -> VisionFilter:
    raw = json.loads(Path(path).read_text())
    return VisionFilter(
        min_tokens=int(raw["min_tokens"]),
        max_tokens=int(raw["max_tokens"]),
        strict_max_tokens=int(raw.get("strict_max_tokens", raw["max_tokens"])),
        drop_substrings_case_insensitive=list(raw["drop_substrings_case_insensitive"]),
    )


def is_caption_candidate(text: str, flt: VisionFilter, strict: bool = False) -> bool:
    words = text.split()
    max_tokens = flt.strict_max_tokens if strict else flt.max_tokens
    if len(words) < flt.min_tokens or len(words) > max_tokens:
        return False
    lower = text.lower()
    return not any(bad.lower() in lower for bad in flt.drop_substrings_case_insensitive)


def normalize_caption(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def extract_caption(row: dict[str, Any], *, skip_conversations: bool = True) -> str | None:
    if skip_conversations and "conversations" in row:
        return None
    for key in ("caption", "captions", "text", "description", "value"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_caption(value)
        if isinstance(value, list):
            strings = [normalize_caption(item) for item in value if isinstance(item, str) and item.strip()]
            if strings:
                return strings[0]
    return None


def extract_image_ref(row: dict[str, Any]) -> str | None:
    for key in ("image", "image_path", "path", "file_name", "filename", "url"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_vl_plan(mixture_path: str | Path) -> list[DatasetSpec]:
    return read_mixture(mixture_path)


class ImageCaptionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path,
        tokenizer: ArcTokenizer,
        *,
        image_root: str | Path | None = None,
        image_size: int = 224,
        visual_tokens: int = 32,
        max_seq_len: int = 1024,
        allow_missing_images: bool = False,
        caption_prefixes: list[str] | None = None,
        prefix_seed: int = 1337,
    ):
        self.manifest = Path(manifest)
        self.tokenizer = tokenizer
        self.image_root = Path(image_root) if image_root else None
        self.image_size = image_size
        self.visual_tokens = visual_tokens
        self.max_seq_len = max_seq_len
        self.allow_missing_images = allow_missing_images
        self.caption_prefixes = DEFAULT_CAPTION_PREFIXES if caption_prefixes is None else caption_prefixes
        self.prefix_seed = prefix_seed
        self.rows = []
        with self.manifest.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("image") and row.get("caption"):
                        self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No image-caption rows found in {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image(self, image_ref: str) -> Path:
        path = Path(image_ref)
        if path.is_absolute() or self.image_root is None:
            return path
        return self.image_root / path

    def _load_image(self, image_ref: str) -> torch.Tensor:
        path = self._resolve_image(image_ref)
        if not path.exists():
            if self.allow_missing_images:
                return torch.zeros(3, self.image_size, self.image_size)
            raise FileNotFoundError(f"Missing image: {path}")
        with Image.open(path) as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
            data = torch.from_numpy(array).permute(2, 0, 1)
        return (data - 0.5) / 0.5

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        rng = random.Random(self.prefix_seed + int(idx))
        text_prefix = rng.choice(self.caption_prefixes) if self.caption_prefixes else ""
        caption_ids = self.tokenizer.encode(text_prefix + str(row["caption"]), add_bos=False, add_eos=True)
        prefix = [self.tokenizer.image_start_id] + [self.tokenizer.pad_id] * self.visual_tokens + [
            self.tokenizer.image_end_id
        ]
        max_caption = max(1, self.max_seq_len + 1 - len(prefix))
        seq = prefix + caption_ids[:max_caption]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)
        labels[: self.visual_tokens + 1] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "image": self._load_image(str(row["image"])),
            "image_insert_position": 1,
        }


def collate_image_caption(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = []
    labels = []
    for item in batch:
        pad = max_len - item["input_ids"].numel()
        input_ids.append(F.pad(item["input_ids"], (0, pad), value=pad_id))
        labels.append(F.pad(item["labels"], (0, pad), value=-100))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "images": torch.stack([item["image"] for item in batch]),
        "image_insert_positions": torch.tensor([item["image_insert_position"] for item in batch], dtype=torch.long),
    }


@dataclass(frozen=True)
class ImageCaptionCollator:
    pad_id: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return collate_image_caption(batch, self.pad_id)
