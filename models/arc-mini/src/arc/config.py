from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ArcConfig:
    model_name: str = "Arc-Mini-VL-v0.1"
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 10
    n_heads: int = 8
    ffn_hidden: int = 1792
    max_seq_len: int = 512
    dropout: float = 0.0
    tie_embeddings: bool = True
    image_size: int = 224
    patch_size: int = 16
    vision_width: int = 256
    vision_layers: int = 3
    vision_heads: int = 4
    vision_resampler_tokens: int = 24

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    @property
    def n_image_patches(self) -> int:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        return (self.image_size // self.patch_size) ** 2


def load_config(path: str | Path) -> ArcConfig:
    data: dict[str, Any] = json.loads(Path(path).read_text())
    allowed = set(ArcConfig.__dataclass_fields__.keys())
    return ArcConfig(**{key: value for key, value in data.items() if key in allowed})
