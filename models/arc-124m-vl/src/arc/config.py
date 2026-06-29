from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ArcConfig:
    model_name: str = "Arc-124M-VL-v0.3"
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ffn_hidden: int = 2176
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True
    image_size: int = 224
    patch_size: int = 16
    vision_width: int = 384
    vision_layers: int = 4
    vision_heads: int = 6
    vision_resampler_tokens: int = 32

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
