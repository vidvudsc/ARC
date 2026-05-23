from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import DatasetSpec, read_mixture


@dataclass(frozen=True)
class TextShardPlan:
    mixture_path: Path
    seq_len: int
    target_tokens: int
    specs: list[DatasetSpec]


def load_text_plan(mixture_path: str | Path, seq_len: int = 1024) -> TextShardPlan:
    path = Path(mixture_path)
    specs = read_mixture(path)
    target = 0
    for spec in specs:
        value = spec.extra.get("target_tokens")
        if isinstance(value, int):
            target += value
    return TextShardPlan(mixture_path=path, seq_len=seq_len, target_tokens=target, specs=specs)


def expected_uint16_size_bytes(tokens: int) -> int:
    return tokens * 2


@dataclass(frozen=True)
class TokenShard:
    path: Path
    tokens: int


def discover_token_shards(path: str | Path) -> list[TokenShard]:
    root = Path(path)
    files = sorted(root.glob("*.bin")) if root.is_dir() else [root]
    shards: list[TokenShard] = []
    for file in files:
        if file.suffix != ".bin":
            continue
        size = file.stat().st_size
        if size % 2 != 0:
            raise ValueError(f"Token shard byte size is not uint16-aligned: {file}")
        shards.append(TokenShard(path=file, tokens=size // 2))
    if not shards:
        raise FileNotFoundError(f"No .bin token shards found under {root}")
    return shards


class PackedMemmapDataset(Dataset[torch.Tensor]):
    def __init__(self, shard_path: str | Path, seq_len: int = 1024, seed: int = 1337, stride: int | None = None):
        self.shards = discover_token_shards(shard_path)
        self.seq_len = seq_len
        self.seed = seed
        self.stride = seq_len if stride is None else int(stride)
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        self.arrays = [np.memmap(shard.path, dtype=np.uint16, mode="r") for shard in self.shards]
        self.cum = np.cumsum(
            [max(0, (len(array) - seq_len - 1) // self.stride + 1) for array in self.arrays],
            dtype=np.int64,
        )
        self.total_windows = int(self.cum[-1])
        if self.total_windows <= 0:
            raise ValueError("Token shards are too small for the requested sequence length")

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        idx = int(idx) % self.total_windows
        shard_idx = int(np.searchsorted(self.cum, idx, side="right"))
        prev = 0 if shard_idx == 0 else int(self.cum[shard_idx - 1])
        local = idx - prev
        array = self.arrays[shard_idx]
        start = local * self.stride
        view = np.array(array[start : start + self.seq_len + 1], dtype=np.int64, copy=True)
        return torch.from_numpy(view)


def split_inputs_labels(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return batch[:, :-1].contiguous().long(), batch[:, 1:].contiguous().long()
