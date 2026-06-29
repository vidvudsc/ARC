from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from typing import Iterator


def to_plain_dict(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    return value


@contextmanager
def timed() -> Iterator[callable]:
    start = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - start

    yield elapsed


def format_count(value: int | float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)

