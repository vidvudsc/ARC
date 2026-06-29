from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset: str
    config: str
    split: str
    weight: float
    role: str
    notes: str = ""
    extra: dict[str, object] = field(default_factory=dict)


def read_mixture(path: str | Path) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            known = {key: raw.pop(key) for key in list(raw.keys()) if key in DatasetSpec.__dataclass_fields__}
            specs.append(DatasetSpec(**known, extra=raw))
    total = sum(spec.weight for spec in specs)
    if not specs:
        raise ValueError(f"No dataset specs in {path}")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Mixture weights in {path} sum to {total}, expected 1.0")
    return specs


def describe_mixture(specs: Iterable[DatasetSpec]) -> str:
    lines = []
    for spec in specs:
        pct = 100 * spec.weight
        lines.append(f"{pct:5.1f}%  {spec.name:24s}  {spec.role}")
    return "\n".join(lines)
