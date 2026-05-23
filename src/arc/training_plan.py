from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .model import ArcModel, count_trainable_parameters


@dataclass(frozen=True)
class Stage2Trainability:
    phase: str
    optimizer_visible_parameters: int
    effective_trainable_parameters: int
    total_parameters: int
    notes: str


def configure_stage2_phase(
    model: ArcModel,
    phase: str,
    image_start_id: int,
    image_end_id: int,
    final_layers: int = 2,
) -> Stage2Trainability:
    total = sum(p.numel() for p in model.parameters())
    if phase == "stage2a":
        model.configure_stage2a_trainable(image_start_id, image_end_id)
        notes = "vision/projector + image boundary embedding rows"
    elif phase == "stage2b":
        model.configure_stage2b_trainable(image_start_id, image_end_id, final_layers=final_layers)
        notes = f"stage2a trainables + final {final_layers} decoder layers"
    elif phase == "stage2c":
        model.configure_stage2c_trainable()
        notes = "full low-LR unfreeze"
    else:
        raise ValueError(f"Unknown Stage 2 phase: {phase}")
    optimizer_visible = count_trainable_parameters(model)
    embedding_params = model.tok_emb.weight.numel()
    boundary_embedding_params = 2 * model.cfg.d_model
    effective = optimizer_visible
    if phase in {"stage2a", "stage2b"}:
        effective = optimizer_visible - embedding_params + boundary_embedding_params
    return Stage2Trainability(
        phase=phase,
        optimizer_visible_parameters=optimizer_visible,
        effective_trainable_parameters=effective,
        total_parameters=total,
        notes=notes,
    )


def load_curriculum(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())
