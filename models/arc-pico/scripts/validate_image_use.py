#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from arc.checkpoint import load_checkpoint
from arc.config import load_config
from arc.data_vision import ImageCaptionDataset, collate_image_caption
from arc.model import ArcModel
from arc.tokenizer import ArcTokenizer


@torch.no_grad()
def batch_loss(model: ArcModel, batch: dict[str, torch.Tensor], device: torch.device) -> float:
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    _, loss = model(
        batch["input_ids"],
        labels=batch["labels"],
        images=batch["images"],
        image_insert_positions=batch["image_insert_positions"],
    )
    return float(loss.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether Arc caption loss depends on the image")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_dir", default="tokenizer_16k")
    parser.add_argument("--image_manifest", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow_missing_images", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = ArcTokenizer.from_dir(args.tokenizer_dir)
    device = torch.device(args.device)
    model = ArcModel(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    dataset = ImageCaptionDataset(
        args.image_manifest,
        tokenizer,
        image_root=args.image_root,
        image_size=cfg.image_size,
        visual_tokens=cfg.vision_resampler_tokens,
        max_seq_len=cfg.max_seq_len,
        allow_missing_images=args.allow_missing_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=lambda batch: collate_image_caption(batch, tokenizer.pad_id),
    )
    correct = []
    wrong = []
    blank = []
    for idx, batch in enumerate(loader):
        if idx >= args.batches:
            break
        correct.append(batch_loss(model, batch, device))
        wrong_batch = dict(batch)
        wrong_batch["images"] = batch["images"].roll(shifts=1, dims=0)
        wrong.append(batch_loss(model, wrong_batch, device))
        blank_batch = dict(batch)
        blank_batch["images"] = torch.zeros_like(batch["images"])
        blank.append(batch_loss(model, blank_batch, device))
    if not correct:
        raise SystemExit("No validation batches produced; increase data or lower batch size")
    metrics = {
        "correct_loss": sum(correct) / len(correct),
        "wrong_loss": sum(wrong) / len(wrong),
        "blank_loss": sum(blank) / len(blank),
        "correct_better_than_wrong": (sum(correct) / len(correct)) < (sum(wrong) / len(wrong)),
        "correct_better_than_blank": (sum(correct) / len(correct)) < (sum(blank) / len(blank)),
        "batches": len(correct),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
