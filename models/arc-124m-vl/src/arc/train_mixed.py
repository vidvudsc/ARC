from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import checkpoint_payload, load_checkpoint, save_checkpoint
from .config import load_config
from .data import describe_mixture
from .data_text import PackedMemmapDataset, expected_uint16_size_bytes, load_text_plan, split_inputs_labels
from .data_vision import DEFAULT_CAPTION_PREFIXES, ImageCaptionDataset, collate_image_caption
from .model import ArcModel, estimate_parameters
from .tokenizer import ArcTokenizer
from .utils import format_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arc mixed text+image pretraining")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mixture", default="configs/stage1_text_mix.jsonl")
    parser.add_argument("--tokenizer_dir", default="tokenizer_32k")
    parser.add_argument("--token_shards", required=True)
    parser.add_argument("--val_token_shards", default=None)
    parser.add_argument("--image_manifest", required=True)
    parser.add_argument("--val_image_manifest", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--out_dir", default="checkpoints_mixed")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32, help="Text batch size per process")
    parser.add_argument("--image_batch_size", type=int, default=None, help="Image batch size per process; defaults to batch_size")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--target_tokens", type=int, default=20_000_000_000)
    parser.add_argument("--image_weight", type=float, default=0.04)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--keep_step_checkpoints", type=int, default=3)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log_format", choices=["json", "pretty"], default="pretty")
    parser.add_argument("--caption_prefixes", default="|".join(DEFAULT_CAPTION_PREFIXES))
    parser.add_argument("--allow_missing_images", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def get_lr(step: int, max_steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return distributed, rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def emit_metrics(metrics: dict[str, object], log_format: str) -> None:
    if log_format == "json":
        print(json.dumps(metrics), flush=True)
        return
    parts = []
    for key in ("step", "batch"):
        if key in metrics:
            parts.append(f"{key} {metrics[key]}")
    if "loss" in metrics:
        parts.append(f"loss {float(metrics['loss']):.4f}")
    if "text_val_loss" in metrics:
        parts.append(f"text_val {float(metrics['text_val_loss']):.4f}")
    if "image_val_loss" in metrics:
        parts.append(f"img_val {float(metrics['image_val_loss']):.4f}")
    if "wrong_image_loss" in metrics:
        parts.append(f"wrong {float(metrics['wrong_image_loss']):.4f}")
    if "blank_image_loss" in metrics:
        parts.append(f"blank {float(metrics['blank_image_loss']):.4f}")
    if "lr" in metrics:
        parts.append(f"lr {float(metrics['lr']):.2e}")
    if "tokens_seen" in metrics:
        parts.append(f"tok {int(metrics['tokens_seen']):,}")
    if "text_batches" in metrics and "image_batches" in metrics:
        parts.append(f"text/image {int(metrics['text_batches'])}/{int(metrics['image_batches'])}")
    if "interval_tokens_per_second" in metrics:
        parts.append(f"it/s {float(metrics['interval_tokens_per_second']):,.0f}")
    if "tokens_per_second" in metrics:
        parts.append(f"avg/s {float(metrics['tokens_per_second']):,.0f}")
    if "peak_memory_gb" in metrics:
        parts.append(f"mem {float(metrics['peak_memory_gb']):.2f}GB")
    if "correct_better_than_wrong" in metrics:
        parts.append(f"correct<wrong {bool(metrics['correct_better_than_wrong'])}")
    if "correct_better_than_blank" in metrics:
        parts.append(f"correct<blank {bool(metrics['correct_better_than_blank'])}")
    print(" | ".join(parts), flush=True)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def next_batch(loader: DataLoader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def choose_batch_kind(
    rng: random.Random,
    image_weight: float,
    *,
    distributed: bool,
    rank: int,
    device: torch.device,
) -> bool:
    if not distributed:
        return rng.random() < image_weight
    flag = torch.empty((), device=device, dtype=torch.int64)
    if is_main_process(rank):
        flag.fill_(1 if rng.random() < image_weight else 0)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def text_forward_loss(
    model: torch.nn.Module,
    batch: torch.Tensor,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, int]:
    input_ids, labels = split_inputs_labels(batch)
    input_ids = input_ids.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        _, loss = model(input_ids, labels=labels)
    return loss, int(input_ids.numel())


def image_forward_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, int]:
    batch = batch_to_device(batch, device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        _, loss = model(
            batch["input_ids"],
            labels=batch["labels"],
            images=batch["images"],
            image_insert_positions=batch["image_insert_positions"],
        )
    return loss, int(batch["input_ids"].numel())


@torch.no_grad()
def validate_text_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    batches: int,
    use_amp: bool,
    distributed: bool,
) -> float:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_batches = torch.tensor(0, device=device, dtype=torch.long)
    data_iter = iter(loader)
    for _ in range(max(0, batches)):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        loss, _ = text_forward_loss(model, batch, device=device, use_amp=use_amp)
        total_loss += loss.detach()
        total_batches += 1
    if distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)
    model.train()
    if int(total_batches.item()) == 0:
        return float("nan")
    return float((total_loss / total_batches.clamp_min(1)).detach().cpu())


@torch.no_grad()
def validate_image_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    batches: int,
    use_amp: bool,
    distributed: bool,
) -> dict[str, float | bool | int]:
    model.eval()
    correct_sum = torch.tensor(0.0, device=device)
    wrong_sum = torch.tensor(0.0, device=device)
    blank_sum = torch.tensor(0.0, device=device)
    count = torch.tensor(0, device=device, dtype=torch.long)
    data_iter = iter(loader)
    for _ in range(max(0, batches)):
        try:
            raw = next(data_iter)
        except StopIteration:
            break
        batch = batch_to_device(raw, device)
        loss, _ = image_forward_loss(model, batch, device=device, use_amp=use_amp)
        correct_sum += loss.detach()
        wrong_batch = dict(batch)
        wrong_batch["images"] = batch["images"].roll(shifts=1, dims=0)
        wrong_loss, _ = image_forward_loss(model, wrong_batch, device=device, use_amp=use_amp)
        wrong_sum += wrong_loss.detach()
        blank_batch = dict(batch)
        blank_batch["images"] = torch.zeros_like(batch["images"])
        blank_loss, _ = image_forward_loss(model, blank_batch, device=device, use_amp=use_amp)
        blank_sum += blank_loss.detach()
        count += 1
    if distributed:
        for value in (correct_sum, wrong_sum, blank_sum, count):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    model.train()
    n = max(int(count.item()), 1)
    correct = float((correct_sum / n).detach().cpu())
    wrong = float((wrong_sum / n).detach().cpu())
    blank = float((blank_sum / n).detach().cpu())
    return {
        "image_val_loss": correct,
        "wrong_image_loss": wrong,
        "blank_image_loss": blank,
        "correct_better_than_wrong": correct < wrong,
        "correct_better_than_blank": correct < blank,
        "batches": int(count.item()),
    }


def rotate_step_checkpoints(out_dir: str | Path, keep: int) -> None:
    if keep < 0:
        return
    root = Path(out_dir)
    checkpoints = sorted(root.glob("arc124_mixed_step_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    for old in checkpoints[keep:]:
        old.unlink(missing_ok=True)


def save_mixed_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: object,
    step: int,
    tokens_seen: int,
    text_batches: int,
    image_batches: int,
) -> None:
    save_checkpoint(
        path,
        checkpoint_payload(
            model=unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            config=asdict(cfg),
            step=step,
            tokens_seen=tokens_seen,
            extra={
                "stage": "mixed_pretrain",
                "text_batches": text_batches,
                "image_batches": image_batches,
            },
        ),
    )


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.image_weight <= 1.0):
        raise SystemExit("--image_weight must be between 0 and 1")
    distributed, rank, local_rank, world_size = setup_distributed()
    cfg = load_config(args.config)
    plan = load_text_plan(args.mixture, seq_len=cfg.max_seq_len)
    params = estimate_parameters(cfg)
    image_batch_size = args.image_batch_size or args.batch_size
    if is_main_process(rank):
        print(f"model: {cfg.model_name}")
        print(f"mode: mixed text+image pretraining")
        print(f"context: {cfg.max_seq_len}")
        print(f"params total: {params['total']:,}")
        print(f"params text+embeddings: {params['text_plus_embeddings']:,}")
        print(f"params vision: {params['vision']:,}")
        print(f"target tokens: {format_count(args.target_tokens)}")
        print(f"text/image batch probability: {1.0 - args.image_weight:.3f}/{args.image_weight:.3f}")
        print(f"text batch size per process: {args.batch_size}")
        print(f"image batch size per process: {image_batch_size}")
        print(f"uint16 token storage at target: {expected_uint16_size_bytes(args.target_tokens) / 1e9:.1f} GB")
        print(f"distributed: {world_size} process(es)")
        print("\ntext mixture:")
        print(describe_mixture(plan.specs))
    if args.dry_run:
        if distributed:
            dist.destroy_process_group()
        return

    tokenizer = ArcTokenizer.from_dir(args.tokenizer_dir)
    if distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision(args.matmul_precision)

    model = ArcModel(cfg, pad_id=tokenizer.pad_id).to(device)
    if args.compile:
        model = torch.compile(model)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    start_step = 0
    tokens_seen = 0
    text_batches = 0
    image_batches = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model"], strict=False)
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0))
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        text_batches = int(ckpt.get("text_batches", 0))
        image_batches = int(ckpt.get("image_batches", 0))
        if is_main_process(rank):
            print(f"resumed checkpoint: {args.resume}")

    text_dataset = PackedMemmapDataset(args.token_shards, seq_len=cfg.max_seq_len)
    text_sampler = DistributedSampler(text_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    text_loader = DataLoader(
        text_dataset,
        batch_size=args.batch_size,
        shuffle=text_sampler is None,
        sampler=text_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        drop_last=True,
    )
    val_text_loader = None
    if args.val_token_shards:
        val_text_dataset = PackedMemmapDataset(args.val_token_shards, seq_len=cfg.max_seq_len)
        val_text_sampler = (
            DistributedSampler(val_text_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
        )
        val_text_loader = DataLoader(
            val_text_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_text_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            drop_last=True,
        )

    caption_prefixes = args.caption_prefixes.split("|")
    image_dataset = ImageCaptionDataset(
        args.image_manifest,
        tokenizer,
        image_root=args.image_root,
        image_size=cfg.image_size,
        visual_tokens=cfg.vision_resampler_tokens,
        max_seq_len=cfg.max_seq_len,
        allow_missing_images=args.allow_missing_images,
        caption_prefixes=caption_prefixes,
    )
    image_sampler = DistributedSampler(image_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    image_loader = DataLoader(
        image_dataset,
        batch_size=image_batch_size,
        shuffle=image_sampler is None,
        sampler=image_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        drop_last=True,
        collate_fn=lambda batch: collate_image_caption(batch, tokenizer.pad_id),
    )
    val_image_loader = None
    if args.val_image_manifest:
        val_image_dataset = ImageCaptionDataset(
            args.val_image_manifest,
            tokenizer,
            image_root=args.image_root,
            image_size=cfg.image_size,
            visual_tokens=cfg.vision_resampler_tokens,
            max_seq_len=cfg.max_seq_len,
            allow_missing_images=args.allow_missing_images,
            caption_prefixes=caption_prefixes,
        )
        val_image_sampler = (
            DistributedSampler(val_image_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            if distributed
            else None
        )
        val_image_loader = DataLoader(
            val_image_dataset,
            batch_size=image_batch_size,
            shuffle=False,
            sampler=val_image_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            drop_last=True,
            collate_fn=lambda batch: collate_image_caption(batch, tokenizer.pad_id),
        )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda" and args.dtype == "bfloat16"
    text_iter = iter(text_loader)
    image_iter = iter(image_loader)
    rng = random.Random(1337 + start_step)
    t0 = time.time()
    last_log_time = t0
    last_log_tokens = tokens_seen
    running_loss = 0.0
    model.train()
    step = start_step
    for step in range(start_step, args.max_steps):
        if distributed:
            if text_sampler is not None:
                text_sampler.set_epoch(step)
            if image_sampler is not None:
                image_sampler.set_epoch(step)
        lr = get_lr(step, args.max_steps, args.lr, args.min_lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0
        step_batch_kind = "mixed"
        for micro_step in range(args.grad_accum):
            use_image = choose_batch_kind(
                rng,
                args.image_weight,
                distributed=distributed,
                rank=rank,
                device=device,
            )
            sync_context = (
                model.no_sync()
                if distributed and isinstance(model, DistributedDataParallel) and micro_step < args.grad_accum - 1
                else nullcontext()
            )
            with sync_context:
                if use_image:
                    batch, image_iter = next_batch(image_loader, image_iter)
                    loss, batch_tokens = image_forward_loss(model, batch, device=device, use_amp=use_amp)
                    image_batches += 1
                    step_batch_kind = "image" if args.grad_accum == 1 else "mixed"
                else:
                    batch, text_iter = next_batch(text_loader, text_iter)
                    loss, batch_tokens = text_forward_loss(model, batch, device=device, use_amp=use_amp)
                    text_batches += 1
                    step_batch_kind = "text" if args.grad_accum == 1 else "mixed"
                loss = loss / args.grad_accum
                loss.backward()
            step_loss += float(loss.detach().cpu())
            step_tokens += batch_tokens
        torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], args.grad_clip)
        optimizer.step()
        scheduler.step()
        step_tokens_tensor = torch.tensor(step_tokens, device=device, dtype=torch.long)
        step_loss_tensor = torch.tensor(step_loss, device=device)
        if distributed:
            dist.all_reduce(step_tokens_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(step_loss_tensor, op=dist.ReduceOp.SUM)
            step_loss_for_log = float((step_loss_tensor / world_size).detach().cpu())
        else:
            step_loss_for_log = step_loss
        tokens_seen += int(step_tokens_tensor.item())
        running_loss = step_loss_for_log if step == start_step else 0.95 * running_loss + 0.05 * step_loss_for_log
        if step % args.log_every == 0 and is_main_process(rank):
            now = time.time()
            elapsed = max(now - t0, 1e-6)
            interval_elapsed = max(now - last_log_time, 1e-6)
            metrics = {
                "step": step,
                "batch": step_batch_kind,
                "loss": running_loss,
                "lr": lr,
                "tokens_seen": tokens_seen,
                "text_batches": text_batches,
                "image_batches": image_batches,
                "tokens_per_second": tokens_seen / elapsed,
                "interval_tokens_per_second": (tokens_seen - last_log_tokens) / interval_elapsed,
            }
            if device.type == "cuda":
                metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            emit_metrics(metrics, args.log_format)
            last_log_time = now
            last_log_tokens = tokens_seen
        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            if is_main_process(rank):
                eval_metrics: dict[str, object] = {"step": step + 1, "tokens_seen": tokens_seen}
            else:
                eval_metrics = {}
            if val_text_loader is not None:
                text_val = validate_text_loss(
                    model,
                    val_text_loader,
                    device=device,
                    batches=args.val_batches,
                    use_amp=use_amp,
                    distributed=distributed,
                )
                if is_main_process(rank):
                    eval_metrics["text_val_loss"] = text_val
            if val_image_loader is not None:
                image_metrics = validate_image_loss(
                    model,
                    val_image_loader,
                    device=device,
                    batches=args.val_batches,
                    use_amp=use_amp,
                    distributed=distributed,
                )
                if is_main_process(rank):
                    eval_metrics.update(image_metrics)
            if is_main_process(rank) and len(eval_metrics) > 2:
                emit_metrics(eval_metrics, args.log_format)
        if is_main_process(rank) and (step + 1) % args.save_every == 0:
            save_mixed_checkpoint(
                Path(args.out_dir) / f"arc124_mixed_step_{step + 1}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                step=step + 1,
                tokens_seen=tokens_seen,
                text_batches=text_batches,
                image_batches=image_batches,
            )
            rotate_step_checkpoints(args.out_dir, args.keep_step_checkpoints)
        if args.target_tokens and tokens_seen >= args.target_tokens:
            break

    if is_main_process(rank):
        save_mixed_checkpoint(
            Path(args.out_dir) / "arc124_mixed_last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            step=step + 1,
            tokens_seen=tokens_seen,
            text_batches=text_batches,
            image_batches=image_batches,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
