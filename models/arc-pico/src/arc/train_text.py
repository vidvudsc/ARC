from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import time
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
from .model import ArcModel, estimate_parameters
from .utils import format_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arc Stage 1 text-only training scaffold")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mixture", default="configs/stage1_text_mix.jsonl")
    parser.add_argument("--checkpoint_plan", default="configs/checkpoints.json")
    parser.add_argument("--token_shards", default=None)
    parser.add_argument("--val_token_shards", default=None)
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--keep_step_checkpoints", type=int, default=3)
    parser.add_argument("--target_tokens", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log_format", choices=["json", "pretty"], default="json")
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
    if "step" in metrics:
        parts.append(f"step {int(metrics['step']):>6}")
    if "loss" in metrics:
        parts.append(f"loss {float(metrics['loss']):.4f}")
    if "val_loss" in metrics:
        parts.append(f"val {float(metrics['val_loss']):.4f}")
    if "lr" in metrics:
        parts.append(f"lr {float(metrics['lr']):.2e}")
    if "tokens_seen" in metrics:
        parts.append(f"tok {int(metrics['tokens_seen']):,}")
    if "interval_tokens_per_second" in metrics:
        parts.append(f"it/s {float(metrics['interval_tokens_per_second']):,.0f}")
    if "tokens_per_second" in metrics:
        parts.append(f"avg/s {float(metrics['tokens_per_second']):,.0f}")
    if "peak_memory_gb" in metrics:
        parts.append(f"mem {float(metrics['peak_memory_gb']):.2f}GB")
    print(" | ".join(parts), flush=True)


def configure_stage1_trainable(model: ArcModel) -> None:
    for param in model.vision.parameters():
        param.requires_grad_(False)


def load_stage1_checkpoint_targets(path: str | Path) -> list[dict[str, int | str]]:
    data = json.loads(Path(path).read_text())
    targets = data.get("stage1", [])
    normalized = []
    for item in targets:
        tokens = int(item["tokens"])
        checkpoint_path = str(item["path"])
        normalized.append({"tokens": tokens, "path": checkpoint_path})
    return sorted(normalized, key=lambda item: int(item["tokens"]))


@torch.no_grad()
def validate(
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
        input_ids, labels = split_inputs_labels(batch)
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(input_ids, labels=labels)
        total_loss += loss.detach()
        total_batches += 1
    if distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)
    model.train()
    if int(total_batches.item()) == 0:
        return float("nan")
    return float((total_loss / total_batches.clamp_min(1)).detach().cpu())


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: object,
    step: int,
    tokens_seen: int,
    extra: dict[str, object] | None = None,
) -> None:
    payload_extra = {"stage": "text"}
    if extra:
        payload_extra.update(extra)
    save_checkpoint(
        path,
        checkpoint_payload(
            model=unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            config=asdict(cfg),
            step=step,
            tokens_seen=tokens_seen,
            extra=payload_extra,
        ),
    )


def rotate_step_checkpoints(out_dir: str | Path, keep: int) -> None:
    if keep < 0:
        return
    root = Path(out_dir)
    checkpoints = sorted(
        root.glob("arc_pico_text_step_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old in checkpoints[keep:]:
        old.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = setup_distributed()
    cfg = load_config(args.config)
    plan = load_text_plan(args.mixture, seq_len=cfg.max_seq_len)
    checkpoint_targets = load_stage1_checkpoint_targets(args.checkpoint_plan)
    params = estimate_parameters(cfg)
    if is_main_process(rank):
        print(f"model: {cfg.model_name}")
        print(f"context: {cfg.max_seq_len}")
        print(f"params total: {params['total']:,}")
        print(f"params text+embeddings: {params['text_plus_embeddings']:,}")
        print(f"params vision: {params['vision']:,}")
        print(f"target tokens: {format_count(plan.target_tokens)}")
        print(f"uint16 token storage: {expected_uint16_size_bytes(plan.target_tokens) / 1e9:.1f} GB")
        print("checkpoint token targets: " + ", ".join(format_count(int(item["tokens"])) for item in checkpoint_targets))
        print(f"distributed: {world_size} process(es)")
        print("\ntext mixture:")
        print(describe_mixture(plan.specs))
    if args.dry_run:
        if distributed:
            dist.destroy_process_group()
        return
    if args.token_shards is None:
        raise SystemExit("Pass --token_shards pointing to uint16 .bin shards from scripts/shard_text.py")

    if distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision(args.matmul_precision)
    model = ArcModel(cfg).to(device)
    configure_stage1_trainable(model)
    if args.compile:
        model = torch.compile(model)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    dataset = PackedMemmapDataset(args.token_shards, seq_len=cfg.max_seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        drop_last=True,
    )
    val_loader = None
    if args.val_token_shards:
        val_dataset = PackedMemmapDataset(args.val_token_shards, seq_len=cfg.max_seq_len)
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            drop_last=False,
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
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0))
        tokens_seen = int(ckpt.get("tokens_seen", 0))

    use_amp = device.type == "cuda" and args.dtype == "bfloat16"
    data_iter = iter(loader)
    t0 = time.time()
    last_log_time = t0
    last_log_tokens = tokens_seen
    running_loss = 0.0
    saved_token_targets = {int(item["tokens"]) for item in checkpoint_targets if int(item["tokens"]) <= tokens_seen}
    target_tokens = args.target_tokens if args.target_tokens is not None else plan.target_tokens
    model.train()
    for step in range(start_step, args.max_steps):
        if sampler is not None:
            sampler.set_epoch(step)
        lr = get_lr(step, args.max_steps, args.lr, args.min_lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0
        for micro_step in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            input_ids, labels = split_inputs_labels(batch)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            step_tokens += int(input_ids.numel())
            sync_context = (
                model.no_sync()
                if distributed and isinstance(model, DistributedDataParallel) and micro_step < args.grad_accum - 1
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    _, loss = model(input_ids, labels=labels)
                    loss = loss / args.grad_accum
                loss.backward()
            step_loss += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], args.grad_clip)
        optimizer.step()
        scheduler.step()
        global_step_tokens = step_tokens * world_size
        tokens_seen += global_step_tokens
        running_loss = step_loss if step == start_step else 0.95 * running_loss + 0.05 * step_loss
        if step % args.log_every == 0 and is_main_process(rank):
            now = time.time()
            elapsed = max(now - t0, 1e-6)
            interval_elapsed = max(now - last_log_time, 1e-6)
            metrics = {
                "step": step,
                "loss": running_loss,
                "lr": lr,
                "tokens_seen": tokens_seen,
                "tokens_per_second": tokens_seen / elapsed,
                "interval_tokens_per_second": (tokens_seen - last_log_tokens) / interval_elapsed,
            }
            if device.type == "cuda":
                metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            emit_metrics(metrics, args.log_format)
            last_log_time = now
            last_log_tokens = tokens_seen
        if args.eval_every > 0 and val_loader is not None and (step + 1) % args.eval_every == 0:
            val_loss = validate(
                model,
                val_loader,
                device=device,
                batches=args.val_batches,
                use_amp=use_amp,
                distributed=distributed,
            )
            if is_main_process(rank):
                emit_metrics({"step": step + 1, "val_loss": val_loss, "tokens_seen": tokens_seen}, args.log_format)
        if is_main_process(rank) and (step + 1) % args.save_every == 0:
            save_training_checkpoint(
                Path(args.out_dir) / f"arc_pico_text_step_{step + 1}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                step=step + 1,
                tokens_seen=tokens_seen,
            )
            rotate_step_checkpoints(args.out_dir, args.keep_step_checkpoints)
        if is_main_process(rank):
            for item in checkpoint_targets:
                token_target = int(item["tokens"])
                if token_target not in saved_token_targets and tokens_seen >= token_target:
                    save_training_checkpoint(
                        item["path"],
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cfg=cfg,
                        step=step + 1,
                        tokens_seen=tokens_seen,
                        extra={"checkpoint_token_target": token_target},
                    )
                    saved_token_targets.add(token_target)
        if target_tokens and tokens_seen >= target_tokens:
            break

    if is_main_process(rank):
        save_training_checkpoint(
            Path(args.out_dir) / "arc_pico_text_last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            step=step + 1,
            tokens_seen=tokens_seen,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
