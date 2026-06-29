from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import checkpoint_payload, load_checkpoint, save_checkpoint
from .config import load_config
from .data import describe_mixture
from .data_text import PackedMemmapDataset, split_inputs_labels
from .data_vision import DEFAULT_CAPTION_PREFIXES, ImageCaptionDataset, collate_image_caption, load_vision_filter, load_vl_plan
from .model import ArcModel, estimate_parameters
from .tokenizer import ArcTokenizer
from .training_plan import configure_stage2_phase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arc Stage 2 image-caption grounding scaffold")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mixture", default="configs/stage2_vl_mix.jsonl")
    parser.add_argument("--filter", default="configs/caption_filter.json")
    parser.add_argument("--checkpoint", default="checkpoints/arc124_text_30b.pt")
    parser.add_argument("--tokenizer_dir", default="tokenizer_32k")
    parser.add_argument("--image_manifest", default=None)
    parser.add_argument("--val_image_manifest", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--text_replay_shards", default=None)
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--phase", choices=["stage2a", "stage2b", "stage2c"], default="stage2a")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--text_batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--text_replay_weight", type=float, default=0.60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--lm_lr_scale", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--keep_step_checkpoints", type=int, default=3)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument(
        "--caption_prefixes",
        default="|".join(DEFAULT_CAPTION_PREFIXES),
        help="Pipe-separated caption prefixes; include an empty segment for no prefix",
    )
    parser.add_argument("--log_format", choices=["json", "pretty"], default="json")
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
        return model.module
    return model


def emit_metrics(metrics: dict[str, object], log_format: str) -> None:
    if log_format == "json":
        print(json.dumps(metrics), flush=True)
        return
    parts = []
    if "step" in metrics:
        parts.append(f"step {int(metrics['step']):>6}")
    if "phase" in metrics:
        parts.append(str(metrics["phase"]))
    if "batch" in metrics:
        parts.append(str(metrics["batch"]))
    if "loss" in metrics:
        parts.append(f"loss {float(metrics['loss']):.4f}")
    if "image_val_loss" in metrics:
        parts.append(f"img_val {float(metrics['image_val_loss']):.4f}")
    if "wrong_image_loss" in metrics:
        parts.append(f"wrong {float(metrics['wrong_image_loss']):.4f}")
    if "blank_image_loss" in metrics:
        parts.append(f"blank {float(metrics['blank_image_loss']):.4f}")
    if "lr" in metrics:
        parts.append(f"lr {float(metrics['lr']):.2e}")
    if "lm_lr" in metrics:
        parts.append(f"lm_lr {float(metrics['lm_lr']):.2e}")
    if "tokens_seen" in metrics:
        parts.append(f"tok {int(metrics['tokens_seen']):,}")
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


def build_optimizer_param_groups(model: torch.nn.Module, *, lr: float, lm_lr_scale: float) -> list[dict[str, object]]:
    base = unwrap_model(model)
    vision_param_ids = {id(param) for param in base.vision.parameters()}
    embedding_param_id = id(base.tok_emb.weight)
    boundary_embedding_only = base._embedding_mask_hook is not None
    groups: dict[str, list[torch.nn.Parameter]] = {
        "vision": [],
        "image_boundary_embeddings": [],
        "lm": [],
    }
    for param in model.parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in vision_param_ids:
            groups["vision"].append(param)
        elif param_id == embedding_param_id and boundary_embedding_only:
            groups["image_boundary_embeddings"].append(param)
        else:
            groups["lm"].append(param)

    param_groups: list[dict[str, object]] = []
    if groups["vision"]:
        param_groups.append({"params": groups["vision"], "lr": lr, "lr_scale": 1.0, "name": "vision"})
    if groups["image_boundary_embeddings"]:
        param_groups.append(
            {
                "params": groups["image_boundary_embeddings"],
                "lr": lr,
                "lr_scale": 1.0,
                "name": "image_boundary_embeddings",
                "weight_decay": 0.0,
            }
        )
    if groups["lm"]:
        param_groups.append({"params": groups["lm"], "lr": lr * lm_lr_scale, "lr_scale": lm_lr_scale, "name": "lm"})
    return param_groups


def next_batch(loader: DataLoader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def image_forward_loss(model: ArcModel, batch: dict[str, torch.Tensor], use_amp: bool, device: torch.device) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        _, loss = model(
            batch["input_ids"],
            labels=batch["labels"],
            images=batch["images"],
            image_insert_positions=batch["image_insert_positions"],
        )
    return loss


def text_forward_loss(model: ArcModel, batch: torch.Tensor, use_amp: bool, device: torch.device) -> tuple[torch.Tensor, int]:
    input_ids, labels = split_inputs_labels(batch)
    input_ids = input_ids.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        _, loss = model(input_ids, labels=labels)
    return loss, int(input_ids.numel())


@torch.no_grad()
def validate_image_loss(
    model: ArcModel,
    loader: DataLoader,
    *,
    device: torch.device,
    batches: int,
    use_amp: bool,
) -> dict[str, float | bool | int]:
    model.eval()
    losses = []
    wrong_losses = []
    blank_losses = []
    data_iter = iter(loader)
    for _ in range(max(0, batches)):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        batch = batch_to_device(batch, device)
        losses.append(float(image_forward_loss(model, batch, use_amp, device).detach().cpu()))
        wrong_batch = dict(batch)
        wrong_batch["images"] = batch["images"].roll(shifts=1, dims=0)
        wrong_losses.append(float(image_forward_loss(model, wrong_batch, use_amp, device).detach().cpu()))
        blank_batch = dict(batch)
        blank_batch["images"] = torch.zeros_like(batch["images"])
        blank_losses.append(float(image_forward_loss(model, blank_batch, use_amp, device).detach().cpu()))
    model.train()
    if not losses:
        return {"image_val_loss": float("nan"), "batches": 0}
    values = torch.tensor(
        [sum(losses), sum(wrong_losses), sum(blank_losses), len(losses)],
        device=device,
        dtype=torch.float32,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count = max(float(values[3].item()), 1.0)
    correct = float(values[0].item() / count)
    wrong = float(values[1].item() / count)
    blank = float(values[2].item() / count)
    return {
        "image_val_loss": correct,
        "wrong_image_loss": wrong,
        "blank_image_loss": blank,
        "correct_better_than_wrong": correct < wrong,
        "correct_better_than_blank": correct < blank,
        "batches": len(losses),
    }


def rotate_step_checkpoints(out_dir: str | Path, phase: str, keep: int) -> None:
    if keep < 0:
        return
    root = Path(out_dir)
    checkpoints = sorted(
        root.glob(f"arc124_vl_{phase}_step_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old in checkpoints[keep:]:
        old.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = setup_distributed()
    cfg = load_config(args.config)
    specs = load_vl_plan(args.mixture)
    flt = load_vision_filter(args.filter)
    params = estimate_parameters(cfg)
    if is_main_process(rank):
        print(f"model: {cfg.model_name}")
        print(f"start checkpoint: {args.checkpoint}")
        print(f"image size: {cfg.image_size}")
        print(f"visual tokens: {cfg.vision_resampler_tokens}")
        print(f"params total: {params['total']:,}")
        print(f"caption token filter: {flt.min_tokens}-{flt.max_tokens} tokens")
        print(f"distributed: {world_size} process(es)")
        print("\nVL mixture:")
        print(describe_mixture(specs))
    model = ArcModel(cfg)
    tokenizer = None
    if Path(args.tokenizer_dir).exists():
        tokenizer = ArcTokenizer.from_dir(args.tokenizer_dir)
        image_start_id = tokenizer.image_start_id
        image_end_id = tokenizer.image_end_id
    else:
        image_start_id = cfg.vocab_size - 4
        image_end_id = cfg.vocab_size - 3
        if is_main_process(rank):
            print(f"tokenizer not found at {args.tokenizer_dir}; dry-run using placeholder image token IDs")
    if is_main_process(rank):
        print("\nStage 2 trainability:")
        for phase in ["stage2a", "stage2b", "stage2c"]:
            summary = configure_stage2_phase(
                model,
                phase,
                image_start_id=image_start_id,
                image_end_id=image_end_id,
                final_layers=2,
            )
            visible_pct = 100 * summary.optimizer_visible_parameters / summary.total_parameters
            effective_pct = 100 * summary.effective_trainable_parameters / summary.total_parameters
            print(
                f"{phase}: {summary.effective_trainable_parameters:,} effective trainable "
                f"({effective_pct:.1f}%), {summary.optimizer_visible_parameters:,} optimizer-visible "
                f"({visible_pct:.1f}%) - {summary.notes}"
            )
    if not args.dry_run:
        if tokenizer is None:
            raise SystemExit("Pass --tokenizer_dir pointing to the Arc tokenizer for real Stage 2 training")
        if args.image_manifest is None:
            raise SystemExit("Pass --image_manifest pointing to filtered image-caption JSONL")
        if distributed and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(args.device)
        if device.type == "cuda":
            torch.set_float32_matmul_precision(args.matmul_precision)
        model = model.to(device)
        if args.resume:
            ckpt = load_checkpoint(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            if is_main_process(rank):
                print(f"resumed checkpoint: {args.resume}")
        elif Path(args.checkpoint).exists():
            ckpt = load_checkpoint(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            if is_main_process(rank):
                print(f"loaded checkpoint: {args.checkpoint}")
        else:
            if is_main_process(rank):
                print(f"checkpoint not found, training from current initialization: {args.checkpoint}")
        configure_stage2_phase(
            model,
            args.phase,
            image_start_id=tokenizer.image_start_id,
            image_end_id=tokenizer.image_end_id,
            final_layers=2,
        )
        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        dataset = ImageCaptionDataset(
            args.image_manifest,
            tokenizer,
            image_root=args.image_root,
            image_size=cfg.image_size,
            visual_tokens=cfg.vision_resampler_tokens,
            max_seq_len=cfg.max_seq_len,
            allow_missing_images=args.allow_missing_images,
            caption_prefixes=args.caption_prefixes.split("|"),
        )
        image_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        image_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
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
            val_dataset = ImageCaptionDataset(
                args.val_image_manifest,
                tokenizer,
                image_root=args.image_root,
                image_size=cfg.image_size,
                visual_tokens=cfg.vision_resampler_tokens,
                max_seq_len=cfg.max_seq_len,
                allow_missing_images=args.allow_missing_images,
                caption_prefixes=args.caption_prefixes.split("|"),
            )
            val_sampler = (
                DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                if distributed
                else None
            )
            val_image_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.persistent_workers and args.num_workers > 0,
                drop_last=True,
                collate_fn=lambda batch: collate_image_caption(batch, tokenizer.pad_id),
            )
        text_loader = None
        if args.text_replay_shards and args.phase == "stage2a" and is_main_process(rank):
            print("ignoring --text_replay_shards during stage2a; Stage 2a is image-only by design")
        if args.text_replay_shards and args.phase != "stage2a":
            text_dataset = PackedMemmapDataset(args.text_replay_shards, seq_len=cfg.max_seq_len)
            text_sampler = (
                DistributedSampler(text_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                if distributed
                else None
            )
            text_loader = DataLoader(
                text_dataset,
                batch_size=args.text_batch_size or args.batch_size,
                shuffle=text_sampler is None,
                sampler=text_sampler,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.persistent_workers and args.num_workers > 0,
                drop_last=True,
            )
        trainable = [param for param in model.parameters() if param.requires_grad]
        optimizer_param_groups = build_optimizer_param_groups(model, lr=args.lr, lm_lr_scale=args.lm_lr_scale)
        optimizer = torch.optim.AdamW(
            optimizer_param_groups,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=device.type == "cuda",
        )
        start_step = 0
        tokens_seen = 0
        if args.resume:
            if ckpt.get("optimizer") is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_step = int(ckpt.get("step", 0))
            tokens_seen = int(ckpt.get("tokens_seen", 0))
        use_amp = device.type == "cuda" and args.dtype == "bfloat16"
        image_iter = iter(image_loader)
        text_iter = iter(text_loader) if text_loader is not None else None
        t0 = time.time()
        last_log_time = t0
        last_log_tokens = tokens_seen
        running_loss = 0.0
        rng = random.Random(1337 + start_step)
        model.train()
        for step in range(start_step, args.max_steps):
            if distributed:
                if image_sampler is not None:
                    image_sampler.set_epoch(step)
                if text_loader is not None and getattr(text_loader, "sampler", None) is not None:
                    text_loader.sampler.set_epoch(step)
            lr = get_lr(step, args.max_steps, args.lr, args.min_lr, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr * float(group.get("lr_scale", 1.0))
            optimizer.zero_grad(set_to_none=True)
            use_text = text_loader is not None and rng.random() < args.text_replay_weight
            if use_text:
                raw_batch, text_iter = next_batch(text_loader, text_iter)
                loss, batch_tokens = text_forward_loss(model, raw_batch, use_amp, device)
                batch_kind = "text"
            else:
                raw_batch, image_iter = next_batch(image_loader, image_iter)
                batch = batch_to_device(raw_batch, device)
                loss = image_forward_loss(model, batch, use_amp, device)
                batch_tokens = int(batch["input_ids"].numel())
                batch_kind = "image"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            tokens_seen += batch_tokens * world_size
            loss_value = float(loss.detach().cpu())
            running_loss = loss_value if step == 0 else 0.95 * running_loss + 0.05 * loss_value
            if step % args.log_every == 0 and is_main_process(rank):
                now = time.time()
                elapsed = max(now - t0, 1e-6)
                interval_elapsed = max(now - last_log_time, 1e-6)
                metrics = {
                    "step": step,
                    "phase": args.phase,
                    "batch": batch_kind,
                    "loss": running_loss,
                    "lr": lr,
                    "lm_lr": lr * args.lm_lr_scale,
                    "tokens_seen": tokens_seen,
                    "tokens_per_second": tokens_seen / elapsed,
                    "interval_tokens_per_second": (tokens_seen - last_log_tokens) / interval_elapsed,
                }
                if device.type == "cuda":
                    metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                emit_metrics(metrics, args.log_format)
                last_log_time = now
                last_log_tokens = tokens_seen
            if args.eval_every > 0 and val_image_loader is not None and (step + 1) % args.eval_every == 0:
                metrics = validate_image_loss(
                    model,
                    val_image_loader,
                    device=device,
                    batches=args.val_batches,
                    use_amp=use_amp,
                )
                metrics.update({"step": step + 1, "tokens_seen": tokens_seen})
                if is_main_process(rank):
                    emit_metrics(metrics, args.log_format)
            if is_main_process(rank) and (step + 1) % args.save_every == 0:
                path = Path(args.out_dir) / f"arc124_vl_{args.phase}_step_{step + 1}.pt"
                save_checkpoint(
                    path,
                    checkpoint_payload(
                        model=unwrap_model(model),
                        optimizer=optimizer,
                        scheduler=None,
                        config=asdict(cfg),
                        step=step + 1,
                        tokens_seen=tokens_seen,
                        extra={"stage": "vl", "phase": args.phase},
                    ),
                )
                rotate_step_checkpoints(args.out_dir, args.phase, args.keep_step_checkpoints)
        if is_main_process(rank):
            save_checkpoint(
                Path(args.out_dir) / f"arc124_vl_{args.phase}_last.pt",
                checkpoint_payload(
                    model=unwrap_model(model),
                    optimizer=optimizer,
                    scheduler=None,
                    config=asdict(cfg),
                    step=step + 1,
                    tokens_seen=tokens_seen,
                    extra={"stage": "vl", "phase": args.phase},
                ),
            )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
