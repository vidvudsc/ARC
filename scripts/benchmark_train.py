#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from arc.config import load_config
from arc.model import ArcModel
from arc.throughput import recommend_token_budget


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return distributed, rank, local_rank, world_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Arc Stage 1 throughput benchmark")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    distributed, rank, local_rank, world_size = setup_distributed()
    cfg = load_config(args.config)
    if distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision(args.matmul_precision)
    model = ArcModel(cfg).to(device)
    for param in model.vision.parameters():
        param.requires_grad_(False)
    if args.compile:
        model = torch.compile(model)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=1e-4,
        fused=device.type == "cuda",
    )
    use_amp = device.type == "cuda" and args.dtype == "bfloat16"
    input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, cfg.max_seq_len), device=device)
    labels = torch.randint(0, cfg.vocab_size, (args.batch_size, cfg.max_seq_len), device=device)

    model.train()
    total_steps = args.warmup + args.steps
    measured_tokens = 0
    start = None
    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step == args.warmup:
            start = time.time()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        if step >= args.warmup:
            measured_tokens += input_ids.numel() * world_size
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.time() - (start or time.time()), 1e-6)
    tokens_per_second = measured_tokens / elapsed
    if distributed:
        value = torch.tensor(tokens_per_second, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        tokens_per_second = float(value.detach().cpu())
    if rank == 0:
        print(
            json.dumps(
                {
                    "model": cfg.model_name,
                    "world_size": world_size,
                    "batch_size_per_process": args.batch_size,
                    "context": cfg.max_seq_len,
                    "steps": args.steps,
                    "warmup": args.warmup,
                    "dtype": args.dtype,
                    "matmul_precision": args.matmul_precision,
                    "compile": args.compile,
                    "tokens_per_second": tokens_per_second,
                    "recommendation": recommend_token_budget(tokens_per_second),
                },
                indent=2,
            )
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
