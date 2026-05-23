#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PRETTY_FIELD_PATTERNS = {
    "step": re.compile(r"(?:^|\|)\s*step\s+(\d+)"),
    "loss": re.compile(r"(?:^|\|)\s*loss\s+([0-9.eE+-]+)"),
    "val_loss": re.compile(r"(?:^|\|)\s*val\s+([0-9.eE+-]+)"),
    "lr": re.compile(r"(?:^|\|)\s*lr\s+([0-9.eE+-]+)"),
    "tokens_seen": re.compile(r"(?:^|\|)\s*tok\s+([0-9,]+)"),
    "interval_tokens_per_second": re.compile(r"(?:^|\|)\s*it/s\s+([0-9,]+)"),
    "tokens_per_second": re.compile(r"(?:^|\|)\s*avg/s\s+([0-9,]+)"),
    "peak_memory_gb": re.compile(r"(?:^|\|)\s*mem\s+([0-9.eE+-]+)GB"),
}


def parse_number(value: str) -> float:
    return float(value.replace(",", ""))


def parse_pretty_line(line: str) -> dict[str, Any] | None:
    if "step" not in line:
        return None
    row: dict[str, Any] = {}
    for key, pattern in PRETTY_FIELD_PATTERNS.items():
        match = pattern.search(line)
        if not match:
            continue
        raw = match.group(1)
        if key in {"step", "tokens_seen"}:
            row[key] = int(raw.replace(",", ""))
        else:
            row[key] = parse_number(raw)
    return row if "step" in row else None


def parse_json_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    return row if "step" in row else None


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            row = parse_json_line(line) or parse_pretty_line(line)
            if row:
                rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = [
        "step",
        "loss",
        "val_loss",
        "image_val_loss",
        "wrong_image_loss",
        "blank_image_loss",
        "lr",
        "tokens_seen",
        "interval_tokens_per_second",
        "tokens_per_second",
        "peak_memory_gb",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_loss(rows: list[dict[str, Any]], out: Path) -> None:
    train = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    val = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    image_val = [(r["step"], r["image_val_loss"]) for r in rows if "image_val_loss" in r]
    wrong = [(r["step"], r["wrong_image_loss"]) for r in rows if "wrong_image_loss" in r]
    blank = [(r["step"], r["blank_image_loss"]) for r in rows if "blank_image_loss" in r]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if train:
        ax.plot(*zip(*train), label="train loss", linewidth=1.8)
    if val:
        ax.plot(*zip(*val), marker="o", label="text val loss", linewidth=1.6)
    if image_val:
        ax.plot(*zip(*image_val), marker="o", label="image val loss", linewidth=1.6)
    if wrong:
        ax.plot(*zip(*wrong), marker=".", label="wrong image loss", linewidth=1.2)
    if blank:
        ax.plot(*zip(*blank), marker=".", label="blank image loss", linewidth=1.2)
    ax.set_title("Arc Training Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_throughput(rows: list[dict[str, Any]], out: Path) -> None:
    interval = [(r["step"], r["interval_tokens_per_second"]) for r in rows if "interval_tokens_per_second" in r]
    avg = [(r["step"], r["tokens_per_second"]) for r in rows if "tokens_per_second" in r]
    mem = [(r["step"], r["peak_memory_gb"]) for r in rows if "peak_memory_gb" in r]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if interval:
        ax.plot(*zip(*interval), label="interval tok/s", linewidth=1.8)
    if avg:
        ax.plot(*zip(*avg), label="avg tok/s", linewidth=1.4)
    ax.set_title("Arc Throughput")
    ax.set_xlabel("step")
    ax.set_ylabel("tokens/sec")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    if mem:
        ax2 = ax.twinx()
        ax2.plot(*zip(*mem), color="#C0392B", alpha=0.7, label="peak memory GB", linewidth=1.2)
        ax2.set_ylabel("GB")
        ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Arc training metrics from JSON or pretty logs")
    parser.add_argument("--log", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--prefix", default="stage1")
    args = parser.parse_args()

    log = Path(args.log)
    out_dir = Path(args.out_dir) if args.out_dir else log.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_log(log)
    if not rows:
        raise SystemExit(f"No metric rows found in {log}")
    csv_path = out_dir / f"{args.prefix}_metrics.csv"
    loss_path = out_dir / f"{args.prefix}_loss.png"
    throughput_path = out_dir / f"{args.prefix}_throughput.png"
    write_csv(rows, csv_path)
    plot_loss(rows, loss_path)
    plot_throughput(rows, throughput_path)
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "loss_png": str(loss_path), "throughput_png": str(throughput_path)}, indent=2))


if __name__ == "__main__":
    main()
