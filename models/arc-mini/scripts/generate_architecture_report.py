#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

from arc.config import load_config
from arc.model import estimate_parameters


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports"
OUT_PDF = OUT_DIR / "arc_mini_vl_architecture_report.pdf"


COLORS = {
    "ink": "#17202A",
    "muted": "#5D6D7E",
    "blue": "#2E86C1",
    "orange": "#D68910",
    "green": "#239B56",
    "purple": "#7D3C98",
    "red": "#C0392B",
    "panel": "#F7F9F9",
    "line": "#D5DBDB",
}


def box(ax, xy, wh, text, *, fc="#FFFFFF", ec="#D5DBDB", fs=10, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.1,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=weight,
        color=COLORS["ink"],
        transform=ax.transAxes,
    )


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=1.4, color=COLORS["muted"]),
    )


def page(pdf, title, subtitle=None):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_axis_off()
    ax.text(0.06, 0.93, title, fontsize=24, fontweight="bold", color=COLORS["ink"], transform=ax.transAxes)
    if subtitle:
        ax.text(0.06, 0.885, subtitle, fontsize=11, color=COLORS["muted"], transform=ax.transAxes)
    ax.plot([0.06, 0.94], [0.06, 0.06], color=COLORS["line"], lw=1, transform=ax.transAxes)
    ax.text(0.06, 0.03, "Arc-Mini-VL v0.1 architecture report", fontsize=8, color=COLORS["muted"], transform=ax.transAxes)
    return fig, ax


def table(ax, x, y, rows, col_widths, row_h=0.055, fs=9):
    for r, row in enumerate(rows):
        yy = y - r * row_h
        for c, cell in enumerate(row):
            xx = x + sum(col_widths[:c])
            fc = "#EBF5FB" if r == 0 else ("#FFFFFF" if r % 2 else COLORS["panel"])
            rect = plt.Rectangle((xx, yy - row_h), col_widths[c], row_h, transform=ax.transAxes, facecolor=fc, edgecolor=COLORS["line"], linewidth=0.7)
            ax.add_patch(rect)
            ax.text(xx + 0.008, yy - row_h / 2, str(cell), va="center", ha="left", fontsize=fs, color=COLORS["ink"], transform=ax.transAxes)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.json")
    params = estimate_parameters(cfg)

    with PdfPages(OUT_PDF) as pdf:
        fig, ax = page(pdf, cfg.model_name, "50M-class cheap multimodal pretraining testbed")
        cards = [
            ("Total", f"{params['total'] / 1e6:.2f}M", COLORS["blue"]),
            ("Text", f"{params['text_plus_embeddings'] / 1e6:.2f}M", COLORS["green"]),
            ("Vision", f"{params['vision'] / 1e6:.2f}M", COLORS["orange"]),
            ("Context", str(cfg.max_seq_len), COLORS["purple"]),
        ]
        for i, (label, value, color) in enumerate(cards):
            box(ax, (0.08 + i * 0.22, 0.58), (0.17, 0.14), f"{value}\n{label}", ec=color, fs=14, weight="bold")
        ax.text(
            0.08,
            0.42,
            "Arc-Mini is a small but meaningfully capable next step after Arc-Pico. It keeps the same tokenizer, data, checkpointing, and image-use validation workflow while giving the decoder enough capacity for a better quality signal.",
            fontsize=13,
            color=COLORS["ink"],
            wrap=True,
            transform=ax.transAxes,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = page(pdf, "Model Flow", "Two input paths, one shared decoder stream")
        box(ax, (0.07, 0.70), (0.17, 0.10), "Text token ids", fc="#EBF5FB", ec=COLORS["blue"], weight="bold")
        box(ax, (0.31, 0.70), (0.22, 0.10), f"Token embedding\n{cfg.vocab_size} x {cfg.d_model}", fc="#EBF5FB", ec=COLORS["blue"])
        box(ax, (0.07, 0.43), (0.17, 0.10), f"Image\n{cfg.image_size} x {cfg.image_size}", fc="#FEF5E7", ec=COLORS["orange"], weight="bold")
        box(ax, (0.31, 0.43), (0.22, 0.10), f"Vision encoder\n{cfg.n_image_patches} patches -> {cfg.vision_resampler_tokens} tokens", fc="#FEF5E7", ec=COLORS["orange"])
        box(ax, (0.59, 0.56), (0.17, 0.11), f"Shared {cfg.d_model}-d\nsequence", fc="#F4ECF7", ec=COLORS["purple"], weight="bold")
        box(ax, (0.81, 0.56), (0.12, 0.11), f"{cfg.n_layers} decoder\nblocks", fc="#E8F8F5", ec=COLORS["green"], weight="bold")
        box(ax, (0.81, 0.34), (0.12, 0.10), f"LM head\n{cfg.d_model} -> vocab", fc="#FDEDEC", ec=COLORS["red"])
        arrow(ax, (0.24, 0.75), (0.31, 0.75))
        arrow(ax, (0.53, 0.75), (0.59, 0.62))
        arrow(ax, (0.24, 0.48), (0.31, 0.48))
        arrow(ax, (0.53, 0.48), (0.59, 0.61))
        arrow(ax, (0.76, 0.61), (0.81, 0.61))
        arrow(ax, (0.87, 0.56), (0.87, 0.44))
        ax.text(0.08, 0.22, f"The image becomes {cfg.vision_resampler_tokens} soft tokens. Those tokens have the same {cfg.d_model}-wide shape as text embeddings, so the decoder can attend over one mixed sequence.", fontsize=12, transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = page(pdf, "Configuration", "Implemented dimensions")
        rows = [
            ("Field", "Value"),
            ("vocab_size", cfg.vocab_size),
            ("d_model", cfg.d_model),
            ("n_layers", cfg.n_layers),
            ("n_heads", cfg.n_heads),
            ("head_dim", cfg.head_dim),
            ("ffn_hidden", cfg.ffn_hidden),
            ("image_size", cfg.image_size),
            ("vision_width", cfg.vision_width),
            ("vision_layers", cfg.vision_layers),
            ("vision_resampler_tokens", cfg.vision_resampler_tokens),
        ]
        table(ax, 0.08, 0.82, rows, [0.32, 0.20], fs=10)
        ax.text(0.58, 0.72, "Training target", fontsize=14, fontweight="bold", transform=ax.transAxes)
        ax.text(0.58, 0.64, "1B mixed tokens/equivalent\n85% text-only\n15% image-caption\nsingle GPU\nunder about 5 EUR target", fontsize=12, transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    meta = {"pdf": str(OUT_PDF), "model": cfg.model_name, "params": params, "pages": 3}
    (OUT_DIR / "arc_mini_vl_architecture_report.summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
