#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

from arc.config import load_config
from arc.model import ArcModel, estimate_parameters


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports"
OUT_PDF = OUT_DIR / "arc124_vl_architecture_report.pdf"

PAGE = (11.0, 8.5)
COLORS = {
    "ink": "#17202A",
    "muted": "#5D6D7E",
    "line": "#D5DBDB",
    "panel": "#F7F9F9",
    "blue": "#2E86C1",
    "green": "#239B56",
    "orange": "#D68910",
    "red": "#C0392B",
    "purple": "#7D3C98",
    "teal": "#117A65",
}


def add_title(ax, title: str, subtitle: str | None = None) -> None:
    ax.text(0.06, 0.93, title, fontsize=24, fontweight="bold", color=COLORS["ink"], transform=ax.transAxes)
    if subtitle:
        ax.text(0.06, 0.885, subtitle, fontsize=11, color=COLORS["muted"], transform=ax.transAxes)


def add_footer(ax, page: int) -> None:
    ax.plot([0.06, 0.94], [0.055, 0.055], color=COLORS["line"], lw=1, transform=ax.transAxes)
    ax.text(0.06, 0.025, "Arc-124M-VL v0.3 architecture report", fontsize=8, color=COLORS["muted"], transform=ax.transAxes)
    ax.text(0.94, 0.025, f"{page}", fontsize=8, color=COLORS["muted"], ha="right", transform=ax.transAxes)


def setup_page(pdf: PdfPages, title: str, subtitle: str | None = None, page: int = 1):
    fig, ax = plt.subplots(figsize=PAGE)
    ax.set_axis_off()
    add_title(ax, title, subtitle)
    add_footer(ax, page)
    return fig, ax


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec=None, color=None, fs=10, weight="normal", radius=0.018):
    ec = ec or COLORS["line"]
    color = color or COLORS["ink"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
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
        color=color,
        fontweight=weight,
        transform=ax.transAxes,
        linespacing=1.25,
    )
    return patch


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.4):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", color=color or COLORS["muted"], lw=lw, shrinkA=2, shrinkB=2),
    )


def section_text(ax, x, y, text, width=70, fs=10, color=None, line_height=0.042):
    color = color or COLORS["ink"]
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip():
            lines.extend(fill(paragraph, width=width).splitlines())
        else:
            lines.append("")
    for i, line in enumerate(lines):
        ax.text(x, y - i * line_height, line, fontsize=fs, color=color, transform=ax.transAxes)
    return y - len(lines) * line_height


def simple_table(ax, x, y, rows, col_widths, row_h=0.055, fs=9, header=True):
    total_w = sum(col_widths)
    for r, row in enumerate(rows):
        yy = y - r * row_h
        bg = "#EAF2F8" if header and r == 0 else ("#FFFFFF" if r % 2 else COLORS["panel"])
        ax.add_patch(Rectangle((x, yy - row_h), total_w, row_h, facecolor=bg, edgecolor=COLORS["line"], lw=0.8, transform=ax.transAxes))
        xx = x
        for c, cell in enumerate(row):
            ax.text(xx + 0.01, yy - row_h * 0.64, str(cell), fontsize=fs, color=COLORS["ink"], transform=ax.transAxes)
            xx += col_widths[c]
            if c < len(row) - 1:
                ax.plot([xx, xx], [yy - row_h, yy], color=COLORS["line"], lw=0.8, transform=ax.transAxes)


def param_groups(model: ArcModel) -> dict[str, int]:
    groups = {
        "token embedding / tied LM head": model.tok_emb.weight.numel(),
        "decoder attention": 0,
        "decoder FFN": 0,
        "decoder norms": 0,
        "vision side": sum(p.numel() for p in model.vision.parameters()),
        "final norm": sum(p.numel() for p in model.norm.parameters()),
    }
    for block in model.blocks:
        groups["decoder attention"] += sum(p.numel() for p in block.attn.parameters())
        groups["decoder FFN"] += sum(p.numel() for p in block.mlp.parameters())
        groups["decoder norms"] += sum(p.numel() for p in block.norm1.parameters()) + sum(p.numel() for p in block.norm2.parameters())
    return groups


def page_cover(pdf, cfg, params):
    fig, ax = setup_page(pdf, "Arc-124M-VL v0.3", "Small from-scratch text + image pretrained base model", 1)
    ax.text(0.06, 0.78, "Project goal", fontsize=13, fontweight="bold", color=COLORS["ink"], transform=ax.transAxes)
    section_text(
        ax,
        0.06,
        0.735,
        "Arc is a compact multimodal base model: text tokens predict the next text token, and image + text prefixes predict caption-style continuations. It is intentionally not an assistant, not an OCR model, and not a document/chart model.",
        width=86,
        fs=11,
        line_height=0.045,
    )
    cards = [
        ("Total params", f"{params['total'] / 1e6:.1f}M", COLORS["blue"]),
        ("Text side", f"{params['text_plus_embeddings'] / 1e6:.1f}M", COLORS["green"]),
        ("Vision side", f"{params['vision'] / 1e6:.1f}M", COLORS["orange"]),
        ("Context", f"{cfg.max_seq_len}", COLORS["purple"]),
    ]
    for i, (label, value, col) in enumerate(cards):
        x = 0.06 + i * 0.22
        box(ax, x, 0.46, 0.18, 0.14, f"{value}\n{label}", fc="#FFFFFF", ec=col, color=col, fs=13, weight="bold")
    box(
        ax,
        0.11,
        0.20,
        0.78,
        0.15,
        "Text tokens and projected visual tokens\nboth enter the decoder as 768-dimensional vectors.",
        fc="#F4F6F7",
        ec=COLORS["line"],
        fs=13,
        weight="bold",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_spec(pdf, cfg, params, groups):
    fig, ax = setup_page(pdf, "1. Configuration And Parameter Budget", "The exact implemented dimensions from config.json and model.py", 2)
    rows = [
        ("Field", "Value", "Meaning"),
        ("vocab_size", cfg.vocab_size, "Number of text tokens"),
        ("d_model", cfg.d_model, "Width of every decoder token vector"),
        ("n_layers", cfg.n_layers, "Number of decoder transformer blocks"),
        ("n_heads", cfg.n_heads, "Attention heads per decoder block"),
        ("head_dim", cfg.head_dim, "Width per attention head"),
        ("ffn_hidden", cfg.ffn_hidden, "SwiGLU/FFN intermediate width"),
        ("image_size", cfg.image_size, "Input image side length"),
        ("patch_size", cfg.patch_size, "Vision patch size"),
        ("vision_resampler_tokens", cfg.vision_resampler_tokens, "Compressed visual tokens inserted into decoder"),
    ]
    simple_table(ax, 0.06, 0.81, rows, [0.22, 0.16, 0.50], row_h=0.055, fs=8.7)

    labels = list(groups.keys())
    values = [groups[k] for k in labels]
    inset = fig.add_axes([0.62, 0.12, 0.27, 0.27])
    inset.pie(values, labels=None, colors=["#2E86C1", "#85C1E9", "#AED6F1", "#D6EAF8", "#F5B041", "#AAB7B8"], startangle=90)
    inset.set_title("Parameter split", fontsize=10)
    ax.text(0.06, 0.16, "Largest parameter groups", fontsize=12, fontweight="bold", color=COLORS["ink"], transform=ax.transAxes)
    rows2 = [("Group", "Parameters", "Share")]
    total = params["total"]
    for name, value in sorted(groups.items(), key=lambda x: -x[1]):
        rows2.append((name, f"{value / 1e6:.2f}M", f"{100 * value / total:.1f}%"))
    simple_table(ax, 0.06, 0.12, rows2, [0.31, 0.17, 0.14], row_h=0.042, fs=8.3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_whole_arch(pdf, cfg):
    fig, ax = setup_page(pdf, "2. Whole-Model Flow", "Two input paths, one shared decoder stream", 3)
    box(ax, 0.08, 0.68, 0.18, 0.10, "Text token ids", fc="#EBF5FB", ec=COLORS["blue"], weight="bold")
    box(ax, 0.31, 0.68, 0.22, 0.10, f"Token embedding\n{cfg.vocab_size} x {cfg.d_model}", fc="#EBF5FB", ec=COLORS["blue"])
    box(ax, 0.08, 0.41, 0.18, 0.10, f"Image\n{cfg.image_size} x {cfg.image_size}", fc="#FEF5E7", ec=COLORS["orange"], weight="bold")
    box(ax, 0.31, 0.41, 0.22, 0.10, f"Vision encoder\n196 patches -> 32 tokens", fc="#FEF5E7", ec=COLORS["orange"])
    box(ax, 0.58, 0.54, 0.18, 0.12, "Shared 768-d\nsequence", fc="#F4ECF7", ec=COLORS["purple"], weight="bold")
    box(ax, 0.80, 0.54, 0.13, 0.12, "12 decoder\nblocks", fc="#E8F8F5", ec=COLORS["teal"], weight="bold")
    box(ax, 0.80, 0.33, 0.13, 0.10, "Tied LM head\n768 -> vocab", fc="#FDEDEC", ec=COLORS["red"])
    box(ax, 0.58, 0.25, 0.18, 0.10, "Next-token\nloss/logits", fc="#FDEDEC", ec=COLORS["red"], weight="bold")
    arrow(ax, 0.26, 0.73, 0.31, 0.73)
    arrow(ax, 0.53, 0.73, 0.58, 0.61)
    arrow(ax, 0.26, 0.46, 0.31, 0.46)
    arrow(ax, 0.53, 0.46, 0.58, 0.59)
    arrow(ax, 0.76, 0.60, 0.80, 0.60)
    arrow(ax, 0.865, 0.54, 0.865, 0.43)
    arrow(ax, 0.80, 0.38, 0.76, 0.30)
    section_text(ax, 0.08, 0.20, "Key idea: image features are not decoded separately. The vision path converts images into 32 soft tokens with the same 768-wide shape as text embeddings. The decoder then attends over one mixed sequence.", width=105, fs=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_decoder_block(pdf, cfg):
    fig, ax = setup_page(pdf, "3. Decoder Block Deep Dive", "The same shape enters and exits every block: B x T x 768", 4)
    y = 0.74
    xs = [0.06, 0.22, 0.40, 0.57, 0.74]
    labels = [
        "Input x0\nB x T x 768",
        "RMSNorm\nstabilize",
        "Causal attention\ntokens read context",
        "Residual add\nx1 = x0 + attention",
        "RMSNorm\nstabilize",
    ]
    for i, lab in enumerate(labels):
        box(ax, xs[i], y, 0.13 if i != 2 else 0.15, 0.10, lab, fc="#FFFFFF", ec=COLORS["blue"], fs=8.7)
        if i < len(labels) - 1:
            arrow(ax, xs[i] + (0.13 if i != 2 else 0.15), y + 0.05, xs[i + 1], y + 0.05)
    box(ax, 0.35, 0.49, 0.18, 0.10, f"SwiGLU FFN\n768 -> {cfg.ffn_hidden} -> 768", fc="#FEF9E7", ec=COLORS["orange"], fs=9)
    box(ax, 0.60, 0.49, 0.18, 0.10, "Residual add\nx2 = x1 + FFN", fc="#E8F8F5", ec=COLORS["green"], fs=9)
    box(ax, 0.83, 0.49, 0.11, 0.10, "Output x2\nB x T x 768", fc="#FFFFFF", ec=COLORS["green"], fs=8.5)
    arrow(ax, 0.805, 0.74, 0.44, 0.59)
    arrow(ax, 0.53, 0.54, 0.60, 0.54)
    arrow(ax, 0.78, 0.54, 0.83, 0.54)
    code = "x0 = block input\nx1 = x0 + attention(RMSNorm(x0))\nx2 = x1 + SwiGLU_FFN(RMSNorm(x1))\nreturn x2"
    box(ax, 0.09, 0.23, 0.37, 0.17, code, fc="#F4F6F7", ec=COLORS["line"], fs=10)
    section_text(ax, 0.52, 0.35, "Attention communicates across tokens. The FFN/MLP processes each token independently after attention has mixed in context. Residual adds preserve the old state and only add learned updates.", width=55, fs=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_attention(pdf, cfg):
    fig, ax = setup_page(pdf, "4. Attention Mechanics", "How 768 dimensions become 12 heads and then merge again", 5)
    box(ax, 0.07, 0.68, 0.16, 0.10, "Input x\nB x T x 768", fc="#EBF5FB", ec=COLORS["blue"], weight="bold")
    box(ax, 0.30, 0.68, 0.18, 0.10, "QKV projection\n768 -> 2304", fc="#EBF5FB", ec=COLORS["blue"])
    box(ax, 0.55, 0.68, 0.16, 0.10, "Split\nQ, K, V", fc="#EBF5FB", ec=COLORS["blue"])
    box(ax, 0.78, 0.68, 0.15, 0.10, "Reshape\n12 x 64", fc="#EBF5FB", ec=COLORS["blue"])
    arrow(ax, 0.23, 0.73, 0.30, 0.73)
    arrow(ax, 0.48, 0.73, 0.55, 0.73)
    arrow(ax, 0.71, 0.73, 0.78, 0.73)
    box(ax, 0.16, 0.46, 0.20, 0.10, "RoPE on Q and K\nposition signal", fc="#F4ECF7", ec=COLORS["purple"])
    box(ax, 0.43, 0.46, 0.20, 0.10, "Scaled dot-product\ncausal attention", fc="#F4ECF7", ec=COLORS["purple"])
    box(ax, 0.70, 0.46, 0.18, 0.10, "Head outputs\nB x 12 x T x 64", fc="#F4ECF7", ec=COLORS["purple"])
    arrow(ax, 0.85, 0.68, 0.26, 0.56)
    arrow(ax, 0.36, 0.51, 0.43, 0.51)
    arrow(ax, 0.63, 0.51, 0.70, 0.51)
    box(ax, 0.24, 0.25, 0.20, 0.10, "Concatenate heads\n12 x 64 = 768", fc="#E8F8F5", ec=COLORS["green"])
    box(ax, 0.56, 0.25, 0.20, 0.10, "Output projection\n768 -> 768", fc="#E8F8F5", ec=COLORS["green"])
    arrow(ax, 0.79, 0.46, 0.34, 0.35)
    arrow(ax, 0.44, 0.30, 0.56, 0.30)
    section_text(ax, 0.08, 0.16, "The split is safe because 768 divides cleanly by 12. The model first learns Q/K/V projections, so heads are not just raw fixed slices of the original embedding. Each head sees a learned view, then the output projection mixes the heads back into one 768-wide token state.", width=112, fs=9.8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_vision(pdf, cfg):
    fig, ax = setup_page(pdf, "5. Vision Path", "A small native image front-end produces decoder-compatible visual tokens", 6)
    steps = [
        (0.06, "Image\n224 x 224 x 3"),
        (0.22, "Patchify\n16 x 16"),
        (0.38, "196 patches\npatch dim 768"),
        (0.54, "Patch projection\n768 -> 384"),
        (0.70, "4 vision blocks\nwidth 384"),
        (0.86, "Resampler\n196 -> 32"),
    ]
    for x, txt in steps:
        box(ax, x, 0.62, 0.12, 0.12, txt, fc="#FEF5E7", ec=COLORS["orange"], fs=8.3)
    for i in range(len(steps) - 1):
        arrow(ax, steps[i][0] + 0.12, 0.68, steps[i + 1][0], 0.68)
    box(ax, 0.39, 0.39, 0.20, 0.12, "Project visual tokens\n384 -> 768", fc="#EBF5FB", ec=COLORS["blue"], fs=10, weight="bold")
    box(ax, 0.66, 0.39, 0.22, 0.12, "Insert into decoder stream\npositions 1..32", fc="#E8F8F5", ec=COLORS["green"], fs=10, weight="bold")
    arrow(ax, 0.92, 0.62, 0.50, 0.51)
    arrow(ax, 0.59, 0.45, 0.66, 0.45)
    section_text(ax, 0.08, 0.26, "The vision encoder is intentionally small: it learns basic object, color, scene, and action grounding. It is not designed for OCR, documents, charts, or fine-grained visual reasoning. The resampler keeps image cost low by giving the decoder 32 visual tokens instead of all 196 patch tokens.", width=108, fs=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_sequence_loss(pdf, cfg):
    fig, ax = setup_page(pdf, "6. Multimodal Sequence And Loss Mask", "The image is a prefix; loss is only on caption/text tokens", 7)
    tokens = [
        ("<image_start>", "#FADBD8"),
        ("visual 1", "#D6EAF8"),
        ("visual ...", "#D6EAF8"),
        ("visual 32", "#D6EAF8"),
        ("<image_end>", "#FADBD8"),
        ("The", "#D5F5E3"),
        ("image", "#D5F5E3"),
        ("shows", "#D5F5E3"),
        ("a dog", "#D5F5E3"),
    ]
    x = 0.05
    for name, color in tokens:
        w = 0.10 if "visual" not in name else 0.095
        box(ax, x, 0.61, w, 0.10, name, fc=color, ec="#AAB7B8", fs=7.5)
        x += w + 0.008
    ax.text(0.06, 0.52, "Loss mask:", fontsize=11, fontweight="bold", color=COLORS["ink"], transform=ax.transAxes)
    box(ax, 0.18, 0.49, 0.36, 0.07, "ignore image boundary + visual positions", fc="#FDEDEC", ec=COLORS["red"], fs=9)
    box(ax, 0.58, 0.49, 0.26, 0.07, "compute caption loss", fc="#E8F8F5", ec=COLORS["green"], fs=9)
    section_text(ax, 0.08, 0.37, "The visual tokens are embeddings inserted into the sequence, not normal vocabulary words. The model is trained to predict the caption continuation, so the loss ignores the visual prefix and starts on the natural-language text.", width=105, fs=10)
    section_text(ax, 0.08, 0.25, "Caption prefixes are varied during Stage 2: empty prefix, 'A photo of', 'The image shows', 'In this scene,', and 'This is'. This teaches image + text prefix -> continuation rather than only image -> caption.", width=105, fs=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_training(pdf):
    fig, ax = setup_page(pdf, "7. Training System", "Stage 1 builds language; Stage 2 teaches image grounding", 8)
    box(ax, 0.08, 0.67, 0.22, 0.12, "Stage 1\ntext-only pretraining\n20B-25B tokens", fc="#EBF5FB", ec=COLORS["blue"], weight="bold")
    box(ax, 0.39, 0.67, 0.22, 0.12, "Stage 2a\nfreeze LM\ntrain vision bridge", fc="#FEF5E7", ec=COLORS["orange"], weight="bold")
    box(ax, 0.70, 0.67, 0.22, 0.12, "Stage 2b\nunfreeze final 2 layers\ntext replay + captions", fc="#E8F8F5", ec=COLORS["green"], weight="bold")
    arrow(ax, 0.30, 0.73, 0.39, 0.73)
    arrow(ax, 0.61, 0.73, 0.70, 0.73)
    rows = [
        ("Phase", "Trainable parts", "Purpose"),
        ("Stage 1", "text decoder only; vision frozen", "learn language continuation"),
        ("Stage 2a", "vision encoder, resampler, projector, image boundary rows", "align images to decoder space"),
        ("Stage 2b", "Stage 2a + final 2 decoder blocks", "learn image/text fusion with low LM LR"),
        ("Stage 2c", "optional full low-LR unfreeze", "only if validation remains stable"),
    ]
    simple_table(ax, 0.07, 0.49, rows, [0.14, 0.35, 0.39], row_h=0.058, fs=8.5)
    section_text(ax, 0.08, 0.15, "The essential validation is correct-image vs wrong-image vs blank-image caption loss. If correct image loss is not lower, the decoder is ignoring visual tokens.", width=108, fs=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_data_compute(pdf):
    fig, ax = setup_page(pdf, "8. Data And Compute Plan", "The current practical execution plan", 9)
    rows = [
        ("Area", "Plan"),
        ("Text data", "60% FineWeb-Edu, 40% DCLM 100BT shuffled"),
        ("Text target", "20B safe, 25B stretch; 30B only if budget allows"),
        ("Vision data", "COCO Captions + AnyModal/Flickr30k"),
        ("Stage 1 hardware", "8x H100 SXM, bf16, torch.compile, batch 32/GPU"),
        ("Measured H100 speed", "about 3.35M tokens/sec after compile warmup"),
        ("Stage 2 hardware", "1x or 2x RTX 4090 should be enough for v0"),
    ]
    simple_table(ax, 0.07, 0.80, rows, [0.22, 0.66], row_h=0.064, fs=9)
    inset = fig.add_axes([0.13, 0.12, 0.33, 0.24])
    labels = ["20B", "25B", "30B"]
    hours = [20e9 / 3.35e6 / 3600, 25e9 / 3.35e6 / 3600, 30e9 / 3.35e6 / 3600]
    inset.bar(labels, hours, color=[COLORS["green"], COLORS["orange"], COLORS["red"]])
    inset.set_ylabel("raw H100 hours")
    inset.set_title("Stage 1 raw train time")
    inset.grid(axis="y", alpha=0.25)
    ax.text(0.55, 0.30, "Budget note", fontsize=12, fontweight="bold", transform=ax.transAxes, color=COLORS["ink"])
    section_text(ax, 0.55, 0.25, "Data sharding is slow but cheap and should happen on a low-cost pod attached to the same network volume. H100 time should be reserved for the actual Stage 1 train.", width=47, fs=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_code_map(pdf):
    fig, ax = setup_page(pdf, "9. Code Map For Reviewers", "Files to inspect for architecture, training, and data flow", 10)
    rows = [
        ("File", "Why it matters"),
        ("src/arc/model.py", "Complete neural architecture: decoder, vision encoder, resampler, forward pass."),
        ("config.json + src/arc/config.py", "Actual dimensions and model settings."),
        ("src/arc/data_vision.py", "Image-caption sequence construction and loss mask."),
        ("src/arc/train_text.py", "Stage 1 DDP text pretraining loop."),
        ("src/arc/train_vl.py", "Stage 2 multimodal training loop and validation."),
        ("src/arc/training_plan.py", "Stage 2 freeze/unfreeze policy."),
        ("src/arc/tokenizer.py", "Byte BPE tokenizer and image special tokens."),
        ("configs/*.jsonl", "Dataset mixture definitions."),
    ]
    simple_table(ax, 0.06, 0.80, rows, [0.28, 0.60], row_h=0.061, fs=8.7)
    section_text(ax, 0.08, 0.17, "Minimal architecture bundle: config.json, src/arc/model.py, src/arc/config.py. Full systems bundle: add data_vision.py, train_text.py, train_vl.py, training_plan.py, tokenizer.py, README.md, and configs.", width=108, fs=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.json")
    params = estimate_parameters(cfg)
    model = ArcModel(cfg)
    groups = param_groups(model)

    with PdfPages(OUT_PDF) as pdf:
        page_cover(pdf, cfg, params)
        page_spec(pdf, cfg, params, groups)
        page_whole_arch(pdf, cfg)
        page_decoder_block(pdf, cfg)
        page_attention(pdf, cfg)
        page_vision(pdf, cfg)
        page_sequence_loss(pdf, cfg)
        page_training(pdf)
        page_data_compute(pdf)
        page_code_map(pdf)

    meta = {
        "pdf": str(OUT_PDF),
        "model": cfg.model_name,
        "params": params,
        "pages": 10,
    }
    (OUT_DIR / "arc124_vl_architecture_report.summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
