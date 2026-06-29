# Arc Model Family

Arc is a small from-scratch multimodal model family. The repo now keeps the three current variants side by side:

```text
models/
  arc-pico/      ~15M parameter cheap pipeline proof
  arc-mini/      ~50M parameter low-budget capability experiment
  arc-124m-vl/   ~124M parameter main target model
```

The goal is not to build an assistant yet. These models are pretrained base models for:

```text
text input       -> next-token prediction
image + text     -> caption-style continuation
```

They are not expected to do reliable instruction following, OCR, chart reading, document understanding, tool use, or long reasoning.

## Models

| Model | Folder | Parameters | Context | Main Purpose |
|---|---:|---:|---:|---|
| Arc-Pico-VL v0.1 | `models/arc-pico` | ~15M | 512 | Cheapest end-to-end pipeline test |
| Arc-Mini-VL v0.1 | `models/arc-mini` | ~50M | 512 | Low-budget text+image learning experiment |
| Arc-124M-VL v0.3 | `models/arc-124m-vl` | ~124M | 1024 | Main planned model |

Each model folder is self-contained enough to run its own local smoke tests and RunPod workflows.

## Architecture Pattern

All three variants use the same basic design:

```text
text tokens -> token embeddings  \
                               -> decoder-only Transformer -> next token
image       -> vision encoder   /
              -> visual tokens
              -> projection into decoder hidden size
```

For image-caption training, an image becomes a small set of learned visual prefix embeddings:

```text
<image_start> [visual embeddings] <image_end> caption tokens
```

Loss is computed on caption/text tokens, not on the visual embeddings.

## Training Data

Text:

```text
60% FineWeb-Edu
40% DCLM 100BT shuffled
```

Vision:

```text
COCO captions
AnyModal/flickr30k
```

The current vision objective is clean caption grounding, not visual instruction tuning.

## Repo Layout

```text
README.md              family-level overview
models/README.md       model index and comparison
models/arc-pico/       15M model code/config/docs
models/arc-mini/       50M model code/config/docs
models/arc-124m-vl/    124M model code/config/docs
```

The root also still contains the current `Arc-124M-VL` working tree files for compatibility with the existing branch history and RunPod scripts. New readers should start from `models/`.

## Current Status

```text
Arc-Pico:
  trained as a cheap proof of the full workflow
  useful as a systems test, not a capable model

Arc-Mini:
  completed a ~1B mixed-token run
  learned measurable image conditioning
  still weak qualitatively

Arc-124M-VL:
  CUDA/DDP smoke-tested
  ready for larger benchmark planning, not yet fully pretrained
```

## Quick Start

Pick a model folder and follow its README:

```bash
cd models/arc-mini
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Each model folder includes:

```text
config.json
configs/
src/
scripts/
runpod/
README.md
```

Large generated artifacts are intentionally not tracked in git:

```text
downloads/
runs/
checkpoints/
data/
*.pt
*.safetensors
```

