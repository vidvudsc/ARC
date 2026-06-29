# Arc Models

This directory contains the current Arc model variants.

## Variant Summary

| Variant | Folder | Size | Why It Exists |
|---|---|---:|---|
| Arc-Pico-VL v0.1 | `arc-pico/` | ~15M params | Proves the full text+image workflow cheaply |
| Arc-Mini-VL v0.1 | `arc-mini/` | ~50M params | Tests whether a low-budget model can learn useful language and image grounding |
| Arc-124M-VL v0.3 | `arc-124m-vl/` | ~124M params | Main project target |

## How To Choose

Use `arc-pico/` when you want to test scripts, data prep, logging, checkpointing, and image-conditioning validation without caring about model quality.

Use `arc-mini/` when you want the cheapest meaningful capability experiment. This is the current best place to test data recipes, tokenizer changes, and training schedules.

Use `arc-124m-vl/` when you are preparing the real run. It is the main model, but it is expensive enough that bugs should be flushed out in Pico or Mini first.

## Common Model Pattern

All three variants are decoder-only language models with a lightweight vision front end:

```text
Text:
  token ids
  -> token embeddings
  -> decoder Transformer
  -> next-token logits

Image + text:
  image
  -> small vision encoder
  -> resampled/projected visual embeddings
  -> same decoder Transformer
  -> caption/text logits
```

The vision tokens are soft embeddings inserted into the decoder stream. They are not normal vocabulary tokens, but after projection they have the same width as text token embeddings.

## What Is Not Included

Generated training outputs are intentionally excluded:

```text
downloads/
runs/
checkpoints/
data/
*.pt
*.pth
*.safetensors
```

Keep those on RunPod volumes, local artifact folders, or Hugging Face model repos instead of committing them here.

