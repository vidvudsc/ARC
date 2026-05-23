# Arc

```text
Arc-124M-VL v0.3
A 124M total-parameter pretrained text+image base model.
```

Arc v0.3 is not an assistant, OCR model, chart model, document model, agent, or instruction-following chatbot.

## Current Readiness

The repository is not ready for the full H100 run yet, but the core training path has passed local CPU tests and a 1x RTX 4090 CUDA bug-flush run.

Verified:

```text
local CPU tiny Stage 1 train
local CPU checkpoint resume
local 2-process CPU DDP Stage 1
full 124M one-step CPU train
1x RTX 4090 bf16 CUDA Stage 1
1x RTX 4090 checkpoint resume
1x RTX 4090 batch_size 16 at 1024 context
FineWeb-Edu streaming smoke
DCLM 100BT shuffled streaming smoke
1M-token HF text shard
COCO streaming image-caption smoke
tiny Stage 2a CUDA image-caption train on COCO
```

Known issues:

```text
Flickr30k via nlphuji/flickr30k fails with current datasets package because it uses a deprecated HF dataset script.
AnyModal/flickr30k is the current replacement candidate; it streams as parquet/native HF and exposes image plus alt_text/original_alt_text captions.
HF/datasets streaming can trigger a Python shutdown crash after successful preprocessing; prep scripts force a clean process exit after flushing output.
```

Still required before the full H100 run:

```text
real tokenizer training
real FineWeb-Edu/DCLM token shards
held-out validation shards
throughput benchmark on 8xH100
2-GPU DDP smoke test on CUDA
8-GPU DDP 200-step benchmark
Stage 2 image shard pipeline at real scale
Flickr30k replacement or compatible prep environment
correct/wrong/blank image validation
checkpoint resume test at realistic scale
```

Target behavior:

```text
Text           -> next-token prediction
Image + prefix -> caption-style continuation
```

Example:

```text
<image_start> [visual tokens] <image_end>
A dog is sitting on a couch next to a blanket.
```

## Model Spec

```text
total params: ~124M
context:      1024
vocab:        32k
precision:    bf16
```

Text decoder:

```text
~113M-116M params
12 layers
768 hidden size
12 attention heads
head_dim 64
SwiGLU
RMSNorm
RoPE
tied input/output embeddings
```

Vision module:

```text
~8M-11M params
image size:       224
patch size:       16
vision dim:       384
vision layers:    4
vision heads:     6
visual tokens:    32 for v0.3
projector:        384 -> 768
```

Only try 64 visual tokens later.

## Multimodal Fusion

Arc is not two separate models glued together. It is one shared decoder stream with two input paths:

```text
text tokens -> token embeddings  \
                               -> same decoder transformer -> next token
image       -> visual embeddings /
```

For image-caption training, the image is converted into LM-compatible prefix embeddings:

```text
image -> vision encoder -> resampler/projector -> 32 visual tokens

combined sequence:
<image_start> [32 visual embeddings] <image_end> caption tokens
```

The visual embeddings are not vocabulary tokens, but they have the same hidden size as text token embeddings:

```text
text token embedding:   768 dims
visual token embedding: 768 dims
```

So the image becomes 32 soft tokens that the decoder can attend to while predicting caption text.

This means v0.3 learns caption-style fusion:

```text
Good:
  "A photo of..."
  "The image shows..."
  basic object/scene/color/action grounding

Weak:
  complex VQA
  instruction following
  object counting
  OCR
  chart/document reasoning
  multi-region comparison
```

Stage 2a mostly learns:

```text
image -> LM-friendly visual prefix
```

This is not just "attach image encoder and hope." Stage 2a explicitly trains the bridge that maps image features into the decoder token space, while the language model stays fixed.

Stage 2b matters because unfreezing the final decoder layers lets the upper language layers adapt to mixed image+text sequences:

```text
visual tokens + text tokens -> shared decoder representations
```

## Stage 1: Text-Only Pretraining

Target:

```text
minimum: 20B tokens
target:  30B tokens
maximum: 30B tokens
```

Data:

```text
60% FineWeb-Edu
40% DCLM 100BT shuffled
```

For 30B total:

```text
18B FineWeb-Edu tokens
12B DCLM tokens
```

Checkpoints:

```text
checkpoints/arc124_text_5b.pt
checkpoints/arc124_text_10b.pt
checkpoints/arc124_text_20b.pt
checkpoints/arc124_text_30b.pt
```

Stage 2 starts from `arc124_text_30b.pt`, or `arc124_text_20b.pt` if throughput is too slow.

Ordinary step checkpoints are crash-recovery checkpoints. By default the trainers keep only the latest 3:

```text
--save_every 1000
--keep_step_checkpoints 3
```

This does not delete fixed billion-token milestones or `*_last.pt`.

## Throughput Gate

Benchmark on the actual 8xH100 setup before a real run.

```text
<2M tok/s:      debug only, do not full-send
2M-2.5M tok/s: train 15B-20B tokens
2.5M-3.5M/s:   train 20B-25B tokens
3.5M+ tok/s:   train full 30B tokens
```

At 30B tokens:

```text
2.5M tok/s ~= 3.3 hours
3.0M tok/s ~= 2.8 hours
4.0M tok/s ~= 2.1 hours
5.0M tok/s ~= 1.7 hours
```

Current training optimizations:

```text
precomputed RoPE cache
PyTorch SDPA attention path
bf16 autocast on CUDA
fused AdamW on CUDA
torch.compile option
DDP no_sync during gradient accumulation
persistent DataLoader workers option
interval tokens/sec logging
peak CUDA memory logging
TF32 matmul precision knob
```

## Stage 2: Image-Text Grounding

Image data:

```text
80% COCO captions
20% Flickr30k captions or replacement clean caption dataset
```

Preferred size:

```text
~600k COCO caption pairs
~150k Flickr30k caption pairs
~750k total image-caption pairs
```

Minimum fallback:

```text
COCO only
or a smaller COCO + Flickr30k subset
```

Current note:

```text
COCO streaming prep has been verified on RunPod.
nlphuji/flickr30k is blocked by current Hugging Face datasets behavior because it uses a deprecated dataset script.
Use AnyModal/flickr30k as the current replacement, or fall back to configs/stage2_vl_mix_coco_only.jsonl.
```

Caption-pair rule:

```text
one training sample = one image + one caption
```

If one image has five captions, those become five possible globally shuffled image-caption examples.

Basic caption hygiene:

```text
normalize whitespace
drop captions shorter than 4 tokens
drop captions longer than 80 tokens
deduplicate exact image-caption pairs
avoid assistant/chat/VQA/OCR/document/chart formatting
```

Stage 2 length:

```text
1B-2B sequence tokens equivalent
60% text replay
40% image-caption
preferred image passes: 3-8
hard cap: 10-12 unless validation still improves
```

## Stage 2 Curriculum

Stage 2a: vision alignment.

Freeze:

```text
LM transformer blocks
normal text embeddings
LM head
```

Train:

```text
vision encoder
visual resampler
image-to-LM projector
<image_start> embedding
<image_end> embedding
visual positional/type embeddings if used
```

Stage 2b: partial LM unfreeze.

```text
unfreeze final 2 transformer layers
train 60% text replay / 40% image-caption
lower LR for LM layers than vision/projector
```

Stage 2c: optional full low-LR unfreeze.

Only do this if text validation loss is stable, image validation is still improving, and there are no signs of language degradation.

## Caption Dataset Filtering

Keep:

```text
natural images
ordinary scenes/objects/actions
English captions
caption-style text
4-80 tokens
visually grounded at 224px
```

Remove:

```text
OCR-heavy examples
documents
charts
graphs
tables
screenshots
UI/webpages
slides
PDF pages
receipts
forms
invoices
code screenshots
memes where text matters
assistant/chat format
question-answer format
captions relying on tiny unreadable details
long synthetic aesthetic descriptions
```

## Training Format

Use caption continuation, not chat.

Correct:

```text
<image_start> [visual tokens] <image_end>
A brown dog is sitting on a couch.
```

Use caption-prefix variation so the model learns image + text prefix -> continuation, not only image -> caption:

```text
<image_start> [visual tokens] <image_end>
A photo of a brown dog lying on a couch.

<image_start> [visual tokens] <image_end>
The image shows a brown dog lying on a couch.

<image_start> [visual tokens] <image_end>
In this scene, a brown dog is lying on a couch.

<image_start> [visual tokens] <image_end>
This is a brown dog lying on a couch.
```

The trainer uses these prefixes by default:

```text
""
"A photo of "
"The image shows "
"In this scene, "
"This is "
```

Wrong:

```text
User: What is in this image?
Assistant: A brown dog is sitting on a couch.
```

Loss masking:

```text
no loss on visual tokens
loss on caption/text tokens
```

## Validation

Track:

```text
text train loss
text validation loss
image-caption validation loss
tokens/sec
gradient norm
NaNs
```

Required image-use test:

```text
caption loss(correct image) < caption loss(wrong image)
caption loss(correct image) < caption loss(blank image)
```

If this fails, the model may be generating plausible captions from text priors instead of actually using visual tokens.

Manual prompts:

```text
<image> A photo of
<image> The image shows
<image> In this scene,
```

## Data and Storage Prep

Before renting H100 time:

```text
tokenize FineWeb-Edu
tokenize DCLM
prepare COCO image-caption pairs
prepare Flickr30k replacement or use COCO-only fallback
resize images to 224/256
make WebDataset shards
make text memmap shards
test checkpoint save/resume
test distributed CUDA launch
test tiny overfit
```

Expected storage:

```text
30B text tokens as uint16: ~60 GB
checkpoints/logs:          ~10-30 GB
image shards:              ~30-100 GB
recommended disk:          150-300 GB
```

## Repository Layout

```text
arc/
  README.md
  pyproject.toml
  config.json

  configs/
    stage1_text_mix.jsonl
    stage2_vl_mix.jsonl
    caption_filter.json

  src/arc/
    model.py
    data.py
    data_text.py
    data_vision.py
    train_text.py
    train_vl.py
    checkpoint.py
    utils.py

  scripts/
    train_tokenizer.py
    shard_text.py
    make_tiny_text_corpus.py
    make_tiny_text_fixture.py
    make_tiny_vl_corpus.py
    prepare_coco.py
    prepare_flickr30k.py
    shard_images.py
    validate_image_use.py
    benchmark_train.py
    benchmark_gate.py
```

## Local Smoke Test

Run this before touching expensive GPUs:

```bash
PYTHONPATH=src python scripts/make_tiny_text_fixture.py \
  --out_dir data/fixtures/tokens \
  --tokens 20000

PYTHONPATH=src python -m arc.train_text \
  --config configs/arc_tiny_smoke.json \
  --token_shards data/fixtures/tokens \
  --out_dir runs/smoke_text \
  --device cpu \
  --dtype float32 \
  --batch_size 2 \
  --val_token_shards data/fixtures/tokens \
  --eval_every 1 \
  --val_batches 1 \
  --max_steps 2 \
  --num_workers 0 \
  --log_every 1
```

For real Stage 1 prep:

```bash
PYTHONPATH=src python scripts/train_tokenizer.py \
  --mixture configs/stage1_text_mix.jsonl \
  --out_dir tokenizer_32k \
  --samples 2000000

PYTHONPATH=src python scripts/shard_text.py \
  --mixture configs/stage1_text_mix.jsonl \
  --tokenizer_dir tokenizer_32k \
  --out_dir data/tokens_stage1 \
  --target_tokens 29900000000 \
  --val_tokens 100000000 \
  --shard_tokens 100000000
```

For an offline data-prep smoke test:

```bash
PYTHONPATH=src python scripts/make_tiny_text_corpus.py \
  --out data/fixtures/text_corpus.jsonl \
  --rows 200

PYTHONPATH=src python scripts/train_tokenizer.py \
  --source_jsonl data/fixtures/text_corpus.jsonl \
  --out_dir runs/tokenizer_smoke \
  --samples 200 \
  --vocab_size 1024

PYTHONPATH=src python scripts/shard_text.py \
  --source_jsonl data/fixtures/text_corpus.jsonl \
  --tokenizer_dir runs/tokenizer_smoke \
  --out_dir runs/tokens_smoke \
  --target_tokens 10000 \
  --val_tokens 2000 \
  --shard_tokens 4000
```

For an offline Stage 2 metadata/trainer smoke test:

```bash
PYTHONPATH=src python scripts/make_tiny_vl_corpus.py \
  --out_dir data/fixtures/vl

PYTHONPATH=src python scripts/prepare_coco.py \
  --source_jsonl data/fixtures/vl/coco_raw.jsonl \
  --out runs/vl_smoke/coco_captions.jsonl \
  --max_examples 10

PYTHONPATH=src python scripts/prepare_flickr30k.py \
  --source_jsonl data/fixtures/vl/flickr30k_raw.jsonl \
  --out runs/vl_smoke/flickr30k_captions.jsonl \
  --max_examples 10

PYTHONPATH=src python scripts/shard_images.py \
  --manifest runs/vl_smoke/coco_captions.jsonl \
  --manifest runs/vl_smoke/flickr30k_captions.jsonl \
  --image_root data/fixtures/vl \
  --out_dir runs/vl_shards_smoke \
  --samples_per_shard 1 \
  --max_samples 2

PYTHONPATH=src python -m arc.train_vl \
  --config configs/arc_tiny_smoke.json \
  --tokenizer_dir runs/tokenizer_smoke \
  --image_manifest runs/vl_smoke/coco_captions.jsonl \
  --val_image_manifest runs/vl_smoke/flickr30k_captions.jsonl \
  --checkpoint runs/smoke_text/arc124_text_last.pt \
  --text_replay_shards runs/tokens_smoke/train \
  --out_dir runs/smoke_vl \
  --device cpu \
  --dtype float32 \
  --batch_size 1 \
  --text_batch_size 1 \
  --max_steps 2 \
  --num_workers 0 \
  --log_every 1 \
  --eval_every 1 \
  --val_batches 1 \
  --phase stage2a \
  --allow_missing_images
```

For a 1x RTX 4090 CUDA smoke test, use a short run and readable logs:

```bash
PYTHONPATH=src python -m arc.train_text \
  --config config.json \
  --token_shards /workspace/arc_test/tokens/train \
  --val_token_shards /workspace/arc_test/tokens/val \
  --out_dir /workspace/arc_test/runs/full124m_cuda \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 16 \
  --max_steps 10 \
  --num_workers 2 \
  --log_every 1 \
  --log_format pretty
```

For a COCO-only Stage 2a CUDA smoke test:

```bash
PYTHONPATH=src python -m arc.train_vl \
  --config configs/arc_tiny_smoke.json \
  --tokenizer_dir /workspace/arc_test/tokenizer_hf_smoke \
  --image_manifest /workspace/arc_test/vl/coco_smoke.jsonl \
  --out_dir /workspace/arc_test/runs/vl_cuda_smoke \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 4 \
  --max_steps 10 \
  --num_workers 2 \
  --phase stage2a \
  --log_every 1 \
  --log_format pretty
```

The smoke fixture uses blank images when `--allow_missing_images` is set, so it only validates code flow. Real image-use validation must be run on actual image files:

```bash
PYTHONPATH=src python scripts/validate_image_use.py \
  --config config.json \
  --checkpoint checkpoints/arc124_vl_base.pt \
  --tokenizer_dir tokenizer_32k \
  --image_manifest data/vl/val_captions.jsonl \
  --image_root data/images \
  --batch_size 8 \
  --batches 20
```

For the real Stage 1 run, launch with `torchrun` after the sharded data and validation shards exist:

First run a synthetic throughput benchmark on the actual pod:

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=8 scripts/benchmark_train.py \
  --config config.json \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 32 \
  --steps 50 \
  --warmup 10 \
  --matmul_precision high \
  --compile
```

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=8 -m arc.train_text \
  --config config.json \
  --mixture configs/stage1_text_mix.jsonl \
  --checkpoint_plan configs/checkpoints.json \
  --token_shards data/tokens_stage1/train \
  --val_token_shards data/tokens_stage1/val \
  --out_dir checkpoints \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 32 \
  --grad_accum 1 \
  --target_tokens 30000000000 \
  --max_steps 200000 \
  --log_every 10 \
  --eval_every 1000 \
  --save_every 1000 \
  --keep_step_checkpoints 3 \
  --matmul_precision high \
  --persistent_workers \
  --compile
```

For Hugging Face datasets that yield image objects instead of local image paths, pass `--image_out_dir` during caption prep:

```bash
PYTHONPATH=src HF_HUB_DISABLE_XET=1 python scripts/prepare_coco.py \
  --dataset jxie/coco_captions \
  --streaming \
  --out data/vl/coco_captions.jsonl \
  --image_out_dir data/images/coco \
  --max_examples 600000
```

Flickr30k note:

```text
nlphuji/flickr30k failed under the tested RunPod datasets package because it uses a deprecated HF dataset script.
Use AnyModal/flickr30k instead. It exposes PIL image rows and caption lists in alt_text/original_alt_text.
Use configs/stage2_vl_mix_coco_only.jsonl only if the replacement fails in your environment.
```

For real Stage 2a on GPUs:

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=8 -m arc.train_vl \
  --config config.json \
  --mixture configs/stage2_vl_mix_coco_only.jsonl \
  --filter configs/caption_filter.json \
  --checkpoint checkpoints/arc124_text_30b.pt \
  --tokenizer_dir tokenizer_32k \
  --image_manifest data/vl/train_captions.jsonl \
  --val_image_manifest data/vl/val_captions.jsonl \
  --image_root data/images \
  --text_replay_shards data/tokens_stage1/train \
  --out_dir checkpoints \
  --phase stage2a \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 32 \
  --text_batch_size 32 \
  --text_replay_weight 0.0 \
  --max_steps 50000 \
  --log_every 10 \
  --eval_every 1000 \
  --val_batches 20 \
  --save_every 1000 \
  --keep_step_checkpoints 3 \
  --matmul_precision high \
  --persistent_workers
```

Stage 2a is image-only by design. Text replay starts in Stage 2b.

To resume Stage 2:

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=8 -m arc.train_vl \
  --config config.json \
  --checkpoint checkpoints/arc124_text_30b.pt \
  --resume checkpoints/arc124_vl_stage2a_step_1000.pt \
  --tokenizer_dir tokenizer_32k \
  --image_manifest data/vl/train_captions.jsonl \
  --val_image_manifest data/vl/val_captions.jsonl \
  --image_root data/images \
  --text_replay_shards data/tokens_stage1/train \
  --out_dir checkpoints \
  --phase stage2a \
  --device cuda
```
