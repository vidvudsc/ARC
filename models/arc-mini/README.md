# Arc-Mini

```text
Arc-Mini-VL v0.1
A ~50M parameter from-scratch text+image base model for cheap capability tests.
```

Arc-Mini is the next step after Arc-Pico. It is still not an assistant, OCR model, chart model, document model, or reasoning model. The goal is a small multimodal base that can learn noticeably better text continuation and caption-style image grounding while staying inside a low rental-GPU budget.

## Model Spec

```text
total params:   ~50.20M
text params:    ~46.41M
vision params:  ~3.79M
vocab:          16,384
context:        512
precision:      bf16 on CUDA
```

Text decoder:

```text
decoder-only Transformer
10 layers
hidden size 512
8 attention heads
head dim 64
SwiGLU FFN dim 1792
RMSNorm
RoPE
tied input/output embeddings
```

Vision path:

```text
image size:       224
patch size:       16
raw patches:      196
vision width:     256
vision layers:    3
vision heads:     4
visual tokens:    24
projection:       256 -> 512
```

## Training Plan

Recommended first paid run:

```text
Stage A: data preparation
  train 16k tokenizer
  shard 1B text tokens
  prepare 100k COCO/Flickr image-caption examples

Stage B: mixed pretraining
  1B counted tokens/equivalent
  85% text-only
  15% image-caption

Stage C: optional continuation
  continue to 2B only if 1B samples look promising
```

## Data

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

Default image prep:

```text
80k COCO caption rows
20k Flickr30k caption rows
1k held-out image-caption validation rows
```

## Local Smoke Test

```bash
cd /Users/vidvudscalitis/Desktop/CODING/MultiModal/arc/models/arc-mini
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

mkdir -p /tmp/arc_mini_smoke

PYTHONPATH=src python scripts/make_tiny_text_corpus.py \
  --out /tmp/arc_mini_smoke/text.jsonl \
  --rows 500

PYTHONPATH=src python scripts/train_tokenizer.py \
  --source_jsonl /tmp/arc_mini_smoke/text.jsonl \
  --out_dir /tmp/arc_mini_smoke/tokenizer_4k \
  --samples 500 \
  --vocab_size 4096

PYTHONPATH=src python scripts/shard_text.py \
  --source_jsonl /tmp/arc_mini_smoke/text.jsonl \
  --tokenizer_dir /tmp/arc_mini_smoke/tokenizer_4k \
  --out_dir /tmp/arc_mini_smoke/tokens \
  --target_tokens 200000 \
  --val_tokens 10000 \
  --shard_tokens 200000

PYTHONPATH=src python -m arc.train_text \
  --config configs/arc_tiny_smoke.json \
  --token_shards /tmp/arc_mini_smoke/tokens/train \
  --val_token_shards /tmp/arc_mini_smoke/tokens/val \
  --out_dir /tmp/arc_mini_smoke/runs/text \
  --device cpu \
  --batch_size 4 \
  --max_steps 5 \
  --log_format pretty
```

## RunPod Workflow

Use one network volume. Prep should try a cheap CPU pod first; if RunPod cannot place CPU in the volume region, use the cheapest available GPU and delete it after prep.

Mac:

```bash
cd /Users/vidvudscalitis/Desktop/CODING/MultiModal/arc/models/arc-mini
bash runpod/package_repo.sh /tmp/arc_mini_runpod.tgz
runpodctl send /tmp/arc_mini_runpod.tgz --code arc-mini
```

Pod:

```bash
cd /workspace
rm -rf /workspace/arc-mini
mkdir -p /workspace/arc-mini
cd /workspace/arc-mini
runpodctl receive arc-mini
tar --no-same-owner -xzf arc_mini_runpod.tgz
bash runpod/install_pod_deps.sh
```

Prepare text shards:

```bash
export HF_TOKEN="hf_read_only_token"
export ARC_VOL="/workspace"
export RUN_NAME="mini_1b"
export TARGET_TOKENS=1000000000
export VAL_TOKENS=2000000
export TOKENIZER_SAMPLES=100000
bash runpod/prep_stage1_tokens.sh
```

Prepare image-caption manifests:

```bash
export HF_TOKEN="hf_read_only_token"
export ARC_VOL="/workspace"
export COCO_EXAMPLES=80000
export FLICKR_EXAMPLES=20000
export VAL_EXAMPLES=1000
bash runpod/prep_mini_vl.sh
```

Start mixed training:

```bash
export ARC_VOL="/workspace"
export RUN_NAME="mini_mixed_1b"
export TOKEN_DIR="/workspace/arc_mini_data/tokens_mini_1b"
export TARGET_TOKENS=1000000000
export IMAGE_WEIGHT=0.15
export BATCH_SIZE=32
export IMAGE_BATCH_SIZE=16
export NPROC_PER_NODE=1
bash runpod/start_stage1_nohup.sh
tail -f /workspace/arc_mini_data/logs/nohup_mini_mixed_1b.log
```

Check status:

```bash
export ARC_VOL="/workspace"
export RUN_NAME="mini_mixed_1b"
bash runpod/status_stage1.sh
```

## Inference

```bash
PYTHONPATH=src python scripts/infer.py \
  --checkpoint downloads/mini_mixed_1b/checkpoints_mini_mixed_1b/arc_mini_mixed_last.pt \
  --tokenizer_dir downloads/mini_mixed_1b/tokenizer_16k \
  --prompt "A photo of " \
  --device cpu \
  --dtype float32
```

## Budget Target

Target: keep the first 1B run under 5 EUR by:

```text
using a cheap prep pod for sharding
using RTX 3090/4090-class GPU for training
keeping all data/checkpoints on the network volume
deleting pods immediately after prep/training
continuing to 2B only after evaluating the 1B checkpoint
```
