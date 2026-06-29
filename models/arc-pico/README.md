# Arc-Pico

```text
Arc-Pico-VL v0.1
A ~15M parameter from-scratch text+image base model for cheap multimodal recipe tests.
```

Arc-Pico is the lab model for Arc. The goal is not high capability. The goal is to prove that the native-ish multimodal workflow works before spending real money on Arc-124M.

Arc-Pico is not an assistant, OCR model, document model, chart model, or reasoning model.

## Why Pico Exists

The larger Arc model is expensive enough that every pipeline bug hurts. Arc-Pico is small enough to train on one rental GPU under about 5 EUR, while still exercising the real system:

```text
tokenizer training
FineWeb-Edu + DCLM text sharding
COCO + Flickr30k image-caption prep
mixed text/image pretraining
checkpoint save/resume
image-conditioning validation
loss/throughput logging
```

## Model Spec

```text
total params:   ~15.27M
text params:    ~13.70M
vision params:  ~1.56M
vocab:          16,384
context:        512
precision:      bf16 on CUDA
```

Text decoder:

```text
decoder-only Transformer
10 layers
hidden size 256
8 attention heads
head dim 32
SwiGLU FFN dim 896
RMSNorm
RoPE
tied input/output embeddings
```

Vision path:

```text
image size:       224
patch size:       16
raw patches:      196
vision width:     192
vision layers:    2
vision heads:     4
visual tokens:    16
projection:       192 -> 256
```

The 16k vocabulary is deliberate. A 32k vocabulary would consume too much of a 15M budget in embeddings.

## Training Plan

Recommended cheap run:

```text
Stage A: optional text bootstrap
  100M text-only tokens

Stage B: main mixed pretraining
  500M sequence tokens/equivalent
  85% text-only
  15% image-caption

Stage C: optional vision polish
  50M-100M tokens/equivalent
  60% image-caption
  40% text replay
```

For the first paid run, skip Stage C unless Stage B behaves well.

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

The default Pico prep script uses about:

```text
80k COCO caption rows
20k Flickr30k caption rows
1k held-out image-caption validation rows
```

This is intentionally smaller than the full Arc plan.

## Local Smoke Test

```bash
cd /Users/vidvudscalitis/Desktop/CODING/MultiModal/arc/models/arc-pico
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

mkdir -p /tmp/arc_pico_smoke

PYTHONPATH=src python scripts/make_tiny_text_corpus.py \
  --out /tmp/arc_pico_smoke/text.jsonl \
  --rows 500

PYTHONPATH=src python scripts/train_tokenizer.py \
  --source_jsonl /tmp/arc_pico_smoke/text.jsonl \
  --out_dir /tmp/arc_pico_smoke/tokenizer_4k \
  --samples 500 \
  --vocab_size 4096

PYTHONPATH=src python scripts/shard_text.py \
  --source_jsonl /tmp/arc_pico_smoke/text.jsonl \
  --tokenizer_dir /tmp/arc_pico_smoke/tokenizer_4k \
  --out_dir /tmp/arc_pico_smoke/tokens \
  --target_tokens 200000 \
  --val_tokens 10000 \
  --shard_tokens 200000

PYTHONPATH=src python -m arc.train_text \
  --config configs/arc_tiny_smoke.json \
  --token_shards /tmp/arc_pico_smoke/tokens/train \
  --val_token_shards /tmp/arc_pico_smoke/tokens/val \
  --out_dir /tmp/arc_pico_smoke/runs/text \
  --device cpu \
  --batch_size 4 \
  --max_steps 5 \
  --log_format pretty
```

## RunPod Cheap Workflow

Use one network volume and one cheap single-GPU pod. The scripts default to `/workspace/arc-pico` and `/workspace/arc_pico_data`.

On your Mac:

```bash
cd /Users/vidvudscalitis/Desktop/CODING/MultiModal/arc/models/arc-pico
bash runpod/package_repo.sh /tmp/arc_pico_runpod.tgz
runpodctl send /tmp/arc_pico_runpod.tgz --code arc-pico
```

On the pod:

```bash
cd /workspace
rm -rf /workspace/arc-pico
mkdir -p /workspace/arc-pico
cd /workspace/arc-pico
runpodctl receive arc-pico
tar --no-same-owner -xzf arc_pico_runpod.tgz
bash runpod/install_pod_deps.sh
```

Prepare text shards:

```bash
export HF_TOKEN="hf_read_only_token"
export ARC_VOL="/workspace"
export RUN_NAME="pico_500m"
export TARGET_TOKENS=500000000
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
bash runpod/prep_pico_vl.sh
```

Start mixed training:

```bash
export ARC_VOL="/workspace"
export RUN_NAME="pico_mixed_500m"
export TOKEN_DIR="/workspace/arc_pico_data/tokens_pico_500m"
export TARGET_TOKENS=500000000
export IMAGE_WEIGHT=0.15
export BATCH_SIZE=64
export IMAGE_BATCH_SIZE=32
export NPROC_PER_NODE=1
bash runpod/start_stage1_nohup.sh
tail -f /workspace/arc_pico_data/logs/nohup_pico_mixed_500m.log
```

Check status:

```bash
export ARC_VOL="/workspace"
export RUN_NAME="pico_mixed_500m"
bash runpod/status_stage1.sh
```

Upload checkpoints to Hugging Face from the pod:

```bash
export HF_TOKEN="hf_write_token"
export HF_REPO_ID="yourname/arc-pico-vl-checkpoints"
export CHECKPOINT_DIR="/workspace/arc_pico_data/checkpoints_pico_mixed_500m"
export REMOTE_PATH="pico_mixed_500m"
bash runpod/upload_checkpoints_hf.sh
```

## Cost Target

As of the current RunPod pricing page, examples include:

```text
RTX A5000: $0.27/hr
L4:        $0.39/hr
RTX 4090:  $0.69/hr
```

The practical target is:

```text
single RTX 4090 or cheaper
80GB network volume
500M mixed tokens/equivalent
under ~5 EUR if the pod is deleted promptly
```

## Expected Output

Arc-Pico should learn tiny-model behaviors:

```text
basic next-token text continuation
very simple caption-style visual grounding
colors, common objects, common scenes
whether correct-image loss beats wrong/blank image loss
```

Failure is useful too. If Pico ignores images, diverges, or bottlenecks on image loading, we fix that here before touching Arc-124M.
