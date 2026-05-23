#!/usr/bin/env bash
set -euo pipefail

ARC_VOL="${ARC_VOL:-/workspace}"
ARC_REPO_DIR="${ARC_REPO_DIR:-$ARC_VOL/arc}"
ARC_DATA_DIR="${ARC_DATA_DIR:-$ARC_VOL/arc_data}"
RUN_NAME="${RUN_NAME:-stage1_mixed_20b}"
TOKEN_DIR="${TOKEN_DIR:-$ARC_DATA_DIR/tokens_stage1_20b}"
IMAGE_MANIFEST="${IMAGE_MANIFEST:-$ARC_DATA_DIR/vl/vl_train.jsonl}"
VAL_IMAGE_MANIFEST="${VAL_IMAGE_MANIFEST:-$ARC_DATA_DIR/vl/vl_val.jsonl}"
OUT_DIR="${OUT_DIR:-$ARC_DATA_DIR/checkpoints_$RUN_NAME}"
TARGET_TOKENS="${TARGET_TOKENS:-20000000000}"
IMAGE_WEIGHT="${IMAGE_WEIGHT:-0.04}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-32}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_STEPS="${MAX_STEPS:-200000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
VAL_BATCHES="${VAL_BATCHES:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
COMPILE_FLAG="${COMPILE_FLAG:---compile}"

cd "$ARC_REPO_DIR"
mkdir -p "$OUT_DIR" "$ARC_DATA_DIR/logs"

echo "RUN_NAME=$RUN_NAME"
echo "TOKEN_DIR=$TOKEN_DIR"
echo "IMAGE_MANIFEST=$IMAGE_MANIFEST"
echo "OUT_DIR=$OUT_DIR"
echo "TARGET_TOKENS=$TARGET_TOKENS"
echo "IMAGE_WEIGHT=$IMAGE_WEIGHT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "IMAGE_BATCH_SIZE=$IMAGE_BATCH_SIZE"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
df -h "$ARC_VOL" || true

VAL_IMAGE_ARGS=()
if [[ -f "$VAL_IMAGE_MANIFEST" ]]; then
  VAL_IMAGE_ARGS=(--val_image_manifest "$VAL_IMAGE_MANIFEST")
fi

PYTHONPATH=src torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m arc.train_mixed \
  --config config.json \
  --token_shards "$TOKEN_DIR/train" \
  --val_token_shards "$TOKEN_DIR/val" \
  --tokenizer_dir "$ARC_DATA_DIR/tokenizer_32k" \
  --image_manifest "$IMAGE_MANIFEST" \
  "${VAL_IMAGE_ARGS[@]}" \
  --out_dir "$OUT_DIR" \
  --device cuda \
  --dtype bfloat16 \
  --batch_size "$BATCH_SIZE" \
  --image_batch_size "$IMAGE_BATCH_SIZE" \
  --image_weight "$IMAGE_WEIGHT" \
  --target_tokens "$TARGET_TOKENS" \
  --max_steps "$MAX_STEPS" \
  --save_every "$SAVE_EVERY" \
  --eval_every "$EVAL_EVERY" \
  --val_batches "$VAL_BATCHES" \
  --num_workers "$NUM_WORKERS" \
  --log_every 10 \
  --log_format pretty \
  --matmul_precision high \
  --persistent_workers \
  $COMPILE_FLAG \
  2>&1 | tee "$ARC_DATA_DIR/logs/train_$RUN_NAME.log"
