#!/usr/bin/env bash
set -euo pipefail

ARC_VOL="${ARC_VOL:-/workspace}"
ARC_REPO_DIR="${ARC_REPO_DIR:-$ARC_VOL/arc}"
ARC_DATA_DIR="${ARC_DATA_DIR:-$ARC_VOL/arc_data}"
RUN_NAME="${RUN_NAME:-stage1_20b}"
TOKEN_DIR="${TOKEN_DIR:-$ARC_DATA_DIR/tokens_$RUN_NAME}"
OUT_DIR="${OUT_DIR:-$ARC_DATA_DIR/checkpoints_$RUN_NAME}"
TARGET_TOKENS="${TARGET_TOKENS:-20000000000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_STEPS="${MAX_STEPS:-200000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
VAL_BATCHES="${VAL_BATCHES:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
COMPILE_FLAG="${COMPILE_FLAG:---compile}"

cd "$ARC_REPO_DIR"
mkdir -p "$OUT_DIR" "$ARC_DATA_DIR/logs"

echo "RUN_NAME=$RUN_NAME"
echo "TOKEN_DIR=$TOKEN_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "TARGET_TOKENS=$TARGET_TOKENS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
df -h "$ARC_VOL" || true

PYTHONPATH=src torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m arc.train_text \
  --config config.json \
  --token_shards "$TOKEN_DIR/train" \
  --val_token_shards "$TOKEN_DIR/val" \
  --out_dir "$OUT_DIR" \
  --device cuda \
  --dtype bfloat16 \
  --batch_size "$BATCH_SIZE" \
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
