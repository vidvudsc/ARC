#!/usr/bin/env bash
set -euo pipefail

ARC_VOL="${ARC_VOL:-/workspace}"
ARC_REPO_DIR="${ARC_REPO_DIR:-$ARC_VOL/arc-pico}"
ARC_DATA_DIR="${ARC_DATA_DIR:-$ARC_VOL/arc_pico_data}"
TARGET_TOKENS="${TARGET_TOKENS:-500000000}"
VAL_TOKENS="${VAL_TOKENS:-2000000}"
SHARD_TOKENS="${SHARD_TOKENS:-50000000}"
TOKENIZER_SAMPLES="${TOKENIZER_SAMPLES:-100000}"
VOCAB_SIZE="${VOCAB_SIZE:-16384}"
RUN_NAME="${RUN_NAME:-pico_500m}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

cd "$ARC_REPO_DIR"
mkdir -p "$ARC_DATA_DIR/logs"

echo "ARC_VOL=$ARC_VOL"
echo "ARC_DATA_DIR=$ARC_DATA_DIR"
echo "TARGET_TOKENS=$TARGET_TOKENS"
echo "VAL_TOKENS=$VAL_TOKENS"
echo "SHARD_TOKENS=$SHARD_TOKENS"
echo "TOKENIZER_SAMPLES=$TOKENIZER_SAMPLES"
df -h "$ARC_VOL" || true

TOKENIZER_DIR="${TOKENIZER_DIR:-$ARC_DATA_DIR/tokenizer_16k}"

if [[ ! -f "$TOKENIZER_DIR/vocab.json" ]]; then
  PYTHONPATH=src python scripts/train_tokenizer.py \
    --mixture configs/stage1_text_mix.jsonl \
    --out_dir "$TOKENIZER_DIR" \
    --samples "$TOKENIZER_SAMPLES" \
    --vocab_size "$VOCAB_SIZE" \
    2>&1 | tee "$ARC_DATA_DIR/logs/tokenizer_16k.log"
else
  echo "Tokenizer already exists at $TOKENIZER_DIR; skipping tokenizer training."
fi

PYTHONPATH=src python scripts/shard_text.py \
  --mixture configs/stage1_text_mix.jsonl \
  --tokenizer_dir "$TOKENIZER_DIR" \
  --out_dir "$ARC_DATA_DIR/tokens_$RUN_NAME" \
  --target_tokens "$TARGET_TOKENS" \
  --val_tokens "$VAL_TOKENS" \
  --shard_tokens "$SHARD_TOKENS" \
  2>&1 | tee "$ARC_DATA_DIR/logs/shard_$RUN_NAME.log"

cat "$ARC_DATA_DIR/tokens_$RUN_NAME/summary.json"
df -h "$ARC_VOL" || true
