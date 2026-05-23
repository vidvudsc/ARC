#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN to a read-only Hugging Face token before running}"

ARC_VOL="${ARC_VOL:-/workspace}"
ARC_REPO_DIR="${ARC_REPO_DIR:-$ARC_VOL/arc}"
ARC_DATA_DIR="${ARC_DATA_DIR:-$ARC_VOL/arc_data}"
TARGET_TOKENS="${TARGET_TOKENS:-20000000000}"
VAL_TOKENS="${VAL_TOKENS:-100000000}"
SHARD_TOKENS="${SHARD_TOKENS:-100000000}"
TOKENIZER_SAMPLES="${TOKENIZER_SAMPLES:-2000000}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
RUN_NAME="${RUN_NAME:-stage1_${TARGET_TOKENS}}"

export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
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

if [[ ! -f "$ARC_DATA_DIR/tokenizer_32k/vocab.json" ]]; then
  PYTHONPATH=src python scripts/train_tokenizer.py \
    --mixture configs/stage1_text_mix.jsonl \
    --out_dir "$ARC_DATA_DIR/tokenizer_32k" \
    --samples "$TOKENIZER_SAMPLES" \
    --vocab_size "$VOCAB_SIZE" \
    2>&1 | tee "$ARC_DATA_DIR/logs/tokenizer_32k.log"
else
  echo "Tokenizer already exists at $ARC_DATA_DIR/tokenizer_32k; skipping tokenizer training."
fi

PYTHONPATH=src python scripts/shard_text.py \
  --mixture configs/stage1_text_mix.jsonl \
  --tokenizer_dir "$ARC_DATA_DIR/tokenizer_32k" \
  --out_dir "$ARC_DATA_DIR/tokens_$RUN_NAME" \
  --target_tokens "$TARGET_TOKENS" \
  --val_tokens "$VAL_TOKENS" \
  --shard_tokens "$SHARD_TOKENS" \
  2>&1 | tee "$ARC_DATA_DIR/logs/shard_$RUN_NAME.log"

cat "$ARC_DATA_DIR/tokens_$RUN_NAME/summary.json"
df -h "$ARC_VOL" || true
