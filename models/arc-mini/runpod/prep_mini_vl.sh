#!/usr/bin/env bash
set -euo pipefail

ARC_VOL="${ARC_VOL:-/workspace}"
ARC_REPO_DIR="${ARC_REPO_DIR:-$ARC_VOL/arc-mini}"
ARC_DATA_DIR="${ARC_DATA_DIR:-$ARC_VOL/arc_mini_data}"
VL_DIR="${VL_DIR:-$ARC_DATA_DIR/vl}"
COCO_EXAMPLES="${COCO_EXAMPLES:-80000}"
FLICKR_EXAMPLES="${FLICKR_EXAMPLES:-20000}"
VAL_EXAMPLES="${VAL_EXAMPLES:-1000}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

cd "$ARC_REPO_DIR"
mkdir -p "$VL_DIR/images/coco" "$VL_DIR/images/flickr30k" "$ARC_DATA_DIR/logs"

echo "ARC_VOL=$ARC_VOL"
echo "ARC_DATA_DIR=$ARC_DATA_DIR"
echo "VL_DIR=$VL_DIR"
echo "COCO_EXAMPLES=$COCO_EXAMPLES"
echo "FLICKR_EXAMPLES=$FLICKR_EXAMPLES"
echo "VAL_EXAMPLES=$VAL_EXAMPLES"
df -h "$ARC_VOL" || true

PYTHONPATH=src python scripts/prepare_coco.py \
  --streaming \
  --out "$VL_DIR/coco_all.jsonl" \
  --image_out_dir "$VL_DIR/images/coco" \
  --max_examples "$COCO_EXAMPLES" \
  2>&1 | tee "$ARC_DATA_DIR/logs/prepare_coco.log"

PYTHONPATH=src python scripts/prepare_flickr30k.py \
  --streaming \
  --out "$VL_DIR/flickr_all.jsonl" \
  --image_out_dir "$VL_DIR/images/flickr30k" \
  --max_examples "$FLICKR_EXAMPLES" \
  2>&1 | tee "$ARC_DATA_DIR/logs/prepare_flickr30k.log"

export VL_DIR VAL_EXAMPLES
python - <<'PY'
import os
import random
from pathlib import Path

vl_dir = Path(os.environ["VL_DIR"])
val_examples = int(os.environ["VAL_EXAMPLES"])
rows = []
for name in ["coco_all.jsonl", "flickr_all.jsonl"]:
    path = vl_dir / name
    if path.exists():
        rows.extend(line for line in path.read_text().splitlines() if line.strip())

rng = random.Random(1337)
rng.shuffle(rows)
val = rows[:val_examples]
train = rows[val_examples:]
(vl_dir / "vl_train.jsonl").write_text("\n".join(train) + ("\n" if train else ""))
(vl_dir / "vl_val.jsonl").write_text("\n".join(val) + ("\n" if val else ""))
print({"train": len(train), "val": len(val), "out": str(vl_dir)})
PY

wc -l "$VL_DIR"/vl_train.jsonl "$VL_DIR"/vl_val.jsonl
df -h "$ARC_VOL" || true
