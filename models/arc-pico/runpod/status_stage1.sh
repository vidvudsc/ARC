#!/usr/bin/env bash
set -euo pipefail

ARC_VOL="${ARC_VOL:-/workspace}"
ARC_DATA_DIR="${ARC_DATA_DIR:-$ARC_VOL/arc_pico_data}"
RUN_NAME="${RUN_NAME:-pico_mixed_500m}"
LOG_DIR="$ARC_DATA_DIR/logs"
LOG="$LOG_DIR/nohup_$RUN_NAME.log"
TRAIN_LOG="$LOG_DIR/train_$RUN_NAME.log"
PID_FILE="$LOG_DIR/nohup_$RUN_NAME.pid"
OUT_DIR="${OUT_DIR:-$ARC_DATA_DIR/checkpoints_$RUN_NAME}"

echo "=== Arc Stage 1 Status ==="
echo "RUN_NAME=$RUN_NAME"
echo "ARC_DATA_DIR=$ARC_DATA_DIR"
echo "OUT_DIR=$OUT_DIR"
date -Is
echo

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  echo "pid: $pid"
  if kill -0 "$pid" 2>/dev/null; then
    echo "process: running"
  else
    echo "process: not running"
  fi
else
  echo "pid: missing ($PID_FILE)"
fi

echo
echo "disk:"
df -h "$ARC_VOL" || true

echo
echo "gpu:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
else
  echo "nvidia-smi not available"
fi

echo
echo "latest checkpoints:"
if [[ -d "$OUT_DIR" ]]; then
  ls -lh "$OUT_DIR"/*.pt 2>/dev/null | tail -n 8 || echo "no checkpoints yet"
else
  echo "checkpoint directory missing"
fi

echo
echo "plots:"
if [[ -f "$TRAIN_LOG" ]]; then
  if PYTHONPATH="${PYTHONPATH:-src}" python scripts/plot_training_log.py \
    --log "$TRAIN_LOG" \
    --out_dir "$LOG_DIR" \
    --prefix "$RUN_NAME" >/tmp/arc_plot_status.json 2>/tmp/arc_plot_status.err; then
    cat /tmp/arc_plot_status.json
  else
    echo "plot generation failed:"
    cat /tmp/arc_plot_status.err || true
  fi
else
  echo "training log missing; no plots yet"
fi

echo
echo "last status lines:"
if [[ -f "$LOG" ]]; then
  grep -E "Arc Stage 1 (START|SUCCESS|FAILED)|Traceback|RuntimeError|CUDA out of memory|error|ERROR|step" "$LOG" | tail -n 30 || tail -n 30 "$LOG"
else
  echo "log missing: $LOG"
fi

echo
echo "log files:"
echo "  nohup: $LOG"
echo "  train: $TRAIN_LOG"
