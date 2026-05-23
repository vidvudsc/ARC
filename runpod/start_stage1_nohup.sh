#!/usr/bin/env bash
set -euo pipefail

ARC_VOL="${ARC_VOL:-/workspace}"
ARC_REPO_DIR="${ARC_REPO_DIR:-$ARC_VOL/arc}"
ARC_DATA_DIR="${ARC_DATA_DIR:-$ARC_VOL/arc_data}"
RUN_NAME="${RUN_NAME:-stage1_20b}"
LOG_DIR="$ARC_DATA_DIR/logs"
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/nohup_$RUN_NAME.log"
PID_FILE="$LOG_DIR/nohup_$RUN_NAME.pid"

cd "$ARC_REPO_DIR"
nohup bash -lc '
  set -o pipefail
  echo "=== Arc Stage 1 START $(date -Is) ==="
  echo "RUN_NAME=${RUN_NAME:-stage1_20b}"
  echo "ARC_VOL=${ARC_VOL:-/workspace}"
  bash runpod/train_stage1.sh
  status=$?
  if [[ $status -eq 0 ]]; then
    echo "=== Arc Stage 1 SUCCESS $(date -Is) exit=$status ==="
  else
    echo "=== Arc Stage 1 FAILED $(date -Is) exit=$status ==="
  fi
  exit $status
' > "$LOG" 2>&1 &
echo $! > "$PID_FILE"

echo "Started Stage 1 run:"
echo "  pid: $(cat "$PID_FILE")"
echo "  log: $LOG"
echo
echo "Watch with:"
echo "  tail -f $LOG"
echo "  bash runpod/status_stage1.sh"
echo
echo "Stop with:"
echo "  kill $(cat "$PID_FILE")"
