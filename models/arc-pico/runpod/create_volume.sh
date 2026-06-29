#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_DATACENTER_ID:?Set RUNPOD_DATACENTER_ID, for example US-GA-1. Check with: runpodctl datacenter list}"

VOLUME_NAME="${VOLUME_NAME:-arc-pico-data}"
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-80}"

runpodctl network-volume create \
  --name "$VOLUME_NAME" \
  --size "$VOLUME_SIZE_GB" \
  --data-center-id "$RUNPOD_DATACENTER_ID"

echo
echo "List volumes with:"
echo "  runpodctl network-volume list"
