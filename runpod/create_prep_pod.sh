#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_NETWORK_VOLUME_ID:?Set RUNPOD_NETWORK_VOLUME_ID. Check with: runpodctl network-volume list}"
: "${RUNPOD_DATACENTER_ID:?Set RUNPOD_DATACENTER_ID to the same datacenter as the network volume}"

POD_NAME="${POD_NAME:-arc-prep}"
GPU_ID="${GPU_ID:-NVIDIA RTX 4000 Ada}"
GPU_COUNT="${GPU_COUNT:-1}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-40}"
TEMPLATE_ID="${TEMPLATE_ID:-runpod-torch-v21}"
TERMINATE_AFTER="${TERMINATE_AFTER:-24h}"

runpodctl pod create \
  --name "$POD_NAME" \
  --template-id "$TEMPLATE_ID" \
  --gpu-id "$GPU_ID" \
  --gpu-count "$GPU_COUNT" \
  --container-disk-in-gb "$CONTAINER_DISK_GB" \
  --network-volume-id "$RUNPOD_NETWORK_VOLUME_ID" \
  --data-center-ids "$RUNPOD_DATACENTER_ID" \
  --ports "22/tcp" \
  --docker-args "sleep infinity" \
  --terminate-after "$TERMINATE_AFTER"

echo
echo "Get pod details and SSH info with:"
echo "  runpodctl pod list"
echo "  runpodctl pod get <pod-id>"
