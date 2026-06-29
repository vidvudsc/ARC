# Arc-Mini RunPod Workflow

This workflow is for a one-GPU Arc-Mini run.

```text
Mac packages repo
-> cheap prep pod attached to network volume
-> train tokenizer and shard 1B text tokens
-> prepare ~100k COCO/Flickr image-caption rows
-> delete prep pod
-> GPU pod attached to same volume
-> mixed text+image pretraining under nohup
-> download or upload checkpoint
-> delete GPU pod
```

Data is stored under:

```text
/workspace/arc_mini_data
```

Repo is expected at:

```text
/workspace/arc-mini
```

## Create Volume

```bash
export RUNPOD_DATACENTER_ID="EU-CZ-1"
export VOLUME_NAME="arc-mini-data"
export VOLUME_SIZE_GB=120
bash runpod/create_volume.sh
```

## Create Prep Pod

Try CPU first from the CLI. If RunPod cannot place it in the same data center as the volume, use the cheapest available GPU.

```bash
runpodctl pod create \
  --name arc-mini-prep \
  --template-id runpod-torch-v240 \
  --compute-type CPU \
  --container-disk-in-gb 20 \
  --network-volume-id "$RUNPOD_NETWORK_VOLUME_ID" \
  --data-center-ids "$RUNPOD_DATACENTER_ID" \
  --ports "22/tcp"
```

GPU fallback:

```bash
export POD_NAME="arc-mini-prep"
export GPU_ID="NVIDIA GeForce RTX 3090"
export CONTAINER_DISK_GB=20
bash runpod/create_prep_pod.sh
```

## Send Repo

Mac:

```bash
cd /Users/vidvudscalitis/Desktop/CODING/MultiModal/arc-mini
bash runpod/package_repo.sh /tmp/arc_mini_runpod.tgz
runpodctl send /tmp/arc_mini_runpod.tgz --code arc-mini
```

Pod:

```bash
cd /workspace
rm -rf /workspace/arc-mini
mkdir -p /workspace/arc-mini
cd /workspace/arc-mini
runpodctl receive arc-mini
tar --no-same-owner -xzf arc_mini_runpod.tgz
bash runpod/install_pod_deps.sh
```

## Prepare Data

```bash
export HF_TOKEN="hf_read_only_token"
export ARC_VOL="/workspace"
export RUN_NAME="mini_1b"
export TARGET_TOKENS=1000000000
export VAL_TOKENS=2000000
export TOKENIZER_SAMPLES=100000
bash runpod/prep_stage1_tokens.sh
```

```bash
export HF_TOKEN="hf_read_only_token"
export ARC_VOL="/workspace"
export COCO_EXAMPLES=80000
export FLICKR_EXAMPLES=20000
export VAL_EXAMPLES=1000
bash runpod/prep_mini_vl.sh
```

Delete the prep pod after this.

## Train

```bash
export ARC_VOL="/workspace"
export RUN_NAME="mini_mixed_1b"
export TOKEN_DIR="/workspace/arc_mini_data/tokens_mini_1b"
export TARGET_TOKENS=1000000000
export IMAGE_WEIGHT=0.15
export BATCH_SIZE=32
export IMAGE_BATCH_SIZE=16
export NPROC_PER_NODE=1
bash runpod/start_stage1_nohup.sh
tail -f /workspace/arc_mini_data/logs/nohup_mini_mixed_1b.log
```

Status:

```bash
export ARC_VOL="/workspace"
export RUN_NAME="mini_mixed_1b"
bash runpod/status_stage1.sh
```

## Upload

```bash
export HF_TOKEN="hf_write_token"
export HF_REPO_ID="yourname/arc-mini-vl-checkpoints"
export CHECKPOINT_DIR="/workspace/arc_mini_data/checkpoints_mini_mixed_1b"
export REMOTE_PATH="mini_mixed_1b"
bash runpod/upload_checkpoints_hf.sh
```
