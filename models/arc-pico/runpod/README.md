# Arc-Pico RunPod Workflow

This workflow is for a one-GPU Arc-Pico run, not the 8x H100 Arc-124M run.

## Shape

```text
Mac packages repo
-> RunPod single-GPU pod
-> install deps
-> train 16k tokenizer
-> shard 500M text tokens
-> prepare ~100k COCO/Flickr image-caption rows
-> run mixed text+image pretraining under nohup
-> upload checkpoints to Hugging Face
-> delete pod
```

Data is stored under:

```text
/workspace/arc_pico_data
```

Repo is expected at:

```text
/workspace/arc-pico
```

## Create Volume

```bash
export RUNPOD_DATACENTER_ID="US-GA-1"
export VOLUME_NAME="arc-pico-data"
export VOLUME_SIZE_GB=80
bash runpod/create_volume.sh
```

## Create Single-GPU Pod

Cheap prep:

```bash
export RUNPOD_DATACENTER_ID="US-GA-1"
export RUNPOD_NETWORK_VOLUME_ID="your-volume-id"
export GPU_ID="NVIDIA RTX A5000"
bash runpod/create_prep_pod.sh
```

Faster train:

```bash
export RUNPOD_DATACENTER_ID="US-GA-1"
export RUNPOD_NETWORK_VOLUME_ID="your-volume-id"
export GPU_ID="NVIDIA GeForce RTX 4090"
bash runpod/create_train_pod.sh
```

## Send Repo

Mac:

```bash
cd /Users/vidvudscalitis/Desktop/CODING/MultiModal/arc-pico
bash runpod/package_repo.sh /tmp/arc_pico_runpod.tgz
runpodctl send /tmp/arc_pico_runpod.tgz --code arc-pico
```

Pod:

```bash
cd /workspace
rm -rf /workspace/arc-pico
mkdir -p /workspace/arc-pico
cd /workspace/arc-pico
runpodctl receive arc-pico
tar --no-same-owner -xzf arc_pico_runpod.tgz
bash runpod/install_pod_deps.sh
```

## Prepare Data

```bash
export HF_TOKEN="hf_read_only_token"
export ARC_VOL="/workspace"
export RUN_NAME="pico_500m"
export TARGET_TOKENS=500000000
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
bash runpod/prep_pico_vl.sh
```

## Train

```bash
export ARC_VOL="/workspace"
export RUN_NAME="pico_mixed_500m"
export TOKEN_DIR="/workspace/arc_pico_data/tokens_pico_500m"
export TARGET_TOKENS=500000000
export IMAGE_WEIGHT=0.15
export BATCH_SIZE=64
export IMAGE_BATCH_SIZE=32
export NPROC_PER_NODE=1
bash runpod/start_stage1_nohup.sh
tail -f /workspace/arc_pico_data/logs/nohup_pico_mixed_500m.log
```

Status:

```bash
export ARC_VOL="/workspace"
export RUN_NAME="pico_mixed_500m"
bash runpod/status_stage1.sh
```

## Upload

```bash
export HF_TOKEN="hf_write_token"
export HF_REPO_ID="yourname/arc-pico-vl-checkpoints"
export CHECKPOINT_DIR="/workspace/arc_pico_data/checkpoints_pico_mixed_500m"
export REMOTE_PATH="pico_mixed_500m"
bash runpod/upload_checkpoints_hf.sh
```
