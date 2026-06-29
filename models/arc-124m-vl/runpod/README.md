# RunPod CLI Workflow

This folder contains shell scripts for the Arc Stage 1 RunPod workflow.

The scripts use documented RunPod CLI commands:

- `runpodctl network-volume create`
- `runpodctl pod create`
- `runpodctl pod list`
- `runpodctl pod get`
- `runpodctl send`
- `runpodctl receive`
- `runpodctl pod delete`

Keep API keys and Hugging Face tokens out of this repo. Export them in your shell.

## Local Setup

```bash
runpodctl config --apiKey YOUR_RUNPOD_API_KEY
runpodctl datacenter list
runpodctl gpu list
runpodctl template search pytorch
```

Choose one datacenter and keep the network volume, prep pod, and H100 pod in that same datacenter.

## Create Network Volume

```bash
export RUNPOD_DATACENTER_ID="US-GA-1"
export VOLUME_NAME="arc-data"
export VOLUME_SIZE_GB=300
bash runpod/create_volume.sh
```

Then get the volume ID:

```bash
runpodctl network-volume list
export RUNPOD_NETWORK_VOLUME_ID="your-volume-id"
```

## Create Cheap Prep Pod

The default uses the documented `runpod-torch-v240` template and a cheap GPU.

```bash
export RUNPOD_DATACENTER_ID="US-GA-1"
export RUNPOD_NETWORK_VOLUME_ID="your-volume-id"
export GPU_ID="NVIDIA RTX 4000 Ada Generation"
bash runpod/create_prep_pod.sh
```

Get SSH info:

```bash
runpodctl pod list
runpodctl pod get <pod-id>
```

## Send Repo To Pod

On your Mac:

```bash
bash runpod/package_repo.sh /tmp/arc_runpod.tgz
runpodctl send /tmp/arc_runpod.tgz --code arc-runpod
```

On the pod:

```bash
cd /workspace
rm -rf /workspace/arc
mkdir -p /workspace/arc
cd /workspace/arc
runpodctl receive arc-runpod
tar --no-same-owner -xzf arc_runpod.tgz
bash runpod/install_pod_deps.sh
```

## Prepare Stage 1 Tokens On Cheap Pod

Use a read-only HF token.

```bash
export HF_TOKEN="hf_read_only_token"
export ARC_VOL="/workspace"
export TARGET_TOKENS=20000000000
export VAL_TOKENS=100000000
export RUN_NAME="stage1_20b"
bash runpod/prep_stage1_tokens.sh
```

For 25B:

```bash
export TARGET_TOKENS=25000000000
export RUN_NAME="stage1_25b"
bash runpod/prep_stage1_tokens.sh
```

Terminate the prep pod after data is prepared. The network volume keeps `/workspace` data.

```bash
runpodctl pod delete <pod-id>
```

## Create H100 Pod

```bash
export RUNPOD_DATACENTER_ID="US-GA-1"
export RUNPOD_NETWORK_VOLUME_ID="your-volume-id"
export GPU_ID="NVIDIA H100 80GB HBM3"
export GPU_COUNT=8
bash runpod/create_h100_pod.sh
```

SSH in, receive/extract the repo if it is not already on the volume, then run deps:

```bash
bash runpod/install_pod_deps.sh
```

## Start Stage 1 Training

Use `nohup` for the real run so a browser or SSH disconnect does not stop training.

For 20B:

```bash
export ARC_VOL="/workspace"
export RUN_NAME="stage1_20b"
export TARGET_TOKENS=20000000000
export BATCH_SIZE=32
export NPROC_PER_NODE=8
bash runpod/start_stage1_nohup.sh
tail -f /workspace/arc_data/logs/nohup_stage1_20b.log
```

For 25B:

```bash
export ARC_VOL="/workspace"
export RUN_NAME="stage1_25b"
export TARGET_TOKENS=25000000000
export BATCH_SIZE=32
export NPROC_PER_NODE=8
bash runpod/start_stage1_nohup.sh
tail -f /workspace/arc_data/logs/nohup_stage1_25b.log
```

## Experimental Mixed Stage 1

The `codex/mixed-pretraining` branch also includes `runpod/train_stage1_mixed.sh`.
Use this only after preparing the text shards, tokenizer, and image-caption
manifest on the same network volume.

Expected files:

```text
/workspace/arc_data/tokenizer_32k/
/workspace/arc_data/tokens_stage1_20b/train/
/workspace/arc_data/tokens_stage1_20b/val/
/workspace/arc_data/vl/vl_train.jsonl
/workspace/arc_data/vl/vl_val.jsonl
```

Launch manually on the H100 pod first; do a short benchmark before a full run:

```bash
export ARC_VOL="/workspace"
export RUN_NAME="stage1_mixed_20b"
export TOKEN_DIR="/workspace/arc_data/tokens_stage1_20b"
export IMAGE_MANIFEST="/workspace/arc_data/vl/vl_train.jsonl"
export VAL_IMAGE_MANIFEST="/workspace/arc_data/vl/vl_val.jsonl"
export TARGET_TOKENS=20000000000
export IMAGE_WEIGHT=0.04
export BATCH_SIZE=32
export IMAGE_BATCH_SIZE=32
export NPROC_PER_NODE=8
export MAX_STEPS=200

bash runpod/train_stage1_mixed.sh
```

If that benchmark looks healthy, raise `MAX_STEPS` for the real run.

## Upload Checkpoints To Hugging Face

Run this later from a cheap pod attached to the same volume.

```bash
export HF_TOKEN="hf_write_token"
export HF_REPO_ID="yourname/arc-124m-vl-checkpoints"
export CHECKPOINT_DIR="/workspace/arc_data/checkpoints_stage1_20b"
export REMOTE_PATH="stage1_20b"
bash runpod/upload_checkpoints_hf.sh
```
