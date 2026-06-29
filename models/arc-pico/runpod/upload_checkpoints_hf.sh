#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN before running}"
: "${HF_REPO_ID:?Set HF_REPO_ID, for example yourname/arc-pico-vl-checkpoints}"

ARC_VOL="${ARC_VOL:-/workspace}"
ARC_DATA_DIR="${ARC_DATA_DIR:-$ARC_VOL/arc_pico_data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ARC_DATA_DIR/checkpoints_pico_mixed_500m}"
REMOTE_PATH="${REMOTE_PATH:-pico_mixed_500m}"

export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export HF_REPO_ID CHECKPOINT_DIR REMOTE_PATH

python - <<'PY' || pip install -U huggingface_hub
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("huggingface_hub") else 1)
PY

python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
folder_path = os.environ.get("CHECKPOINT_DIR")
path_in_repo = os.environ.get("REMOTE_PATH", "stage1")
token = os.environ["HUGGING_FACE_HUB_TOKEN"]

api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=folder_path,
    path_in_repo=path_in_repo,
    commit_message=f"Upload Arc checkpoints to {path_in_repo}",
)
print(f"Uploaded {folder_path} to hf://{repo_id}/{path_in_repo}")
PY
