#!/usr/bin/env bash
set -euo pipefail

cd "${ARC_REPO_DIR:-/workspace/arc-pico}"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(i, torch.cuda.get_device_name(i), round(props.total_memory / 1e9, 2), "GB")
PY

pip install -U pip
pip install datasets tokenizers numpy pillow tqdm huggingface_hub
pip install -e . --no-deps

python - <<'PY'
from arc.config import load_config
from arc.model import estimate_parameters
cfg = load_config("config.json")
print(estimate_parameters(cfg))
PY
