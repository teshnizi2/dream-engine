#!/bin/bash
# One-time environment setup for dream-engine on ALICE. Run it ON THE LOGIN NODE (compute nodes may have no
# internet):   cd ~/dream-engine && bash alice/setup.sh
# Creates ~/dream-engine/env (conda-forge only, Reza's ~/.condarc has `nodefaults`) and pip-installs the pinned
# training dependencies. Nothing outside ~/dream-engine is touched.
set -euo pipefail
cd "$HOME/dream-engine"
module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -d env ]; then
  conda create -y -p ./env python=3.12 -c conda-forge --override-channels
fi
conda activate ./env
# TORCH_INDEX (optional) picks a CUDA build matching the cluster driver, e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash alice/setup.sh
if [ -n "${TORCH_INDEX:-}" ]; then
  pip install --no-cache-dir --index-url "$TORCH_INDEX" torch==2.14.1
fi
pip install --no-cache-dir -r alice/requirements.txt
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda, "(no GPU on the login node is expected)")
PY
echo "setup done: $HOME/dream-engine/env"
