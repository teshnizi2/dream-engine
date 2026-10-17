#!/bin/bash
#SBATCH --job-name=dreamprb
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
# Inference-setting probe for one checkpoint:  RUN=abl_B sbatch alice/probe.slurm
set -euo pipefail
cd "$HOME/dream-engine"
module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/dream-engine/env"
mkdir -p results
python tools/probe_sampler.py --ckpt "runs/${RUN:?RUN}/final_ema.pt" --name "$RUN" --device cuda ${PROBE_ARGS:-}
