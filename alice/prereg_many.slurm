#!/bin/bash
#SBATCH --job-name=dreamprg
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# The pre-registered evaluation (docs/PREREG-v3.md) of several runs with identical code, 64 episodes x 512 frames each (~3 min each):
#   sbatch --export=ALL,RUNS=w2_B+full1 alice/prereg_many.slurm
# Writes results/prereg_<run>.json and .npz. A run's game and context length come from its checkpoint.
set -euo pipefail
cd "$HOME/dream-engine"
module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/dream-engine/env"
mkdir -p results
IFS='+' read -ra LIST <<< "${RUNS:?RUNS}"
for r in "${LIST[@]}"; do
  echo "== $r"
  CK="runs/$r/${CKPT:-final_ema.pt}"
  python eval_prereg.py --ckpt "$CK" --name "${NAME_PREFIX:-}$r" --episodes "${PREREG_EPISODES:-64}" --device cuda
done
