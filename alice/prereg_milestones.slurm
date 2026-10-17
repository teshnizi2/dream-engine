#!/bin/bash
#SBATCH --job-name=dreammil
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Exploratory learning curve: the pre-registered evaluation (64 episodes x 512 frames, ~2 min each) of the milestone checkpoints of one run
#   sbatch --export=ALL,RUN=full1,STEPS="20000 40000 60000" alice/prereg_milestones.slurm
# Writes results/prereg_<run>_s<step>.json and .npz (not used in any pre-registered decision).
set -euo pipefail
cd "$HOME/dream-engine"
module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/dream-engine/env"
mkdir -p results
for s in ${STEPS:?STEPS}; do
  echo "== ${RUN:?RUN} step $s"
  python eval_prereg.py --ckpt "runs/$RUN/ema_step$s.pt" --name "${RUN}_s$s" --episodes "${PREREG_EPISODES:-64}" --keep 0 --device cuda
done
