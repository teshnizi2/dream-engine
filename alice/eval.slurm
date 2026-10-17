#!/bin/bash
#SBATCH --job-name=dreameval
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
# Final evaluation (all numbers + figures) of one run:   RUN=full1 sbatch alice/eval.slurm
# Writes results/eval_$NAME.json, results/probe_$NAME.json(+grid png) and media/{drift.png,rollout_grid.png,rollout.gif}.
set -euo pipefail
cd "$HOME/dream-engine"
module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/dream-engine/env"
mkdir -p results media
CK="runs/${RUN:?RUN}/${CKPT:-final_ema.pt}"
NAME="${NAME:-$RUN}"
python eval.py --ckpt "$CK" --name "$NAME" --device cuda --episodes "${EPISODES:-16}" --long "${LONG:-512}" --n-val 256 --bank 20000
python tools/probe_sampler.py --ckpt "$CK" --name "$NAME" --device cuda
