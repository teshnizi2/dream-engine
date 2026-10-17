#!/bin/bash
#SBATCH --job-name=dreamevm
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
# Same evaluation code for several runs (apples to apples), no figures:  RUNS="w2_B+w3_ft_plain" sbatch --export=ALL,RUNS=w2_B+w3_ft_plain alice/eval_many.slurm
# Writes results/eval2_<run>.json
set -euo pipefail
cd "$HOME/dream-engine"
module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/dream-engine/env"
mkdir -p results
IFS='+' read -ra LIST <<< "${RUNS:?RUNS}"
for r in "${LIST[@]}"; do
  echo "== $r"
  python eval.py --ckpt "runs/$r/final_ema.pt" --name "eval2_$r" --no-figs --device cuda --episodes "${EPISODES:-16}" --long "${LONG:-256}" --n-val 256 --bank 12000
done
