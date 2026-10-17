#!/bin/bash
#SBATCH --job-name=dreameng
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#
# Resumable training job (each job is capped at 4 h by gpu-short, train.py stops itself with a checkpoint
# after TIME_LIMIT_MIN minutes and every job continues from runs/$RUN/ckpt_last.pt).
#
#   cd ~/dream-engine
#   sbatch --time=00:20:00 --export=ALL,RUN=smoke,STEPS=600,TIME_LIMIT_MIN=12 alice/train.slurm     # smoke
#   bash alice/submit_chain.sh full 3                                                                # full run, 3 chained jobs
#
# Settings come from the environment (defaults below). Training uses bf16 autocast on the A100.
set -euo pipefail
cd "$HOME/dream-engine"
mkdir -p logs runs
module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/dream-engine/env"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

RUN=${RUN:-full}
STEPS=${STEPS:-120000}
BATCH=${BATCH:-128}
LR=${LR:-3e-4}
# lists use '-' (sbatch --export cannot carry commas inside a value): CHANNELS=32-64-96-128 DEPTHS=2-2-2-2
CHANNELS=${CHANNELS:-32-64-96-128}
DEPTHS=${DEPTHS:-2-2-2-2}
EMA=${EMA:-0.9995}
WORKERS=${WORKERS:-14}
TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-228}
EVAL_EVERY=${EVAL_EVERY:-2000}
CKPT_EVERY=${CKPT_EVERY:-1000}
# v3: NUM_CTX = context frames K (default 4 = v2), GAME = game spec json (default game/spec_v2.json = the v2 game)
NUM_CTX=${NUM_CTX:-4}
GAME=${GAME:-game/spec_v2.json}

echo "job $SLURM_JOB_ID on $(hostname) | run $RUN | steps $STEPS | batch $BATCH | $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python train.py --out "runs/$RUN" --steps "$STEPS" --batch "$BATCH" --lr "$LR" --channels "${CHANNELS//-/,}" \
  --depths "${DEPTHS//-/,}" --ema "$EMA" --device cuda --amp bf16 --workers "$WORKERS" --resume \
  --time-limit-min "$TIME_LIMIT_MIN" --eval-every "$EVAL_EVERY" --eval-n 128 --ckpt-every "$CKPT_EVERY" \
  --milestone-every 20000 --log-every 100 --num-ctx "$NUM_CTX" --game "$GAME" ${INIT_FROM:+--init-from "$INIT_FROM"} ${UNROLL:+--unroll "$UNROLL"} ${BLUR_PROB:+--ctx-blur-prob "$BLUR_PROB"} ${BLUR_MAX:+--ctx-blur-max "$BLUR_MAX"} ${ROLLOUT_CTX:+--rollout-ctx "$ROLLOUT_CTX"} ${ROLLOUT_PROB:+--rollout-prob "$ROLLOUT_PROB"} ${EXTRA_ARGS:-}
if [ "${RUN_EVAL:-0}" = "1" ] && [ -f "runs/$RUN/final_ema.pt" ]; then
  mkdir -p results
  python eval.py --ckpt "runs/$RUN/final_ema.pt" --name "$RUN" --no-figs --episodes "${EVAL_EPISODES:-8}" --long "${EVAL_LONG:-256}" --device cuda
fi
# the pre-registered evaluation of docs/PREREG-v3.md (64 episodes x 512 frames, ~3 min on an A100)
if [ "${RUN_PREREG:-0}" = "1" ] && [ -f "runs/$RUN/final_ema.pt" ]; then
  mkdir -p results
  python eval_prereg.py --ckpt "runs/$RUN/final_ema.pt" --name "$RUN" --game "$GAME" --episodes "${PREREG_EPISODES:-64}" --device cuda
fi
echo "finished $(date)"
