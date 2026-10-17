# Running dream-engine on ALICE (Leiden)

Everything here was used for the results in `../README.md`; the numbers below are measured (`../RUNS.md`,
`../results/training_summary.json`). One A100 80GB per job on partition `gpu-short` (4 h limit), `ssh alice2`.

| | |
|---|---|
| code on the cluster | `~/dream-engine` (rsync from the Mac, see below); nothing outside this folder is touched |
| environment | `bash alice/setup.sh` on the **login node** (it has internet): conda-forge Python 3.12 in `~/dream-engine/env`, pip-installed torch 2.14.1 (cu130 build; the A100 nodes' driver is 610), numpy, pillow, pytest, matplotlib, imageio |
| smoke | `sbatch --time=00:20:00 --export=ALL,RUN=smoke,STEPS=3000,TIME_LIMIT_MIN=10 alice/train.slurm` (4.75 min, 12.2 it/s at batch 128) |
| full run | `bash alice/submit_chain.sh full1 2 STEPS=150000 EMA=0.9995 BATCH=128 LR=3e-4 CHANNELS=32-64-96-128 DEPTHS=2-2-2-2 EVAL_EVERY=5000 RUN_EVAL=1` |
| evaluation | `sbatch --export=ALL,RUN=full1,NAME=full1_n64,EPISODES=64,LONG=512 alice/eval.slurm` (numbers + the 12-setting sampler probe, ~5 min on an A100; the closing eval of the training job takes 1 min), `alice/eval_many.slurm` (several runs, same code, ~1 min each), `alice/probe.slurm` (sampler settings only). The figures are drawn locally from the saved json/npz: `node tools/render_drift.mjs`, `python eval.py --figs-only` |

Notes that cost time once:

* `sbatch --export` splits on commas, so lists are passed with dashes (`CHANNELS=32-64-96-128`) and converted in the script.
* `squeue` truncates job names; use `squeue -h -u $USER -o "%i %j %T %M %R"`. The Slurm account is shared with the other
  visual-series builders (jobs named `rf-*` are not ours), and the per-user GPU quota (`QOSMaxGRESPerUser`) makes extra jobs wait.
* Every job is resumable: `train.py --resume` continues from `runs/<RUN>/ckpt_last.pt`, the LR schedule is a function of the
  global step, and a job stops itself with a checkpoint after `TIME_LIMIT_MIN` (default 228 min) or on SIGTERM.
  Chained jobs after a finished run exit immediately (the main run needed one job, so its chained successor was cancelled).
* Training data is generated on the fly by 14 DataLoader workers per job (16 CPUs requested); no dataset is copied.

## Cost estimate for a full-quality run (measured throughput, batch 128, bf16)

| model | params | it/s | 150k steps | note |
|---|---|---|---|---|
| widths 32-64-96-128, 2 blocks per level (used) | 6.77M | 12.6 | 3.3 h: **measured**, one job of 3 h 20 min (RUNS.md row 9, 3.35 GPU-h) | final EMA: 34.0 dB one step ahead |
| widths 64x4 (DIAMOND size) | 4.47M | 9.1 | 4.6 h (two chained jobs) | 20k-step ablation: within 0.4 dB of the model above |
| widths 32-64-96-128, 1 block per level | 4.51M | 18.6 | 2.2 h | within 0.4 dB at 20k steps |
| unrolled training, 3 steps | 6.77M | 4.5 | 3x the compute per step | no drift benefit measured |

Copy results back with `rsync -az alice2:dream-engine/results/ results/ablation/` and
`rsync -az alice2:dream-engine/runs/<RUN>/final_ema.pt runs/`.

Evaluation cost, measured: the closing evaluation inside the training job (8 episodes x 256 frames) 1 min; 64 episodes x 512 frames plus the
probe 6 min on an A100 (`eval.py` alone 3 min); the same 8 episodes x 512 frames on the Apple M4 (MPS) 9 min; the probe on the M4 about 10 min.
The per-user GPU cap on ALICE is shared with other jobs, so a queued evaluation waited 15 to 90 minutes.
