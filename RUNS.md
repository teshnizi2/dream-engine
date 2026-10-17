# RUNS.md

Append-only. One row per experiment: one group of Slurm jobs that answers one question. Never edit or reorder old rows; a correction is a new row that cites the old row number, so row numbers stay stable ids. Record smoke tests and cancelled jobs too, marked as such, and keep them out of the results. State the gap in the paper's own units.

Verdict words: MATCH | MISMATCH | UNDECIDED | FAILED | SMOKE | CANCELLED

| # | Date (start) | Job id(s) | Cluster / partition / hardware | Commit or version | Config | Metric (ours vs reference) | Verdict | Wall / GPU-h | Notes file |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-17 | 5246018 | ALICE / gpu-short / 1x A100 80GB PCIe (node874) | f63d5cc + working tree (train.py later committed as a273277) | smoke: widths 32-64-96-128, 2 res blocks/level, 6.77M params, batch 128, lr 3e-4 cosine, bf16, 3000 steps, ctx-noise 0.5/0.9, EMA 0.9995 | 1-step PSNR 24.9 dB / SSIM 0.73 (EMA, 3 Euler steps, 128 val windows) vs copy-last-frame baseline 22.4 dB / 0.565; loss 1.10 -> 0.029; samples are recognisable corridors | SMOKE | 4.75 min / 0.08 GPU-h (12.2 it/s) | runs/smoke_alice/metrics.jsonl, runs/smoke_alice/samples/ (copy of the log: results/alice_smoke_metrics.jsonl) |
