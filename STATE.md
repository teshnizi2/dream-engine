# dream-engine: handoff note (STATE)

Last updated: 2026-10-17 (resumed after the first Claude Code process exited mid-benchmark).

## Goal
A small action-conditioned diffusion world model (DIAMOND-style EDM U-Net) trained on a first-person
raycaster maze. In the browser (onnxruntime-web, WebGPU) next to the real game: REAL vs DREAM, same
keypresses. Spec: `BRIEF.md`. Shared rules: `../visual-series-RULES.md` (re-read it: the ALICE rule changed).

## Status (verified = produced by a run, saved in `results/`)
| Piece | Status | Evidence |
|---|---|---|
| Game twins (Py `game/raycaster.py`, JS `game/raycaster.mjs`, shared `game/spec.json`) | DONE | `results/game_parity.json`: 100.0000% identical pixels, max abs diff 0, state diff 0 after 100 steps (808 trajectory frames + 300 random poses + 4 edge poses) |
| Data (`data/policy.py`, `data/loader.py`, `data/coverage.py`) | DONE | `results/data_throughput.json` (~1.6k frames/s/process), `results/coverage.json|png` (normalised entropy 0.995, no empty cell) |
| Model (`model/unet.py`, `denoiser.py`, `sampler.py`, `metrics.py`) | DONE, tested | 47 Python tests green (`pytest tests/`), 4 node tests green (`node --test tests/*.test.mjs`) |
| `train.py` (resumable, EMA, bf16 on CUDA, logs) | TODO | |
| Local MPS smoke run (<= 20 min) | TODO | |
| `export.py` (ONNX opset 17, batch 1, static) + numeric check | TODO | |
| Browser demo `site/` | TODO | |
| ALICE: smoke job, then full chained training | TODO (approved by Reza, see below) | |
| Eval (1-step PSNR/SSIM, 32/128/512-frame drift), media, README/POST | TODO | |

## Decisions (change only with a reason written here)
1. **Action space is factorised** instead of "6 discrete": (move: none/fwd/back) x (strafe: none/left/right) x
   (turn: none/left/right) = 27 combos, fed to the model as one-hot 3+3+3 (= sum of three embeddings).
   Reason: smooth controls (hold W and A/D for arcs); Q/E strafe is kept only if its rollouts are as good as
   the others (check per-action PSNR; otherwise hide Q/E in the UI and say so in the README).
2. **No dataset on disk.** The Mac has ~9.6 GB free (96% full), 1M frames would be 12 GB. Trajectories are
   generated on the fly (`data.loader.BatchStream`, ~1.6k frames/s per process).
3. **Bit-exact JS/Python twins**: the simulation uses only + - * / floor abs sqrt on float64 and a unit-vector
   heading rotated by a precomputed (cos, sin); no libm trig in the loop. Result: max abs diff 0.
4. **Map**: 16x16, 119 open cells, regions coloured by wall type (blue west wall of the hall with a doorway,
   amber pillar at (7,7), window gallery along the north wall, red-brick east room, green west wing, purple SW,
   ochre SE). Start pose: west corridor looking east through the doorway at the pillar.
5. **Physics**: move 0.12 tile/step, strafe 0.10, turn 5 deg/step, collision radius 0.22 with sliding,
   plane 0.8 (FOV 77 deg), fog f = 1/(1+(0.16 d)^2). Turn first, then move along the new heading.
6. **Model recipe** (read `vendor/diamond`, MIT): EDM preconditioning sigma_data 0.5, offset noise 0.3, training
   sigma lognormal(-0.4, 1.2) clipped to [2e-3, 20], DIAMOND U-Net (depths 2 per level, mid-block attention at 8x8,
   AdaGroupNorm, cond 256), K = 4 context frames by channel concat. Added: context-noise augmentation
   (sigma_c ~ U(0, 0.5) w.p. 0.9, embedded by an MLP). Sampler: Euler (DIAMOND) or Heun, sigma_min 2e-3,
   sigma_max 5, rho 7. Default init sigma_max*N(0,I) (Karras); DIAMOND's code starts from N(0,I): pick by validation.
7. **Candidate sizes** (measured with `scratchpad` bench): DIAMOND-size (64,64,64,64): 4.47M params, 3.04 GMACs
   per forward; (32,64,96,128): 6.77M params, 1.49 GMACs. All widths must be multiples of 32 (GroupNorm groups).
8. **ONNX**: legacy exporter at opset 17 so GroupNorm lowers to InstanceNormalization (the WebGPU EP has no
   GroupNormalization). onnxruntime-web pinned to 1.30.0 (published 2026-09-14, > 2 weeks old).

## Machine notes
- Apple M4 MPS is slow and shared: DIAMOND-size train step at B=32 takes ~1.7 s (about 0.35 TFLOPS effective;
  other builders keep the GPU 20-60% busy). A benchmark at B=64 was killed (exit 137, likely memory). Keep local
  batches <= 32 and run only one heavy job at a time.
- Disk: ~9.6 GB free. Keep checkpoints small (~50 MB each), media modest.

## Open for Reza
- ALICE budget: approved for the long dream-engine run (his words, relayed by the coordinator: "Yes, you can submit
  the long ALICE runs for the dream-engine training and the high-res fractal renders."). Default taken: one short
  smoke job, then one resumable run of chained 4 h jobs on `gpu-short` / `a100:1` with a hard cap on total hours
  (see `alice/`); GPU-hours used will be reported.
- Strafe (Q/E): kept in the action space by default; hidden in the UI if it hurts quality (see decision 1).

## Next steps
1. `git commit` this milestone. Write `train.py`; local MPS smoke run (loss falls, samples look like corridors).
2. `export.py` + ONNX-vs-PyTorch check; browser benchmark of the architecture (random weights are enough for speed).
3. `alice/setup.sh`, `alice/train.slurm`; smoke job on ALICE, then the full chained run; `RUNS.md` row per experiment.
4. Browser demo with the smoke checkpoint while ALICE trains (invoke `frontend-design` first), then swap in the best model.
5. Eval + media (`media/dream-1080x1350.mp4`, `hero.png`, `drift.png`), README, POST, `verification-before-completion`.
