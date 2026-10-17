# dream-engine: handoff note (STATE)

Last updated: 2026-10-17 17:45 (main ALICE run still training, ETA 18:33; see "Next steps").

## Goal
A small action-conditioned diffusion world model (DIAMOND-style EDM U-Net) trained on a first-person raycaster maze.
In the browser (onnxruntime-web, WebGPU) next to the real game: REAL vs DREAM, same keypresses. Spec: `BRIEF.md`.
Shared rules: `../visual-series-RULES.md` (ALICE is approved for this project's training, one row per experiment in `RUNS.md`).

## Status (verified = produced by a run, saved in `results/`)
| Piece | Status | Evidence |
|---|---|---|
| Game twins (Py `game/raycaster.py`, JS `game/raycaster.mjs`, shared `game/spec.json`) | DONE | `results/game_parity.json`: 100.0000% identical pixels, max abs diff 0, state diff 0 after 100 steps (re-run after the texture redesign) |
| Data (`data/`) | DONE | `results/data_throughput.json`, `results/coverage.json|png` (entropy 0.995) |
| Model, `train.py` (resumable, EMA, bf16, unroll / blur / rollout-context options), `eval.py`, `export.py` | DONE, tested | `pytest tests` 75 passed, `node --test tests/*.test.mjs` 8 passed, `node tools/site_smoke.mjs` 14/14 |
| Local MPS smoke run | DONE (stopped at step ~290: shared GPU, 0.3 it/s) | `results/local_mps_smoke_metrics.jsonl` |
| ALICE smoke job 5246018 | DONE | RUNS.md row 1 |
| Browser demo `site/` | WORKS with the w2_B (20k steps) model on the new game; final model pending | `tools/site_smoke.mjs`; ONNX in browser vs PyTorch: fp32 1.3e-6, fp16 1.3e-2 (`results/onnx_browser_verification.json`) |
| Main ALICE run `full1` (150k steps, B arch, game v2) | RUNNING: Slurm 5246233 (+ chained no-op 5246234), ~12.6 it/s, ends ~18:30 | `~/dream-engine/runs/full1` on ALICE |
| Final eval, probe, export, fps, video, hero, drift figure, README, POST | TODO after full1 | `alice/eval.slurm`, `tools/*`, `tools/fill_readme.py` |

## Decisions (change only with a reason written here)
1. **Action space is factorised** (move x strafe x turn = 27 combos, one-hot 3+3+3) instead of "6 discrete": hold W + A/D for arcs.
   Strafe works (hold-key test beats freezing) but is the weakest action (rarest in the data).
2. **No dataset on disk** (Mac has ~10 GB free): trajectories are generated on the fly.
3. **Bit-exact JS/Python twins**: no libm trig in the simulation loop; heading is a unit vector rotated by a precomputed (cos, sin).
4. **Game v2 (commit 17cf67f): learnable textures.** Wave 1 (game v1, fine textures) saturated at ~28 dB 1-step for every
   architecture. Error attribution (`results/error_attribution_*.json`): window-stripe pixels were 5% of the pixels and 58% of the squared error (first ad-hoc estimate on 128 windows: 65%). Textures are now
   distance pre-filtered ("mip" 2..7 cells fades them to their mean colour), window panes are 3 per face, fewer thin lines.
   Same recipe, 20k steps: 27.9 -> 30.7 dB. Window panes are still 5% of pixels and 36.5% of the error.
5. **Model**: DIAMOND U-Net, widths 32-64-96-128, 2 res blocks per level, mid-block attention, cond 256, K = 4 frames, EDM
   (sigma_data 0.5, offset noise 0.3, lognormal(-0.4, 1.2)), context-noise augmentation sigma_c ~ U(0, 0.5) p 0.9. 6.77M params.
   Width/depth did not matter at 20k steps (A 64x4: 28.0, B: 27.9, depth-1: 27.7, no context noise: 27.7 dB on game v1).
6. **Sampler**: Euler only. Heun with 2 or 3 steps diverges on this schedule (5 / 13 dB). 1 step is as good as 3 (30.7 vs 30.0 dB)
   and 3x faster; `unit` init (N(0,1) like the DIAMOND code) beats Karras init by +1.3 dB at 3 steps. Demo defaults are
   written into `site/models/dream_meta.json` by `export.py --ctx-sigma --init --default-steps`.
7. **Drift stabilisers tested as 4k-step fine-tunes of w2_B** (RUNS.md): plain control, DIAMOND-style unroll 3, context blur,
   rollout-context (scheduled sampling, p 0.5). None improved the tracking horizon (mean PSNR over 32 frames 20.2-20.4 dB for the
   first three, 18.6 for rollout-context); blur reduced collapse onto flat wall close-ups (nearest-real-frame PSNR 27.5 vs 30.1,
   real 24.2); rollout-context sharpened late frames but made them blobby and cost 3 dB. Run-to-run noise is ~0.4 dB.
   Plan: after full1, one optional blur fine-tune; pick by tracking + validity; otherwise ship full1 as is.
8. **Honest picture of the dream**: it follows the real game for ~8-16 frames, then dreams a different but game-like place; late
   frames get smoother. PSNR vs the real game is meaningless after divergence, so `eval.py` also reports the nearest-real-frame
   validity, sharpness and tracking-horizon numbers. The demo has a manual "Wake up", an optional "Auto wake-up" (every 2 s) and
   marks wake-ups on the drift chart. The video re-seeds every 2 s and says so in a caption.
9. **ONNX / browser**: legacy exporter at opset 17 (GroupNorm -> InstanceNorm), conv_in padded 15 -> 16 channels (WebGPU NHWC conv
   is wrong for 15 input channels), SiLU fused to QuickGelu, ORT basic optimisation. WebGPU graph capture + fp16: 13.8 MB model.
   onnxruntime-web pinned 1.30.0 from jsDelivr. WASM fallback runs in a worker (`proxy`), inputs copied.

## Machine / cluster notes
- Apple M4: GPU shared with the other builders (often 60-99% busy, one other builder runs pytest on it): every fps number must carry
  that caveat (`tools/browser_fps.mjs` records `ioreg` utilisation). Local MPS training is ~100x slower than one A100.
- ALICE: Slurm account shared with the reasoning-fractals builder (never cancel `rf-*`); per-user GPU quota makes extra jobs wait
  (`QOSMaxGRESPerUser`). `~/dream-engine` holds code, env, runs, logs, results. `alice/README.md` has commands + cost table.
- Disk: ~10 GB free locally; `runs/` and `tools/bench_models/` are gitignored scratch.

## Open for Reza
- **Licence** of this repo's own code is not chosen (README says so; DIAMOND's MIT notice is in `THIRD_PARTY.md`).
- **Nothing is deployed or posted.** To publish, `site/` is a static folder (needs the `site/models/*.onnx` files, which are
  gitignored until the final export is committed with `git add -f`).
- The page loads onnxruntime-web (28 MB wasm) from jsDelivr and fonts from Google Fonts; self-hosting them is a privacy/offline
  choice for Reza. First load downloads ~42 MB (runtime + model).
- Auto wake-up default is OFF (pure imagination); the video uses re-seeding every 2 s and says so. His call if the post should show
  the raw drift instead.

## Next steps
1. `full1` (Slurm 5246233) ends ~18:33 and runs `eval.py` inline in the same allocation (8 episodes x 256 frames, primary config 2 steps /
   `unit` / ctx 0.05 = the new `eval.py` defaults, no figures). The chained no-op job 5246234 was cancelled while pending (it would only
   have repeated that eval). The ALICE queue is shared with the other builder (12-GPU per-user cap, ~100 `rf-render` tasks queued ahead),
   so any extra ALICE job waits hours: do the rest locally. `5246644` (eval_many of the old runs) is still pending; cancel it if it has
   not started when the README is built (the README tables then use the existing `results/ablation/eval_*.json`).
2. `rsync alice2:dream-engine/runs/full1/final_ema.pt runs/full1_final_ema.pt` and `alice2:dream-engine/results/eval_full1.json results/`;
   then the local full eval with figures: `python eval.py --ckpt runs/full1_final_ema.pt --name full1_local --device mps --long 512 --episodes 8`
   (<= 20 min; falls back to cpu). README numbers come from this file, the ALICE one is the cross-check.
3. `python export.py --ckpt ... --out site/models --name dream --fp16 --verify --ctx-sigma 0.05 --init unit --default-steps 2`;
   `python tools/make_eval_summary.py`; `node tools/verify_onnx_browser.mjs`; `python tools/crosscheck.py ...`;
   `node tools/browser_fps.mjs` (quiet GPU window) + `--backend wasm --dtype fp32`; `node tools/browser_profile.mjs`.
4. Media: `node tools/capture_video.mjs --out media/dream-1080x1350.mp4 --wake-every 32 --wake-until <tick> --notes "0=...|<tick>=..."`
   (+ `--square 1`), hero still, `media/drift.png`, rollout grid/gif (written by the local eval).
5. `python tools/training_summary.py ...`, RUNS.md rows (main run, inline eval, cancelled chain job), `python tools/fill_readme.py ...`,
   POST.md, full test run, `git add -f site/models/dream_*.onnx`, commit.
