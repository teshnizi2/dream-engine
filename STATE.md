# dream-engine: handoff note (STATE)

Last updated: 2026-10-17 21:10 (v3 iteration in progress, see "v3 iteration" below; v2 is finished and tagged `v2-final`).

## v3 iteration (task: `BRIEF-v3.md`; pre-registration: `docs/PREREG-v3.md`)
Goal: fix the flat-wall collapse of long dreams. Hypothesis: observability (a close-up wall says nothing about where you are).
v2 is the fallback and stays reproducible (`git checkout v2-final`; `game/spec_v2.json` is byte-identical to its spec).

| Step | Status | Evidence |
|---|---|---|
| Tag `v2-final` on 43d7de4 | DONE | `git tag` |
| Context length K as a parameter (train, eval, export, tests) and `--game` spec selection | DONE | `tests/test_context_length.py`; train.py `--num-ctx --game`; checkpoints record both |
| Flat-wall metric + pre-registered evaluation | DONE | `model/flatwall.py`, `eval_prereg.py`, `tools/flatwall_justify.py`; constants justified on 32 v2 episodes (M4): `results/prereg_full1_m4_32ep.*`, `results/flatwall_justification_full1_m4_32ep.json` |
| Game v3 (position tint + 4 face shades), Python/JS twins | DONE | `game/spec_v3.json`, `tests/test_game_v3.py`, parity 100.0000% / max diff 0 for v2 and v3 (`results/game_parity*.json`) |
| Design evidence (no model) | DONE | `results/signature_design_v3.json`, `results/observability_v2_v3.json` |
| `docs/PREREG-v3.md` committed before any v3 training | see git log | |
| ALICE: baselines, ablation (a) v2/K8, (b) v3/K4, (c) v3/K8 at 20k steps, then the full run | not started | RUNS.md |

Decisions (v3):
* **D12. The collapse metric has a persistence window.** The real game is flat-wall (no floor/ceiling visible) in 23% of its frames, so the
  brief's single-frame definition has a floor of 0.175 for a perfectly calibrated but untracked dream against v2's 0.375 and could never be
  halved. "Stuck on a flat wall" = the last 32 frames all flat; window, tolerance and threshold justified on v2 in the PREREG (section 4);
  the single-frame rate is reported too.
* **D13. Every model wakes up at real frame 8** (K = 8 needs 8 real frames), always on the first n of 64 fixed trajectories, so v2 and v3 see
  the same paths and K = 4 / K = 8 models are scored on the same frames.
* **D14. v3 tint:** three smooth waves of the hit position, one per colour channel (amplitude 12 grey levels, periods >= 5 cells), found by a
  seeded search for maximal separation of the 142 visible wall faces; faces shaded W 1.00 / E 0.90 / N 0.78 / S 0.66; the pillar keeps its look.
  Chosen on a model-free proxy (nearest-neighbour pose recovery from flat-wall frames: within 1.5 cells 46% -> 91%), not on any trained model.
* **D15. Larger collision radius: not done** (it cannot stop the wall filling the view; it would change the dynamics so v2 and v3 episodes
  would differ).

## Goal
A small action-conditioned diffusion world model (DIAMOND-style EDM U-Net) trained on a first-person raycaster maze.
In the browser (onnxruntime-web, WebGPU) next to the real game: REAL vs DREAM, same keypresses. Spec: `BRIEF.md`.
Shared rules: `../visual-series-RULES.md` (ALICE was approved for this project's training and evaluation, one row per experiment in `RUNS.md`).

## Status (verified = produced by a run, saved in `results/`)
| Piece | Status | Evidence |
|---|---|---|
| Game twins (Py `game/raycaster.py`, JS `game/raycaster.mjs`, shared `game/spec.json`) | DONE | `results/game_parity.json`: 100.0000% identical pixels, max abs diff 0 |
| Data (`data/`) | DONE | `results/data_throughput.json`, `results/coverage.json|png` |
| Model, `train.py` (resumable, EMA, bf16, unroll / blur / rollout-context options), `eval.py`, `export.py` | DONE, tested | `pytest tests`, `node --test tests/*.test.mjs`, `node tools/site_smoke.mjs` (see the verification note at the end) |
| Smoke runs | DONE | local MPS run stopped at step 275 (GPU shared, 0.35 it/s, loss 1.10 -> 0.22); ALICE smoke 5246018: 3000 steps in 4.5 min, 24.9 dB one step ahead (`results/*_smoke_metrics.jsonl`, RUNS.md row 1) |
| Main training run `full1` (150k steps, 6.77M params) | DONE | Slurm 5246233, 3.35 GPU-h, RUNS.md row 9; `runs/full1_final_ema.pt` (gitignored), metrics in `runs/full1_logs/` |
| Final evaluation (64 episodes x 512 frames on an A100, 16 episodes of probe, M4 cross-check) | DONE | `results/eval_full1_n64.json` (primary), `results/probe_full1_a100.json`, `results/eval_full1_local.json` (M4), RUNS.md row 11 |
| ONNX export + verification (fp32 / fp16, PyTorch vs ORT CPU vs WebGPU vs WASM) | DONE | `results/onnx_export_dream.json`, `results/onnx_browser_verification.json`, `results/browser_vs_python_1step.json` |
| Browser demo `site/` with the final model | DONE | 14/14 smoke checks; 28 fps median (2 steps), `results/browser_fps.json`, WASM fallback 7 fps `results/browser_fps_wasm.json`, profile `results/browser_profile.json` |
| Media | DONE | `media/dream-1080x1350.mp4` (22 s), `media/dream-1080x1080.mp4`, `media/hero.png`, `media/drift.png`, `media/rollout_grid.png`, `media/rollout.gif`, `media/sampler_steps.png` |
| Docs | DONE | `README.md` (generated by `tools/fill_readme.py`, every number from a results json), `POST.md` (119 words, never posted), `RUNS.md`, this file |

## Decisions (change only with a reason written here)
1. **Action space is factorised** (move x strafe x turn = 27 combos, one-hot 3+3+3) instead of "6 discrete": hold W + A/D for arcs.
2. **No dataset on disk**: trajectories are generated on the fly (the Mac has ~10 GB free).
3. **Bit-exact JS/Python twins**: no libm trig in the simulation loop; heading is a unit vector rotated by a precomputed (cos, sin).
4. **Game v2 (commit 17cf67f): learnable textures.** Four architectures on game v1 landed within 0.4 dB; error attribution showed
   the window stripes were 5% of the pixels and 58% of the squared error. Redesigning the textures gave +2.8 dB at the same recipe.
5. **Model**: DIAMOND U-Net, widths 32-64-96-128, 2 res blocks per level, mid-block attention, K = 4 frames, EDM (sigma_data 0.5,
   lognormal(-0.4, 1.2)), context-noise augmentation sigma_c ~ U(0, 0.5) p 0.9. 6.77M params, 150k steps, batch 128, lr 3e-4 cosine to 3e-5.
6. **Sampler**: Euler only (Heun with 2-3 steps diverges here). Demo default: 2 steps, `unit` start, context noise 0.05 (written into
   `site/models/dream_meta.json` by `export.py`). 2 steps score 34.0 dB one step ahead against 33.7 for 1 step and are visibly cleaner
   (`media/sampler_steps.png`); the drift probe cannot separate 1, 2 and 3 steps (differences within noise). 2 steps cost 29 ms/frame.
7. **Drift stabilisers do not help.** 4k-step fine-tunes of the 20k-step model (plain, DIAMOND unroll 3, context blur, scheduled sampling):
   no variant beats the plain control beyond noise. RUNS.md row 6 claimed that blur reduces collapse onto flat walls; row 10 shows this
   does not replicate under the demo sampler (29.8 vs 29.9 dB), so the blur fine-tune of `full1` was not run.
8. **What the dream is.** It follows the real game for a median of 16 frames (middle half 8 to 30), then dreams a different but game-like
   place; late frames get smoother and often collapse onto flat wall views; a 360 degree spin does not return to the start view (no map
   memory). The headline metric is the per-episode median tracking horizon: the mean curve's crossing of 20 dB is unstable (21 on 64
   episodes, 37 on another 16) and the mean PSNR over 32 frames moves between 22.0 and 24.5 dB with the episode sample (`README.md`,
   "How stable are these numbers?"). More training helps drift too (20k to 150k steps: tracking horizon and 32-frame PSNR up, see the README table).
9. **ONNX / browser**: legacy exporter at opset 17 (GroupNorm -> InstanceNorm), conv_in padded 15 -> 16 channels (WebGPU NHWC conv is wrong for
   15 input channels), SiLU fused to QuickGelu, ORT basic optimisation. WebGPU graph capture + fp16: 13.8 MB model. onnxruntime-web pinned
   1.30.0 from jsDelivr. WASM fallback runs in a worker (`proxy`), inputs copied.
10. **Video**: the dream is re-seeded from the real game every 30 game ticks (1.25 s of the 24 fps video) and the caption says so; the live
    demo is pure imagination until "Wake up" (Auto wake-up is an option, default OFF). Hero still = tick 520 of the same render (amber pillar).
11. **Figures come from json**: `eval.py` saves per-frame curves in `results/eval_<name>.json` and the first rollouts in `results/rollouts_<name>.npz`
    (gitignored); `tools/render_drift.mjs` draws `media/drift.png` with the site's fonts; `eval.py --figs-only` rebuilds the grid and gif.

## Machine / cluster notes
- Apple M4: GPU shared with the other builders (often 30-90% busy): every fps number carries that caveat (`tools/browser_fps.mjs` records `ioreg` utilisation).
  Local MPS evaluation of the final model took 9 minutes; local training would be ~100x slower than one A100.
- ALICE: Slurm account shared with the reasoning-fractals builder (never cancel `rf-*`); the per-user cap is 12 GPUs, so extra jobs can wait for hours or minutes.
  `~/dream-engine` holds code, env, runs, logs, results. `alice/README.md` has commands + measured cost table.
- Disk: ~10 GB free locally; `runs/`, `results/rollouts_*.npz` and `tools/bench_models/` are gitignored scratch.

## Open for Reza
- **Licence** of this repo's own code is not chosen (README says so; DIAMOND's MIT notice is in `THIRD_PARTY.md`).
- **Nothing is deployed or posted.** To publish, `site/` is a static folder; the model files `site/models/dream_{fp16,fp32}.onnx` are committed with `git add -f`.
- The page loads onnxruntime-web (28 MB wasm) from jsDelivr and fonts from Google Fonts; self-hosting them is a privacy/offline choice. First load downloads ~42 MB.
- Auto wake-up default is OFF (pure imagination, honest but it collapses onto flat walls after a while); the video re-seeds every 1.25 s and says so.
  A "collapse guard" that wakes the dream when a frame goes flat would look better but would hide the failure mode: your call.
- Only headless Edge (WebGPU) was tested; Safari and Firefox are untested.
- ALICE: the planned blur fine-tune and `eval_many` runs were done or skipped as described in RUNS.md rows 10-12; nothing else is queued.

## Next steps (none required)
Ideas if this is continued: train on human play data; a second map to test generalisation; condition on a longer context or an explicit map
to fix the 360 degree spin; self-host the runtime and fonts before publishing.

## Verification gate (run fresh on 2026-10-17, 19:14-19:42)
- `pytest tests -q`: 82 passed. `node --test tests/*.test.mjs` (after regenerating the parity fixture): 10 passed, 0 failed.
- `node tools/site_smoke.mjs`: 14/14 checks (also from a fresh `git clone` of the committed repo, which carries the ONNX models); `node tools/site_fuzz.mjs`: 181 random interactions, 0 errors.
- `ffprobe` on both videos: H.264, yuv420p, 24 fps, 523 frames, 21.8 s, no audio stream (1080x1350 and 1080x1080).
- README built by `tools/fill_readme.py` with no unresolved tokens; every number in it names its results json. Local git only: nothing pushed, deployed or posted.
