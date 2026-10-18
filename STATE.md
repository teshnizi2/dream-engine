# dream-engine: handoff note (STATE)

Last updated: 2026-10-18 01:30 (v3 iteration finished: **v3 not adopted, v2 stays shipped**; see "v3 iteration" below. v2 is tagged `v2-final`).

## v3 iteration (task: `BRIEF-v3.md`; pre-registration: `docs/PREREG-v3.md`): a negative result
Goal: fix the flat-wall collapse of long dreams. Hypothesis: observability (a close-up wall says nothing about where you are). Intervention: game v3
(every wall face gets a unique low-frequency colour signature and one of four face shades) and a longer context (K = 8). Verdict by the pre-registered rule:
**not adopted**. `full2` (v3, K = 4, 150k steps) against `full1` (v2) on the same 64 episodes x 512 frames: collapse rate 0.332 against 0.159 (rule: at most half; ratio 2.09,
paired 95% interval 1.72 to 2.56) and median tracking horizon 19 against 20 frames (rule: not shorter). v2 is fallback and shipped, reproducible with `git checkout v2-final`
(`game/spec_v2.json` is byte-identical to its spec). Nothing is deployed, pushed or posted.

| Step | Status | Evidence |
|---|---|---|
| Tag `v2-final` on 43d7de4 | DONE | `git tag` |
| Context length K as a parameter (train, eval, export, tests) and `--game` spec selection | DONE | `tests/test_context_length.py`; `train.py --num-ctx --game`; checkpoints record both |
| Flat-wall metric + pre-registered evaluation | DONE | `model/flatwall.py`, `eval_prereg.py`, `tools/flatwall_justify.py`; constants justified on 32 v2 episodes (M4): `results/prereg_full1_m4_32ep.*`, `results/flatwall_justification_full1_m4_32ep.json` |
| Game v3 (position tint + 4 face shades), Python/JS twins | DONE | `game/spec_v3.json`, `tests/test_game_v3.py`, parity 100.0000% / max diff 0 for v2 and v3 (`results/game_parity*.json`) |
| Design evidence (no trained model) | DONE | `results/signature_design_v3.json`, `results/observability_v2_v3.json` (flat-wall views localise the player within 1.5 cells 46% of the time in v2, 91% in v3) |
| `docs/PREREG-v3.md` committed before any v3 training | DONE (5b3f468; six logged entries in section 8) | `git log docs/PREREG-v3.md` |
| ALICE smoke (v3, K = 8) | DONE (found and fixed: `final_ema.pt` did not record the game) | job 5247835, RUNS.md row 13 |
| v2 baselines under the pre-registered protocol | DONE | job 5247836, rows 14, `results/prereg_w2_B.json`, `results/prereg_full1.json` |
| 20k-step ablation: (a) v2 K = 8, (b) v3 K = 4, (c) v3 K = 8, and the selection rule | DONE: v3k4 selected | jobs 5247837-39, row 15, `results/prereg_ablation.json` |
| Full run `full2` (v3, K = 4, 150k steps) and the adoption rule | DONE: not adopted | job 5247963, rows 18-19, `results/prereg_final_full2_vs_full1.json` |
| Exploratory: milestone learning curves, flat-spell statistics, flat-wall escape, signature reliance | DONE | rows 16-17, 20, `results/flat_spells.json`, `results/signature_use_*.json`, `media/collapse_curve.png` |
| Standard `eval.py` + probe of `full2` | DONE (not used for any decision) | job 5248723, row 21 |
| Site and tools take K from the model meta (tested with a random K = 8 model) | DONE | `tools/site_smoke.mjs --root <dir>`: 16/16 for K = 4 and K = 8 |

Pre-registered numbers (64 episodes x 512 frames, wake-up at real frame 8, Euler 2 steps, unit start, context noise 0.05; PSNR levels are not comparable across the two games):
| | `full1` (v2) | `full2` (v3) |
|---|---|---|
| collapse rate (stuck on a flat wall for 32 frames while the real frame is not, frames 64-512) | 0.159 (0.128 to 0.199) | 0.332 (0.274 to 0.395) |
| median tracking horizon, frames until the 5-frame mean PSNR < 20 dB (middle half) | 20 (10.5 to 33) | 19 (10.8 to 38) |
| 360 degree spin return PSNR | 15.0 dB | 15.5 dB |
| one step ahead (copy-last) | 34.1 dB (24.2) | 34.4 dB (24.7) |
Single-frame collapse 0.346 / 0.429, window-64 collapse 0.062 / 0.238, flat-wall share of dream frames 0.49 / 0.60 (the real game: 0.25).

20k-step ablation (baseline `w2_B` = v2, K = 4; median horizon / collapse rate): baseline 14.0 / 0.667; (a) v2 K = 8: 13.5 / 0.670 (a longer context alone does nothing);
(b) v3 K = 4: 15.5 / 0.542 (ratio 0.81); (c) v3 K = 8: 13.0 / 0.498 (ratio 0.75); only (b) had a median horizon at least the baseline's, so the rule picked it.

Exploratory (declared in the PREREG log before the results they describe): the collapse is a failure to leave: per flat frame the dream leaves a flat wall with probability 0.5% (`w2_B`),
3.3% (`full1`), 2.0% (`full2`), the real game 6.3%. The collapse rate falls with training in v2 (0.60 at 20k to 0.16 at 150k); v3 follows it to 60k steps and then stalls near 0.33,
while its tracking horizon is at least v2's at 6 of 7 milestones. The v3 model does read the signature (shifting it costs 0.13 dB in the escape test at 20k steps and 0.85 dB at 150k) and
still collapses. Collapsed frames are smooth colour fields (per-frame colour std about 6 grey levels against about 21 for a real wall close-up).

Decisions (v3):
* **D12. The collapse metric has a persistence window.** The real game is flat-wall (no floor/ceiling visible) in 23% of its frames, so the brief's single-frame definition has a floor of 0.175
  for a perfectly calibrated but untracked dream against v2's 0.375 and could never be halved. "Stuck on a flat wall" = the last 32 frames all flat; window, tolerance and threshold justified on v2 (PREREG section 4);
  the single-frame rate is reported too.
* **D13. Every model wakes up at real frame 8** (K = 8 needs 8 real frames), always on the first n of 64 fixed trajectories, so v2 and v3 see the same paths and K = 4 / K = 8 models are scored on the same frames.
* **D14. v3 tint:** three smooth waves of the hit position, one per colour channel (amplitude 12 grey levels, periods >= 5 cells), found by a seeded search for maximal separation of the 142 visible wall faces;
  faces shaded W 1.00 / E 0.90 / N 0.78 / S 0.66; the pillar keeps its colour. Chosen on a model-free proxy, not on any trained model.
* **D15. Larger collision radius: not done** (it cannot stop the wall filling the view; it would change the dynamics so v2 and v3 episodes would differ).
* **D16. v3 is not adopted** because the pre-registered rule says so; v3 code, game, tools and results stay in the repo (`game/spec_v3.json`, `eval_prereg.py`, ...). `game/spec.json` is still v2 (`game/make_spec.py: SHIPPED = 2`).
* **D17. What the result does not show:** every row is one training run; the paired intervals cover the sampling of episodes, not the variation between training runs, and the sign of the v3 effect changed between 20k (ratio 0.81)
  and 150k steps (2.09). It does not show that v3 causes more collapse, nor that no other signature design would help.

Open for Reza (v3):
* Whether to spend compute on seeds: three seeds each of v2 and v3 (about 3.4 GPU-hours per run, about 20 GPU-hours) would measure the run-to-run variation of the collapse rate, the biggest unknown behind D17.
* The video still re-seeds the dream every 1.25 s and says so; the live demo is pure imagination until "Wake up". A collapse guard remains your call (see below).
* Ideas not tried (the brief allowed one bounded iteration): training on the model's own long rollouts, an objective that penalises staying on a flat wall, an explicit map or longer-than-8 memory.
* The site now reads the context length from the model meta, so a K = 8 model could be dropped in (tested with a random-weights model, 16/16 smoke checks).

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

## Deployed and published (2026-10-18, approved by Reza)
- Live: https://dream.teshnizi.dev (also https://dream-engine-two.vercel.app; plain dream-engine.vercel.app is somebody else's site). Vercel project `dream-engine` (team hmkhd2-1413s-projects, `prj_7gKwOzbIZvRyeYbMy0x66q4w5VQs`), static deploy of `site/`, no build step; the model files `site/models/dream_{fp16,fp32}.onnx` are committed with `git add -f` and served from there.
- Code and weights: https://github.com/teshnizi2/dream-engine (public, MIT licence, chosen by Reza; DIAMOND's MIT notice stays in `THIRD_PARTY.md`).
- Redeploy after changing `site/`: `cd site && vercel deploy --prod --yes --scope hmkhd2-1413s-projects` (`site/.vercel/` is gitignored), then `git push`.
- Checked live: all 15 served files are byte-identical to `site/` (SHA-256, including both ONNX files); in the in-app browser the page reported `[dream] ready webgpu 8017 ms`, backend WebGPU fp16 (captured), 2 steps per frame, drift PSNR 33.8 dB, and the 375 px layout has no horizontal scroll. The in-app pane was hidden and shared the GPU, so its 19 fps is not a speed measurement (the README numbers stand). The only console message besides the warm-up log is a harmless warning that a preloaded onnxruntime wasm file was not used within a few seconds.
- Share card: `site/og.jpg` (1200x630, a crop of `media/hero.png`) plus Open Graph, Twitter and canonical tags in `site/index.html`.
- Hosting limits: Vercel Hobby includes 100 GB/month of Fast Data Transfer for the whole team (it also serves www.studypuff.com); sustained overage can lead to paused deployments (Vercel warns first; fix = upgrade to Pro or cut traffic). The fp16 model is the only large file served by Vercel (13.1 MB); the 28 MB onnxruntime wasm comes from jsDelivr, so roughly 7,000 first visits use the allowance. If a post takes off, move the heavy files to a host with free egress or upgrade the team to Pro.

## Open for Reza
- **Licence**: MIT, chosen by Reza on 2026-10-18 (`LICENSE`, `package.json`, README); DIAMOND's MIT notice is in `THIRD_PARTY.md`.
- **Deployed 2026-10-18, nothing posted.** See the section above.
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

## Verification gate after the v3 iteration (run fresh on 2026-10-18, 01:00-01:45)
- `python tests/make_parity_fixture.py && pytest tests -q`: 175 passed, exit 0. `node --test tests/*.test.mjs`: 20 passed, 0 failed (parity 100.0000% identical pixels, max abs diff 0, for the v2 and the v3 game, and
  the JS flat-wall background fraction equals the Python one on 300 poses per game).
- `node tools/site_smoke.mjs`: 16/16 checks (shipped v2 model, K = 4; also 16/16 on scratch sites with a random K = 8 model and with a smoke-trained v3 K = 8 model); `node tools/site_fuzz.mjs`: 177 random interactions, 0 errors.
- `git status` clean; `git tag v2-final` on 43d7de4; `media/`, `site/models/` and `results/eval_full1_n64.json` are byte-identical to the tag (only `media/collapse_curve.png` was added); no git remote, nothing pushed, deployed or posted; no attribution lines in any commit message.
- README rebuilt by `tools/fill_readme.py` with no unresolved tokens and no diff against the committed one; every number in its v3 section names a results json.
- Order of events: `docs/PREREG-v3.md` committed 2026-10-17 20:54:47 (5b3f468); the first v3 training job started 20:55:35.

