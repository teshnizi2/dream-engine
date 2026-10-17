# Dream Engine

**A game that only exists in a neural network's dream.** A small diffusion world model, trained on a first-person
raycaster maze, runs in your browser tab and *is* the game: on the left the real engine, on the right the network
imagining every frame from the same keypresses, with no game code.

![hero](media/hero.png)

* Video (1080x1350, 20 s): `media/dream-1080x1350.mp4` · Drift figure: `media/drift.png` · Rollout grid: `media/rollout_grid.png`
* Run it: `cd site && python3 -m http.server 8000`, open <http://localhost:8000> in Chrome or Edge (WebGPU; falls back to WASM).
* Nothing here is deployed or posted. Everything below was measured; each number names the file it came from.

## What it is

| | |
|---|---|
| Game | deterministic Wolfenstein-style raycaster, 64x64 RGB, one fixed hand-designed 16x16 map with four landmarks (amber pillar, blue doorway, red-brick room, window gallery), continuous position and heading, 27 key combinations (move x strafe x turn) |
| Twins | the same game in Python (numpy, vectorised) and JavaScript, **100.0000% identical pixels, max abs diff 0** over 1112 frames (`results/game_parity.json`) |
| Model | DIAMOND-style EDM U-Net, **6.77M parameters** (widths 32-64-96-128, 2 residual blocks per level, attention in the 8x8 middle block), conditioned on the last 4 frames (channel concat) and the last 4 actions (adaptive GroupNorm); context-noise augmentation from GameNGen |
| Data | never stored: every training batch is generated on the fly by a sticky-random exploration policy (1699 frames/s per process; coverage entropy 0.995 of 1.0, no open cell without samples; `results/coverage.png`) |
| Training | ALICE (Leiden), one A100 per job, bf16, batch 128: **150,000 steps = 0.8 GPU-hours** for the main run, 4.4 GPU-hours in total including ablations (`RUNS.md`) |
| Browser | onnxruntime-web 1.30.0, WebGPU with graph capture (fp16), WASM fallback; **19 fps with 3 denoising steps, 8 with 1** on an Apple M4 that was shared with other jobs (see "Speed") |

## Results

All results are on held-out data from the game's own exploration policy unless stated. Scripts: `eval.py`,
`tools/probe_sampler.py`, `tools/browser_fps.mjs`, `export.py`; raw numbers in `results/`.

### One-step prediction (the real last 4 frames + actions -> the next frame)

256 fixed validation windows, context noise 0.1, init as in `eval.py`.

| method | PSNR dB | SSIM | model calls | browser ms/frame |
|---|---|---|---|---|
| copy the last frame (baseline) | 23.9 | 0.660 | 0 |  |
| Euler, 1 step | 30.7 | 0.896 | 1 | 121 ms |
| Euler, 2 steps | 30.8 | 0.898 | 2 | 236 ms |
| Euler, 3 steps (DIAMOND) | 30.0 | 0.884 | 3 | 323 ms |
| Heun, 2 steps (unstable here) | 5.0 | 0.034 | 3 |  |
| Heun, 3 steps (unstable here) | 12.8 | 0.158 | 5 |  |

Source: `results/ablation/eval_w2_B.json`, `/private/tmp/claude-501/-Users-teshnizi-Random-projects/dde2b447-a63a-4b03-bd23-7e8fad594ebf/scratchpad/dream/fps_test.json`.

### Drift: pure imagination against the true game

Rollouts of 256 frames from 4 real frames, 8 episodes (exploration-policy actions), Euler 3 steps, context noise 0.05. An unrelated frame scores about 12.1 dB, so the dream follows the real game for roughly the first 8-16 frames (a quarter to half a second) and then wanders to a different place.

| frames since wake-up | 1 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| dream (PSNR dB) | 30.7 | 26.0 | 21.7 | 18.4 | 17.6 | 17.6 | 13.9 | 14.3 |
| dream (SSIM) | 0.88 | 0.80 | 0.72 | 0.56 | 0.60 | 0.61 | 0.57 | 0.53 |
| freeze the last real frame (PSNR dB) | 23.1 | 20.3 | 17.7 | 14.3 | 14.2 | 15.3 | 13.3 | 13.5 |

![drift](media/drift.png)

### Is it still a game frame?

PSNR of every dreamed frame to its nearest neighbour among a bank of real frames, averaged over the rollout: **29.7 dB** for dream frames against **24.2 dB** for the true frames of the same episodes. High values late in the rollout mean the dream settles on simple, game-like views (often a wall close-up): it stays valid but loses detail (`sharpness_ratio_last32` in the probe json). Source: `results/ablation/eval_w2_B.json`.

### Do the controls work? (hold one key for 24 frames from random poses)

| key | dream @8 frames | freeze @8 | dream @24 | freeze @24 |
|---|---|---|---|---|
| forward | 23.8 | 18.7 | 20.1 | 16.5 |
| back | 21.0 | 18.0 | 17.7 | 15.6 |
| turn left | 22.0 | 15.1 | 16.2 | 13.6 |
| turn right | 22.7 | 16.3 | 15.7 | 13.7 |
| strafe left | 20.4 | 18.1 | 16.7 | 16.4 |
| strafe right | 21.3 | 17.8 | 17.8 | 16.2 |
| forward+left | 21.0 | 15.3 | 15.9 | 14.1 |
| forward+right | 24.0 | 16.9 | 17.2 | 14.0 |

PSNR dB against the real game; "freeze" repeats the last real frame. Every control beats freezing; strafing is the weakest (it is the rarest action in the training data).

### Sampler and context-noise settings

12 inference settings (1-3 Euler steps, two initialisations, context noise 0 or 0.1) over 16 episodes x 128 frames: mean PSNR over the first 32 frames ranges from 20.1 to 21.7 dB, so the setting matters little; the best was 1 step(s), init `edm`, context noise 0.0. Source: `results/ablation/probe_w2_B.json`.

### ONNX export (opset 17, batch 1, static shapes) against PyTorch

| precision | size MB | max abs diff, ONNX Runtime CPU | max abs diff, WebGPU in the browser |
|---|---|---|---|
| fp32 | 27.3 | 2.8e-06 | 1.3e-06 |
| fp16 | 13.8 | 2.4e-03 | 1.3e-02 |

Outputs are in [-1, 1]; 1 grey level = 0.0078. Sources: `results/onnx_export_dream.json`, `results/onnx_browser_verification.json`.

### Speed in the browser (Apple M4, headless Edge, WebGPU fp16 with graph capture)

| denoising steps | ms per frame, median window | fps, median window | ms per frame, best window | fps, best window | GPU busy (%) around the windows | windows |
|---|---|---|---|---|---|---|
| 1 | 120.6 | 8 | 117.9 | 8 | 98-100 | 2 |
| 2 | 235.9 | 25 | 33.1 | 25 | 80-99 | 2 |
| 3 | 323.2 | 19 | 41.0 | 19 | 80-99 | 2 |

The game tick is capped at 30 per second, so the best windows read 30 fps. **The M4's GPU was shared with other jobs on this machine during all runs** (the busy column comes from `ioreg`), which is why the median windows are slower than the best ones; a quiet GPU should behave like the best windows. Source: `/private/tmp/claude-501/-Users-teshnizi-Random-projects/dde2b447-a63a-4b03-bd23-7e8fad594ebf/scratchpad/dream/fps_test.json` (`tools/browser_fps.mjs`).

## What was tried (and what did not help)

#### Architecture and context noise (game v1, 20k steps each)

| variant | params | 1-step PSNR dB | SSIM | rollout PSNR @32 frames | copy-last baseline |
|---|---|---|---|---|---|
| widths 32-64-96-128, 2 blocks (used) | 6.77M | 27.9 | 0.835 | 16.7 | 22.2 |
| same, no context noise | 6.77M | 27.6 | 0.831 | 17.1 | 22.2 |
| widths 64x4, 2 blocks (DIAMOND size) | 4.47M | 28.0 | 0.843 | 17.9 | 22.2 |
| widths 32-64-96-128, 1 block | 4.51M | 27.7 | 0.824 | 18.3 | 22.2 |

All four land within 0.4 dB: width, depth and context noise were not the bottleneck.

#### The game, not the network, was the ceiling

| game version (same model and recipe, 20k steps) | 1-step PSNR dB | copy-last baseline dB | share of squared error on window-pane pixels |
|---|---|---|---|
| v1: fine textures | 27.9 | 22.2 | 58% (5% of pixels) |
| v2: distance pre-filtered, wide window panes | 30.7 | 23.9 | 36% (5% of pixels) |

Window stripes were 5% of the pixels but 58% of the model's squared error; redesigning them (3 panes per wall, fading to their mean colour with distance) cut that share to 36%. Source: `results/error_attribution_*.json` (`tools/error_attribution.py`).

#### Can training fix drift? (4k-step fine-tunes of the 20k-step model, game v2)

| fine-tune | 1-step PSNR dB | mean PSNR over the first 32 rollout frames | nearest-real-frame PSNR (real frames) | late-frame sharpness vs real |
|---|---|---|---|---|
| plain fine-tune (control) | 30.8 | 20.4 | 30.1 (24.2) | n/a |
| DIAMOND-style unroll 3 | 30.8 | 20.3 | 29.5 (24.2) | 0.23 |
| context blur augmentation | 30.8 | 20.2 | 27.5 (24.2) | n/a |
| scheduled sampling, 3 own frames, p 0.5 | 27.9 | 18.6 | 28.7 (24.2) | 0.54 |

None improved how long the dream tracks the real game (run-to-run noise is about 0.4 dB). Context blur reduced the collapse onto flat wall views; scheduled sampling sharpened late frames but made them blobby and cost 3 dB.

## How to run

```bash
# the demo (any static server works; WebGPU needs Chrome/Edge 113+, otherwise it runs on WASM at a few fps)
cd site && python3 -m http.server 8000

# tests: 60+ python tests (game, data, model, sampler, export, training) and the JS twin / helpers
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tests/make_parity_fixture.py && .venv/bin/python -m pytest tests -q && node --test tests/*.test.mjs
npm install && npm run smoke                       # end-to-end browser smoke test (headless Edge, WebGPU)

# train (local smoke on Apple MPS, or ALICE): see alice/ for the cluster scripts
.venv/bin/python train.py --out runs/smoke --steps 1200 --batch 32 --device mps --time-limit-min 18
.venv/bin/python eval.py --ckpt runs/full1/final_ema.pt --name full1
.venv/bin/python export.py --ckpt runs/full1/final_ema.pt --out site/models --name dream --fp16 --verify
```

URL options: `?steps=1|2|3`, `?backend=wasm`, `?dtype=fp32`, `?capture=1` (deterministic poster layout used for the video).
Keys: WASD / arrows move and turn, Q/E strafe, R wake up, T dream only, P pause, 1-3 denoising steps.

## Honest limits

* **One fixed map, 64x64 pixels.** The network has memorised this maze (the way a person knows their own flat). It does not generalise to other levels, and nothing about it is a general game engine.
* **It drifts.** Errors compound frame after frame: the dream tracks the real game for about the first 8-16 frames and then goes to a different place. It stays *game-like* (see the validity table) but is no longer the same place as the real panel, so "PSNR against the real game" stops being meaningful after that point. "Wake up" re-seeds it from the real game, and the page marks every wake-up on the drift chart.
* **It gets smoother over time.** Late in a long rollout the dreamed frames lose detail and settle on simple views (often a wall close-up): the probe reports their sharpness as a fraction of the real frames'. Edges are always softer than the real engine's pixel-exact ones, and thin features (window frames, brick joints, the rings on the pillar) are the first to blur.
* **Out-of-distribution play is weaker.** Training data is a sticky random walk with wall avoidance, not human play. Strafing (Q/E) is the rarest action and the weakest; spinning in place or pressing into walls for a long time is further from the training distribution.
* **Small, single-run evidence.** One architecture trained at one scale; every comparison is a single run per setting, and differences below about 0.5 dB are within run-to-run noise (an extra 4k plain fine-tuning steps moved the rollout means by 0.4 dB). Evaluation is on the game's own trajectories; there is no human study of how convincing it feels.
* **Speed is hardware-dependent.** The fps numbers are from one Apple M4 whose GPU was shared with other jobs, in headless Edge. Other GPUs, browsers and power states differ. Without WebGPU the page falls back to WASM at a few frames per second.
* **Toy scale on purpose.** GameNGen fine-tunes a Stable Diffusion model and Genie is trained on internet video at far larger scale; this is a few million parameters, hours of A100 time, one map. It shows the recipe, not the state of the art.
* **Heun is not a verdict on Heun.** With two or three Heun steps and the sigma range 2e-3 to 5 the sampler diverged here (5-13 dB), so Euler steps (as in DIAMOND) are used; more steps or a different schedule would likely work.

## Repo map

```
game/        raycaster.py (reference), raycaster.mjs (JS twin), spec.json (map, physics, palette: single source of truth)
data/        exploration policy, on-the-fly trajectory generator, coverage heatmap
model/       U-Net (DIAMOND blocks), EDM denoiser with context noise / unrolled / blur augmentation, samplers, metrics
train.py     resumable training (EMA, bf16, signals, time limit)      eval.py   drift / validity / control evaluation
export.py    ONNX export (static, opset 17) + fp16 + numeric checks   alice/     setup.sh, train.slurm, submit_chain.sh, probe.slurm
site/        the static demo (index.html, js/, models/)               tools/    benchmarks, capture, smoke test, probes
tests/       python + node tests (TDD)                                results/  every number quoted here, as json (+ figures)
RUNS.md      one row per ALICE experiment                             STATE.md  handoff note       POST.md  LinkedIn draft (never posted)
```

## Credits and licences

* **DIAMOND** (Alonso, Jelley, Micheli, Kanervisto, Storkey, Pearce, Fleuret, NeurIPS 2024; MIT): U-Net blocks, EDM
  preconditioning and the 3-step Euler sampler are re-implemented from `eloialonso/diamond` (read in `vendor/`).
* **EDM** (Karras, Aittala, Aila, Laine 2022); **GameNGen** (Valevski, Leviathan, Arar, Fruchter 2024), context-noise augmentation.
* **onnxruntime-web** 1.30.0 (Microsoft, MIT). Fonts: Bricolage Grotesque and JetBrains Mono (SIL OFL), via Google Fonts.
* No external datasets: the data is generated by the game in this repository. Full third-party notices: `THIRD_PARTY.md`. The licence for this repository's own code is Reza's call and has not been set.
