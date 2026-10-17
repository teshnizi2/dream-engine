# dream-engine: a game that only exists in a neural network's dream

Read `~/Random projects/visual-series-RULES.md` first. Those rules apply here.

## The one-line pitch
A small action-conditioned diffusion world model, trained on a first-person maze game, runs in the
browser and *is* the game. Left panel: the real game engine. Right panel: the neural net imagining the
next frame from your keypresses, at interactive frame rates, with no game code on the right.

## Why this, why now
World models are one of the hottest areas on GitHub in Aug to Oct 2026 (AlayaWorld, open-dreamer,
HY-World 2.0, WorldPlay2, Zing, FLUX 3 Action, ForgeWM). All of them are big research code bases. None
is *playable in a browser tab*. GameNGen (Doom) and DIAMOND (Atari, CS:GO) showed the recipe; we make
a tiny honest version anyone can try.

## Design (decided; change only with a written reason in STATE.md)

### Game: `game/` (Python + JS twins)
- A deterministic Wolfenstein-style raycaster, 64×64 RGB.
- One fixed, hand-designed 16×16 map that is fun to walk. It has:
  - corridors, rooms and loops;
  - distinct wall colours per region;
  - N/S vs E/W shading and distance fog;
  - floor and ceiling gradients;
  - 3 or 4 landmarks (a glowing amber pillar, a blue doorway, a red room, a window-like stripe
    pattern), so the dream is recognisable.
- Continuous position and heading.
- Discrete actions (6): noop, forward, back, turn-left, turn-right, plus strafe-left/strafe-right if
  they don't hurt quality.
- Collision with sliding.
- Python (numpy, vectorised over a batch) and JS implementations must produce the same frames. Test
  this TDD: fixed seeds/trajectories, ≥99.5% identical pixels, with max abs diff reported.

### Data: `data/`
- An exploration policy: sticky random actions (geometric durations), biased toward forward and turns,
  with wall-avoidance and random respawns, so the map is covered evenly.
- Generate trajectories on the fly in the training DataLoader (infinite data), or pre-generate
  ~1M frames to a uint8 memmap if that's faster. Measure it.
- Log a coverage heatmap of the map.

### Model: `model/` (follow DIAMOND, Alonso et al. NeurIPS 2024; read `eloialonso/diamond` for details)
- EDM preconditioning (Karras et al. 2022): σ_data = 0.5, training σ log-normal (P_mean = −0.4,
  P_std = 1.2, or DIAMOND's values).
- U-Net, ~4–10M params:
  - levels 64→32→16→8, 2 res blocks per level, attention at 8×8 at most;
  - input = noisy next frame ⊕ last K = 4 frames (channel concat);
  - conditioning = noise-level embedding + embeddings of the last K actions, via adaptive GroupNorm.
- Context-noise augmentation (GameNGen): noise the context frames at a random level during
  training, and condition on that level. At inference, use a small fixed level. This fights
  autoregressive drift.
- Sampler: 3 Euler steps (DIAMOND) with the EDM σ schedule (σ_min = 2e-3, σ_max ≈ 5, ρ = 7). Also
  try 2 steps with Heun and compare quality vs speed.
- Validation: PSNR/SSIM of 1-step predictions, plus autoregressive rollouts of 32/128/512 frames
  against the true game (drift curves), saved as GIF/MP4 grids.

### Training: `train.py`
- Local MPS smoke run, ≤ 20 min: loss falls, and samples look like corridors.
- The full run is for ALICE (A100, `gpu-short`, `--gres=gpu:a100:1`, 4 h per job, so it must be
  resumable from checkpoints, use bf16, and log to file). Write `alice/train.slurm` and
  `alice/setup.sh` (env creation on the cluster: module load Miniconda3/24.7.1-0, then
  `conda create ... -c conda-forge --override-channels`, or a uv venv). Do NOT submit. Report the
  expected wall time.
- If the local smoke model is already decent, also run one longer local training (≤ 90 min, in the
  background, after checking nothing else heavy is using the GPU), so the browser demo has a real
  model even before ALICE.

### Export: `export.py`
- ONNX (opset ≥ 17), batch 1, static shapes.
- fp16 if onnxruntime-web WebGPU handles it, otherwise fp32.
- Check ONNX vs PyTorch outputs numerically.

### Browser: `site/` (invoke `frontend-design` first)
- onnxruntime-web, pinned version, WebGPU execution provider with a wasm fallback; show which one
  is active.
- Keys: WASD/arrows (plus Q/E strafe if present). On-screen touch buttons on mobile.
- Two panels:
  - REAL: the JS raycaster, from the same start;
  - DREAM: the autoregressive neural rollout, seeded with 4 real frames, then pure imagination.
  - Both are upscaled with nearest-neighbour to ~384–512 px, with a subtle optional scanline effect.
- Readouts: model params, ms/frame, fps, steps per frame, drift (PSNR vs REAL), frames since wake-up.
- Buttons: "Wake up" (reseed from real), "Dream only" (hide REAL), sampler steps 1–3.
- An "About" drawer covering how it works, how it differs from GameNGen/Genie (scale!), the honest
  limits (toy game, fixed map, drift), and credits.
- The target is ≥ 15 fps on the M4 in Edge/Chrome with WebGPU. Measure it and report it.

### Media
- `?capture=1` or a scripted run: a 1080×1350 side-by-side gameplay video (~20 s), the same key
  sequence on both panels, with labels REAL / DREAM: `media/dream-1080x1350.mp4`.
- A hero still: `media/hero.png`.
- Drift-curve figure: `media/drift.png`.

### Docs
`README.md`, `STATE.md`, `POST.md`.

## Done means
- Game twins are tested.
- The training pipeline works and the smoke model learned.
- The ONNX model runs in the browser with measured fps.
- REAL vs DREAM demo and media exist.
- ALICE scripts are ready, with an estimate for the full-quality model.
- Everything honest in the README.
