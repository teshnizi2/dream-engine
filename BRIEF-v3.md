# dream-engine v3: fix the flat-wall collapse (one bounded iteration)

Read first: `STATE.md`, then `README.md`, `RUNS.md`, `alice/README.md` and `../visual-series-RULES.md`.
v2 (HEAD 43d7de4) is finished and verified. **Tag it `v2-final` before changing anything; v2 is the
fallback and must stay reproducible.**

## The problem
- The dream tracks the real game for a median of only 16 frames.
- Long dreams collapse onto flat, featureless wall views and stay there.
- A 360° spin does not return to the start view.

The video hides this by re-seeding the dream every 1.25 s, which weakens the post's hook ("every
frame is imagined").

The likely cause is observability, not capacity:
- with K = 4 context frames, a close-up wall carries no information about where the player is;
- walls within a colour region are identical;
- N and S faces (and E and W) look the same.

Once the context is all wall, the model has nothing to localise from, and the flat-wall mode becomes
absorbing. The drift stabilisers already tried (unroll, blur, scheduled sampling) did not help, which
is consistent with this explanation.

## The change (game v3 + context)
1. **Self-localising walls.** Give every wall *cell face* a unique, low-frequency colour signature.
   - Keep each region's base hue (the look stays recognisable), but vary hue/value smoothly with the
     cell's world coordinates. Any wall view, even a close-up, then tells you where you are.
   - Shade the four face directions (N/S/E/W) distinctly, so a wall view also gives the heading
     quadrant.
   - No new high-frequency texture: the v2 finding was that stripes cost error.
   - Keep the landmarks.
   - Keep the Python/JS twins bit-exact, and keep the parity test green.
2. **Optionally,** a slightly larger collision radius, so the camera can't press its nose into a wall
   (only if it doesn't break corridors).
3. **Context length.** Test K = 8 against K = 4.

## Pre-register before training (write `docs/PREREG-v3.md` and commit it before any v3 result exists)
- **Collapse rate.** Define a "flat-wall frame" metric computable on both real and dream frames (e.g. no
  visible floor/ceiling rows, or low row-variance; pick it and justify it on v2 rollouts). The collapse
  rate is the fraction of dream frames at t ∈ [64, 512] that are flat-wall while the matching real
  frame is not.
- Per-episode median tracking horizon (PSNR > 20 dB), as in v2.
- Spin-return: PSNR between the start view and the view after a 360° turn, compared against v2.
- **Adoption rule:** adopt v3 only if the collapse rate drops by at least half and the tracking horizon
  doesn't get worse. Otherwise keep v2 and document v3 as a negative result. PSNR levels across game
  versions aren't directly comparable, so say that.

## Runs (ALICE is approved for dream-engine training; follow the how-to in the rules file)
- Ablation at 20k steps each:
  - (a) v2 game, K = 8;
  - (b) v3 game, K = 4;
  - (c) v3 game, K = 8.
  Evaluate with the pre-registered metrics on ≥ 16 episodes.
- Full run (150k steps, the same recipe as `full1`) of the best variant. Then the final evaluation on
  64 episodes × 512 frames.
- One `RUNS.md` row per experiment.
- Don't cancel the reasoning-fractals jobs (`rf-*`) that share the account.

## If v3 is adopted
Re-export ONNX (fp16 + fp32, verified as before), update `site/` (the context buffer for K, model meta),
re-measure fps, and re-render the media:
- the video with **long pure-dream stretches** (re-seed only every ~5 s, or not at all if it holds up;
  the caption must state the real interval);
- the hero, the drift figure and the rollout grid;
- the README via `tools/fill_readme.py`;
- POST.md.

Keep the honest framing: report what still fails.

## Report back (≤ 250 words)
The pre-registered numbers for v2 vs v3, whether v3 was adopted, fps, media paths, ALICE job ids and
GPU-hours.
