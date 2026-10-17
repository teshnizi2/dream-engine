# Pre-registration: dream-engine v3 (the flat-wall collapse)

Written and committed **before any model has been trained on the v3 game or with 8 context frames**, and before any number that
could decide the outcome exists. What exists at the time of writing: the v2 model (`v2-final` tag, 150k steps), a 32-episode
evaluation of it that was used only to define the metric (section 4), the v3 game code and the model-free evidence for its design
(section 2). Nothing below may be changed after a v3 result is seen; a change goes into the deviation log (section 8) with its reason
and the commit that made it.

## 1. Question

v2 tracks the real game for a median of 16 frames and then dreams a different, game-like place. Late in a long dream the frames
collapse onto flat wall views and stay there; a full 360 degree turn does not return to the start view.

**Hypothesis (observability).** With K = 4 context frames, a close-up wall carries no information about where the player is: walls
inside one colour region are identical, and so are N and S faces (and E and W faces). Once the context is all wall, the model has
nothing to localise from and the flat-wall state is absorbing. If that is the cause, then (i) giving every wall face a unique,
low-frequency colour signature and distinct face shades (game v3) lets a close-up say where you are and which way you face, and the
collapse rate falls; (ii) a longer context alone (K = 8 on the unchanged game) does not, because eight flat-wall frames are as
uninformative as four. The drift stabilisers tried in v2 (unrolled training, context blur, scheduled sampling) did not help, which is
consistent with this explanation but is not a test of it.

## 2. The intervention

* **Game v3** (`game/spec_v3.json`, built by `game/make_spec.py`, renderer in `game/raycaster.py` and its twin `game/raycaster.mjs`).
  It adds, and only adds, two things to v2 (a test checks that every other key of the spec is unchanged):
  1. a smooth colour tint on every wall that depends on the world position of the hit point: channel c of the offset is
     `12 * wave(ax_c * x + ay_c * y + phase_c)` grey levels, `wave` a smooth periodic function in [-1, 1], with periods of at least 5
     cells (low frequency, no new pixel-scale texture). Each region keeps its base colour; the landmark pillar keeps its exact amber look
     (tint gain 0);
  2. four distinct shades for the four wall-face directions, W 1.00, E 0.90, N 0.78, S 0.66 (v2: E/W 1.00 and N/S 0.74).
  The map, physics, floor, ceiling, fog, textures and landmarks are unchanged, so the real game's trajectories and its floor/ceiling pixels
  are identical in v2 and v3 (tested), and the flat-wall flags of the real frames are the same in both.
  The twins stay bit-exact: 100.0000% identical pixels, max abs difference 0, on 1112 fixed frames per game
  (`results/game_parity.json`, `results/game_parity_v3.json`). The v2 game stays reproducible bit for bit (`game/spec_v2.json` is
  byte-identical to the `v2-final` spec; golden frame hashes are tested).
* **Design evidence that needs no trained model.** The three (ax, ay, phase) triples were found by a seeded search that maximises the
  smallest colour difference between any two visible faces of the same wall type (`tools/design_signature.py`,
  `results/signature_design_v3.json`): the 142 visible wall faces are all at least 3.9 grey levels apart and neighbouring cells along a
  wall at least 4.6 apart. A model-free test of how much a flat wall view says about the pose (`tools/observability.py`,
  `results/observability_v2_v3.json`: nearest-neighbour pose recovery from noisy flat-wall frames among 20,000 rendered frames) gives, for
  v2 and v3:

  | | v2 | v3 |
  |---|---|---|
  | position error of the recovered pose, median (cells) | 1.89 | 0.25 |
  | recovered position within 1.5 cells | 46% | 91% |
  | recovered heading within 45 degrees | 71% | 96% |

  This says the information is in the frames. It does not say a network learns to use it; that is what the experiments test.
* **Context length K = 8** against K = 4 (the model takes K frames and K actions; 6.77M parameters at K = 4, about 3k more at K = 8).
* **Larger collision radius: not done.** It was optional. Any radius that keeps the one-cell corridors passable (below 0.5) still lets the
  wall fill the view (it does for any camera closer than one cell), and it would change the dynamics, so the v2 and v3 episodes
  would no longer be the same paths. Recorded here so that it is not read as an omission.

## 3. Evaluation protocol (`eval_prereg.py`, identical for every model)

* The same 64 trajectories of the exploration policy (seed 123, no warm-up; always generated as 64 and sliced, so the first n are the same
  for every n and for both games).
* Every model wakes up at **real frame 8**: a K = 8 model sees real frames 0 to 7, a K = 4 model frames 4 to 7. Then it dreams 512
  frames in pure imagination from the real action stream, and each dreamed frame t (t = 1 is real frame 8) is scored against real
  frame 8 + t - 1.
* Sampler: Euler, 2 steps, `unit` start, context noise 0.05 (the demo's setting), EMA weights, on an A100.
* The ablation runs and the final models are all evaluated on all 64 episodes.
* Integrity check on the pairing: the flat-wall flags of the real frames of a v3 evaluation must equal those of the v2 evaluation (same
  paths, same geometry); the report states whether they do.

## 4. Metrics

**M1, tracking horizon (as in v2).** Per episode, the first frame whose 5-frame average PSNR against the real game is below 20 dB (an
episode that never falls below counts as 513). Report the median over the 64 episodes (and the quartiles, and the same at 25 dB).

**M2, collapse rate.**
* A frame is **flat-wall** if at most 5% of its pixels are floor or ceiling. A pixel is floor/ceiling if its colour is within 4 grey
  levels (max over channels) of the game's own background colour for its row (`World.bg_rows`; the game paints floor and ceiling by
  row, the same in v2 and v3). This is a statement about geometry, not texture, so adding colour to the walls cannot change it, and it
  can be computed on a real frame, a dreamed frame and a frame of either game version (`model/flatwall.py`).
* A dream is **stuck on a flat wall** at frame t if the last 32 frames (t - 31 ... t) are all flat-wall.
* The **collapse rate** is the fraction of dream frames at t in [64, 512] that are stuck on a flat wall while the matching real frame
  is not (the real game's stuck flag uses the same 32-frame window).
* Why a persistence window rather than the single-frame flag. The real game is itself flat-wall in 23% of its frames (the exploration
  policy often stands against walls), so a dream that had exactly the real game's share of flat frames but did not track it would already
  score p(1-p) = 0.175 on the single-frame version against v2's 0.375: halving would be impossible by construction. What the v2 dream does
  wrong is to stay: its flat-wall share rises from the real game's level at frames 1 to 32 (18% against 16%) to about 50% from frame
  65 on while the real game stays at about 23%. A 32-frame window is about the 92nd percentile of the length of a flat spell in the real
  game (24 frames at the 90th percentile, 47 at the 95th), so the real game is rarely "stuck" (5.8% of frames) and the v2 dream often is
  (22%).
* Evidence, from the 32-episode v2 evaluation on the Apple M4 (`results/prereg_full1_m4_32ep.*`, `results/flatwall_justification_full1_m4_32ep.json`,
  contact sheet `results/flatwall_sheet_full1_m4_32ep.png`), used only to choose these constants:

  | window (frames) | dream stuck share | real stuck share | collapse rate (95% interval over episodes) | rate of a dream that matched the real game's stuck share but did not track it |
  |---|---|---|---|---|
  | 1 | 0.505 | 0.227 | 0.375 (0.315 to 0.438) | 0.175 |
  | 16 | 0.332 | 0.089 | 0.282 (0.227 to 0.346) | 0.081 |
  | **32 (chosen)** | 0.220 | 0.058 | **0.183 (0.133 to 0.248)** | 0.054 |
  | 64 | 0.104 | 0.029 | 0.085 (0.040 to 0.147) | 0.028 |

  The colour tolerance does not matter on the plateau (dream flat share 0.47 at tolerances 2, 4, 6 and 10 grey levels, 8 saved episodes;
  the real game's value does not change at all); the flat threshold sits where the dream's background-fraction distribution has its mass
  at zero (45% of late dream frames show no floor or ceiling at all, against 17% of real frames, and 50% against 23% at the 5% threshold). The constants are fixed: tolerance 4, flat
  threshold 5%, window 32, frames 64 to 512 (the values in `model/flatwall.py`).
* Also reported, not used for any decision: the single-frame collapse rate (window 1) and the window-64 version, the dream's and the real
  game's flat shares, and the collapse rate per episode.

**M3, spin-return.** Hold "turn right" for a full revolution (72 steps of 5 degrees) from 32 random poses after K standing-still frames;
report the PSNR between the start view and the dream's frame 72 (the real game returns exactly, 100 dB).

**Secondary, reported but not decisive:** one-step PSNR/SSIM on 256 fixed windows (all models predict the same 9th frame of a 9-frame
window), mean PSNR over the first 32 dreamed frames, PSNR curve at frames 1, 4, 8, 16, 32, 64, 128, 256, 512, the copy-last-frame
baseline and the unrelated-frame floor for each game.

**Not comparable across games.** PSNR is measured against the frames of the game the model was trained on. v3 frames carry more colour
variation than v2 frames, so the copy-last baseline, the unrelated-frame floor and the PSNR levels differ between the games; a 20 dB
horizon in v3 is not exactly the same quality as a 20 dB horizon in v2. M1 compares them with the same 20 dB threshold as the brief
requires, and every report states this next to the numbers (with each game's copy-last and unrelated-frame reference values).

## 5. Experiments

| run | game | K | steps | role |
|---|---|---|---|---|
| `w2_B` | v2 | 4 | 20,000 | baseline (exists: RUNS.md row 5), re-evaluated with the protocol above |
| `v2k8` (a) | v2 | 8 | 20,000 | longer context alone |
| `v3k4` (b) | v3 | 4 | 20,000 | signatures alone |
| `v3k8` (c) | v3 | 8 | 20,000 | both |
| `full2` | the selected variant | | 150,000 | the same recipe as `full1` (widths 32-64-96-128, 2 blocks per level, batch 128, lr 3e-4 cosine to 3e-5, bf16, EMA 0.9995, context noise U(0, 0.5) with p 0.9, seed 0) |
| `full1` | v2 | 4 | 150,000 | the shipped v2 model (RUNS.md row 9), re-evaluated with the protocol above |

The ablation runs use the recipe of `w2_B` (single run each, seed 0; differences between single runs of this size are partly noise and
are reported with bootstrap intervals over episodes).

**Which variant gets the full run.** Among (a), (b), (c): eligible = median M1 horizon at least that of `w2_B`. Take the eligible variant
with the lowest M2 collapse rate; if two collapse rates differ by less than 0.02 absolute, take the one with the longer median horizon;
if still tied, the smaller K, then v3 over v2. If none is eligible, take the lowest collapse rate overall. The full run goes ahead
whatever the ablation shows (the README already shows that 20k-step drift numbers say little about 150k-step ones).

## 6. Adoption rule

Compared on the same 64 episodes, `full2` against `full1` (both with `eval_prereg.py`):

* **A.** collapse rate of `full2` is at most half that of `full1` (point estimates, window 32), and
* **B.** the median M1 tracking horizon of `full2` is not below that of `full1` (point estimates).

The same rule applies whichever variant `full2` is (if it is variant (a), the game stays v2 and only the context length changes).
Adopt only if A and B hold. Then: re-export ONNX (fp16 and fp32, verified as before), update `site/` (context buffer, model meta), re-measure
fps, re-render the media (the video with long pure-dream stretches, the hero, the drift figure, the rollout grid), regenerate the README and
POST.md. Otherwise v2 stays shipped and v3 is documented in the README and in STATE.md as a negative result, with all numbers below.

Paired bootstrap intervals (over episodes, since both models see the same paths) are reported for the ratio of collapse rates and the
difference of median horizons. They do not change the rule: a pass whose interval includes "no change" is adopted and described as weak
evidence.

## 7. Reporting commitments and limits

* Every number defined above is reported for every run, whatever the outcome, with the run's RUNS.md row and the commit.
* One training run per variant: a single seed. The 150k-step comparison is one run against one run. A job that dies for infrastructure
  reasons (time limit, node failure) is resumed or re-run with the same seed and code; that is not a new variant.
* No tuning on the evaluation episodes, no further v3 variants after seeing results in this iteration, no changes to the constants above.
* The v2 reference in this document is from 32 episodes on the Apple M4 and only fixed the constants; the numbers that enter the adoption
  rule are the 64-episode A100 evaluations of `full1` and `full2`.
* What would count as support for the hypothesis: v3 variants (b, c) cut the collapse rate relative to `w2_B` and (a) does not.
  What would count against it: no variant cuts it (then observability in the sense tested here is not the cause, or the signatures
  are not learned within this budget; the report will say which of these the evidence cannot separate).

## 8. Deviation log

(empty at the time of writing)
