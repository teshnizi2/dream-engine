# POST.md (LinkedIn draft, never posted)

Attach `media/dream-1080x1350.mp4` (or the 1080x1080 cut). Second image: `media/drift.png`.

## Draft (119 words)

I trained a 6.8M-parameter network to be a game.

Left: the real engine, plain JavaScript. Right: a diffusion model imagining every frame from the same keypresses. No game code runs on the right. 28 fps, in a browser tab.

It tracks the real game for about 16 frames (half a second), then dreams a different place that still looks like the game. Spin a full circle and it does not come back to the start view: no map memory.

The surprise: the bottleneck was my game, not the network. Window stripes were 5% of the pixels and 58% of the error. Fixing the textures gained 2.8 dB. Four architectures landed within 0.4 dB.

DIAMOND's recipe, 150k steps, 3.3 A100-hours.

## Three alternative hooks

1. No game code runs on the right side of this video.
2. I tried four architectures and they all scored within 0.4 dB. The problem was the window textures in my game.
3. This network never saw the game's code or its map, only pixels and keypresses. Now it plays the game, for about half a second at a time.

## Where each number comes from

| claim | file |
|---|---|
| 6.77M parameters, 150k steps | `results/eval_full1_n64.json` (`params`, `step`) |
| 3.3 A100-hours for the main run | `results/training_summary.json` (`main_gpu_hours`) |
| 28 fps with 2 denoising steps (median window, M4 shared with other jobs) | `results/browser_fps.json` |
| tracks about 16 frames (median of 64 episodes, PSNR above 20 dB; half the episodes diverge between frames 8 and 30) | `results/eval_full1_n64.json` (`rollout_primary.per_episode_frames_until_below_20db`) |
| a full 360-degree spin returns at 18.0 dB (the real game: 100 dB) | `results/eval_full1_n64.json` (`spin_360`) |
| window stripes: 5% of the pixels, 58% of the squared error | `results/error_attribution_abl_B_game_v1.json` |
| texture fix: +2.8 dB at the same recipe (27.9 to 30.7 dB, one step ahead) | `results/ablation/eval_abl_B.json`, `eval_w2_B.json` |
| four architectures within 0.4 dB | `results/ablation/eval_abl_*.json` |
| "the video re-seeds the dream every 1.25 s" (caption in the video) | `tools/capture_video.mjs --wake-every 30`, 24 fps |

Say in the comments that the video re-seeds the dream from the real game every 30 game ticks (1.25 s of video); in the live demo it is pure imagination until you press Wake up.

## Follow-up material (v3 attempt; not part of the 120-word draft, never posted)

If someone asks about the "collapses onto flat walls" part: I tried to fix it with a pre-registered experiment. Every wall face got its own colour signature (and each of the four directions its own shade), so that even a close-up of a wall says where you are, and I also tried a longer context. It did not work: the collapse rate was 0.33 for the new game against 0.16 for the shipped one (the rule was "at most half"), although the network clearly learned to read the colours (shifting them cost it about 0.85 dB). A longer context alone changed nothing (0.67 against 0.67 at 20k steps). So v2 stays the demo and the video still re-seeds the dream every 1.25 s. Single training runs, one map, 64 episodes: a negative result, not a proof that it cannot work.

| claim | file |
|---|---|
| collapse rate (stuck on a flat wall for 32 frames, frames 64 to 512, 64 episodes): 0.33 for the signature game, 0.16 for the shipped one; the rule was "at most half" | `results/prereg_final_full2_vs_full1.json` |
| longer context alone: 0.670 against 0.667 at 20k steps | `results/prereg_ablation.json` |
| the network reads the signature: 0.85 dB lost in the escape test when it is shifted | `results/signature_use_full2.json` |
| the dream leaves a flat wall in 3.3% of frames (shipped model), the real game in 6.3% | `results/flat_spells.json` |
