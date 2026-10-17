#!/usr/bin/env python
"""The pre-registered evaluation of docs/PREREG-v3.md.

Every model, whatever its context length K and game version, is rolled out in pure imagination from the same real
frame (the 8th, so that a K = 8 model has 8 real frames behind it and a K = 4 model the last 4 of them) on the same
episodes, and scored against the same real frames:

  * per-episode tracking horizon (first frame whose 5-frame average PSNR is below 20 dB), median over episodes;
  * collapse rate (model/flatwall.py): the fraction of dream frames at t in [64, 512] that are stuck on a flat wall
    while the matching real frame is not;
  * spin-return: PSNR between the start view and the dream's view after a full 360 degree turn;
  * (secondary) one-step PSNR on common target frames, mean PSNR over the first 32 frames.

Per-frame arrays (PSNR, background fraction of dream and real frames) are saved next to the json, so the CPU-side numbers can be
recomputed with `analyze` without a GPU.

    python eval_prereg.py --ckpt runs/full1/final_ema.pt --name full1 --episodes 64 --device cuda
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import eval as E  # noqa: E402  (helpers: tracking horizons, spin test, uint8 conversion)
from export import ckpt_game, load_ema  # noqa: E402
from game import raycaster as rc  # noqa: E402
from model import evaluate as ev  # noqa: E402
from model import flatwall as fw  # noqa: E402

WAKE = 8                  # real frame at which every model wakes up (a K = 8 model needs 8 real frames behind it)
HORIZON = 512             # dreamed frames per episode
N_TRAJ = 64               # trajectories generated (seed 123); an evaluation uses the first `--episodes` of them
TRAJ_SEED = 123
SAMPLER = {"steps": 2, "init": "unit", "ctx_sigma": 0.05}     # the demo's sampler, the same for every model
PSNR_AT = (1, 4, 8, 16, 32, 64, 128, 256, 512)


def _horizon(psnr_all, thr):
    per = E.tracking_horizons(psnr_all, thr)
    s = E.horizon_summary(per, psnr_all.shape[1])
    lo, hi = fw.bootstrap_ci(per, stat=np.median)
    return {"definition": f"first frame whose 5-frame window averages below {thr:g} dB, per episode; never = horizon + 1",
            **s, "median_ci95": [lo, hi], "per_episode": [int(v) for v in per]}


def _collapse(dream_flat, real_flat, window, max_bg):
    c = fw.collapse_rate(dream_flat, real_flat, window)
    lo, hi = fw.bootstrap_ci(c["per_episode"])
    return {"rate": c["rate"], "rate_ci95": [lo, hi], "window": window, "t_range": c["t_range"],
            "dream_stuck_share": c["dream_stuck_share"], "real_stuck_share": c["real_stuck_share"],
            "flat_max_bg": max_bg, "bg_tol": fw.BG_TOL, "per_episode": [float(v) for v in c["per_episode"]]}


def analyze(psnr_all, bg_dream, bg_real, window=fw.STUCK_WINDOW, max_bg=fw.FLAT_MAX_BG) -> dict:
    """Everything that can be computed from the per-frame arrays [episodes, frames]: PSNR, and the background fraction of the
    dreamed and of the matching real frames."""
    psnr_all = np.asarray(psnr_all, dtype=float)
    fd, fr = fw.flat_flags(bg_dream, max_bg), fw.flat_flags(bg_real, max_bg)
    lo, hi = fw.T_RANGE
    mean_curve = psnr_all.mean(0)
    pe32 = psnr_all[:, :32].mean(1)
    return {
        "tracking_horizon_20db": _horizon(psnr_all, 20.0),
        "tracking_horizon_25db": _horizon(psnr_all, 25.0),
        "mean_psnr_first32": float(pe32.mean()),
        "mean_psnr_first32_ci95": list(fw.bootstrap_ci(pe32)),
        "psnr_at": E.summarize(mean_curve, PSNR_AT),
        "collapse": _collapse(fd, fr, window, max_bg),
        "collapse_plain": _collapse(fd, fr, 1, max_bg),
        "dream_flat_share": float(fd[:, lo - 1:hi].mean()) if fd.shape[1] >= lo else float("nan"),
        "real_flat_share": float(fr[:, lo - 1:hi].mean()) if fr.shape[1] >= lo else float("nan"),
    }


def flat_escape(world, denoise, K, n=64, horizon=24, seed=21, steps=2, ctx_sigma=0.05, init="unit"):
    """Exploratory diagnostic (docs/PREREG-v3.md, M4; not part of any decision): the mechanism behind the collapse in isolation. Start
    facing a wall closely (a flat-wall frame, no floor or ceiling), K identical frames of context, then hold 'turn right' for `horizon`
    frames (5 degrees each). The real game turns away from the wall and shows what is around the corner; a dream that cannot tell where
    it is has nothing to go on, a dream that can localise from the wall's colour does. Reports PSNR against the real game at a few
    frames (and the freeze baseline) and how often the dream is still on a flat wall at the end, next to the real game."""
    from data.policy import ExplorationPolicy
    rng = np.random.default_rng(seed)
    pool = ExplorationPolicy(world, 8 * n, rng, respawn_prob=0.0).random_states(8 * n)
    flat = fw.flat_flags(fw.background_fraction(world.render(pool), world.bg_rows))
    keep = np.nonzero(flat)[0][:n]
    st = rc.State(*[v[keep] for v in pool])
    m = len(keep)
    frames = [world.render(st)] * K
    acts = [np.zeros((m, 3), np.int64)] * (K - 1)
    for _ in range(horizon):
        a = np.tile(np.array((0, 0, 2), np.int64), (m, 1))
        acts.append(a)
        st = world.step(st, a)
        frames.append(world.render(st))
    traj = {"frames": ev._to_unit(np.stack(frames, 1)), "actions": torch.from_numpy(np.stack(acts, 1))}
    r = ev.rollout_metrics(denoise, traj, horizon=horizon, steps=steps, order=1, ctx_sigma=ctx_sigma, init=init, batch=m, return_frames=True, K=K)
    fz, _ = E.frozen_baseline(traj, horizon, K=K)
    dream_flat = fw.flat_flags(fw.background_fraction(E.to_u8(r["frames"]), world.bg_rows))
    real_flat = fw.flat_flags(fw.background_fraction(E.to_u8(traj["frames"][:, K:K + horizon]), world.bg_rows))
    at = [t for t in (4, 8, 16, 24) if t <= horizon]
    return {"definition": f"start on a flat-wall frame, {K} identical context frames, hold turn-right for {horizon} frames; exploratory, not pre-registered as decisive",
            "episodes": int(m), "start_frames_flat_share": float(fw.flat_flags(fw.background_fraction(E.to_u8(traj["frames"][:, :1]), world.bg_rows)).mean()),
            "dream_psnr_at": {str(t): float(r["psnr"][t - 1]) for t in at}, "freeze_psnr_at": {str(t): float(fz[t - 1]) for t in at},
            "dream_psnr_mean": float(r["psnr"].mean()), "freeze_psnr_mean": float(fz.mean()),
            "dream_flat_share_at_end": float(dream_flat[:, -1].mean()), "real_flat_share_at_end": float(real_flat[:, -1].mean()),
            "dream_flat_share_last8": float(dream_flat[:, -8:].mean()), "real_flat_share_last8": float(real_flat[:, -8:].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--game", default=None, help="game spec json (default: the game the checkpoint was trained on)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--episodes", type=int, default=64, help="episodes scored (the first ones of the %d trajectories)" % N_TRAJ)
    ap.add_argument("--n-traj", type=int, default=N_TRAJ, help="trajectories generated; keep the default so that every model sees the same episodes")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--chunk", type=int, default=16, help="episodes rolled out at once")
    ap.add_argument("--spin-n", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--keep", type=int, default=8, help="episodes whose dreamed and real frames are saved (uint8) for figures")
    ap.add_argument("--window", type=int, default=fw.STUCK_WINDOW)
    ap.add_argument("--out-dir", default="results", help="where the json and npz go (relative to the repository root)")
    ap.add_argument("--only-flat-escape", action="store_true", help="add the exploratory flat-escape diagnostic to an existing prereg json and stop")
    a = ap.parse_args()
    dev = E.pick_device(a.device)
    model, step = load_ema(a.ckpt)
    model = model.to(dev).eval()
    K = model.cfg.num_ctx
    assert K <= WAKE, "the wake-up frame must leave room for the context"
    game = a.game or ckpt_game(a.ckpt)
    world = rc.World(rc.load_spec(game))
    if a.only_flat_escape:
        path = (pathlib.Path(a.out_dir) if pathlib.Path(a.out_dir).is_absolute() else ROOT / a.out_dir) / f"prereg_{a.name}.json"
        prev = json.loads(path.read_text())
        assert prev["num_ctx"] == K and prev["game"] == game, "the json belongs to another model or game"
        prev["flat_escape"] = flat_escape(world, model.denoise, K, steps=SAMPLER["steps"], ctx_sigma=SAMPLER["ctx_sigma"], init=SAMPLER["init"])
        path.write_text(json.dumps(prev, indent=1))
        print(json.dumps(prev["flat_escape"]), flush=True)
        return
    t0 = time.time()
    out = {"script": "python eval_prereg.py " + " ".join(sys.argv[1:]), "checkpoint": a.ckpt, "step": step,
           "params": sum(p.numel() for p in model.parameters()), "num_ctx": K, "game": game, "device": str(dev),
           "protocol": {"wake_frame": WAKE, "episodes": a.episodes, "of_trajectories": a.n_traj, "trajectory_seed": TRAJ_SEED,
                        "horizon": a.horizon, "sampler": SAMPLER, "flat_wall": {"bg_tol": fw.BG_TOL, "max_bg": fw.FLAT_MAX_BG,
                                                                                 "window": a.window, "t_range": list(fw.T_RANGE)}}}

    traj = ev.eval_trajectories_t(world, n=a.n_traj, length=WAKE + a.horizon, seed=TRAJ_SEED)
    sub = ev.wake_slice({k: v[:a.episodes] for k, v in traj.items()}, WAKE, K)
    psnr, bg_d, bg_r, keep_d, keep_g = [], [], [], [], []
    for i in range(0, a.episodes, a.chunk):
        chunk = {k: v[i:i + a.chunk] for k, v in sub.items()}
        r = ev.rollout_metrics(model.denoise, chunk, horizon=a.horizon, steps=SAMPLER["steps"], order=1, ctx_sigma=SAMPLER["ctx_sigma"],
                               init=SAMPLER["init"], batch=a.chunk, return_frames=True, K=K, seed=1000 * i)
        dream_u8 = E.to_u8(r["frames"])
        real_u8 = E.to_u8(chunk["frames"][:, K:K + a.horizon])
        psnr.append(r["psnr_all"])
        bg_d.append(fw.background_fraction(dream_u8, world.bg_rows))
        bg_r.append(fw.background_fraction(real_u8, world.bg_rows))
        if i < a.keep:
            keep_d.append(dream_u8[:a.keep - i])
            keep_g.append(real_u8[:a.keep - i])
        print(f"rolled out episodes {i}..{i + len(dream_u8) - 1} ({time.time() - t0:.0f} s)", flush=True)
    psnr, bg_d, bg_r = np.concatenate(psnr), np.concatenate(bg_d), np.concatenate(bg_r)
    out.update(analyze(psnr, bg_d, bg_r, window=a.window))
    # reference points that make PSNR levels readable (they differ between the games): freeze the last real frame / an unrelated real frame
    fz_p, _ = E.frozen_baseline(sub, a.horizon, K=K)
    out["baseline_freeze_last_real_frame"] = {"psnr_at": E.summarize(fz_p, PSNR_AT)}
    out["baseline_unrelated_frame_psnr"] = float(E.uncorrelated_floor(sub, a.horizon, K=K).mean())

    # spin-return: a full revolution in place; the real game closes the loop (100 dB), a dream with no map memory does not
    out["spin_360"] = E.spin_test(world, model.denoise, n=a.spin_n, steps=SAMPLER["steps"], ctx_sigma=SAMPLER["ctx_sigma"],
                                  init=SAMPLER["init"], K=K)
    # one step ahead on common target frames (9-frame windows: every K <= 8 predicts the same 9th frame)
    val = ev.val_windows(world, n=a.n_val, seed=999, K=max(K, WAKE))
    m = ev.one_step_metrics(model.denoise, val, steps=SAMPLER["steps"], ctx_sigma=SAMPLER["ctx_sigma"], init=SAMPLER["init"], K=K)
    base = ev.one_step_metrics(lambda x, s, ctx, act, sc: ctx[:, -1], val, steps=1, ctx_sigma=0.0, K=K)
    out["one_step"] = {"psnr": m["psnr"], "ssim": m["ssim"], "copy_last_psnr": base["psnr"], "n_windows": a.n_val}
    try:                                                   # exploratory diagnostic: must never cost the pre-registered numbers
        out["flat_escape"] = flat_escape(world, model.denoise, K, steps=SAMPLER["steps"], ctx_sigma=SAMPLER["ctx_sigma"], init=SAMPLER["init"])
    except Exception as e:                                 # noqa: BLE001
        out["flat_escape"] = {"error": repr(e)[:300]}
        print("flat_escape failed:", repr(e), flush=True)
    out["eval_wall_seconds"] = time.time() - t0

    res = pathlib.Path(a.out_dir)
    res = res if res.is_absolute() else ROOT / res
    res.mkdir(parents=True, exist_ok=True)
    (res / f"prereg_{a.name}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(res / f"prereg_{a.name}.npz", psnr=psnr.astype(np.float32), bg_dream=bg_d.astype(np.float32),
                        bg_real=bg_r.astype(np.float32))
    if keep_d:
        np.savez_compressed(res / f"rollouts_prereg_{a.name}.npz", dream=np.concatenate(keep_d), gt=np.concatenate(keep_g))
    summ = {"name": a.name, "K": K, "horizon_median": out["tracking_horizon_20db"]["median"],
            "collapse": out["collapse"]["rate"], "collapse_plain": out["collapse_plain"]["rate"],
            "spin": out["spin_360"]["dream_frame72_vs_start_view_psnr"], "one_step": out["one_step"]["psnr"]}
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
