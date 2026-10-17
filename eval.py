#!/usr/bin/env python
"""Evaluate a trained checkpoint against the true game.

  * 1-step prediction PSNR/SSIM on fixed validation windows, with the copy-last-frame baseline;
  * sampler ablations (Euler 1/2/3 steps, Heun 2 steps) and context-noise level / init ablations;
  * per-action-type 1-step PSNR (is strafing as good as walking?);
  * autoregressive rollouts of 32 / 128 / 512 frames in pure imagination (drift curves), against two references:
    freezing the last real frame, and the "uncorrelated" floor (the true frame of another episode).

Writes results/eval_<name>.json and the figures media/drift.png, media/rollout_grid.png (+ media/rollout.gif).

    python eval.py --ckpt runs/full/final_ema.pt --name full
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

from game import raycaster as rc  # noqa: E402
from model import evaluate as ev  # noqa: E402
from model import metrics  # noqa: E402
from export import load_ema  # noqa: E402

TOKENS = {"bg": "#0e1117", "panel": "#151a23", "line": "#2e3a52", "text": "#e8ecf5", "muted": "#8893ab", "amber": "#f59e0b"}


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def summarize(curve, steps=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
    return {str(s): float(curve[s - 1]) for s in steps if s <= len(curve)}


def grad_mag(x):
    """Mean absolute horizontal+vertical gradient of frames in [-1,1] (a blur / detail measure)."""
    gx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    gy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return float((gx + gy) / 2)


def first_below(curve, thr):
    idx = np.nonzero(np.asarray(curve) < thr)[0]
    return int(idx[0]) + 1 if len(idx) else len(curve) + 1


def action_class(a):
    """Name of the action type for per-action analysis; a = (move, strafe, turn)."""
    mv, st, tu = int(a[0]), int(a[1]), int(a[2])
    if st:
        return "strafe"
    if mv == 2:
        return "back"
    if mv == 1 and tu:
        return "forward+turn"
    if mv == 1:
        return "forward"
    if tu:
        return "turn"
    return "noop"


HOLD_ACTIONS = {"forward": (1, 0, 0), "back": (2, 0, 0), "turn_left": (0, 0, 1), "turn_right": (0, 0, 2),
                "strafe_left": (0, 1, 0), "strafe_right": (0, 2, 0), "forward+left": (1, 0, 1), "forward+right": (1, 0, 2)}


def hold_action_rollouts(world, denoise, n=24, horizon=24, steps=1, ctx_sigma=0.05, init="edm", seed=5, device="cpu"):
    """Hold one key for `horizon` frames from n random poses (4 standing-still frames as context): how well does the
    dream follow each control? Returns PSNR at a few frame indices per action, and the copy-the-last-frame baseline."""
    from data.policy import ExplorationPolicy
    rng = np.random.default_rng(seed)
    st0 = ExplorationPolicy(world, n, rng, respawn_prob=0.0).random_states(n)
    res = {}
    for name, act in HOLD_ACTIONS.items():
        st = st0
        frames = [world.render(st)]
        acts = []
        for _ in range(3):
            acts.append(np.zeros((n, 3), np.int64))
            frames.append(world.render(st))                       # noop: the same frame
        for _ in range(horizon):
            a = np.tile(np.array(act, np.int64), (n, 1))
            acts.append(a)
            st = world.step(st, a)
            frames.append(world.render(st))
        fr = ev._to_unit(np.stack(frames, 1))                      # [n, T, 3, 64, 64]
        traj = {"frames": fr, "actions": torch.from_numpy(np.stack(acts, 1))}
        r = ev.rollout_metrics(denoise, traj, horizon=horizon, steps=steps, order=1, ctx_sigma=ctx_sigma, init=init, batch=n)
        fz, _ = frozen_baseline(traj, horizon)
        idx = [i for i in (1, 4, 8, 16, 24) if i <= horizon]
        res[name] = {"dream_psnr_at": {str(i): float(r["psnr"][i - 1]) for i in idx},
                     "freeze_psnr_at": {str(i): float(fz[i - 1]) for i in idx}}
    return res


def spin_test(world, denoise, n=16, steps=1, ctx_sigma=0.05, init="edm", seed=11, device="cpu"):
    """Hold 'turn right' for a full revolution (72 steps of 5 degrees) from random poses: the real game ends on exactly the
    start view. Reports how close the dream gets after 8 / 24 / 36 / 72 frames and the PSNR of the dream's frame 72 against
    the start view (a world with memory would return to it)."""
    from data.policy import ExplorationPolicy
    rng = np.random.default_rng(seed)
    st = ExplorationPolicy(world, n, rng, respawn_prob=0.0).random_states(n)
    frames = [world.render(st)] * 4
    acts = [np.zeros((n, 3), np.int64)] * 3
    for _ in range(72):
        a = np.tile(np.array((0, 0, 2), np.int64), (n, 1))
        acts.append(a)
        st = world.step(st, a)
        frames.append(world.render(st))
    fr = ev._to_unit(np.stack(frames, 1))
    traj = {"frames": fr, "actions": torch.from_numpy(np.stack(acts, 1))}
    r = ev.rollout_metrics(denoise, traj, horizon=72, steps=steps, order=1, ctx_sigma=ctx_sigma, init=init, batch=n, return_frames=True)
    start = fr[:, 3]
    back = metrics.psnr(((r["frames"][:, 71] + 1) / 2), ((start + 1) / 2)).clamp(max=100)
    real_back = metrics.psnr(((fr[:, 4 + 71] + 1) / 2), ((start + 1) / 2)).clamp(max=100)
    return {"episodes": n, "dream_psnr_at": {str(k): float(r["psnr"][k - 1]) for k in (1, 8, 24, 36, 72)},
            "dream_frame72_vs_start_view_psnr": float(back.mean()), "real_frame72_vs_start_view_psnr": float(real_back.mean())}


def frozen_baseline(traj, horizon, K=4):
    fr = traj["frames"]
    last = fr[:, K - 1:K].expand(-1, horizon, -1, -1, -1)
    gt = fr[:, K:K + horizon]
    p = metrics.psnr(((last + 1) / 2).flatten(0, 1), ((gt + 1) / 2).flatten(0, 1)).view(len(fr), horizon).clamp(max=100)
    s = metrics.ssim(((last + 1) / 2).flatten(0, 1), ((gt + 1) / 2).flatten(0, 1)).view(len(fr), horizon)
    return p.mean(0).numpy(), s.mean(0).numpy()


def uncorrelated_floor(traj, horizon, K=4):
    fr = traj["frames"][:, K:K + horizon]
    other = fr.roll(1, dims=0)
    p = metrics.psnr(((other + 1) / 2).flatten(0, 1), ((fr + 1) / 2).flatten(0, 1)).view(len(fr), horizon).clamp(max=100)
    return p.mean(0).numpy()


def save_drift_figure(path, curves, floor, frozen, nn_dream=None, nn_real=None, horizon_marks=(32, 128, 512)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = TOKENS
    fig, axes = plt.subplots(2 if nn_dream is not None else 1, 1, figsize=(7.2, 6.4 if nn_dream is not None else 4.2), dpi=200,
                             gridspec_kw={"height_ratios": [3, 2]} if nn_dream is not None else None, sharex=True)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["panel"])
    x = np.arange(1, len(floor) + 1)
    ax.plot(x, floor, color=t["muted"], lw=1.0, ls=":", label="unrelated frame (floor)")
    ax.plot(x, frozen, color=t["muted"], lw=1.2, ls="--", label="freeze last real frame")
    palette = [t["amber"], "#60a5fa", "#34d399", "#f472b6"]
    for (name, c), col in zip(curves.items(), palette):
        ax.plot(np.arange(1, len(c) + 1), c, color=col, lw=2.0 if col == t["amber"] else 1.3, label=name)
    for h in horizon_marks:
        if h <= len(floor):
            ax.axvline(h, color=t["line"], lw=0.8)
            ax.text(h, ax.get_ylim()[1], f" {h}", color=t["muted"], fontsize=7, va="top")
    ax.set_xscale("log")
    ax.set_ylabel("PSNR vs the real game (dB)", color=t["text"], fontsize=8)
    ax.legend(fontsize=7, facecolor=t["panel"], edgecolor=t["line"], labelcolor=t["text"], loc="upper right")
    ax.set_title("Dream drift: same actions, no game code", color=t["text"], fontsize=10, loc="left")
    if nn_dream is not None:
        ax2 = axes[1]
        ax2.set_facecolor(t["panel"])
        xx = np.arange(1, len(nn_dream) + 1)
        ax2.plot(xx, nn_real, color=t["muted"], lw=1.2, ls="--", label="real frames (reference)")
        ax2.plot(xx, nn_dream, color=t["amber"], lw=1.8, label="dream frames")
        ax2.set_ylabel("PSNR to the nearest\nreal frame (dB)", color=t["text"], fontsize=8)
        ax2.legend(fontsize=7, facecolor=t["panel"], edgecolor=t["line"], labelcolor=t["text"], loc="lower left")
        ax2.set_title("Is it still a game frame? (valid even where it no longer matches the true frame)", color=t["text"], fontsize=8, loc="left")
    axes[-1].set_xlabel("frames since wake-up (autoregressive, no real frames)", color=t["text"], fontsize=8)
    for a_ in axes:
        a_.tick_params(colors=t["muted"], labelsize=7)
        for sp in a_.spines.values():
            sp.set_color(t["line"])
        a_.grid(color=t["line"], lw=0.4, alpha=0.6)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_rollout_grid(path, gt, dream, steps=(0, 7, 15, 31, 63, 127, 255, 511), episodes=4, scale=3):
    from PIL import Image
    steps = [s for s in steps if s < dream.shape[1]]
    rows = []
    for e in range(min(episodes, len(dream))):
        for src in (gt, dream):
            row = torch.cat([src[e, s] for s in steps], dim=2)        # [3, 64, 64*len]
            rows.append(row)
        rows.append(torch.full_like(rows[-1][:, :4], -0.8))             # spacer
    img = torch.cat(rows, dim=1)
    arr = ((img + 1) / 2 * 255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr).resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST).save(path)


def save_rollout_gif(path, gt, dream, episodes=3, frames=128, scale=3, every=2):
    import imageio.v2 as imageio
    n = min(frames, dream.shape[1])
    ims = []
    for t in range(0, n, every):
        cols = []
        for e in range(min(episodes, len(dream))):
            cols.append(torch.cat([gt[e, t], torch.full((3, 64, 3), -0.8), dream[e, t]], dim=2))
        img = torch.cat(cols, dim=1)
        arr = ((img + 1) / 2 * 255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
        arr = np.kron(arr, np.ones((scale, scale, 1), dtype=np.uint8))
        ims.append(arr)
    imageio.mimsave(path, ims, duration=0.07 * every, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--long", type=int, default=512, help="longest rollout (frames)")
    ap.add_argument("--episodes", type=int, default=8, help="episodes for the longest rollout")
    ap.add_argument("--bank", type=int, default=12000, help="size of the real-frame bank for the validity metric")
    ap.add_argument("--steps", type=int, default=1, help="Euler steps of the primary rollout configuration (the demo default)")
    ap.add_argument("--init", default="edm", choices=["edm", "unit"], help="sampler start of the primary configuration")
    ap.add_argument("--ctx-sigma", type=float, default=-1.0, help="context noise of the primary configuration (negative: probe 0/0.05/0.1/0.2)")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smaller sweeps (smoke checks)")
    a = ap.parse_args()
    dev = pick_device(a.device)
    model, step = load_ema(a.ckpt)
    model = model.to(dev).eval()
    world = rc.World()
    n_params = sum(p.numel() for p in model.parameters())
    out = {"script": "python eval.py " + " ".join(sys.argv[1:]), "checkpoint": a.ckpt, "step": step, "params": n_params,
           "device": str(dev), "n_val_windows": a.n_val}
    t_start = time.time()

    # ---------------- 1-step
    val = ev.val_windows(world, n=a.n_val, seed=999)
    copy = lambda x, s, ctx, act, sc: ctx[:, -1]
    base = ev.one_step_metrics(copy, val, steps=1, ctx_sigma=0.0)
    out["baseline_copy_last_frame"] = {"psnr": base["psnr"], "ssim": base["ssim"]}
    sweep = {}
    for init in ("edm", "unit"):
        for cs in ((0.0, 0.1) if a.quick else (0.0, 0.05, 0.1, 0.2)):
            m = ev.one_step_metrics(model.denoise, val, steps=3, ctx_sigma=cs, init=init)
            sweep[f"euler3_init={init}_ctx={cs}"] = {"psnr": m["psnr"], "ssim": m["ssim"]}
    out["one_step_ctx_init_sweep"] = sweep
    samplers = {}
    sampler_cs = a.ctx_sigma if a.ctx_sigma >= 0 else 0.1
    out["one_step_samplers_config"] = {"ctx_sigma": sampler_cs, "init": a.init}
    for name, steps, order in (("euler1", 1, 1), ("euler2", 2, 1), ("euler3", 3, 1), ("heun2", 2, 2), ("heun3", 3, 2)):
        torch.manual_seed(0)
        t = time.time()
        m = ev.one_step_metrics(model.denoise, val, steps=steps, order=order, ctx_sigma=sampler_cs, init=a.init)
        if dev.type == "mps":
            torch.mps.synchronize()
        samplers[name] = {"psnr": m["psnr"], "ssim": m["ssim"], "nfe": steps if order == 1 else 2 * steps - 1,
                          "eval_seconds": time.time() - t}
    out["one_step_samplers_ctx0.1"] = samplers

    # per last-action-type PSNR
    m = ev.one_step_metrics(model.denoise, val, steps=a.steps, ctx_sigma=sampler_cs, init=a.init)
    mb = ev.one_step_metrics(copy, val, steps=1, ctx_sigma=0.0)
    cls = [action_class(x) for x in val["actions"][:, 3].numpy()]
    per = {}
    for c in sorted(set(cls)):
        idx = np.array([i for i, k in enumerate(cls) if k == c])
        per[c] = {"n": int(len(idx)), "psnr": float(m["psnr_per_window"][idx].mean()),
                  "copy_last_psnr": float(mb["psnr_per_window"][idx].mean())}
    out["one_step_by_last_action"] = per
    print("1-step done", json.dumps({k: v for k, v in out.items() if k.startswith("baseline")}), flush=True)

    # ---------------- rollouts
    L = a.long
    traj = ev.eval_trajectories_t(world, n=a.episodes * 2, length=4 + L, seed=123)
    # choose the context-noise level that gives the best 128-frame drift (the 1-step score alone prefers 0)
    probe = {k: v[:8] for k, v in traj.items()}
    cs_scores = {}
    if a.ctx_sigma >= 0:
        best_cs = a.ctx_sigma
    else:
        for cs in ((0.0, 0.1) if a.quick else (0.0, 0.05, 0.1, 0.2)):
            r = ev.rollout_metrics(model.denoise, probe, horizon=min(128, L), steps=a.steps, ctx_sigma=cs, init=a.init)
            cs_scores[str(cs)] = float(r["psnr"].mean())
        best_cs = float(max(cs_scores, key=cs_scores.get))
    out["ctx_sigma_probe_mean_psnr_128"] = cs_scores
    out["chosen_ctx_sigma"] = best_cs
    out["primary_config"] = {"steps": a.steps, "order": 1, "init": a.init, "ctx_sigma": best_cs}
    print("ctx_sigma probe", cs_scores, "-> primary config", out["primary_config"], flush=True)

    curves = {}
    long_traj = {k: v[:a.episodes] for k, v in traj.items()}
    r = ev.rollout_metrics(model.denoise, long_traj, horizon=L, steps=a.steps, ctx_sigma=best_cs, init=a.init, return_frames=True, batch=8)
    curves[f"{n_params / 1e6:.1f}M EDM, {a.steps} Euler step(s), ctx noise {best_cs}"] = r["psnr"]
    out["rollout_primary"] = {"config": out["primary_config"],"psnr_curve": r["psnr"].tolist(), "ssim_curve": r["ssim"].tolist(),
                             "psnr_at": summarize(r["psnr"]), "ssim_at": summarize(r["ssim"]),
                             "mean_psnr_first32": float(r["psnr"][:32].mean()),
                             "mean_psnr_first128": float(r["psnr"][:128].mean()),
                             "mean_psnr_all": float(r["psnr"].mean()), "episodes": a.episodes, "horizon": L}
    out["rollout_primary"]["frames_until_psnr_below_25db"] = first_below(r["psnr"], 25.0)
    out["rollout_primary"]["frames_until_psnr_below_20db"] = first_below(r["psnr"], 20.0)
    out["rollout_primary"]["sharpness_ratio_last32"] = grad_mag(r["frames"][:, -32:]) / grad_mag(long_traj["frames"][:, 4 + L - 32:4 + L])
    # does the dream stay a valid game frame even when it no longer matches the true frame?
    bank = ev.make_frame_bank(world, n=a.bank if not a.quick else 4000)
    gt_long = long_traj["frames"][:, 4:4 + L]
    nn_dream = ev.nearest_real_psnr(r["frames"].flatten(0, 1).to(dev), bank).view(a.episodes, L).mean(0).numpy()
    nn_real = ev.nearest_real_psnr(gt_long.flatten(0, 1).to(dev), bank).view(a.episodes, L).mean(0).numpy()
    out["validity_nearest_real_frame_psnr"] = {
        "note": "PSNR of each frame to its nearest neighbour among %d real frames; real = the true game frames of the same episodes" % len(bank["frames"]),
        "dream_at": summarize(nn_dream), "real_at": summarize(nn_real),
        "dream_mean": float(nn_dream.mean()), "real_mean": float(nn_real.mean())}
    print("validity (nearest real frame PSNR): dream %.1f vs real %.1f" % (nn_dream.mean(), nn_real.mean()), flush=True)
    for h, n_ep in ((32, 32), (128, 16)):
        if h <= L:
            rr = ev.rollout_metrics(model.denoise, {k: v[:n_ep] for k, v in traj.items()}, horizon=h, steps=a.steps,
                                    ctx_sigma=best_cs, init=a.init)
            out[f"rollout_{h}"] = {"episodes": n_ep, "mean_psnr": float(rr["psnr"].mean()), "final_psnr": float(rr["psnr"][-1]),
                                   "mean_ssim": float(rr["ssim"].mean()), "final_ssim": float(rr["ssim"][-1])}
    for name, steps, order in (("euler1", 1, 1), ("euler2", 2, 1), ("euler3", 3, 1), ("heun2", 2, 2)):
        rr = ev.rollout_metrics(model.denoise, {k: v[:8] for k, v in traj.items()}, horizon=min(128, L), steps=steps, order=order,
                                ctx_sigma=best_cs, init=a.init)
        out[f"rollout128_{name}"] = {"mean_psnr": float(rr["psnr"].mean()), "mean_psnr_first32": float(rr["psnr"][:32].mean()),
                                     "final_psnr": float(rr["psnr"][-1])}
    fz_p, fz_s = frozen_baseline(long_traj, L)
    floor = uncorrelated_floor(long_traj, L)
    out["baseline_freeze_last_real_frame"] = {"psnr_at": summarize(fz_p), "ssim_at": summarize(fz_s)}
    out["baseline_unrelated_frame_psnr"] = float(floor.mean())
    out["hold_one_key"] = hold_action_rollouts(world, model.denoise, steps=a.steps, ctx_sigma=best_cs, init=a.init, device=str(dev))
    out["spin_360"] = spin_test(world, model.denoise, steps=a.steps, ctx_sigma=best_cs, init=a.init)
    print("360-degree spin:", out["spin_360"], flush=True)
    print("hold-one-key PSNR@8:", {k: round(v["dream_psnr_at"]["8"], 1) for k, v in out["hold_one_key"].items()}, flush=True)
    out["eval_wall_seconds"] = time.time() - t_start

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / f"eval_{a.name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("baseline_copy_last_frame", "one_step_samplers_ctx0.1", "rollout_primary")
                      if k in out}, indent=1)[:3000])
    if not a.no_figs:
        media = ROOT / "media"
        media.mkdir(exist_ok=True)
        save_drift_figure(media / "drift.png", {k: v for k, v in list(curves.items())[:1]}, floor, fz_p, nn_dream, nn_real)
        gen = r["frames"]
        gt = long_traj["frames"][:, 4:4 + L]
        save_rollout_grid(media / "rollout_grid.png", gt, gen)
        save_rollout_gif(media / "rollout.gif", gt, gen)
        print("figures written to media/")


if __name__ == "__main__":
    main()
