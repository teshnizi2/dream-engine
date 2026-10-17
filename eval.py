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
from export import ckpt_game, load_ema  # noqa: E402


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


def tracking_horizons(psnr_all, thr, win=5):
    """Per episode: the first frame whose `win`-frame window averages below `thr` dB (episodes that never cross: horizon + 1)."""
    out = []
    for row in np.asarray(psnr_all, dtype=float):
        sm = np.convolve(row, np.ones(win) / win, mode="valid")          # sm[i] = mean of frames i+1 .. i+win
        idx = np.nonzero(sm < thr)[0]
        out.append(int(idx[0]) + 1 if len(idx) else len(row) + 1)
    return np.array(out)


def horizon_summary(per, horizon):
    return {"episodes": int(len(per)), "median": float(np.median(per)), "p25": float(np.percentile(per, 25)),
            "p75": float(np.percentile(per, 75)), "share_never_below": float(np.mean(np.asarray(per) > horizon))}


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


def hold_action_rollouts(world, denoise, n=24, horizon=24, steps=1, ctx_sigma=0.05, init="edm", seed=5, device="cpu", K=4):
    """Hold one key for `horizon` frames from n random poses (K standing-still frames as context): how well does the
    dream follow each control? Returns PSNR at a few frame indices per action, and the copy-the-last-frame baseline."""
    from data.policy import ExplorationPolicy
    rng = np.random.default_rng(seed)
    st0 = ExplorationPolicy(world, n, rng, respawn_prob=0.0).random_states(n)
    res = {}
    for name, act in HOLD_ACTIONS.items():
        st = st0
        frames = [world.render(st)]
        acts = []
        for _ in range(K - 1):
            acts.append(np.zeros((n, 3), np.int64))
            frames.append(world.render(st))                       # noop: the same frame
        for _ in range(horizon):
            a = np.tile(np.array(act, np.int64), (n, 1))
            acts.append(a)
            st = world.step(st, a)
            frames.append(world.render(st))
        fr = ev._to_unit(np.stack(frames, 1))                      # [n, T, 3, 64, 64]
        traj = {"frames": fr, "actions": torch.from_numpy(np.stack(acts, 1))}
        r = ev.rollout_metrics(denoise, traj, horizon=horizon, steps=steps, order=1, ctx_sigma=ctx_sigma, init=init, batch=n, K=K)
        fz, _ = frozen_baseline(traj, horizon, K=K)
        idx = [i for i in (1, 4, 8, 16, 24) if i <= horizon]
        res[name] = {"dream_psnr_at": {str(i): float(r["psnr"][i - 1]) for i in idx},
                     "freeze_psnr_at": {str(i): float(fz[i - 1]) for i in idx}}
    return res


def spin_test(world, denoise, n=16, steps=1, ctx_sigma=0.05, init="edm", seed=11, device="cpu", K=4):
    """Hold 'turn right' for a full revolution (72 steps of 5 degrees) from random poses (K standing-still frames as
    context): the real game ends on exactly the start view. Reports how close the dream gets after 8 / 24 / 36 / 72 frames
    and the PSNR of the dream's frame 72 against the start view (a world with memory would return to it)."""
    from data.policy import ExplorationPolicy
    rng = np.random.default_rng(seed)
    st = ExplorationPolicy(world, n, rng, respawn_prob=0.0).random_states(n)
    frames = [world.render(st)] * K
    acts = [np.zeros((n, 3), np.int64)] * (K - 1)
    for _ in range(72):
        a = np.tile(np.array((0, 0, 2), np.int64), (n, 1))
        acts.append(a)
        st = world.step(st, a)
        frames.append(world.render(st))
    fr = ev._to_unit(np.stack(frames, 1))
    traj = {"frames": fr, "actions": torch.from_numpy(np.stack(acts, 1))}
    r = ev.rollout_metrics(denoise, traj, horizon=72, steps=steps, order=1, ctx_sigma=ctx_sigma, init=init, batch=n, return_frames=True, K=K)
    start = fr[:, K - 1]
    back = metrics.psnr(((r["frames"][:, 71] + 1) / 2), ((start + 1) / 2)).clamp(max=100)
    real_back = metrics.psnr(((fr[:, K + 71] + 1) / 2), ((start + 1) / 2)).clamp(max=100)
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


def to_u8(x):
    """[E, T, 3, H, W] in [-1, 1] -> [E, T, H, W, 3] uint8"""
    return ((x + 1) / 2 * 255).round().clamp(0, 255).byte().permute(0, 1, 3, 4, 2).numpy()


def from_u8(a):
    return torch.from_numpy(a).permute(0, 1, 4, 2, 3).float() / 255 * 2 - 1


def rollout_figures(name):
    """media/rollout_grid.png and media/rollout.gif from results/rollouts_<name>.npz (written by a full eval)."""
    z = np.load(ROOT / "results" / f"rollouts_{name}.npz")
    gt, gen = from_u8(z["gt"]), from_u8(z["dream"])
    media = ROOT / "media"
    media.mkdir(exist_ok=True)
    save_rollout_grid(media / "rollout_grid.png", gt, gen)
    save_rollout_gif(media / "rollout.gif", gt, gen)


def _tile(x, size):
    """[3, 64, 64] in [-1, 1] -> PIL image, nearest-neighbour upscaled"""
    from PIL import Image
    a = ((x + 1) / 2 * 255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(a).resize((size, size), Image.NEAREST)


def save_rollout_grid(path, gt, dream, steps=(0, 7, 15, 31, 63, 127, 255, 511), episodes=4, scale=2):
    """Real (grey label) above dream (amber label), same actions, at the given frames since wake-up."""
    from PIL import Image, ImageDraw, ImageFont
    steps = [s for s in steps if s < dream.shape[1]]
    n_ep = min(episodes, len(dream))
    tile, gap, left, top, pad, ep_gap = 64 * scale, 6, 84, 58, 22, 20
    bg, line, muted, amber = (14, 17, 23), (46, 58, 82), (136, 147, 171), (245, 158, 11)
    W = left + len(steps) * (tile + gap) - gap + pad
    H = top + n_ep * (2 * tile + gap + ep_gap) - ep_gap + pad
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    f_small, f_lab = ImageFont.load_default(size=15), ImageFont.load_default(size=17)
    d.text((left, 20), "frames since wake-up (4 real frames, then pure imagination from the same keypresses)", font=f_small, fill=muted, anchor="ls")
    for j, st in enumerate(steps):
        d.text((left + j * (tile + gap) + tile // 2, top - 12), str(st + 1), font=f_lab, fill=(232, 236, 245), anchor="ms")
    for e in range(n_ep):
        y0 = top + e * (2 * tile + gap + ep_gap)
        for r, (src, lab, col) in enumerate(((gt, "REAL", muted), (dream, "DREAM", amber))):
            y = y0 + r * (tile + gap)
            d.text((left - 14, y + tile // 2), lab, font=f_lab, fill=col, anchor="rm")
            for j, st in enumerate(steps):
                img.paste(_tile(src[e, st], tile), (left + j * (tile + gap), y))
        if e < n_ep - 1:
            yl = y0 + 2 * tile + gap + ep_gap // 2
            d.line([(left, yl), (W - pad, yl)], fill=line, width=1)
    img.save(path)


def save_rollout_gif(path, gt, dream, episodes=3, frames=128, scale=3, every=2):
    """REAL | DREAM side by side for a few episodes, labelled, with the frame counter."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    n = min(frames, dream.shape[1])
    n_ep = min(episodes, len(dream))
    tile, gap, head, foot, colgap = 64 * scale, 4, 30, 30, 18
    bg, muted, amber = (14, 17, 23), (136, 147, 171), (245, 158, 11)
    W = n_ep * (2 * tile + gap) + (n_ep - 1) * colgap
    f = ImageFont.load_default(size=16)
    ims = []
    for t in range(0, n, every):
        img = Image.new("RGB", (W, head + tile + foot), bg)
        d = ImageDraw.Draw(img)
        for e in range(n_ep):
            x = e * (2 * tile + gap + colgap)
            d.text((x + 2, head - 8), "REAL", font=f, fill=muted, anchor="ls")
            d.text((x + tile + gap + 2, head - 8), "DREAM", font=f, fill=amber, anchor="ls")
            img.paste(_tile(gt[e, t], tile), (x, head))
            img.paste(_tile(dream[e, t], tile), (x + tile + gap, head))
        d.text((2, head + tile + foot - 8), f"frame {t + 1} since wake-up", font=f, fill=muted, anchor="ls")
        ims.append(np.asarray(img))
    imageio.mimsave(path, ims, duration=0.07 * every, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--name", default="model")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--long", type=int, default=512, help="longest rollout (frames)")
    ap.add_argument("--episodes", type=int, default=8, help="episodes for the longest rollout")
    ap.add_argument("--bank", type=int, default=12000, help="size of the real-frame bank for the validity metric")
    ap.add_argument("--steps", type=int, default=2, help="Euler steps of the primary rollout configuration (the demo default)")
    ap.add_argument("--init", default="unit", choices=["edm", "unit"], help="sampler start of the primary configuration (the demo default)")
    ap.add_argument("--ctx-sigma", type=float, default=0.05, help="context noise of the primary configuration (the demo default; negative: probe 0/0.05/0.1/0.2)")
    ap.add_argument("--game", default=None, help="game spec json (default: the game the checkpoint was trained on)")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smaller sweeps (smoke checks)")
    ap.add_argument("--figs-only", action="store_true", help="rebuild media/rollout_grid.png and rollout.gif from results/rollouts_<name>.npz")
    a = ap.parse_args()
    if a.figs_only:
        rollout_figures(a.name)
        return
    if not a.ckpt:
        ap.error("--ckpt is required")
    dev = pick_device(a.device)
    model, step = load_ema(a.ckpt)
    model = model.to(dev).eval()
    K = model.cfg.num_ctx                                   # context frames: 4 in v2; the dream wakes up after K real frames
    game = a.game or ckpt_game(a.ckpt)
    world = rc.World(rc.load_spec(game))
    n_params = sum(p.numel() for p in model.parameters())
    out = {"script": "python eval.py " + " ".join(sys.argv[1:]), "checkpoint": a.ckpt, "step": step, "params": n_params,
           "device": str(dev), "n_val_windows": a.n_val, "num_ctx": K, "game": game}
    t_start = time.time()

    # ---------------- 1-step
    val = ev.val_windows(world, n=a.n_val, seed=999, K=K)
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
    cls = [action_class(x) for x in val["actions"][:, K - 1].numpy()]
    per = {}
    for c in sorted(set(cls)):
        idx = np.array([i for i, k in enumerate(cls) if k == c])
        per[c] = {"n": int(len(idx)), "psnr": float(m["psnr_per_window"][idx].mean()),
                  "copy_last_psnr": float(mb["psnr_per_window"][idx].mean())}
    out["one_step_by_last_action"] = per
    print("1-step done", json.dumps({k: v for k, v in out.items() if k.startswith("baseline")}), flush=True)

    # ---------------- rollouts
    L = a.long
    traj = ev.eval_trajectories_t(world, n=a.episodes * 2, length=K + L, seed=123)
    # choose the context-noise level that gives the best 128-frame drift (the 1-step score alone prefers 0)
    probe = {k: v[:8] for k, v in traj.items()}
    cs_scores = {}
    if a.ctx_sigma >= 0:
        best_cs = a.ctx_sigma
    else:
        for cs in ((0.0, 0.1) if a.quick else (0.0, 0.05, 0.1, 0.2)):
            r = ev.rollout_metrics(model.denoise, probe, horizon=min(128, L), steps=a.steps, ctx_sigma=cs, init=a.init, K=K)
            cs_scores[str(cs)] = float(r["psnr"].mean())
        best_cs = float(max(cs_scores, key=cs_scores.get))
    out["ctx_sigma_probe_mean_psnr_128"] = cs_scores
    out["chosen_ctx_sigma"] = best_cs
    out["primary_config"] = {"steps": a.steps, "order": 1, "init": a.init, "ctx_sigma": best_cs}
    print("ctx_sigma probe", cs_scores, "-> primary config", out["primary_config"], flush=True)

    curves = {}
    long_traj = {k: v[:a.episodes] for k, v in traj.items()}
    r = ev.rollout_metrics(model.denoise, long_traj, horizon=L, steps=a.steps, ctx_sigma=best_cs, init=a.init, return_frames=True, batch=8, K=K)
    curves[f"{n_params / 1e6:.1f}M EDM, {a.steps} Euler step(s), ctx noise {best_cs}"] = r["psnr"]
    out["rollout_primary"] = {"config": out["primary_config"],"psnr_curve": r["psnr"].tolist(), "ssim_curve": r["ssim"].tolist(),
                             "psnr_at": summarize(r["psnr"]), "ssim_at": summarize(r["ssim"]),
                             "mean_psnr_first32": float(r["psnr"][:32].mean()),
                             "mean_psnr_first128": float(r["psnr"][:128].mean()),
                             "mean_psnr_all": float(r["psnr"].mean()), "episodes": a.episodes, "horizon": L}
    out["rollout_primary"]["frames_until_psnr_below_25db"] = first_below(r["psnr"], 25.0)
    out["rollout_primary"]["frames_until_psnr_below_20db"] = first_below(r["psnr"], 20.0)
    for thr in (25, 20):
        out["rollout_primary"][f"per_episode_frames_until_below_{thr}db"] = {
            "definition": f"first frame whose 5-frame window averages below {thr} dB, per episode; never = horizon + 1",
            **horizon_summary(tracking_horizons(r["psnr_all"], float(thr)), L)}
    pe32 = r["psnr_all"][:, :32].mean(1)
    out["rollout_primary"]["mean_psnr_first32_sem"] = float(pe32.std(ddof=1) / np.sqrt(len(pe32))) if len(pe32) > 1 else None
    out["rollout_primary"]["sharpness_ratio_last32"] = grad_mag(r["frames"][:, -32:]) / grad_mag(long_traj["frames"][:, K + L - 32:K + L])
    # does the dream stay a valid game frame even when it no longer matches the true frame?
    bank = ev.make_frame_bank(world, n=a.bank if not a.quick else 4000)
    gt_long = long_traj["frames"][:, K:K + L]
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
                                    ctx_sigma=best_cs, init=a.init, K=K)
            out[f"rollout_{h}"] = {"episodes": n_ep, "mean_psnr": float(rr["psnr"].mean()), "final_psnr": float(rr["psnr"][-1]),
                                   "mean_ssim": float(rr["ssim"].mean()), "final_ssim": float(rr["ssim"][-1])}
    for name, steps, order in (("euler1", 1, 1), ("euler2", 2, 1), ("euler3", 3, 1), ("heun2", 2, 2)):
        rr = ev.rollout_metrics(model.denoise, {k: v[:8] for k, v in traj.items()}, horizon=min(128, L), steps=steps, order=order,
                                ctx_sigma=best_cs, init=a.init, K=K)
        out[f"rollout128_{name}"] = {"mean_psnr": float(rr["psnr"].mean()), "mean_psnr_first32": float(rr["psnr"][:32].mean()),
                                     "final_psnr": float(rr["psnr"][-1])}
    fz_p, fz_s = frozen_baseline(long_traj, L, K=K)
    floor = uncorrelated_floor(long_traj, L, K=K)
    out["baseline_freeze_last_real_frame"] = {"psnr_at": summarize(fz_p), "ssim_at": summarize(fz_s)}
    out["baseline_unrelated_frame_psnr"] = float(floor.mean())
    pa = r["psnr_all"]
    rnd = lambda v: [round(float(x), 3) for x in v]
    out["figure_curves"] = {"note": "per-frame curves behind media/drift.png (tools/render_drift.mjs); frame 1 = first dreamed frame",
                            "episodes": a.episodes, "horizon": L,
                            "dream_mean": rnd(r["psnr"]), "dream_p25": rnd(np.percentile(pa, 25, axis=0)),
                            "dream_p75": rnd(np.percentile(pa, 75, axis=0)),
                            "freeze_mean": rnd(fz_p), "unrelated_mean": rnd(floor), "nn_dream": rnd(nn_dream), "nn_real": rnd(nn_real)}
    out["hold_one_key"] = hold_action_rollouts(world, model.denoise, steps=a.steps, ctx_sigma=best_cs, init=a.init, device=str(dev), K=K)
    out["spin_360"] = spin_test(world, model.denoise, steps=a.steps, ctx_sigma=best_cs, init=a.init, K=K)
    print("360-degree spin:", out["spin_360"], flush=True)
    print("hold-one-key PSNR@8:", {k: round(v["dream_psnr_at"]["8"], 1) for k, v in out["hold_one_key"].items()}, flush=True)
    out["eval_wall_seconds"] = time.time() - t_start

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / f"eval_{a.name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("baseline_copy_last_frame", "one_step_samplers_ctx0.1", "rollout_primary")
                      if k in out}, indent=1)[:3000])
    if not a.no_figs:
        gt = long_traj["frames"][:, K:K + L]
        np.savez_compressed(res / f"rollouts_{a.name}.npz", gt=to_u8(gt[:4]), dream=to_u8(r["frames"][:4]))
        rollout_figures(a.name)
        print("rollout figures written to media/; the drift figure comes from: node tools/render_drift.mjs --eval results/eval_%s.json" % a.name)


if __name__ == "__main__":
    main()
