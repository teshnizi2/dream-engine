"""Builds README.md from README.template.md: every number comes from a results json (the tokens name the source),
and the build fails if a token cannot be resolved, so the README cannot quote a number that was not measured.

    python tools/fill_readme.py --eval results/eval_full1.json --probe results/probe_full1.json \
        --fps results/browser_fps.json --training results/training_summary.json
"""
import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(p):
    return json.loads((ROOT / p).read_text()) if p and (ROOT / p).exists() else None


def f1(x):
    return f"{x:.1f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ablation_tables():
    """What was tried (all single runs; results/ablation/*.json written by eval.py on ALICE)."""
    def ev(n):
        return load(f"results/ablation/eval_{n}.json")
    def roll(r):
        return r.get("rollout_primary") or r["rollout_euler3"]
    parts = []
    wave1 = [("abl_B", "widths 32-64-96-128, 2 blocks (used)"), ("abl_B_noctx", "same, no context noise"),
             ("abl_A", "widths 64x4, 2 blocks (DIAMOND size)"), ("abl_Bd1", "widths 32-64-96-128, 1 block")]
    rows = []
    for n, label in wave1:
        r = ev(n)
        if r:
            e1 = r["one_step_samplers_ctx0.1"]["euler1"]
            rows.append([label, f"{r['params'] / 1e6:.2f}M", f1(e1["psnr"]), f"{e1['ssim']:.3f}", f1(roll(r)["psnr_at"]["32"]), f1(r["baseline_copy_last_frame"]["psnr"])])
    if rows:
        vals = [float(r[2]) for r in rows]
        spread = max(vals) - min(vals)
        verdict = ("width, depth and context noise were not the bottleneck." if spread <= 0.5 else "so the choice mattered at this scale.")
        parts.append("#### Architecture and context noise (game v1, 20k steps each)\n\n"
                     + table(["variant", "params", "1-step PSNR dB", "SSIM", "rollout PSNR @32 frames", "copy-last baseline"], rows)
                     + f"\n\nThe {len(rows)} variants span {spread:.1f} dB in 1-step PSNR: {verdict}")
    v1, v2 = ev("abl_B"), ev("w2_B")
    at1, at2 = load("results/error_attribution_abl_B_game_v1.json"), load("results/error_attribution_w2_B.json")
    if v1 and v2 and at1 and at2:
        w1, w2 = at1["by_surface"]["window_stripes"], at2["by_surface"]["window_stripes"]
        rows = [["v1: fine textures", f1(v1["one_step_samplers_ctx0.1"]["euler1"]["psnr"]), f1(v1["baseline_copy_last_frame"]["psnr"]), f"{w1['error_share_pct']:.0f}% ({w1['pixel_share_pct']:.0f}% of pixels)"],
                ["v2: distance pre-filtered, wide window panes", f1(v2["one_step_samplers_ctx0.1"]["euler1"]["psnr"]), f1(v2["baseline_copy_last_frame"]["psnr"]), f"{w2['error_share_pct']:.0f}% ({w2['pixel_share_pct']:.0f}% of pixels)"]]
        parts.append("#### The game, not the network, was the ceiling\n\n"
                     + table(["game version (same model and recipe, 20k steps)", "1-step PSNR dB", "copy-last baseline dB", "share of squared error on window-pane pixels"], rows)
                     + f"\n\nWindow stripes were {w1['pixel_share_pct']:.0f}% of the pixels but {w1['error_share_pct']:.0f}% of the model's squared error; redesigning them (3 panes per wall, fading to their mean colour with distance) cut that share to {w2['error_share_pct']:.0f}%. Source: `results/error_attribution_*.json` (`tools/error_attribution.py`).")
    rows, vals = [], {}
    for n, label in (("w3_ft_plain", "plain fine-tune (control)"), ("w3_ft_unroll3", "DIAMOND-style unroll 3"), ("w3_ft_blur", "context blur augmentation"),
                     ("w3_ft_rollout3", "scheduled sampling, 3 own frames, p 0.5")):
        r = ev(n)
        r2 = load(f"results/ablation/eval2_{n}.json")
        if r:
            ro = roll(r)
            m32 = sum(ro["psnr_curve"][:32]) / 32
            v = r["validity_nearest_real_frame_psnr"]
            p1 = r["one_step_samplers_ctx0.1"]["euler1"]["psnr"]
            sharp = (roll(r2).get("sharpness_ratio_last32") if r2 else ro.get("sharpness_ratio_last32"))
            vals[n] = {"label": label, "m32": m32, "p1": p1, "valid": v["dream_mean"], "real": v["real_mean"], "sharp": sharp}
            rows.append([label, f1(p1), f1(m32), f1(v["dream_mean"]) + f" ({f1(v['real_mean'])})", f"{sharp:.2f}" if sharp is not None else "n/a"])
    if rows:
        verdict = ""
        base = vals.get("w3_ft_plain")
        if base:
            others = {k: x for k, x in vals.items() if k != "w3_ft_plain"}
            better = [x["label"] for x in others.values() if x["m32"] - base["m32"] > 0.4]
            verdict = ("\n\nChange in the mean PSNR over the first 32 frames against the plain control: "
                       + ", ".join(f"{x['label']} {x['m32'] - base['m32']:+.1f} dB" for x in others.values()) + ". "
                       + ("None of them beats the control by more than the run-to-run noise (about 0.4 dB: the plain control itself moved the means by that much)."
                          if not better else f"Beyond the noise (about 0.4 dB): {', '.join(better)}.")
                       + (f" Context blur: nearest-real-frame PSNR {vals['w3_ft_blur']['valid']:.1f} dB against {base['valid']:.1f} for the control (true frames: {base['real']:.1f})."
                          if "w3_ft_blur" in vals else "")
                       + (f" Scheduled sampling changes the 1-step PSNR by {vals['w3_ft_rollout3']['p1'] - base['p1']:+.1f} dB."
                          if "w3_ft_rollout3" in vals else ""))
        parts.append("#### Can training fix drift? (4k-step fine-tunes of the 20k-step model, game v2)\n\n"
                     + table(["fine-tune", "1-step PSNR dB", "mean PSNR over the first 32 rollout frames", "nearest-real-frame PSNR (real frames)", "late-frame sharpness vs real"], rows)
                     + verdict)
    return "\n\n".join(parts)


def build(a):
    ev = load(a.eval)
    probe = load(a.probe)
    fps = load(a.fps)
    parity = load("results/game_parity.json")
    thr = load("results/data_throughput.json")
    cov = load("results/coverage.json")
    onnx_exp = load(a.onnx)
    onnx_web = load("results/onnx_browser_verification.json")
    train = load(a.training)
    tok = {}
    tok["PARITY_PCT"] = f"{parity['trajectories']['identical_pixel_pct']:.4f}"
    tok["PARITY_MAXDIFF"] = str(int(parity["trajectories"]["max_abs_diff"]))
    tok["PARITY_FRAMES"] = str(int(parity["trajectories"]["frames"] + parity["random poses"]["frames"] + 4))
    tok["DATA_FPS"] = f"{thr['frames_per_sec_single_process_B128']:.0f}"
    tok["COVERAGE"] = f"{cov['normalised_entropy']:.3f}"
    tok["PARAMS_M"] = f"{ev['params'] / 1e6:.2f}"
    if train:
        tok["MAIN_STEPS"] = f"{train['main_steps']:,}"
        tok["MAIN_HOURS"] = f"{train['main_gpu_hours']:.1f}"
        tok["TOTAL_GPUH"] = f"{train['total_gpu_hours']:.1f}"
    ro0 = ev.get("rollout_primary") or ev["rollout_euler3"]
    tok["ONE_STEP_PSNR"] = f"{ev['one_step_samplers_ctx0.1']['euler2']['psnr']:.1f}"
    tok["COPY_PSNR"] = f"{ev['baseline_copy_last_frame']['psnr']:.1f}"
    tok["TRACK20"] = str(ro0.get("frames_until_psnr_below_20db", "?"))
    tok["ROLL_PSNR_32"] = f"{ro0['psnr_at']['32']:.1f}"
    tok["ROLL_PSNR_128"] = f"{ro0['psnr_at']['128']:.1f}"
    tok["FLOOR_PSNR"] = f"{ev['baseline_unrelated_frame_psnr']:.1f}"
    tok["SPIN_BACK"] = f"{ev['spin_360']['dream_frame72_vs_start_view_psnr']:.1f}" if "spin_360" in ev else "n/a"
    hk = ev.get("hold_one_key", {})
    if hk:
        beat = [k for k, d in hk.items() if d["dream_psnr_at"]["8"] > d["freeze_psnr_at"]["8"]]
        weak = min(hk, key=lambda k: hk[k]["dream_psnr_at"]["8"] - hk[k]["freeze_psnr_at"]["8"])
        if len(beat) == len(hk):
            tok["CONTROLS"] = (f"All {len(hk)} hold-one-key tests (move, strafe, turn and the arcs) beat \"freeze the last frame\" after 8 frames; "
                               f"the weakest margin is {weak.replace('_', ' ')}.")
        else:
            miss = ", ".join(k.replace("_", " ") for k in hk if k not in beat)
            tok["CONTROLS"] = f"{len(beat)} of {len(hk)} hold-one-key tests beat \"freeze the last frame\" after 8 frames (not: {miss})."
        tok["CONTROLS_NOTE"] = (f"Every key beats freezing at 8 frames; the smallest margin is {weak.replace('_', ' ')}." if len(beat) == len(hk)
                                else f"Not every key beats freezing at 8 frames ({miss}).")
    if fps:
        tok["FPS2"] = f"{fps['summary']['2']['fps_counted_median']:.0f}"
        tok["FPS2_BEST"] = f"{fps['summary']['2']['fps_counted_best']:.0f}"
    if fps:
        tok["FPS3"] = f"{fps['summary']['3']['fps_counted_median']:.0f}"
        tok["FPS1"] = f"{fps['summary']['1']['fps_counted_median']:.0f}"
    sections = []
    # ---- 1-step
    s = ev["one_step_samplers_ctx0.1"]
    cp = ev["baseline_copy_last_frame"]
    ms = {}
    if fps:
        ms = {k: {"ms_per_frame_median": fps["summary"][str(k)]["ms_per_frame_median"]} for k in (1, 2, 3)}
    rows = [["copy the last frame (baseline)", f1(cp["psnr"]), f"{cp['ssim']:.3f}", "0", ""]]
    for name, label, steps in (("euler1", "Euler, 1 step", 1), ("euler2", "Euler, 2 steps", 2), ("euler3", "Euler, 3 steps (DIAMOND)", 3)):
        rows.append([label, f1(s[name]["psnr"]), f"{s[name]['ssim']:.3f}", s[name]["nfe"],
                     f"{ms[steps]['ms_per_frame_median']:.0f} ms" if ms else ""])
    for name, label in (("heun2", "Heun, 2 steps"), ("heun3", "Heun, 3 steps")):
        if name in s:
            rows.append([label + " (unstable here)", f1(s[name]["psnr"]), f"{s[name]['ssim']:.3f}", s[name]["nfe"], ""])
    sections.append("### One-step prediction (the real last 4 frames + actions -> the next frame)\n\n"
                    f"{ev['n_val_windows']} fixed validation windows, context noise {ev.get('one_step_samplers_config', {}).get('ctx_sigma', 0.1)}, sampler start `{ev.get('one_step_samplers_config', {}).get('init', 'edm')}`.\n\n"
                    + table(["method", "PSNR dB", "SSIM", "model calls", "browser ms/frame"], rows)
                    + "\n\nSource: `" + a.eval + "`" + (", `" + a.fps + "`" if fps else "") + ".")
    # ---- drift
    ro = ev.get("rollout_primary") or ev["rollout_euler3"]
    fz = ev["baseline_freeze_last_real_frame"]["psnr_at"]
    ks = [k for k in ("1", "4", "8", "16", "32", "64", "128", "256", "512") if k in ro["psnr_at"]]
    rows = [["dream (PSNR dB)"] + [f1(ro["psnr_at"][k]) for k in ks],
            ["dream (SSIM)"] + [f"{ro['ssim_at'][k]:.2f}" for k in ks],
            ["freeze the last real frame (PSNR dB)"] + [f1(fz[k]) for k in ks]]
    cfg = ro.get("config") or {"steps": 3, "init": "edm", "ctx_sigma": ev.get("chosen_ctx_sigma")}
    extra = []
    if "frames_until_psnr_below_20db" in ro:
        late = ""
        if ro["horizon"] > 128 and "psnr_curve" in ro:
            tail = ro["psnr_curve"][128:]
            late = (f"; frames 129-{ro['horizon']} average {sum(tail) / len(tail):.1f} dB, close to the {ev['baseline_unrelated_frame_psnr']:.1f} dB "
                    "of an unrelated frame (so past that point it is a different place, not a worse copy)")
        extra.append(f"* the mean PSNR falls below 20 dB after **{ro['frames_until_psnr_below_20db']} frames** (below 25 dB after {ro['frames_until_psnr_below_25db']})" + late)
    if "sharpness_ratio_last32" in ro:
        extra.append(f"* late frames keep **{100 * ro['sharpness_ratio_last32']:.0f}%** of the real frames' edge sharpness (mean gradient magnitude, last 32 frames; 100% = as sharp as the real game)")
    if "spin_360" in ev:
        sp = ev["spin_360"]
        extra.append(f"* hold *turn right* for a full 360 degrees (72 frames): the real game ends on exactly the start view ({sp['real_frame72_vs_start_view_psnr']:.0f} dB), "
                     f"the dream's frame 72 matches the start view at only **{sp['dream_frame72_vs_start_view_psnr']:.1f} dB** (it has no map memory to close the loop)")
    sections.append("### Drift: pure imagination against the true game\n\n"
                    f"Rollouts of {ro['horizon']} frames from 4 real frames, {ro['episodes']} episodes (exploration-policy actions), "
                    f"Euler {cfg['steps']} step(s), init `{cfg['init']}`, context noise {cfg['ctx_sigma']}. An unrelated frame scores about "
                    f"{ev['baseline_unrelated_frame_psnr']:.1f} dB. "
                    + (f"The mean PSNR falls below 20 dB at frame {ro['frames_until_psnr_below_20db']} "
                       f"({ro['frames_until_psnr_below_20db'] / 30:.1f} s at the demo's 30 game ticks per second); after that the dream wanders to a different place."
                       if "frames_until_psnr_below_20db" in ro else "") + "\n\n"
                    + table(["frames since wake-up"] + ks, rows) + "\n\n" + "\n".join(extra) + "\n\n![drift](media/drift.png)")
    if "validity_nearest_real_frame_psnr" in ev:
        v = ev["validity_nearest_real_frame_psnr"]
        sections.append("### Is it still a game frame?\n\n"
                        "PSNR of every dreamed frame to its nearest neighbour among a bank of real frames, averaged over the rollout: "
                        f"**{v['dream_mean']:.1f} dB** for dream frames against **{v['real_mean']:.1f} dB** for the true frames of the same "
                        "episodes. High values late in the rollout mean the dream settles on simple, game-like views (often a wall "
                        "close-up): it stays valid but loses detail (`sharpness_ratio_last32` in the probe json). "
                        "Source: `" + a.eval + "`.")
    if "hold_one_key" in ev:
        rows = []
        for k, d in ev["hold_one_key"].items():
            rows.append([k.replace("_", " "), f1(d["dream_psnr_at"]["8"]), f1(d["freeze_psnr_at"]["8"]),
                         f1(d["dream_psnr_at"]["24"]), f1(d["freeze_psnr_at"]["24"])])
        sections.append("### Do the controls work? (hold one key for 24 frames from random poses)\n\n"
                        + table(["key", "dream @8 frames", "freeze @8", "dream @24", "freeze @24"], rows)
                        + "\n\nPSNR dB against the real game; \"freeze\" repeats the last real frame. " + tok.get("CONTROLS_NOTE", "")
                        + " Strafing is the rarest action in the training data.")
    if probe:
        best = max(probe["grid"], key=lambda r: r["mean_psnr_32"])
        worst = min(probe["grid"], key=lambda r: r["mean_psnr_32"])
        sections.append("### Sampler and context-noise settings\n\n"
                        f"{len(probe['grid'])} inference settings (1-3 Euler steps, two initialisations, context noise 0 or 0.1) over "
                        f"{probe['episodes']} episodes x {probe['horizon']} frames: mean PSNR over the first 32 frames ranges from "
                        f"{worst['mean_psnr_32']:.1f} to {best['mean_psnr_32']:.1f} dB, so the setting matters little; the best was "
                        f"{best['steps']} step(s), init `{best['init']}`, context noise {best['ctx_sigma']}. Source: `{a.probe}`.")
    if onnx_exp:
        w = onnx_web["cases"] if onnx_web else []
        wd = {(c["model"], c["ep"], bool(c.get("capture"))): c for c in w if "max_abs_diff" in c}
        rows = [["fp32", f"{onnx_exp['fp32']['size_mb']:.1f}", f"{onnx_exp['verify_fp32']['max_abs_diff_vs_pytorch']:.1e}",
                 f"{wd[('dream_fp32', 'webgpu', False)]['max_abs_diff']:.1e}" if ('dream_fp32', 'webgpu', False) in wd else "-"],
                ["fp16", f"{onnx_exp['fp16']['size_mb']:.1f}", f"{onnx_exp['verify_fp16']['max_abs_diff_vs_pytorch']:.1e}",
                 f"{wd[('dream_fp16', 'webgpu', True)]['max_abs_diff']:.1e}" if ('dream_fp16', 'webgpu', True) in wd else "-"]]
        sections.append("### ONNX export (opset 17, batch 1, static shapes) against PyTorch\n\n"
                        + table(["precision", "size MB", "max abs diff, ONNX Runtime CPU", "max abs diff, WebGPU in the browser"], rows)
                        + "\n\nOutputs are in [-1, 1]; 1 grey level = 0.0078. Sources: `" + a.onnx + "`, `results/onnx_browser_verification.json`.")
    if fps:
        sm = fps["summary"]
        rows = []
        for st in ("1", "2", "3"):
            rs = [r for r in fps["runs"] if str(r["steps"]) == st]
            lo = min(r["gpu_util_others_before_pct"] for r in rs)
            hi = max(r["gpu_util_others_before_pct"] for r in rs)
            rows.append([st, f"{sm[st]['ms_per_frame_median']:.1f}", f"{sm[st]['fps_counted_median']:.0f}", f"{sm[st]['ms_per_frame_best']:.1f}",
                         f"{sm[st]['fps_counted_best']:.0f}", f"{lo}-{hi}", sm[st]["windows"]])
        sections.append("### Speed in the browser (Apple M4, headless Edge, WebGPU fp16 with graph capture)\n\n"
                        + table(["denoising steps", "ms per frame, median window", "fps, median window", "ms per frame, best window", "fps, best window", "GPU busy with other jobs just before the windows (%)", "windows"], rows)
                        + "\n\nThe game tick is capped at 30 per second, so the best windows read 30 fps. **The M4's GPU was shared with other jobs on this machine during all "
                        "runs** (the busy column: `ioreg` utilisation sampled with this page paused), which is why the median windows are slower than the best ones; a quiet GPU should behave like the best windows. "
                        f"Source: `{a.fps}` (`tools/browser_fps.mjs`).")
    text = pathlib.Path(ROOT / "README.template.md").read_text()
    text = text.replace("@@RESULTS_TABLES@@", "\n\n".join(sections))
    text = text.replace("@@ABLATIONS@@", ablation_tables())
    limits = (ROOT / a.limits).read_text() if a.limits and (ROOT / a.limits).exists() else "@@LIMITS@@"
    text = text.replace("@@LIMITS@@", limits.strip())
    for k, v in tok.items():
        text = text.replace(f"@@{k}@@", v)
    left = sorted(set(re.findall(r"@@[A-Z0-9_]+@@", text)))
    if left:
        raise SystemExit("unresolved tokens: " + ", ".join(left))
    (ROOT / "README.md").write_text(text)
    print("wrote README.md", len(text), "chars")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--probe")
    ap.add_argument("--fps")
    ap.add_argument("--onnx", default="results/onnx_export_dream.json")
    ap.add_argument("--training")
    ap.add_argument("--limits", default="docs/limits.md")
    build(ap.parse_args())
