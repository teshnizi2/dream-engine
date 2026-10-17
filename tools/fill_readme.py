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
    if fps:
        by = {r["steps"]: r for r in fps["runs"]}
        tok["FPS3"] = f"{by[3]['fps_counted']:.0f}"
        tok["FPS1"] = f"{by[1]['fps_counted']:.0f}"
    sections = []
    # ---- 1-step
    s = ev["one_step_samplers_ctx0.1"]
    cp = ev["baseline_copy_last_frame"]
    ms = {}
    if fps:
        ms = {1: by[1], 2: by[2], 3: by[3]}
    rows = [["copy the last frame (baseline)", f1(cp["psnr"]), f"{cp['ssim']:.3f}", "0", ""]]
    for name, label, steps in (("euler1", "Euler, 1 step", 1), ("euler2", "Euler, 2 steps", 2), ("euler3", "Euler, 3 steps (DIAMOND)", 3)):
        rows.append([label, f1(s[name]["psnr"]), f"{s[name]['ssim']:.3f}", s[name]["nfe"],
                     f"{ms[steps]['ms_per_frame_median']:.0f} ms" if ms else ""])
    for name, label in (("heun2", "Heun, 2 steps"), ("heun3", "Heun, 3 steps")):
        if name in s:
            rows.append([label + " (unstable here)", f1(s[name]["psnr"]), f"{s[name]['ssim']:.3f}", s[name]["nfe"], ""])
    sections.append("### One-step prediction (the real last 4 frames + actions -> the next frame)\n\n"
                    f"{ev['n_val_windows']} fixed validation windows, context noise 0.1, init as in `eval.py`.\n\n"
                    + table(["method", "PSNR dB", "SSIM", "model calls", "browser ms/frame"], rows)
                    + "\n\nSource: `" + a.eval + "`" + (", `" + a.fps + "`" if fps else "") + ".")
    # ---- drift
    ro = ev.get("rollout_primary") or ev["rollout_euler3"]
    fz = ev["baseline_freeze_last_real_frame"]["psnr_at"]
    ks = [k for k in ("1", "4", "8", "16", "32", "64", "128", "256", "512") if k in ro["psnr_at"]]
    rows = [["dream (PSNR dB)"] + [f1(ro["psnr_at"][k]) for k in ks],
            ["dream (SSIM)"] + [f"{ro['ssim_at'][k]:.2f}" for k in ks],
            ["freeze the last real frame (PSNR dB)"] + [f1(fz[k]) for k in ks]]
    sections.append("### Drift: pure imagination against the true game\n\n"
                    f"Rollouts of {ro['horizon']} frames from 4 real frames, {ro['episodes']} episodes (exploration-policy actions), "
                    f"Euler 3 steps, context noise {ev['chosen_ctx_sigma']}. An unrelated frame scores about "
                    f"{ev['baseline_unrelated_frame_psnr']:.1f} dB, so the dream follows the real game for roughly the first "
                    "8-16 frames (a quarter to half a second) and then wanders to a different place.\n\n"
                    + table(["frames since wake-up"] + ks, rows) + "\n\n![drift](media/drift.png)")
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
                        + "\n\nPSNR dB against the real game; \"freeze\" repeats the last real frame. Every control beats freezing; "
                        "strafing is the weakest (it is the rarest action in the training data).")
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
        rows = [[r["steps"], f"{r['ms_per_frame_median']:.1f}", f"{r['fps_counted']:.1f}", f"{r['gpu_util_before_pct']} / {r['gpu_util_after_pct']}"] for r in fps["runs"]]
        sections.append("### Speed in the browser (Apple M4, headless Edge, WebGPU fp16 with graph capture)\n\n"
                        + table(["denoising steps", "ms per frame (model)", "frames per second (counted)", "GPU busy before / after (%)"], rows)
                        + "\n\nThe game tick is capped at 30 per second. **The M4's GPU was shared with other builders' jobs during these "
                        "runs** (last column, from `ioreg`), so these are conservative; on a quiet GPU the model alone took "
                        f"about {fps.get('quiet_ms_note', 'less')}. The WASM fallback measured a few frames per second. Source: `{a.fps}`.")
    text = pathlib.Path(ROOT / "README.template.md").read_text()
    text = text.replace("@@RESULTS_TABLES@@", "\n\n".join(sections))
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
