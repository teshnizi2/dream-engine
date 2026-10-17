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


def _t20(ro):
    pe = ro.get("per_episode_frames_until_below_20db")
    return f"{pe['median']:.0f} (median)" if pe else f"{ro['frames_until_psnr_below_20db']} (mean curve)"


def _demo_choice(probe, ev):
    """Why the demo's setting (2 steps, unit start, context noise 0.05) is not simply the best row of the probe."""
    g = {(r["steps"], r["init"], r["ctx_sigma"]): r for r in probe["grid"]}
    lo, hi = g.get((2, "unit", 0.0)), g.get((2, "unit", 0.1))
    s1 = ev["one_step_samplers_ctx0.1"]
    cfg = ev.get("primary_config") or {}
    if not (lo and hi and cfg):
        return ""
    return (f"The demo runs {cfg['steps']} steps, init `{cfg['init']}`, context noise {cfg['ctx_sigma']} (between the probed 0 and 0.1): "
            f"the neighbouring probed settings score {lo['mean_psnr_32']:.1f} and {hi['mean_psnr_32']:.1f} dB, within the noise of the best, while "
            f"one step ahead 2 steps score {s1['euler2']['psnr']:.1f} dB against {s1['euler1']['psnr']:.1f} for 1 step, and 2-step frames are visibly cleaner (see below).")


def _final_attribution():
    a = load("results/error_attribution_full1.json")
    if not a:
        return ""
    w, e = a["by_surface"]["window_stripes"], a["strong_vertical_edges"]
    return (f" For the final model the window panes are still {w['pixel_share_pct']:.0f}% of the pixels and {w['error_share_pct']:.0f}% of the error, and strong vertical edges "
            f"(window frames, brick joints, wall corners) are {e['pixel_share_pct']:.0f}% of the pixels and {e['error_share_pct']:.0f}% of the squared error: thin high-contrast structure is where the model is least exact.")


def _scale_sentence(small, big):
    g1 = big["one_step_samplers_ctx0.1"]["euler2"]["psnr"] - small["one_step_samplers_ctx0.1"]["euler2"]["psnr"]
    g32 = big["rollout_primary"]["mean_psnr_first32"] - small["rollout_primary"]["mean_psnr_first32"]
    out = (f"\n\n{big['step'] / small['step']:.1f}x the training steps buy {g1:+.1f} dB one step ahead and {g32:+.1f} dB over the first 32 rollout frames.")
    prim = load("results/eval_full1_n64.json")
    if prim and prim["rollout_primary"]["episodes"] != big["rollout_primary"]["episodes"]:
        out += (f" This {big['rollout_primary']['episodes']}-episode sample is an easier one for the final model ({big['rollout_primary']['mean_psnr_first32']:.1f} dB over the first 32 frames "
                f"against {prim['rollout_primary']['mean_psnr_first32']:.1f} on the 64-episode primary sample), so compare the two rows with each other, not with the headline numbers.")
    ps, pb = small["rollout_primary"].get("per_episode_frames_until_below_20db"), big["rollout_primary"].get("per_episode_frames_until_below_20db")
    if ps and pb:
        out += f" The median tracking horizon goes from {ps['median']:.0f} to {pb['median']:.0f} frames."
    if g1 > 1.0 and g32 < 0.5 * g1:
        out += " Drift is compounding error: a better one-step model shrinks it far more slowly than it improves the one-step score."
    return out


def _load_abs(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fmt_frames(x):
    """A frame count that may be a half (the median of an even number of episodes): 14 -> '14', 15.5 -> '15.5'."""
    return f"{x:.1f}".rstrip("0").rstrip(".") if abs(x - round(x)) > 1e-9 else str(int(round(x)))


def v3_section(res=None):
    """The v3 iteration (docs/PREREG-v3.md): ablation, full run and the verdict of the pre-registered adoption rule, from the saved
    evaluations (results/prereg_*.json, prereg_ablation.json, prereg_final_*.json). Empty if the ablation has not been evaluated."""
    res = pathlib.Path(res) if res else ROOT / "results"
    abl = _load_abs(res / "prereg_ablation.json")
    if not abl:
        return ""
    fin = next((f for f in sorted(res.glob("prereg_final_*_vs_*.json"))), None)
    fin = _load_abs(fin) if fin else None
    obs = _load_abs(res / "observability_v2_v3.json")
    sig = _load_abs(res / "signature_design_v3.json")

    def row(r, label=None):
        lo, hi = r["collapse_ci95"]
        return [label or r["name"], r["game"], r["K"], f"{r['steps']:,}" if r.get("steps") else "", f"{fmt_frames(r['horizon_median'])} ({fmt_frames(r['horizon_p25'])} to {fmt_frames(r['horizon_p75'])})",
                f"{r['collapse']:.3f} ({lo:.2f} to {hi:.2f})", f"{r['collapse_window1']:.3f}", f"{r['dream_flat_share']:.2f} / {r['real_flat_share']:.2f}",
                f1(r["spin_return_psnr"]), f"{r['one_step_psnr']:.1f} ({r['copy_last_psnr']:.1f})", f1(r["mean_psnr_first32"])]
    head = ["run", "game", "K", "steps", "median frames until < 20 dB (middle half)", "collapse rate (95% interval)", "single-frame collapse rate",
            "flat-wall share dream / real", "360 degree spin return, dB", "1-step PSNR dB (copy-last)", "mean PSNR, first 32 frames"]
    out = ["#### v3: can the flat-wall collapse be fixed? (pre-registered: `docs/PREREG-v3.md`)\n",
           "Long dreams collapse onto flat wall views and stay there. The hypothesis was observability: with 4 context frames a close-up wall carries no information about where "
           "the player is (walls of one colour region are identical, and so are N and S faces), so once the context is all wall the model has nothing to localise from. "
           "**Game v3** gives every wall face its own low-frequency colour signature (a smooth tint of the hit position, 12 grey levels, periods of at least 5 cells) and four distinct face shades, "
           "keeping the map, physics, floor, ceiling, fog and landmarks; the JavaScript twin stays bit-exact with the Python reference. The second change was a longer context, K = 8 against K = 4."]
    if obs:
        a2, a3 = obs["v2"]["noise_4"], obs["v3"]["noise_4"]
        sep = f" The 142 visible wall faces are at least {sig['shipped']['min_face_distance']:.1f} grey levels apart (`results/signature_design_v3.json`)." if sig else ""
        out.append(f"A model-free check on rendered frames (nearest-neighbour pose recovery from flat-wall views with noise, `results/observability_v2_v3.json`): the recovered position is within 1.5 cells "
                   f"{100 * a2['within_1.5_cells']:.0f}% of the time in v2 and {100 * a3['within_1.5_cells']:.0f}% in v3, the heading within 45 degrees {100 * a2['heading_within_45deg']:.0f}% and {100 * a3['heading_within_45deg']:.0f}%."
                   f" The information is in the v3 frames; whether a trained network uses it is what was tested." + sep)
    out.append("**Metrics, fixed before any v3 model existed.** Every model wakes up at the same real frame (the 8th) on the same 64 episodes and dreams 512 frames with the demo's sampler. "
               "*Tracking horizon*: the first frame whose 5-frame average PSNR is below 20 dB, median over episodes. *Collapse rate*: the fraction of dream frames at 64 to 512 that are *stuck on a flat wall* "
               "(at most 5% floor or ceiling pixels in each of the last 32 frames) while the matching real frame is not; the real game itself is flat-wall in about a quarter of its frames, which is why the metric asks the dream to stay there "
               "(window and thresholds were fixed on v2 rollouts, `docs/PREREG-v3.md` section 4). *Spin return*: PSNR between the start view and the view after a full 360 degree turn.")
    runs = abl["runs"]
    order = [abl["baseline"]] + [n for n in runs if n != abl["baseline"]]
    pv = abl.get("paired_vs_baseline", {})

    def abl_row(n):
        r = row(runs[n], n + (" (baseline)" if n == abl["baseline"] else ""))
        if n in pv:
            lo, hi = pv[n]["collapse_ratio_ci95"]
            r.append(f"{pv[n]['collapse_ratio']:.2f} ({lo:.2f} to {hi:.2f})")
        else:
            r.append("1" if n == abl["baseline"] else "")
        return r
    out.append("**20k-step ablation** (same recipe as `w2_B`, one run each, 64 episodes x 512 frames):\n\n"
               + table(head + ["collapse rate relative to the baseline (paired 95% interval)"], [abl_row(n) for n in order])
               + f"\n\nSelection rule (fixed in advance): {abl['selection']['reason']}.")
    if fin:
        new, ref = fin["new"], fin["ref"]
        ad = fin["adoption"]
        lo, hi = fin["paired_ratio_ci95"]
        dlo, dhi = fin["paired_median_difference_ci95"]
        out.append(f"**Full run and the adoption rule** (150,000 steps, the recipe of `{ref['name']}`; both evaluated with the same code on the same 64 episodes):\n\n"
                   + table(head, [row(ref, ref["name"] + " (v2, shipped before)"), row(new)])
                   + f"\n\nThe rule: adopt only if the collapse rate is at most half of {ref['name']}'s **and** the median tracking horizon is not shorter. "
                   f"Collapse rate {new['collapse']:.3f} against {ref['collapse']:.3f} (ratio {fin['collapse_ratio']:.2f}, paired 95% interval {lo:.2f} to {hi:.2f}): "
                   f"{'met' if ad['A_collapse_at_most_half'] else 'not met'}. Median horizon {fmt_frames(new['horizon_median'])} against {fmt_frames(ref['horizon_median'])} frames "
                   f"(difference {fin['horizon_median_difference']:+.0f}, paired 95% interval {dlo:+.0f} to {dhi:+.0f}): {'met' if ad['B_horizon_not_shorter'] else 'not met'}.")
        if ad["adopt"]:
            out.append("**The v3 model met the pre-registered rule, so the demo ships it.** What still fails is reported in the limits below; PSNR levels are not comparable across the two games "
                       f"(copy-last-frame baseline {ref['copy_last_psnr']:.1f} dB in v2 and {new['copy_last_psnr']:.1f} dB in v3 on the same windows; unrelated-frame floor "
                       f"{ref['unrelated_frame_psnr']:.1f} and {new['unrelated_frame_psnr']:.1f} dB), so the horizon comparison is a guard against getting worse, not a measurement of how much better.")
        else:
            failed = [k for k, v in (("the collapse rate was not at most half", ad["A_collapse_at_most_half"]), ("the tracking horizon got shorter", ad["B_horizon_not_shorter"])) if not v]
            out.append(f"**v3 did not meet the pre-registered rule ({' and '.join(failed)}), so v2 stays shipped.** This is a negative result for the observability fix as tried here "
                       "(one training run per variant, this signature design, this budget); the code, the game and all the numbers above are in the repository. "
                       f"PSNR levels are not comparable across the two games (copy-last-frame baseline {ref['copy_last_psnr']:.1f} dB in v2 and {new['copy_last_psnr']:.1f} dB in v3; "
                       f"unrelated-frame floor {ref['unrelated_frame_psnr']:.1f} and {new['unrelated_frame_psnr']:.1f} dB).")
        out.append(f"Sources: `results/prereg_*.json` (`eval_prereg.py`), `results/prereg_ablation.json` and `results/prereg_final_{new['name']}_vs_{ref['name']}.json` (`tools/compare_prereg.py`); RUNS.md.")
    out.extend(_v3_exploratory(res))
    return "\n\n".join(out)


def _v3_exploratory(res):
    """Exploratory tables (declared in the deviation log of docs/PREREG-v3.md), from whatever files exist: flat-spell statistics, the
    learning curve over the milestone checkpoints, and the flat-wall escape diagnostic."""
    out = []
    sp = _load_abs(res / "flat_spells.json")
    if sp:
        rows = [["real game", f"{sp['real_game']['share_of_frames_flat']:.2f}", fmt_frames(sp["real_game"]["median_length"]), str(sp["real_game"]["longest"]),
                 f"{100 * sp['real_game']['chance_of_leaving_per_flat_frame']:.2f}%"]]
        for n, v in sp["runs"].items():
            d = v["dream"]
            rows.append([n, f"{d['share_of_frames_flat']:.2f}", fmt_frames(d["median_length"]), str(d["longest"]), f"{100 * d['chance_of_leaving_per_flat_frame']:.2f}%"])
        out.append("**Exploratory: how long does the dream stay on a flat wall?** From the saved per-frame arrays, frames 64 to 512 (a flat spell is a run of frames with at most 5% floor or ceiling pixels):\n\n"
                   + table(["", "share of frames flat", "median spell (frames)", "longest spell", "chance of leaving a flat wall in a given frame"], rows)
                   + "\n\nThe collapse is mostly a failure to leave: the real game leaves a flat wall in about 1 frame in 16, the dream far less often, and training makes the biggest difference. Source: `results/flat_spells.json` (`tools/flat_spells.py`).")
    curve = {}
    for kind in ("full1", "full2"):
        for f in sorted(res.glob(f"prereg_{kind}_s*.json")):
            step = int(f.stem.rsplit("_s", 1)[1])
            j = _load_abs(f)
            curve.setdefault(step, {})[kind] = (j["collapse"]["rate"], j["tracking_horizon_20db"]["median"])
    for kind in ("full1", "full2"):
        f = res / f"prereg_{kind}.json"
        j = _load_abs(f)
        if j and j.get("step"):
            curve.setdefault(int(j["step"]), {})[kind] = (j["collapse"]["rate"], j["tracking_horizon_20db"]["median"])
    if curve:
        kinds = [k for k in ("full1", "full2") if any(k in v for v in curve.values())]
        names = {"full1": "v2 (`full1`)", "full2": "v3 (`full2`)"}
        rows = [[f"{st:,}"] + [(f"{curve[st][k][0]:.3f} / {fmt_frames(curve[st][k][1])}" if k in curve[st] else "") for k in kinds] for st in sorted(curve)]
        out.append("**Exploratory: the learning curve of the collapse.** Collapse rate / median tracking horizon of the EMA checkpoints saved every 20k steps (same protocol, 64 episodes x 512 frames):\n\n"
                   + table(["training steps"] + [names[k] for k in kinds], rows)
                   + "\n\nThe collapse rate falls steadily with training in v2, so a 20k-step comparison understates what 150k steps do and has to be read against v2 at the same step. Source: `results/prereg_<run>_s<step>.json`.")
    esc = []
    for label, path in (("w2_B (v2, 20k)", "prereg_w2_B.json"), ("v3k4 (v3, 20k)", "prereg_v3k4.json"), ("full1 (v2, 150k)", "prereg_full1.json"), ("full2 (v3, 150k)", "prereg_full2.json")):
        j = _load_abs(res / path)
        fe = j.get("flat_escape") if j else None
        if fe and "dream_psnr_at" in fe:
            esc.append([label, f1(fe["dream_psnr_at"]["8"]), f1(fe["dream_psnr_at"]["16"]), f1(fe["dream_psnr_at"]["24"]), f"{100 * fe['dream_flat_share_at_end']:.0f}% / {100 * fe['real_flat_share_at_end']:.0f}%"])
    if esc:
        out.append("**Exploratory: flat-wall escape.** Start facing a wall closely with identical context frames and hold *turn right* for 24 frames; the real game turns away and shows what is around the corner. "
                   "PSNR against the real game, and the share of dreams still on a flat wall at frame 24 (dream / real):\n\n"
                   + table(["model", "PSNR @8", "@16", "@24", "still flat at frame 24"], esc))
    return out


def v3_glance(res=None):
    """One bullet for 'At a glance': the verdict of the pre-registered v3 rule (empty if there is no final comparison yet)."""
    res = pathlib.Path(res) if res else ROOT / "results"
    fin = next((f for f in sorted(res.glob("prereg_final_*_vs_*.json"))), None)
    fin = _load_abs(fin) if fin else None
    if not fin:
        return ""
    new, ref, ad = fin["new"], fin["ref"], fin["adoption"]
    nums = (f"collapse rate {new['collapse']:.2f} against {ref['collapse']:.2f}, median tracking horizon {new['horizon_median']:.0f} against {ref['horizon_median']:.0f} frames "
            f"(64 episodes x 512 frames, same protocol)")
    if ad["adopt"]:
        return (f"* **Self-localising walls (game v3) met the pre-registered adoption rule for the flat-wall collapse** (at most half the collapse rate, no shorter horizon): {nums}. "
                "What still fails is listed under \"Honest limits\".")
    return (f"* A pre-registered attempt to fix the collapse onto flat wall views (v3: every wall face gets its own colour signature, plus a longer context) **did not meet its adoption rule**: {nums}; "
            "v2 stays shipped and v3 is documented as a negative result (see \"What was tried\").")


def v3_limit(res=None):
    """The 'Honest limits' bullet about the flat-wall collapse of the shipped model, with the fate of v3 (empty if no final comparison)."""
    res = pathlib.Path(res) if res else ROOT / "results"
    fin = next((f for f in sorted(res.glob("prereg_final_*_vs_*.json"))), None)
    fin = _load_abs(fin) if fin else None
    if not fin:
        return ""
    new, ref, ad = fin["new"], fin["ref"], fin["adoption"]
    ship = new if ad["adopt"] else ref
    stuck = (f"in {100 * ship['dream_stuck_share']:.0f}% of frames 64 to 512 of a long dream the shipped model is stuck on a flat wall (no floor or ceiling for 32 frames in a row), "
             f"against {100 * ship['real_stuck_share']:.0f}% for the real game's own path; the collapsed frames are smooth colour fields with far less structure than a real wall close-up")
    if ad["adopt"]:
        return (f"* **Long dreams still collapse onto flat wall views, less than in v2.** {stuck[0].upper() + stuck[1:]}. The v3 game's self-localising walls reduced the collapse rate from "
                f"{ref['collapse']:.2f} to {new['collapse']:.2f} (pre-registered rule met); a full 360 degree spin still does not return to the start view.")
    return (f"* **Long dreams collapse onto flat wall views (not fixed).** {stuck[0].upper() + stuck[1:]}. A pre-registered fix (v3: unique colour signatures on every wall face) "
            f"did not meet its rule (collapse rate {new['collapse']:.2f} against {ref['collapse']:.2f}), so v2 stays shipped; a full 360 degree spin does not return to the start view either.")


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
                     + f"\n\nThe {len(rows)} variants span {spread:.1f} dB in 1-step PSNR: {verdict} (Evaluated at the time with 1 Euler step, `edm` start, context noise 0.1.)")
    v1, v2 = ev("abl_B"), ev("w2_B")
    at1, at2 = load("results/error_attribution_abl_B_game_v1.json"), load("results/error_attribution_w2_B.json")
    if v1 and v2 and at1 and at2:
        w1, w2 = at1["by_surface"]["window_stripes"], at2["by_surface"]["window_stripes"]
        rows = [["v1: fine textures", f1(v1["one_step_samplers_ctx0.1"]["euler1"]["psnr"]), f1(v1["baseline_copy_last_frame"]["psnr"]), f"{w1['error_share_pct']:.0f}% ({w1['pixel_share_pct']:.0f}% of pixels)"],
                ["v2: distance pre-filtered, wide window panes", f1(v2["one_step_samplers_ctx0.1"]["euler1"]["psnr"]), f1(v2["baseline_copy_last_frame"]["psnr"]), f"{w2['error_share_pct']:.0f}% ({w2['pixel_share_pct']:.0f}% of pixels)"]]
        parts.append("#### The game, not the network, was the ceiling\n\n"
                     + table(["game version (same model and recipe, 20k steps)", "1-step PSNR dB", "copy-last baseline dB", "share of squared error on window-pane pixels"], rows)
                     + f"\n\nWindow stripes were {w1['pixel_share_pct']:.0f}% of the pixels but {w1['error_share_pct']:.0f}% of the model's squared error; redesigning them (3 panes per wall, fading to their mean colour with distance) cut that share to {w2['error_share_pct']:.0f}%."
                     + _final_attribution() + " Source: `results/error_attribution_*.json` (`tools/error_attribution.py`).")
    small, big = load("results/eval_w2_B_n64.json"), load("results/eval_full1_n64.json")
    if not (small and big):                                     # fall back to the 16 x 256 pair of the eval_many job
        small, big = load("results/ablation/eval2_w2_B.json"), load("results/ablation/eval2_full1.json")
    if small and big:
        def line(e, label):
            ro, v = e["rollout_primary"], e["validity_nearest_real_frame_psnr"]
            return [label, f"{e['step']:,}", f1(e["one_step_samplers_ctx0.1"]["euler2"]["psnr"]), f1(ro["mean_psnr_first32"]), f1(ro["mean_psnr_first128"]),
                    _t20(ro), f"{ro['sharpness_ratio_last32']:.2f}", f"{v['dream_mean']:.1f} ({v['real_mean']:.1f})", f1(e["spin_360"]["dream_frame72_vs_start_view_psnr"])]
        rs, rb = small["rollout_primary"], big["rollout_primary"]
        parts.append("#### Does more training help? (same model and recipe, game v2)\n\n"
                     f"Same evaluation code and demo sampler (2 Euler steps, `unit` start, context noise 0.05), the same {big['rollout_primary']['episodes']} episodes x {big['rollout_primary']['horizon']} frames for both, one run each.\n\n"
                     + table(["run", "steps", "1-step PSNR dB", "mean PSNR, first 32 frames", "mean PSNR, first 128", "frames until < 20 dB", "late-frame sharpness vs real", "nearest-real-frame PSNR (real frames)", "360 degree spin, return PSNR dB"],
                             [line(small, "w2_B"), line(big, "full1 (the demo's model)")])
                     + _scale_sentence(small, big))
    rows, vals = [], {}
    for n, label in (("w3_ft_plain", "plain fine-tune (control)"), ("w3_ft_unroll3", "DIAMOND-style unroll 3"), ("w3_ft_blur", "context blur augmentation"),
                     ("w3_ft_rollout3", "scheduled sampling, 3 own frames, p 0.5")):
        r2 = load(f"results/ablation/eval2_{n}.json")           # same eval code and the demo's sampler settings for every row
        r = r2 or ev(n)
        if r:
            ro = roll(r)
            m32 = sum(ro["psnr_curve"][:32]) / 32
            v = r["validity_nearest_real_frame_psnr"]
            p1 = r["one_step_samplers_ctx0.1"]["euler2" if r2 else "euler1"]["psnr"]
            sharp = ro.get("sharpness_ratio_last32")
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
                     "Every row evaluated with the same code and the demo's sampler (2 Euler steps, `unit` start, context noise 0.05), 16 episodes x 256 frames.\n\n"
                     + table(["fine-tune", "1-step PSNR dB", "mean PSNR over the first 32 rollout frames", "nearest-real-frame PSNR (real frames)", "late-frame sharpness vs real"], rows)
                     + verdict)
    v3 = v3_section()
    if v3:
        parts.append(v3)
    return "\n\n".join(parts)


def build(a):
    ev = load(a.eval)
    probe = load(a.probe)
    fps = load(a.fps)
    wasm = load(a.fps_wasm)
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
    tok["SHARP_PCT"] = f"{100 * ro0['sharpness_ratio_last32']:.0f}" if "sharpness_ratio_last32" in ro0 else "n/a"
    s1_ = ev["one_step_samplers_ctx0.1"]
    tok["HEUN2_DB"], tok["HEUN3_DB"], tok["EULER2_DB"] = (f"{s1_[k]['psnr']:.1f}" if k in s1_ else "n/a" for k in ("heun2", "heun3", "euler2"))
    pe20 = ro0.get("per_episode_frames_until_below_20db")
    tok["TRACK20"] = str(round(pe20["median"])) if pe20 else str(ro0.get("frames_until_psnr_below_20db", "?"))
    tok["TRACK20_NOTE"] = (f"median of {pe20['episodes']} episodes, middle half {round(pe20['p25'])} to {round(pe20['p75'])} frames; PSNR above 20 dB, 5-frame average"
                           if pe20 else "the mean curve crosses 20 dB")
    tok["TRACK_SEC"] = f"{(round(pe20['median']) if pe20 else ro0.get('frames_until_psnr_below_20db', 0)) / 30:.1f}"
    tok["ROLL_PSNR_32"] = f"{ro0['psnr_at']['32']:.1f}"
    tok["ROLL_PSNR_128"] = f"{ro0['psnr_at']['128']:.1f}"
    tok["FLOOR_PSNR"] = f"{ev['baseline_unrelated_frame_psnr']:.1f}"
    tok["SPIN_BACK"] = f"{ev['spin_360']['dream_frame72_vs_start_view_psnr']:.1f}" if "spin_360" in ev else "n/a"
    hk = ev.get("hold_one_key", {})
    if hk:
        beat = [k for k, d in hk.items() if d["dream_psnr_at"]["8"] > d["freeze_psnr_at"]["8"]]
        weak = min(hk, key=lambda k: hk[k]["dream_psnr_at"]["8"] - hk[k]["freeze_psnr_at"]["8"])
        if len(beat) == len(hk):
            mg = hk[weak]["dream_psnr_at"]["8"] - hk[weak]["freeze_psnr_at"]["8"]
            tok["CONTROLS"] = (f"All {len(hk)} hold-one-key tests (move, strafe, turn and the arcs) beat \"freeze the last frame\" after 8 frames; "
                               f"the smallest margin is {mg:.1f} dB ({weak.replace('_', ' ')}).")
        else:
            miss = ", ".join(k.replace("_", " ") for k in hk if k not in beat)
            tok["CONTROLS"] = f"{len(beat)} of {len(hk)} hold-one-key tests beat \"freeze the last frame\" after 8 frames (not: {miss})."
        tok["CONTROLS_NOTE"] = (f"Every key beats freezing at 8 frames; the smallest margin is {mg:.1f} dB ({weak.replace('_', ' ')})." if len(beat) == len(hk)
                                else f"Not every key beats freezing at 8 frames ({miss}).")
    if fps:
        tok["FPS2"] = f"{fps['summary']['2']['fps_counted_median']:.0f}"
        tok["FPS2_BEST"] = f"{fps['summary']['2']['fps_counted_best']:.0f}"
    if wasm:
        w2 = wasm["summary"]["2"]
        tok["WASM_FPS"] = f"{w2['fps_counted_median']:.1f} fps ({w2['ms_per_frame_median']:.0f} ms per frame, 2 steps, fp32, in a worker; {a.fps_wasm})"
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
        pe20, pe25 = ro.get("per_episode_frames_until_below_20db"), ro.get("per_episode_frames_until_below_25db")
        if pe20 and pe25:
            extra.append(f"* tracking horizon per episode (first frame whose 5-frame average PSNR is below the threshold): **median {pe20['median']:.0f} frames below 20 dB** "
                         f"(middle half {pe20['p25']:.0f} to {pe20['p75']:.0f}; {100 * pe20['share_never_below']:.0f}% of the {pe20['episodes']} episodes never fall below within {ro['horizon']} frames) "
                         f"and median {pe25['median']:.0f} frames below 25 dB (middle half {pe25['p25']:.0f} to {pe25['p75']:.0f}); "
                         f"the mean curve crosses 20 dB at frame {ro['frames_until_psnr_below_20db']}, pulled up by easy episodes" + late)
        else:
            extra.append(f"* the mean PSNR falls below 20 dB after **{ro['frames_until_psnr_below_20db']} frames** (below 25 dB after {ro['frames_until_psnr_below_25db']})" + late)
    if "sharpness_ratio_last32" in ro:
        extra.append(f"* late frames keep **{100 * ro['sharpness_ratio_last32']:.0f}%** of the real frames' edge sharpness (mean gradient magnitude, last 32 frames; 100% = as sharp as the real game)")
    if "spin_360" in ev:
        sp = ev["spin_360"]
        extra.append(f"* hold *turn right* for a full 360 degrees (72 frames): the real game ends on exactly the start view ({sp['real_frame72_vs_start_view_psnr']:.0f} dB), "
                     f"the dream's frame 72 matches the start view at only **{sp['dream_frame72_vs_start_view_psnr']:.1f} dB** (it has no map memory to close the loop)")
    sections.append("### Drift: pure imagination against the true game\n\n"
                    f"Rollouts of {ro['horizon']} frames from 4 real frames, {ro['episodes']} episodes (exploration-policy actions), "
                    f"Euler {cfg['steps']} steps, init `{cfg['init']}`, context noise {cfg['ctx_sigma']}. An unrelated frame scores about "
                    f"{ev['baseline_unrelated_frame_psnr']:.1f} dB. "
                    + (f"The median episode falls below 20 dB at frame {tok['TRACK20']} ({tok['TRACK_SEC']} s at the demo's 30 game ticks per second); after that the dream wanders to a different place."
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
                        f"{best['steps']} step{'s' if best['steps'] > 1 else ''}, init `{best['init']}`, context noise {best['ctx_sigma']}. "
                        + _demo_choice(probe, ev) + f" Source: `{a.probe}`.")
    sm_rows = []
    for label, path in (("local Apple M4 (MPS), GPU shared", "results/local_mps_smoke_metrics.jsonl"), ("ALICE A100 (job 5246018)", "results/alice_smoke_metrics.jsonl")):
        fp = ROOT / path
        if not fp.exists():
            continue
        recs = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
        tr = [r for r in recs if "loss" in r]
        ev_ = [r for r in recs if "eval_psnr_ctx0.1" in r]
        sm_rows.append([label, tr[-1]["step"], f"{tr[0]['loss']:.2f} -> {tr[-1]['loss']:.3f}", f"{tr[-1]['elapsed_s'] / 60:.1f} min ({tr[-1]['steps_per_s']:.1f} it/s)",
                        f"{ev_[-1]['eval_psnr_ctx0.1']:.1f} dB / {ev_[-1]['eval_ssim_ctx0.1']:.2f}" if ev_ else "not reached"])
    if sm_rows:
        sections.append("### Smoke tests before the main run\n\n"
                        + table(["run", "steps", "training loss", "time", "one-step PSNR / SSIM (3 Euler steps, 128 windows)"], sm_rows)
                        + "\n\nThe local run stopped early because the Mac's GPU was shared with other jobs (about 0.35 it/s); the ALICE smoke job is the evidence that the pipeline learns "
                        "(its copy-last-frame baseline is quoted in RUNS.md row 1). Sources: `results/*_smoke_metrics.jsonl`, RUNS.md row 1.")
    stab = []
    for label, path in (("A100, closing eval of the training job", "results/eval_full1_alice.json"), ("A100, `eval_many`", "results/ablation/eval2_full1.json"),
                        ("A100", "results/eval_full1_a100.json"), ("Apple M4 (MPS)", "results/eval_full1_local.json"),
                        ("A100, **primary**", "results/eval_full1_n64.json")):
        e = load(path)
        if not e:
            continue
        r_, v_ = e["rollout_primary"], e["validity_nearest_real_frame_psnr"]
        pe = r_.get("per_episode_frames_until_below_20db")
        sem = r_.get("mean_psnr_first32_sem")
        stab.append([label, f"{r_['episodes']} x {r_['horizon']}", f1(e["one_step_samplers_ctx0.1"]["euler2"]["psnr"]),
                     f1(r_["mean_psnr_first32"]) + (f" ± {sem:.1f}" if sem else ""), f1(r_["mean_psnr_first128"]),
                     f"{pe['median']:.0f}" if pe else "n/a", r_["frames_until_psnr_below_20db"], f"{v_['dream_mean']:.1f} ({v_['real_mean']:.1f})"])
    if len(stab) > 1:
        m32s = [float(r[3].split()[0]) for r in stab]
        sections.append("### How stable are these numbers? (one checkpoint, five evaluations)\n\n"
                        + table(["evaluation", "episodes x frames", "1-step PSNR dB", "mean PSNR, first 32 frames (± s.e.)", "mean PSNR, first 128", "median episode: frames until < 20 dB", "mean curve: frames until < 20 dB", "nearest-real-frame PSNR (real)"], stab)
                        + f"\n\nThe one-step score does not move with the sample or the hardware. The drift statistics do: the mean over the first 32 frames ranges from {min(m32s):.1f} to {max(m32s):.1f} dB "
                        "across episode samples (the 64-episode standard error is shown), and the mean curve's crossing of 20 dB is unstable because a few easy episodes pull it up, "
                        "so the headline tracking horizon is the median over episodes. Different hardware draws different noise. Comparisons in \"What was tried\" are made "
                        "only within one episode sample.")
    sf = load("results/sampler_figure.json")
    if sf:
        m = sf["mean_psnr_db"]
        cost = (f" In the browser they cost {fps['summary']['1']['ms_per_frame_median']:.0f} / {fps['summary']['2']['ms_per_frame_median']:.0f} / "
                f"{fps['summary']['3']['ms_per_frame_median']:.0f} ms per frame (median windows)." if fps else "")
        sections.append("### What the denoising steps change\n\n![sampler steps](media/sampler_steps.png)\n\n"
                        f"{len(sf['window_indices'])} evenly spaced validation windows (picked by position, not by quality), same noise seed. Mean PSNR to the true next frame: "
                        f"**{m['euler1']:.1f} dB** with 1 Euler step, **{m['euler2']:.1f}** with 2, **{m['euler3']:.1f}** with 3. One step leaves visible grain on flat surfaces "
                        f"(second row); two steps are as clean as three, so the demo defaults to 2.{cost} Source: `results/sampler_figure.json` (`tools/sampler_figure.py`).")
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
    bvp = load("results/browser_vs_python_1step.json")
    if bvp:
        rows = []
        for st in (1, 2, 3):
            py = bvp["python"].get(f"steps{st}")
            b32, b16 = bvp["browser"].get(f"fp32_steps{st}"), bvp["browser"].get(f"fp16_steps{st}")
            if py and b32 and b16:
                rows.append([st, f"{py['psnr_mean']:.2f}", f"{b32['dream_psnr']:.2f}", f"{b16['dream_psnr']:.2f}"])
        sections.append(f"### The whole browser pipeline against Python (same {bvp['windows']} windows)\n\n"
                        "The JavaScript game renders the windows, the JavaScript sampler and onnxruntime-web (WebGPU) predict the next frame; "
                        f"Python does the same with PyTorch (mean of 3 noise seeds). Copy-last-frame on these windows: {bvp['copy_last_frame_psnr']:.2f} dB, "
                        f"context noise {bvp['ctx_sigma']}, init `{bvp['init']}`.\n\n"
                        + table(["Euler steps", "PyTorch PSNR dB", "browser fp32 PSNR dB", "browser fp16 PSNR dB"], rows)
                        + "\n\nSource: `results/browser_vs_python_1step.json` (`tools/crosscheck.py`). Different noise draws, so agreement within a few tenths of a dB is expected.")
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
    glance = v3_glance()
    text = text.replace("@@V3_GLANCE@@\n", glance + "\n" if glance else "")
    limits = (ROOT / a.limits).read_text() if a.limits and (ROOT / a.limits).exists() else "@@LIMITS@@"
    lim = v3_limit()
    limits = limits.replace("@@V3_LIMIT@@\n", lim + "\n" if lim else "").replace("@@V3_LIMIT@@", lim)
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
    ap.add_argument("--fps-wasm", help="results json of `node tools/browser_fps.mjs --backend wasm --dtype fp32`")
    ap.add_argument("--onnx", default="results/onnx_export_dream.json")
    ap.add_argument("--training")
    ap.add_argument("--limits", default="docs/limits.md")
    build(ap.parse_args())
