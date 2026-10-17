#!/usr/bin/env python
"""Applies the selection rule and the adoption rule of docs/PREREG-v3.md to saved evaluations (results/prereg_<run>.json + .npz).

    python tools/compare_prereg.py ablation --baseline w2_B --variants v2k8 v3k4 v3k8     # which variant gets the full run
    python tools/compare_prereg.py final --new full2 --ref full1                          # the adoption rule
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import flatwall as fw  # noqa: E402

TIE = 0.02          # collapse rates closer than this (absolute) are a tie


def _game(path: str) -> str:
    return "v3" if "v3" in pathlib.Path(path).name else "v2"


def load_row(res: pathlib.Path, name: str) -> dict:
    j = json.loads((res / f"prereg_{name}.json").read_text())
    return {"name": name, "K": j["num_ctx"], "game": _game(j["game"]), "steps": j.get("step"),
            "horizon_median": j["tracking_horizon_20db"]["median"], "horizon_p25": j["tracking_horizon_20db"]["p25"],
            "horizon_p75": j["tracking_horizon_20db"]["p75"], "horizon_median_ci95": j["tracking_horizon_20db"]["median_ci95"],
            "horizon_25db_median": j["tracking_horizon_25db"]["median"],
            "collapse": j["collapse"]["rate"], "collapse_ci95": j["collapse"]["rate_ci95"],
            "collapse_window1": j["collapse_plain"]["rate"], "dream_stuck_share": j["collapse"]["dream_stuck_share"],
            "real_stuck_share": j["collapse"]["real_stuck_share"], "dream_flat_share": j["dream_flat_share"],
            "real_flat_share": j["real_flat_share"], "spin_return_psnr": j["spin_360"]["dream_frame72_vs_start_view_psnr"],
            "one_step_psnr": j["one_step"]["psnr"], "copy_last_psnr": j["one_step"]["copy_last_psnr"],
            "mean_psnr_first32": j["mean_psnr_first32"], "unrelated_frame_psnr": j.get("baseline_unrelated_frame_psnr"),
            "episodes": j["tracking_horizon_20db"]["episodes"]}


def select_variant(variants: dict, baseline: dict) -> dict:
    """The rule of section 5: eligible = median horizon not below the baseline's; lowest collapse rate wins; rates within 0.02 are a tie
    broken by the longer median horizon, then the smaller K, then v3 over v2; if none is eligible, the lowest collapse rate overall."""
    eligible = [n for n, v in variants.items() if v["horizon_median"] >= baseline["horizon_median"]]
    pool = eligible or list(variants)
    best = min(variants[n]["collapse"] for n in pool)
    tied = [n for n in pool if variants[n]["collapse"] - best < TIE]
    winner = sorted(tied, key=lambda n: (-variants[n]["horizon_median"], variants[n]["K"], 0 if variants[n]["game"] == "v3" else 1))[0]
    reason = (f"{len(eligible)} of {len(variants)} variants have a median horizon of at least the baseline's ({baseline['horizon_median']}); "
              if eligible else "no variant has a median horizon of at least the baseline's: the lowest collapse rate overall goes ahead; ")
    reason += f"collapse rates within {TIE} of the lowest ({best:.3f}): {tied}; chosen: {winner}"
    return {"winner": winner, "eligible": eligible, "tied": tied, "reason": reason}


def adoption(new: dict, ref: dict) -> dict:
    a = new["collapse"] <= 0.5 * ref["collapse"]
    b = new["horizon_median"] >= ref["horizon_median"]
    return {"adopt": bool(a and b), "A_collapse_at_most_half": bool(a), "B_horizon_not_shorter": bool(b)}


def paired_ratio_ci(a, b, n=4000, seed=0, alpha=0.05):
    """Interval of mean(a) / mean(b) when episodes are resampled together (both models see the same paths)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a[idx].mean(1) / b[idx].mean(1)
    r = r[np.isfinite(r)]
    return float(np.quantile(r, alpha / 2)), float(np.quantile(r, 1 - alpha / 2))


def paired_median_diff_ci(a, b, n=4000, seed=0, alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    d = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def real_flags_agree(bg_real_a, bg_real_b) -> float:
    """Share of (episode, frame) where the real frames of two evaluations agree on 'flat wall' (1.0 if the pairing is intact)."""
    return float((fw.flat_flags(bg_real_a) == fw.flat_flags(bg_real_b)).mean())


def _paired(res, new: str, ref: str, rows: dict) -> dict:
    """Collapse-rate ratio and median-horizon difference of `new` against `ref` with intervals that resample the episodes jointly."""
    jn, jr = (json.loads((res / f"prereg_{n}.json").read_text()) for n in (new, ref))
    ratio = rows[new]["collapse"] / rows[ref]["collapse"] if rows[ref]["collapse"] else float("nan")
    return {"collapse_ratio": ratio, "collapse_ratio_ci95": list(paired_ratio_ci(jn["collapse"]["per_episode"], jr["collapse"]["per_episode"])),
            "horizon_median_difference": rows[new]["horizon_median"] - rows[ref]["horizon_median"],
            "horizon_median_difference_ci95": list(paired_median_diff_ci(jn["tracking_horizon_20db"]["per_episode"], jr["tracking_horizon_20db"]["per_episode"]))}


def ablation(res, baseline: str, variants: list) -> dict:
    res = pathlib.Path(res)
    rows = {n: load_row(res, n) for n in [baseline, *variants]}
    sel = select_variant({n: rows[n] for n in variants}, rows[baseline])
    base_bg = np.load(res / f"prereg_{baseline}.npz")["bg_real"]
    pairing = {n: real_flags_agree(np.load(res / f"prereg_{n}.npz")["bg_real"], base_bg) for n in variants}
    return {"baseline": baseline, "runs": rows, "selection": sel, "pairing_check_real_flat_flags_agree": pairing,
            "paired_vs_baseline": {n: _paired(res, n, baseline, rows) for n in variants}}


def final(res, new: str, ref: str) -> dict:
    res = pathlib.Path(res)
    jn, jr = (json.loads((res / f"prereg_{n}.json").read_text()) for n in (new, ref))
    rn, rr = load_row(res, new), load_row(res, ref)
    zn, zr = np.load(res / f"prereg_{new}.npz"), np.load(res / f"prereg_{ref}.npz")
    pe_n, pe_r = jn["collapse"]["per_episode"], jr["collapse"]["per_episode"]
    h_n, h_r = jn["tracking_horizon_20db"]["per_episode"], jr["tracking_horizon_20db"]["per_episode"]
    return {"new": rn, "ref": rr, "adoption": adoption(rn, rr),
            "collapse_ratio": rn["collapse"] / rr["collapse"] if rr["collapse"] else float("nan"),
            "paired_ratio_ci95": list(paired_ratio_ci(pe_n, pe_r)),
            "horizon_median_difference": rn["horizon_median"] - rr["horizon_median"],
            "paired_median_difference_ci95": list(paired_median_diff_ci(h_n, h_r)),
            "pairing_check_real_flat_flags_agree": real_flags_agree(zn["bg_real"], zr["bg_real"])}


def table(rows: dict) -> str:
    cols = ["name", "game", "K", "steps", "horizon_median", "collapse", "collapse_window1", "spin_return_psnr", "one_step_psnr", "mean_psnr_first32"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows.values():
        out.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ablation")
    a.add_argument("--baseline", required=True)
    a.add_argument("--variants", nargs="+", required=True)
    f = sub.add_parser("final")
    f.add_argument("--new", required=True)
    f.add_argument("--ref", required=True)
    args = ap.parse_args()
    res = ROOT / "results"
    if args.cmd == "ablation":
        out = ablation(res, args.baseline, args.variants)
        (res / "prereg_ablation.json").write_text(json.dumps(out, indent=1))
        print(table(out["runs"]))
        print(out["selection"]["reason"])
    else:
        out = final(res, args.new, args.ref)
        (res / f"prereg_final_{args.new}_vs_{args.ref}.json").write_text(json.dumps(out, indent=1))
        print(table({"ref": out["ref"], "new": out["new"]}))
        print(json.dumps({k: v for k, v in out.items() if k not in ("new", "ref")}, indent=1))


if __name__ == "__main__":
    main()
