"""tools/compare_prereg.py: the selection rule and the adoption rule of docs/PREREG-v3.md, applied to saved evaluations."""
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def row(h, c, K=4, game="v2"):
    return {"horizon_median": h, "collapse": c, "K": K, "game": game}


# ---------------------------------------------------------------- which variant gets the full run

def test_the_eligible_variant_with_the_lowest_collapse_rate_wins():
    import compare_prereg as C
    base = row(12, 0.20)
    r = C.select_variant({"a": row(14, 0.18, 8), "b": row(13, 0.08, 4, "v3"), "c": row(10, 0.02, 8, "v3")}, base)
    assert r["winner"] == "b" and r["eligible"] == ["a", "b"] and "c" not in r["eligible"]


def test_a_variant_with_a_shorter_horizon_than_the_baseline_is_not_eligible_even_if_it_collapses_least():
    import compare_prereg as C
    r = C.select_variant({"a": row(11, 0.01), "b": row(12, 0.10, 4, "v3")}, row(12, 0.20))
    assert r["winner"] == "b"


def test_collapse_rates_within_0_02_are_a_tie_broken_by_the_longer_horizon():
    import compare_prereg as C
    r = C.select_variant({"a": row(15, 0.10, 8), "b": row(13, 0.09, 4, "v3"), "c": row(20, 0.105, 8, "v3"), "d": row(40, 0.12, 8, "v3")}, row(12, 0.20))
    assert r["winner"] == "c" and r["tied"] == ["a", "b", "c"]            # d is 0.03 above the lowest rate: not in the tie, whatever its horizon
    r = C.select_variant({"a": row(15, 0.10, 8), "b": row(13, 0.07, 4, "v3")}, row(12, 0.20))          # 0.03 apart: not a tie
    assert r["winner"] == "b"


def test_ties_in_collapse_and_horizon_prefer_the_smaller_context_then_the_v3_game():
    import compare_prereg as C
    r = C.select_variant({"a": row(15, 0.10, 8, "v2"), "b": row(15, 0.10, 4, "v3")}, row(12, 0.20))
    assert r["winner"] == "b"
    r = C.select_variant({"a": row(15, 0.10, 4, "v2"), "b": row(15, 0.10, 4, "v3")}, row(12, 0.20))
    assert r["winner"] == "b"


def test_when_no_variant_is_eligible_the_lowest_collapse_rate_overall_goes_ahead():
    import compare_prereg as C
    r = C.select_variant({"a": row(8, 0.15), "b": row(9, 0.12, 4, "v3")}, row(12, 0.20))
    assert r["winner"] == "b" and r["eligible"] == [] and "no variant" in r["reason"]


# ---------------------------------------------------------------- adoption

def test_adoption_needs_half_the_collapse_rate_and_a_horizon_that_is_not_shorter():
    import compare_prereg as C
    v2 = row(16, 0.18)
    assert C.adoption(row(16, 0.09), v2)["adopt"] is True                       # exactly half, equal horizon: both pass
    assert C.adoption(row(30, 0.05), v2)["adopt"] is True
    r = C.adoption(row(15, 0.05), v2)
    assert r["adopt"] is False and r["A_collapse_at_most_half"] is True and r["B_horizon_not_shorter"] is False
    r = C.adoption(row(20, 0.10), v2)
    assert r["adopt"] is False and r["A_collapse_at_most_half"] is False and r["B_horizon_not_shorter"] is True


# ---------------------------------------------------------------- paired bootstrap over episodes

def test_paired_bootstrap_intervals_are_reproducible_and_bracket_the_point_estimates():
    import compare_prereg as C
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 0.1, 64)
    b = a * 2 + rng.uniform(0.0, 0.05, 64)
    lo, hi = C.paired_ratio_ci(a, b, seed=1)
    assert lo < a.mean() / b.mean() < hi and (lo, hi) == C.paired_ratio_ci(a, b, seed=1)
    ha = rng.integers(5, 60, 64)
    hb = ha - 3
    lo, hi = C.paired_median_diff_ci(ha, hb, seed=1)
    assert lo <= np.median(ha) - np.median(hb) <= hi


def test_the_real_game_flat_flags_of_two_evaluations_are_compared_for_the_pairing_check():
    import compare_prereg as C
    a = np.full((4, 100), 0.3)
    a[0, 10:20] = 0.0
    b = a.copy()
    assert C.real_flags_agree(a, b) == 1.0
    b[1, 5] = 0.0
    assert C.real_flags_agree(a, b) == pytest.approx(1 - 1 / 400)


# ---------------------------------------------------------------- end to end on saved evaluations

def fake_eval(tmp_path, name, K, game, horizon_drop, collapse_from, E=8, L=520):
    psnr = np.full((E, L), 30.0, dtype=np.float32)
    psnr[:, horizon_drop:] = 12.0
    bg_real = np.full((E, L), 0.3, dtype=np.float32)
    bg_dream = bg_real.copy()
    bg_dream[:, collapse_from:] = 0.0
    np.savez(tmp_path / f"prereg_{name}.npz", psnr=psnr, bg_dream=bg_dream, bg_real=bg_real)
    import eval_prereg as P
    j = {"num_ctx": K, "game": f"game/spec_{game}.json", "checkpoint": f"runs/{name}/final_ema.pt", "step": 1,
         **P.analyze(psnr, bg_dream, bg_real), "spin_360": {"dream_frame72_vs_start_view_psnr": 15.0, "real_frame72_vs_start_view_psnr": 100.0},
         "one_step": {"psnr": 33.0, "ssim": 0.9, "copy_last_psnr": 24.0}, "baseline_unrelated_frame_psnr": 12.0,
         "baseline_freeze_last_real_frame": {"psnr_at": {"8": 18.0}}}
    (tmp_path / f"prereg_{name}.json").write_text(json.dumps(j))


def test_compare_runs_on_saved_evaluations_applies_both_rules(tmp_path):
    import compare_prereg as C
    fake_eval(tmp_path, "w2_B", 4, "v2", horizon_drop=12, collapse_from=60)
    fake_eval(tmp_path, "v2k8", 8, "v2", horizon_drop=14, collapse_from=70)
    fake_eval(tmp_path, "v3k4", 4, "v3", horizon_drop=14, collapse_from=300)
    fake_eval(tmp_path, "v3k8", 8, "v3", horizon_drop=10, collapse_from=500)
    out = C.ablation(tmp_path, baseline="w2_B", variants=["v2k8", "v3k4", "v3k8"])
    assert out["selection"]["winner"] == "v3k4"                       # v3k8 collapses least but its horizon is shorter than the baseline's
    assert set(out["runs"]) == {"w2_B", "v2k8", "v3k4", "v3k8"} and out["runs"]["v3k4"]["K"] == 4 and out["runs"]["v3k4"]["game"] == "v3"
    assert out["pairing_check_real_flat_flags_agree"] == {"v2k8": 1.0, "v3k4": 1.0, "v3k8": 1.0}      # every run saw the same real frames
    pv = out["paired_vs_baseline"]["v3k4"]                                                     # paired over the same episodes
    assert pv["collapse_ratio"] < 1 and pv["collapse_ratio_ci95"][0] <= pv["collapse_ratio"] <= pv["collapse_ratio_ci95"][1]
    assert pv["horizon_median_difference"] == 2 and len(pv["horizon_median_difference_ci95"]) == 2
    fake_eval(tmp_path, "full1", 4, "v2", horizon_drop=16, collapse_from=60)
    fake_eval(tmp_path, "full2", 4, "v3", horizon_drop=20, collapse_from=400)
    fin = C.final(tmp_path, "full2", "full1")
    assert fin["adoption"]["adopt"] is True and fin["pairing_check_real_flat_flags_agree"] == 1.0
    assert 0 < fin["paired_ratio_ci95"][0] <= fin["collapse_ratio"] <= fin["paired_ratio_ci95"][1]
