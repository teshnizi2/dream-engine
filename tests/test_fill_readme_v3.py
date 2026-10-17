"""The v3 section of the README (tools/fill_readme.py) is built only from saved results and states the verdict of the pre-registered rule."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))


def populate(tmp_path, adopt):
    import compare_prereg as C
    from tests.test_compare_prereg import fake_eval
    for name, K, game, drop, cf in (("w2_B", 4, "v2", 12, 60), ("v2k8", 8, "v2", 14, 70), ("v3k4", 4, "v3", 14, 300), ("v3k8", 8, "v3", 10, 500),
                                   ("full1", 4, "v2", 16, 60), ("full2", 4, "v3", 20 if adopt else 12, 400 if adopt else 62)):
        fake_eval(tmp_path, name, K, game, drop, cf)
    (tmp_path / "prereg_ablation.json").write_text(json.dumps(C.ablation(tmp_path, "w2_B", ["v2k8", "v3k4", "v3k8"]), default=float))
    (tmp_path / "prereg_final_full2_vs_full1.json").write_text(json.dumps(C.final(tmp_path, "full2", "full1"), default=float))
    (tmp_path / "observability_v2_v3.json").write_text(json.dumps({
        "v2": {"noise_4": {"pos_err_median": 1.89, "within_1.5_cells": 0.46, "heading_within_45deg": 0.71, "n_flat_queries": 1248}},
        "v3": {"noise_4": {"pos_err_median": 0.25, "within_1.5_cells": 0.91, "heading_within_45deg": 0.96, "n_flat_queries": 1248}}}))


def test_v3_section_reports_the_ablation_the_selection_and_an_adopting_verdict(tmp_path):
    import fill_readme as F
    populate(tmp_path, adopt=True)
    text = F.v3_section(tmp_path)
    for needle in ("w2_B", "v2k8", "v3k4", "v3k8", "full1", "full2", "collapse rate", "median", "PREREG-v3.md"):
        assert needle in text, needle
    assert "met the pre-registered rule" in text and "not met" not in text
    assert "not comparable" in text.lower()


def test_v3_section_states_a_negative_result_plainly(tmp_path):
    import fill_readme as F
    populate(tmp_path, adopt=False)
    text = F.v3_section(tmp_path)
    assert "did not meet the pre-registered rule" in text and "v2 stays" in text


def test_v3_section_is_empty_without_results(tmp_path):
    import fill_readme as F
    assert F.v3_section(tmp_path) == ""
