"""tools/design_signature.py (how the v3 wave parameters were found) and tools/observability.py (can a flat-wall view tell where
you are?): the evidence for game v3 that needs no trained model."""
import json
import pathlib
import sys

import numpy as np
import pytest

from game import raycaster as rc

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture(scope="module")
def world3():
    return rc.World(rc.load_spec(ROOT / "game" / "spec_v3.json"))


# ---------------------------------------------------------------- design_signature

def test_the_committed_v3_waves_give_every_visible_face_its_own_colour(world3):
    import design_signature as D
    faces = D.exposed_faces(world3)
    assert len(faces) == 142
    sep = D.separation(world3, faces, world3.tint["waves"], world3.tint["amplitude"])
    assert sep["min_face_distance"] >= 3.5 and sep["min_neighbour_step"] >= 4.0, sep


def test_face_colours_in_the_design_tool_match_the_games_own_face_colour(world3):
    import design_signature as D
    faces = D.exposed_faces(world3)
    cols = D.face_colours(world3, faces, world3.tint["waves"], world3.tint["amplitude"])
    for i in (0, 17, 60, 141):
        cx, cy, face = faces[i][:3]
        assert np.allclose(cols[i], world3.face_colour(int(cx), int(cy), int(face)))


def test_a_random_search_with_hill_climbing_never_gets_worse_and_is_reproducible(world3):
    import design_signature as D
    a = D.search(world3, amplitude=12.0, seed=3, n_random=40, n_climb=30)
    b = D.search(world3, amplitude=12.0, seed=3, n_random=40, n_climb=30)
    assert a["waves"] == b["waves"] and a["score"] == b["score"]
    assert a["score"] >= a["score_after_random_search"]
    assert all(abs(w[0]) <= 0.2 + 1e-12 and abs(w[1]) <= 0.2 + 1e-12 for w in a["waves"]), "periods of at least 5 cells (low frequency)"


# ---------------------------------------------------------------- observability

def test_flat_wall_views_localise_the_player_much_better_in_v3_than_in_v2():
    import observability as O
    r2 = O.observability(rc.World(rc.load_spec(ROOT / "game" / "spec_v2.json")), n_bank=3000, n_query=800, noise=4.0, seed=0)
    r3 = O.observability(rc.World(rc.load_spec(ROOT / "game" / "spec_v3.json")), n_bank=3000, n_query=800, noise=4.0, seed=0)
    assert r2["n_flat_queries"] >= 100 and r3["n_flat_queries"] >= 100
    assert r3["within_1.5_cells"] > r2["within_1.5_cells"] + 0.2
    assert r3["heading_within_45deg"] > r2["heading_within_45deg"] + 0.1
    assert r3["pos_err_median"] < r2["pos_err_median"]


def test_the_saved_design_run_reproduces_the_shipped_waves():
    """results/signature_design_v3.json is the output of `python tools/design_signature.py` (seed 0): its amplitude-12 search is
    the tint that game/make_spec.py ships."""
    d = json.loads((ROOT / "results" / "signature_design_v3.json").read_text())
    spec = rc.load_spec(ROOT / "game" / "spec_v3.json")
    run = next(s for s in d["search"] if s["amplitude"] == spec["tint"]["amplitude"])
    assert run["waves"] == spec["tint"]["waves"] and d["shipped"]["waves"] == spec["tint"]["waves"] and d["seed"] == 0
