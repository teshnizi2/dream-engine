"""tools/signature_use.py: does a v3 model use the colour signature to localise? (exploratory; PREREG deviation log 5)"""
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))


def test_the_shifted_game_differs_only_in_the_tint_phase():
    import signature_use as S
    from game import raycaster as rc
    spec = rc.load_spec(ROOT / "game" / "spec_v3.json")
    shifted = S.shifted_spec(spec)
    assert shifted["map"] == spec["map"] and shifted["tint"]["face_shade"] == spec["tint"]["face_shade"]
    assert shifted["tint"]["amplitude"] == spec["tint"]["amplitude"]
    assert all(a[:2] == b[:2] and a[2] != b[2] for a, b in zip(shifted["tint"]["waves"], spec["tint"]["waves"]))
    w, ws = rc.World(spec), rc.World(shifted)
    st = rc.state_from_pose(np.array([8.5]), np.array([5.5]), np.radians(np.array([-90.0])))
    a, b = w.render(st), ws.render(st)
    assert not np.array_equal(a, b)                                  # a different colour code ...
    assert np.array_equal(w.bg_rows, ws.bg_rows)                     # ... on the same geometry


def test_a_denoiser_that_only_copies_colours_does_not_care_which_signature_it_sees():
    import signature_use as S

    def copy_last(x, sigma, ctx, act, s_c):
        return ctx[:, -1]

    r = S.signature_use(copy_last, K=4, n=6, steps=1, ctx_sigma=0.0, init="edm")
    assert abs(r["true_signature"]["dream_psnr_at"]["8"] - r["shifted_signature"]["dream_psnr_at"]["8"]) < 1.5
    assert set(r) >= {"true_signature", "shifted_signature", "psnr_drop_at_8", "psnr_drop_mean"}
