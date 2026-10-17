#!/usr/bin/env python
"""How the v3 wave parameters were found (game/make_spec.py: TINT_WAVES).

The tint of game v3 is, per colour channel, amplitude * wave(ax * x + ay * y + phase) of the world position of the wall hit point.
Three (ax, ay, phase) triples are chosen to maximise the smallest colour difference between any two visible wall faces of the same
type (and between neighbouring cells along a wall), under a low-frequency constraint (|ax|, |ay| <= 0.2: periods of at least 5 cells).
Seeded random search followed by hill climbing; the colours are those of `World.face_colour` (face shade x (base + tint)).

    python tools/design_signature.py --amplitude 12          # writes results/signature_design_v3.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from game import raycaster as rc  # noqa: E402

AX_MAX = 0.2                       # cycles per cell along each axis: periods >= 5 cells
FACE_NAMES = ["W", "E", "N", "S"]


def exposed_faces(world):
    """[(cx, cy, face, wall id, centre x, centre y)] for every wall-cell face that borders an open cell: the faces a player can see."""
    g, N = world.grid, world.N
    out = []
    for cy in range(N):
        for cx in range(N):
            if g[cy, cx] == 0:
                continue
            for face, (dx, dy), (px, py) in ((rc.FACE_W, (-1, 0), (cx, cy + .5)), (rc.FACE_E, (1, 0), (cx + 1, cy + .5)),
                                             (rc.FACE_N, (0, -1), (cx + .5, cy)), (rc.FACE_S, (0, 1), (cx + .5, cy + 1))):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < N and 0 <= ny < N and g[ny, nx] == 0:
                    out.append((cx, cy, face, int(g[cy, cx]), px, py))
    return out


def _with_waves(world, waves, amplitude):
    w = rc.World.__new__(rc.World)
    w.__dict__.update(world.__dict__)
    w.tint = {**world.tint, "waves": [list(map(float, v)) for v in waves], "amplitude": float(amplitude)}
    return w


def face_colours(world, faces, waves, amplitude):
    """[n, 3] colours of the faces (shade x (primary colour + gain * tint at the face centre)) for the given wave parameters."""
    w = _with_waves(world, waves, amplitude)
    F = np.array(faces, dtype=float)
    wid, face = F[:, 3].astype(int), F[:, 2].astype(int)
    base = w.rgb1[wid] + w.tint_gain[wid][:, None] * w.tint_offsets(F[:, 4], F[:, 5])
    return base * np.asarray(w.tint["face_shade"])[face][:, None]


def _neighbours(faces):
    key = {(int(f[0]), int(f[1]), int(f[2])): i for i, f in enumerate(faces)}
    pairs = []
    for (cx, cy, face), i in key.items():
        j = key.get((cx + 1, cy, face)) if face in (rc.FACE_N, rc.FACE_S) else key.get((cx, cy + 1, face))
        if j is not None:
            pairs.append((i, j))
    return np.array(pairs)


def separation(world, faces, waves, amplitude):
    """Smallest colour distance between two faces of the same wall type, and between adjacent cells along a wall."""
    col = face_colours(world, faces, waves, amplitude)
    wid = np.array([f[3] for f in faces])
    dmin = np.inf
    for t in np.unique(wid):
        c = col[wid == t]
        if len(c) > 1:
            d = np.linalg.norm(c[:, None] - c[None], axis=-1)
            dmin = min(dmin, d[np.triu_indices(len(c), 1)].min())
    nb = _neighbours(faces)
    steps = np.linalg.norm(col[nb[:, 0]] - col[nb[:, 1]], axis=1)
    return {"min_face_distance": float(dmin), "min_neighbour_step": float(steps.min()), "p10_neighbour_step": float(np.percentile(steps, 10))}


def _score(world, faces, waves, amplitude):
    s = separation(world, faces, waves, amplitude)
    return min(s["min_face_distance"], s["min_neighbour_step"])


def search(world, amplitude=12.0, seed=0, n_random=30000, n_climb=6000, rng=None):
    """Random search over (ax, ay, phase) per channel, then hill climbing from the best; returns the waves rounded to 4 decimals.
    Pass `rng` to continue one random stream across several amplitudes (what `main` does)."""
    rng = np.random.default_rng(seed) if rng is None else rng
    faces = exposed_faces(world)
    best, best_s = None, -1.0
    for _ in range(n_random):
        par = np.stack([rng.uniform(-AX_MAX, AX_MAX, 3), rng.uniform(-AX_MAX, AX_MAX, 3), rng.uniform(0, 1, 3)], 1)
        s = _score(world, faces, par, amplitude)
        if s > best_s:
            best, best_s = par, s
    after_random = best_s
    step = 0.02
    for _ in range(n_climb):
        cand = best + rng.normal(0, step, best.shape) * np.array([1, 1, 2.5])
        cand[:, :2] = np.clip(cand[:, :2], -AX_MAX, AX_MAX)
        s = _score(world, faces, cand, amplitude)
        if s > best_s:
            best, best_s = cand, s
    waves = np.round(best, 4).tolist()
    return {"amplitude": amplitude, "waves": waves, "score": float(best_s), "score_after_random_search": float(after_random),
            **separation(world, faces, waves, amplitude)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amplitude", type=float, nargs="+", default=[8.0, 10.0, 12.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=30000)
    ap.add_argument("--n-climb", type=int, default=6000)
    a = ap.parse_args()
    world = rc.World(rc.load_spec(ROOT / "game" / "spec_v3.json"))
    faces = exposed_faces(world)
    shipped = world.tint
    rng = np.random.default_rng(a.seed)               # one stream for the amplitudes in turn: reproduces the shipped A = 12 waves
    out = {"script": "python tools/design_signature.py " + " ".join(sys.argv[1:]), "visible_faces": len(faces),
           "faces_by_wall_id": {str(t): int(sum(1 for f in faces if f[3] == t)) for t in sorted({f[3] for f in faces})},
           "constraint": f"|ax|, |ay| <= {AX_MAX} (periods >= {1 / AX_MAX:g} cells); score = min(smallest same-type face distance, smallest neighbour step)",
           "shipped": {"amplitude": shipped["amplitude"], "waves": shipped["waves"], **separation(world, faces, shipped["waves"], shipped["amplitude"])},
           "seed": a.seed, "search": [search(world, amp, a.seed, a.n_random, a.n_climb, rng=rng) for amp in a.amplitude]}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "signature_design_v3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"shipped": out["shipped"], "search": [{k: v for k, v in s.items() if k != "waves"} for s in out["search"]]}, indent=1))


if __name__ == "__main__":
    main()
