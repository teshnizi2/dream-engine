"""Generates the fixture the JS twin is tested against (tests/parity.test.mjs).

Fixed seeds. Python is the reference: it records initial states, action sequences, the state after every
step and every rendered frame. The JS test replays the same actions and compares states and pixels.
Fixtures are written to tests/fixtures/ (gitignored, ~15 MB) and regenerated on demand.

    python tests/make_parity_fixture.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from game import raycaster as rc  # noqa: E402
from model.sampler import build_sigmas  # noqa: E402

OUT = ROOT / "tests" / "fixtures"


def sticky_actions(rng, T, N):
    """Sticky random (move, strafe, turn) triples, shape [T, N, 3]."""
    acts = np.zeros((T, N, 3), dtype=np.int64)
    cur = np.stack([rng.integers(0, 3, N), rng.integers(0, 3, N), rng.integers(0, 3, N)], 1)
    for t in range(T):
        change = rng.random((N, 3)) < 0.15
        new = np.stack([rng.integers(0, 3, N), rng.integers(0, 3, N), rng.integers(0, 3, N)], 1)
        cur = np.where(change, new, cur)
        acts[t] = cur
    return acts


def to_lists(st):
    return {k: np.asarray(v).tolist() for k, v in st._asdict().items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w = rc.World()
    rng = np.random.default_rng(1234)
    free = w.free_cells()
    N, T = 8, 100
    idx = rng.integers(0, len(free), N)
    st = rc.state_from_pose(free[idx, 0] + rng.uniform(0.25, 0.75, N), free[idx, 1] + rng.uniform(0.25, 0.75, N),
                            rng.uniform(0, 2 * np.pi, N))
    acts = sticky_actions(rng, T, N)
    states = [st]
    frames = [w.render(st)]
    for t in range(T):
        st = w.step(st, acts[t])
        states.append(st)
        frames.append(w.render(st))
    traj_frames = np.stack(frames)                           # [T+1, N, 64, 64, 3]

    # random poses, including axis-aligned headings and exact cell-boundary positions
    M = 300
    idx = rng.integers(0, len(free), M)
    px = free[idx, 0] + rng.uniform(0.0, 1.0, M)
    py = free[idx, 1] + rng.uniform(0.0, 1.0, M)
    ang = rng.uniform(0, 2 * np.pi, M)
    px[:20] = free[idx[:20], 0] + 0.5
    py[:20] = free[idx[:20], 1] + 0.5
    ang[:20] = np.radians(np.arange(20) * 90.0)              # 0, 90, 180, 270, 0, ... (exact dx or dy ~ 6e-17)
    px[20:30] = free[idx[20:30], 0]                          # exactly on a cell boundary
    py[20:30] = free[idx[20:30], 1]
    pose = rc.state_from_pose(px, py, ang)
    pose_frames = w.render(pose)
    # exact-zero direction components (rdx == 0 on the centre ray)
    ex = rc.State(np.array([2.5, 8.5, 8.5, 3.5]), np.array([7.5, 9.5, 5.5, 7.5]),
                  np.array([1.0, 0.0, 0.0, -1.0]), np.array([0.0, -1.0, 1.0, 0.0]))
    edge_frames = w.render(ex)

    spec = rc.load_spec()
    meta = {
        "n_traj": N, "T": T, "n_pose": M,
        "actions": acts.tolist(),
        "init": to_lists(states[0]),
        "states": {k: [np.asarray(getattr(s, k)).tolist() for s in states] for k in rc.State._fields},
        "pose": to_lists(pose),
        "edge": to_lists(ex),
        "frame_shape": [spec["height"], spec["width"], 3],
        "sigmas": {str(n): build_sigmas(n, 2e-3, 5.0, 7.0).double().tolist() for n in (1, 2, 3)},
    }
    (OUT / "parity.json").write_text(json.dumps(meta))
    blob = np.concatenate([traj_frames.reshape(-1, 64, 64, 3), pose_frames, edge_frames]).astype(np.uint8)
    (OUT / "parity_frames.bin").write_bytes(blob.tobytes())
    print(f"wrote {OUT} frames={blob.shape[0]} ({blob.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
