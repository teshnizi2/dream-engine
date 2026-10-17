"""On-the-fly trajectory generation (infinite data) for training and fixed trajectories for evaluation.

The game is cheap to simulate, so nothing is stored on disk (the Mac has ~10 GB free; 1M frames of
64x64x3 uint8 would be 12 GB). `TrajectorySource` yields short windows from independent games started
at random poses; `BatchStream` wraps it as a torch IterableDataset so worker processes generate in
parallel with the GPU.

    python -m data.loader --bench      # measure throughput, writes results/data_throughput.json
"""
import argparse
import json
import pathlib
import time

import numpy as np

from game import raycaster as rc
from data.policy import ExplorationPolicy

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TrajectorySource:
    def __init__(self, world, batch_size, seq_len, seed, warmup_max=40, respawn_prob=0.0):
        self.world, self.B, self.T = world, batch_size, seq_len
        self.rng = np.random.default_rng(seed)
        self.policy = ExplorationPolicy(world, batch_size, self.rng, respawn_prob=respawn_prob)
        self.warmup_max = warmup_max

    def next_batch(self):
        w, B, T = self.world, self.B, self.T
        st = self.policy.random_states(B)
        self.policy.reset()
        for _ in range(int(self.rng.integers(0, self.warmup_max + 1))):
            st = w.step(st, self.policy.act(st))
        poses = [np.stack(st, 1)]
        frames = [w.render(st)]
        acts = []
        for _ in range(T - 1):
            a = self.policy.act(st)
            st = w.step(st, a)
            st = self.policy.maybe_respawn(st)
            acts.append(a)
            poses.append(np.stack(st, 1))
            frames.append(w.render(st))
        return {
            "frames": np.stack(frames, 1),           # uint8 [B, T, 64, 64, 3]
            "actions": np.stack(acts, 1),            # int64 [B, T-1, 3]; actions[:, t] leads frame t -> t+1
            "poses": np.stack(poses, 1),             # float64 [B, T, 4] = x, y, dx, dy
        }


def eval_trajectories(world, n, length, seed=123, warmup=0):
    """Fixed exploration-policy trajectories (no respawn) for validation / drift curves."""
    src = TrajectorySource(world, n, length, seed, warmup_max=warmup, respawn_prob=0.0)
    return src.next_batch()


try:
    import torch

    class BatchStream(torch.utils.data.IterableDataset):
        """Yields dicts of torch tensors forever; each DataLoader worker gets its own seed."""

        def __init__(self, batch_size, seq_len, base_seed=0):
            self.batch_size, self.seq_len, self.base_seed = batch_size, seq_len, base_seed

        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            src = TrajectorySource(rc.World(), self.batch_size, self.seq_len, seed=self.base_seed * 1000 + wid)
            while True:
                b = src.next_batch()
                yield {"frames": torch.from_numpy(b["frames"]), "actions": torch.from_numpy(b["actions"])}
except ImportError:  # data generation works without torch
    pass


def _bench():
    w = rc.World()
    out = {"script": "python -m data.loader --bench"}
    for B in (32, 128, 512):
        src = TrajectorySource(w, B, 5, seed=0)
        src.next_batch()
        t = time.time()
        n = 0
        while time.time() - t < 3:
            src.next_batch()
            n += B * 5
        out[f"frames_per_sec_single_process_B{B}"] = n / (time.time() - t)
    t = time.time()
    st = TrajectorySource(w, 512, 2, 0).policy.random_states(512)
    k = 0
    while time.time() - t < 2:
        w.render(st)
        k += 512
    out["render_only_frames_per_sec_B512"] = k / (time.time() - t)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_throughput.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    if ap.parse_args().bench:
        _bench()
