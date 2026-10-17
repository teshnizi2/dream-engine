"""Browser-vs-Python consistency check of the whole demo pipeline (JS game + JS sampler + onnxruntime-web WebGPU) on
identical validation windows. Writes results/browser_vs_python_1step.json.
    python tools/crosscheck.py --ckpt runs/full1_final_ema.pt --ctx-sigma 0.05 --init unit"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.loader import TrajectorySource  # noqa: E402
from export import load_ema  # noqa: E402
from game import raycaster as rc  # noqa: E402
from model import evaluate as ev  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ctx-sigma", type=float, default=0.05)
ap.add_argument("--init", default="unit")
ap.add_argument("--n", type=int, default=64)
a = ap.parse_args()
w = rc.World()
b = TrajectorySource(w, a.n, 5, seed=999).next_batch()
wins = [{"pose": b["poses"][i, 0].tolist(), "actions": b["actions"][i].tolist()} for i in range(a.n)]
tmp = pathlib.Path(tempfile.mkdtemp()) / "wins.json"
tmp.write_text(json.dumps(wins))
val = {"frames": ev._to_unit(b["frames"]), "actions": torch.from_numpy(b["actions"])}
model, step = load_ema(a.ckpt)
out = {"script": "tools/crosscheck.py " + " ".join(sys.argv[1:]), "checkpoint_step": step, "windows": a.n,
       "ctx_sigma": a.ctx_sigma, "init": a.init, "python": {}, "browser": {}}
copy = ev.one_step_metrics(lambda x, s, c, ac, sc: c[:, -1], val, steps=1, ctx_sigma=0.0)
out["copy_last_frame_psnr"] = copy["psnr"]
for steps in (1, 2, 3):
    r = [ev.one_step_metrics(model.denoise, val, steps=steps, ctx_sigma=a.ctx_sigma, init=a.init, seed=s)["psnr"] for s in range(3)]
    out["python"][f"steps{steps}"] = {"psnr_per_noise_seed": r, "psnr_mean": sum(r) / 3}
for dtype in ("fp32", "fp16"):
    for steps in (1, 2, 3):
        res = subprocess.run(["node", str(ROOT / "tools/capture_eval.mjs"), "--windows", str(tmp), "--steps", str(steps), "--dtype", dtype],
                             capture_output=True, text=True, cwd=ROOT)
        line = [l for l in res.stdout.splitlines() if l.startswith("{")][-1]
        out["browser"][f"{dtype}_steps{steps}"] = json.loads(line)
(ROOT / "results" / "browser_vs_python_1step.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v if k != "browser" else {kk: round(vv["dream_psnr"], 3) for kk, vv in v.items()}) for k, v in out.items() if k != "script"}, indent=1)[:1500])
