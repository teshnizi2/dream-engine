"""Distils results/eval_<name>.json into the small json the demo page reads (site/models/eval_summary.json), so every
number the page shows about drift traces to a measured run.   python tools/make_eval_summary.py --eval results/eval_full1.json"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--eval", required=True)
ap.add_argument("--out", default=str(ROOT / "site/models/eval_summary.json"))
a = ap.parse_args()
r = json.loads(pathlib.Path(a.eval).read_text())
ro = r["rollout_euler3"]
summary = {
    "source": a.eval, "checkpoint_step": r["step"], "params": r["params"],
    "psnr_at": {k: v for k, v in ro["psnr_at"].items()},
    "floor": r["baseline_unrelated_frame_psnr"],
    "copy_last": r["baseline_copy_last_frame"],
    "one_step_euler1": r["one_step_samplers_ctx0.1"]["euler1"],
    "horizon": ro["horizon"], "episodes": ro["episodes"],
}
if "validity_nearest_real_frame_psnr" in r:
    summary["validity"] = {k: r["validity_nearest_real_frame_psnr"][k] for k in ("dream_mean", "real_mean")}
pathlib.Path(a.out).write_text(json.dumps(summary, indent=1))
print("wrote", a.out)
