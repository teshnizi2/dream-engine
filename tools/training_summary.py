"""GPU-hour accounting from Slurm (sacct on ALICE): every dream-engine job (names dream*) of this account, one A100 each.
Writes results/training_summary.json.   python tools/training_summary.py --main full1 --main-steps 150000 --main-jobs 5246233,5246234"""
import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--since", default="2026-10-17")
ap.add_argument("--main", default="full1")
ap.add_argument("--main-steps", type=int, required=True)
ap.add_argument("--main-jobs", required=True, help="comma separated Slurm job ids of the main run")
a = ap.parse_args()
cmd = f'sacct -S {a.since} -u $USER -o JobID,JobName%20,Elapsed,State,Start -X -n --parsable2'
raw = subprocess.run(["ssh", "-o", "BatchMode=yes", "alice2", cmd], capture_output=True, text=True, check=True).stdout
jobs = []
for line in raw.strip().splitlines():
    jid, name, elapsed, state, start = line.split("|")
    if not name.startswith("dream"):
        continue                                                       # never count the other builder's rf-* jobs
    parts = elapsed.split("-")
    days = int(parts[0]) if len(parts) == 2 else 0
    h, m, s = (int(x) for x in parts[-1].split(":"))
    hours = days * 24 + h + m / 60 + s / 3600
    jobs.append({"job": jid, "name": name, "elapsed": elapsed, "state": state, "start": start, "gpu_hours": hours})
main_ids = set(a.main_jobs.split(","))
out = {
    "script": "tools/training_summary.py " + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in vars(a).items()),
    "gpu": "NVIDIA A100 80GB PCIe, one per job", "jobs": jobs,
    "main_run": a.main, "main_steps": a.main_steps,
    "main_gpu_hours": sum(j["gpu_hours"] for j in jobs if j["job"] in main_ids),
    "total_gpu_hours": sum(j["gpu_hours"] for j in jobs),
    "n_jobs": len(jobs),
}
(ROOT / "results" / "training_summary.json").write_text(json.dumps(out, indent=1))
print(f"{len(jobs)} jobs, total {out['total_gpu_hours']:.2f} GPU-h, main run {out['main_gpu_hours']:.2f} GPU-h")
