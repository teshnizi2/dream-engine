#!/usr/bin/env python
"""Export the trained denoiser to ONNX (static shapes, batch 1, opset 17) for onnxruntime-web, optionally convert
to fp16, verify against PyTorch, and write test vectors for the in-browser self-test.

Graph interface (all float32; fp16 variants keep float32 inputs/outputs):
    x [1,3,64,64]     noisy next frame at noise level sigma
    sigma [1]         noise level (the EDM preconditioning is inside the graph)
    ctx [1,12,64,64]  the 4 context frames (already noised with ctx_sigma by the caller), in [-1, 1]
    act [1,4,9]       one-hot actions (move 3 | strafe 3 | turn 3) of the last 4 steps
    ctx_sigma [1]     noise level that was added to the context
    -> denoised [1,3,64,64]  D(x; sigma), clamped to [-1, 1]

    python export.py --ckpt runs/full/final_ema.pt --out site/models --name dream --fp16 --verify
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.denoiser import Denoiser, DenoiserConfig, encode_actions  # noqa: E402

INPUTS = ["x", "sigma", "ctx", "act", "ctx_sigma"]


class OnnxDenoiser(torch.nn.Module):
    def __init__(self, model: Denoiser):
        super().__init__()
        self.m = model

    def forward(self, x, sigma, ctx, act, ctx_sigma):
        return self.m.denoise_flat(x, sigma, ctx, act, ctx_sigma)


def example_inputs(seed=0, sigma=0.7, ctx_sigma=0.1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 3, 64, 64, generator=g)
    ctx = torch.rand(1, 12, 64, 64, generator=g) * 2 - 1
    act = encode_actions(torch.randint(0, 3, (1, 4, 3), generator=g))
    return x, torch.tensor([sigma]), ctx, act, torch.tensor([ctx_sigma])


def export_onnx(model: Denoiser, path: str, opset: int = 17) -> None:
    model = model.eval().cpu()
    torch.onnx.export(OnnxDenoiser(model), example_inputs(), path, input_names=INPUTS, output_names=["denoised"],
                      opset_version=opset, dynamo=False, do_constant_folding=True)


def to_fp16(src: str, dst: str) -> None:
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(src)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, dst)


def describe(path: str) -> dict:
    import onnx
    g = onnx.load(path)
    ops = {}
    for n in g.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return {"n_nodes": len(g.graph.node), "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
            "size_mb": pathlib.Path(path).stat().st_size / 1e6,
            "opset": max(o.version for o in g.opset_import if o.domain in ("", "ai.onnx"))}


def load_ema(ckpt_path: str) -> Denoiser:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck["config"]
    cfg = DenoiserConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()})
    m = Denoiser(cfg)
    m.load_state_dict(ck["ema"])
    return m.eval(), ck.get("step", -1)


def verify(model: Denoiser, onnx_path: str, cases) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for xs in cases:
        with torch.no_grad():
            ref = model.denoise_flat(*xs).numpy()
        out = sess.run(None, {n: v.numpy().astype(np.float32) for n, v in zip(INPUTS, xs)})[0]
        worst = max(worst, float(np.abs(out - ref).max()))
    return {"max_abs_diff_vs_pytorch": worst, "cases": len(cases)}


def real_cases(model: Denoiser, n=3):
    """Test inputs built from real game windows at several noise levels (also saved for the browser self-test)."""
    from game import raycaster as rc
    from model import evaluate as ev
    w = ev.val_windows(rc.World(), n=n, seed=4242)
    cases = []
    for i, (sigma, cs) in enumerate([(5.0, 0.1), (0.3, 0.1), (0.02, 0.0)][:n]):
        ctx = w["frames"][i:i + 1, :4]
        tgt = w["frames"][i:i + 1, 4]
        g = torch.Generator().manual_seed(100 + i)
        x = tgt + torch.randn(tgt.shape, generator=g) * sigma
        act = encode_actions(w["actions"][i:i + 1, :4])
        cases.append((x, torch.tensor([sigma]), ctx.reshape(1, 12, 64, 64), act, torch.tensor([cs])))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="checkpoint with an 'ema' state dict (ckpt_last.pt or final_ema.pt)")
    ap.add_argument("--random-init", action="store_true", help="export random weights (speed benchmarks only)")
    ap.add_argument("--channels", default="32,64,96,128")
    ap.add_argument("--depths", default="2,2,2,2")
    ap.add_argument("--out", default="site/models")
    ap.add_argument("--name", default="dream")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.random_init:
        torch.manual_seed(0)
        cfg = DenoiserConfig(channels=tuple(int(c) for c in a.channels.split(",")),
                             depths=tuple(int(c) for c in a.depths.split(",")),
                             attn=tuple(False for _ in a.channels.split(",")))
        model, step = Denoiser(cfg).eval(), -1
    else:
        model, step = load_ema(a.ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    p32 = out / f"{a.name}_fp32.onnx"
    export_onnx(model, str(p32))
    report = {"script": "python export.py " + " ".join(sys.argv[1:]), "checkpoint_step": step, "params": n_params,
              "fp32": describe(str(p32))}
    if a.fp16:
        p16 = out / f"{a.name}_fp16.onnx"
        to_fp16(str(p32), str(p16))
        report["fp16"] = describe(str(p16))
    if a.verify:
        cases = [example_inputs(0, 0.7, 0.1), example_inputs(1, 0.01, 0.0), example_inputs(2, 5.0, 0.3)]
        if not a.random_init:
            cases = real_cases(model) + cases
        report["verify_fp32"] = verify(model, str(p32), cases)
        if a.fp16:
            report["verify_fp16"] = verify(model, str(p16), cases)
        if not a.random_init:
            # test vectors for the browser self-test: per case [x | ctx | act | out] as float32 + a json index
            blob, index = [], []
            for xs in real_cases(model):
                with torch.no_grad():
                    ref = model.denoise_flat(*xs).numpy()
                parts = [xs[0].numpy(), xs[2].numpy(), xs[3].numpy(), ref]
                index.append({"sigma": float(xs[1][0]), "ctx_sigma": float(xs[4][0]),
                              "sizes": [int(p.size) for p in parts]})
                blob.append(np.concatenate([p.ravel() for p in parts]).astype(np.float32))
            (out / "testvec.bin").write_bytes(np.concatenate(blob).tobytes())
            (out / "testvec.json").write_text(json.dumps(index))
    meta = {"params": n_params, "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in model.cfg.__dict__.items()},
            "checkpoint_step": step, "sigma_max": 5.0, "sigma_min": 2e-3, "rho": 7.0, "sigma_data": model.cfg.sigma_data,
            "sigma_offset_noise": model.cfg.sigma_offset_noise}
    (out / f"{a.name}_meta.json").write_text(json.dumps(meta, indent=1))
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / f"onnx_export_{a.name}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("fp32", "fp16")}, indent=1))
    for k in ("fp32", "fp16"):
        if k in report:
            print(k, "nodes", report[k]["n_nodes"], "size MB", round(report[k]["size_mb"], 2), "ops", report[k]["ops"])


if __name__ == "__main__":
    main()
