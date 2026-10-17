"""ONNX export: static batch-1 graph, opset >= 17, numerically equal to PyTorch (fp32) and close in fp16."""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

import export as ex
from model.denoiser import Denoiser, DenoiserConfig, encode_actions
from model.unet import ResBlock, SelfAttention2d

# unusual widths on purpose: exercises GroupNorm group fallback and the skip-connection concatenations
CFG = DenoiserConfig(channels=(16, 32, 48, 64), depths=(1, 1, 1, 1), cond_channels=64)


def make_model():
    torch.manual_seed(0)
    m = Denoiser(CFG).eval()
    torch.nn.init.normal_(m.inner.conv_out.weight, std=0.05)          # fresh nets ignore conditioning: randomise
    for mod in m.modules():
        if isinstance(mod, ResBlock):
            torch.nn.init.normal_(mod.conv2.weight, std=0.05)
        if isinstance(mod, SelfAttention2d):
            torch.nn.init.normal_(mod.out_proj.weight, std=0.05)
    return m


def inputs(seed=0, sigma=0.7, ctx_sigma=0.1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 3, 64, 64, generator=g)
    ctx = torch.rand(1, 12, 64, 64, generator=g) * 2 - 1
    act = encode_actions(torch.randint(0, 3, (1, 4, 3), generator=g))
    return x, torch.tensor([sigma]), ctx, act, torch.tensor([ctx_sigma])


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    m = make_model()
    p32 = d / "m32.onnx"
    ex.export_onnx(m, str(p32))
    return m, p32, d


def run(path, xs):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    return sess, sess.run(None, {n: v.numpy().astype(np.float32) for n, v in zip(names, xs)})[0]


def test_graph_interface_is_static_batch_one_opset_17_plus(exported):
    _, p, _ = exported
    g = onnx.load(str(p))
    assert max(o.version for o in g.opset_import if o.domain in ("", "ai.onnx")) >= 17
    ins = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in g.graph.input}
    assert ins == {"x": [1, 3, 64, 64], "sigma": [1], "ctx": [1, 12, 64, 64], "act": [1, 4, 9], "ctx_sigma": [1]}
    outs = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim] for o in g.graph.output}
    assert outs == {"denoised": [1, 3, 64, 64]}
    ops = {n.op_type for n in g.graph.node}
    assert "GroupNormalization" not in ops, "not supported by the onnxruntime-web WebGPU EP"


def test_fp32_onnx_matches_pytorch(exported):
    m, p, _ = exported
    for seed, sigma, cs in [(0, 0.7, 0.1), (1, 0.01, 0.0), (2, 5.0, 0.3)]:
        xs = inputs(seed, sigma, cs)
        with torch.no_grad():
            ref = m.denoise_flat(*xs).numpy()
        _, out = run(p, xs)
        assert np.abs(out - ref).max() < 2e-4, (seed, np.abs(out - ref).max())


def test_fp16_onnx_stays_close_to_fp32(exported):
    m, p, d = exported
    p16 = d / "m16.onnx"
    ex.to_fp16(str(p), str(p16))
    xs = inputs(3, 0.5, 0.1)
    with torch.no_grad():
        ref = m.denoise_flat(*xs).numpy()
    _, out = run(p16, xs)
    assert np.abs(out - ref).max() < 3e-2, np.abs(out - ref).max()
    assert p16.stat().st_size < 0.6 * p.stat().st_size


def test_op_histogram_and_size_are_reported(exported):
    _, p, _ = exported
    info = ex.describe(str(p))
    assert info["n_nodes"] > 50 and "Conv" in info["ops"] and info["size_mb"] > 0
