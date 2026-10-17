"""Debug helper: expose every node output of an ONNX graph as a graph output, so the browser can compare each
intermediate tensor between the WebGPU and WASM execution providers and find the first diverging op.
    python tools/make_debug_model.py site/models/dream_fp32.onnx tools/bench_models/debug_all.onnx"""
import json
import sys

import onnx
from onnx import TensorProto, helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
init = {i.name for i in g.initializer}
existing = {o.name for o in g.output}
index = []
for i, n in enumerate(g.node):
    for o in n.output:
        if o and o not in existing and o not in init:
            g.output.append(helper.make_tensor_value_info(o, TensorProto.FLOAT, None))
            index.append({"i": i, "op": n.op_type, "out": o, "name": n.name})
onnx.save(m, dst)
json.dump(index, open(dst + ".index.json", "w"))
print("outputs added:", len(index))
