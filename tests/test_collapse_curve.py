"""tools/render_collapse_curve.py: media/collapse_curve.png from the milestone evaluations (results/prereg_<run>_s<step>.json)."""
import pathlib
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))


def test_the_curve_is_drawn_from_milestone_and_final_evaluations(tmp_path):
    import render_collapse_curve as R
    from tests.test_compare_prereg import fake_eval
    for st, cf in ((20000, 60), (40000, 120), (60000, 200)):
        fake_eval(tmp_path, f"full1_s{st}", 4, "v2", horizon_drop=12, collapse_from=cf)
        fake_eval(tmp_path, f"full2_s{st}", 4, "v3", horizon_drop=12, collapse_from=cf + 100)
    fake_eval(tmp_path, "full1", 4, "v2", horizon_drop=16, collapse_from=300)
    fake_eval(tmp_path, "full2", 4, "v3", horizon_drop=16, collapse_from=400)
    import json
    for n in ("full1", "full2"):                                           # the final json carries the step count
        j = json.loads((tmp_path / f"prereg_{n}.json").read_text()); j["step"] = 150000
        (tmp_path / f"prereg_{n}.json").write_text(json.dumps(j))
    curves = R.curves(tmp_path)
    assert [s for s, _ in curves["full1"]] == [20000, 40000, 60000, 150000] and curves["full2"][-1][0] == 150000
    assert curves["full1"][0][1] > curves["full1"][-1][1]                   # collapse falls as the collapse starts later
    out = tmp_path / "curve.png"
    R.draw(curves, out, target=0.079)
    im = Image.open(out)
    assert im.size[0] >= 800 and im.size[1] >= 400 and np.asarray(im).std() > 5
