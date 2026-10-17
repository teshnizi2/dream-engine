"""Writes game/spec.json: the single source of truth shared by the Python and JS game twins.

Everything the two implementations need (map, physics, palette, fog, texture parameters) lives in
the JSON, so the two engines cannot drift apart through copy-pasted constants. Derived numbers (the
turn rotation) are computed here once and written with full precision.

    python game/make_spec.py
"""
import json
import math
import pathlib

HERE = pathlib.Path(__file__).parent

# 16 x 16, '.' = open floor, letters = wall types (see "wall_chars"). Hand-designed:
#   * a 6x5 central hall with a glowing amber pillar (A) and a blue west wall (B) with a doorway,
#   * a window gallery (W) along the whole north wall,
#   * a red-brick room (R) on the east, a mossy green wing (G) in the west, purple (P) and ochre (Y)
#     rooms in the south, stone (S) elsewhere. Walls are coloured by region so the dream can tell
#     where it is from the colours alone.
MAP = [
    "GGGGWWWWWWWWWWWS",
    "G..............Y",
    "G...GS.SSSSSSS.Y",
    "G...GS.SSSSSSS.Y",
    "SG.GSS.SSSSSSS.Y",
    "SG.GB......SRR.R",
    "SG.GB......R...R",
    "SG.....A...R...R",
    "SG.GB..........R",
    "SG.GB......R...R",
    "SG.GPSSSS.SYR.RS",
    "P....PSSS.Y....Y",
    "P..............Y",
    "P....PSSSSY....Y",
    "P....PSSSSY....Y",
    "SPPPPSSSSSSYYYYS",
]


def build():
    turn_deg = 5.0
    a = math.radians(turn_deg)
    spec = {
        "version": 1,
        "size": 16,
        "width": 64,
        "height": 64,
        "plane": 0.8,                 # camera plane half-length: horizontal FOV = 2*atan(0.8) = 77.3 deg
        "move_speed": 0.12,           # tiles per step (forward/back)
        "strafe_speed": 0.10,
        "turn_deg": turn_deg,
        "turn_cos": math.cos(a),
        "turn_sin": math.sin(a),
        "radius": 0.22,               # collision margin along the movement axis
        "fog_k": 0.16,                # f = 1 / (1 + (k d)^2)
        "fog_color": [14, 17, 23],    # = page background #0e1117 (almost)
        "side_shade": 0.74,           # multiplier on N/S-facing wall sides
        "ceiling": {"horizon": [30, 38, 58], "top": [10, 13, 22]},
        "floor": {"horizon": [52, 44, 40], "bottom": [26, 22, 22]},
        "wall_chars": {"S": 1, "R": 2, "B": 3, "A": 4, "W": 5, "G": 6, "P": 7, "Y": 8},
        "mip": [2.0, 7.0],           # textures fade to their mean colour between 2 and 7 cells (no sub-pixel aliasing)
        "walls": {
            "1": {"name": "stone", "rgb": [112, 124, 148], "rgb2": [86, 96, 120], "emit": 0.0,
                  "pattern": "blocks", "p": [2, 2, 0.06, 0.05, 0.11]},
            "2": {"name": "red_brick", "rgb": [168, 56, 50], "rgb2": [112, 40, 40], "emit": 0.0,
                  "pattern": "blocks", "p": [4, 2, 0.10, 0.06, 0.15]},
            "3": {"name": "blue_door", "rgb": [48, 104, 206], "rgb2": [112, 160, 236], "emit": 0.0,
                  "pattern": "frame", "p": [0.1, 0.36]},
            "4": {"name": "amber_pillar", "rgb": [245, 158, 11], "rgb2": [255, 205, 100], "emit": 0.75,
                  "pattern": "pillar", "p": [3, 0.12, 0.12]},
            "5": {"name": "window_stripes", "rgb": [44, 54, 80], "rgb2": [168, 204, 240], "emit": 0.55,
                  "pattern": "windows", "p": [3, 0.16, 0.84, 0.18, 0.82, 0.44]},
            "6": {"name": "moss", "rgb": [64, 138, 92], "rgb2": [46, 106, 72], "emit": 0.0,
                  "pattern": "moss", "p": [5, 0.4]},
            "7": {"name": "purple", "rgb": [128, 92, 176], "rgb2": [92, 66, 132], "emit": 0.0,
                  "pattern": "dado", "p": [0.62, 0.04, 0.38]},
            "8": {"name": "ochre", "rgb": [200, 154, 86], "rgb2": [158, 116, 64], "emit": 0.0,
                  "pattern": "blocks", "p": [2, 2, 0.05, 0.04, 0.09]},
        },
        "map": MAP,
        # first frame: west corridor, looking east through the blue doorway at the amber pillar
        "start": {"x": 2.5, "y": 7.5, "dx": 1.0, "dy": 0.0},
        "actions": {
            "move": ["none", "forward", "back"],
            "strafe": ["none", "left", "right"],
            "turn": ["none", "left", "right"],
        },
    }
    return spec


if __name__ == "__main__":
    spec = build()
    for r in spec["map"]:
        assert len(r) == 16, r
    (HERE / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    print("wrote", HERE / "spec.json")
