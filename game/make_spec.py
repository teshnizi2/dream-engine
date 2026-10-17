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


# Game v3 (docs/PREREG-v3.md): self-localising walls. A tint on every wall that depends on the world position of the hit
# point, plus four distinct face shades. Channel c of the tint is  amplitude * wave(ax_c * x + ay_c * y + phase_c)  grey
# levels, `wave` a smooth periodic function in [-1, 1] (period 1 in its argument) built from + - * floor abs only, so the
# JavaScript twin reproduces it bit for bit. The (ax, ay, phase) triples were found by `tools/design_signature.py` (seeded
# random search + hill climb): they maximise the smallest colour difference between any two visible faces of the same wall
# type, with periods of at least 5 cells (low frequency). The landmark pillar keeps its exact amber look (tint_gain 0): its four
# faces are told apart by the face shades alone.
TINT_AMPLITUDE = 12.0
TINT_WAVES = [[-0.1377, -0.1216, 0.2249], [-0.2, -0.1857, 0.131], [-0.1073, 0.1133, 0.9753]]
FACE_SHADE = [1.0, 0.9, 0.78, 0.66]      # the face of the wall cell that the ray hits: W (ray travels east), E, N (ray travels south), S
TINT_GAIN = {"4": 0.0}                    # per wall id, default 1.0


def build(version=2):
    """The game spec. version 2 is the frozen v2 game (game/spec_v2.json, byte-identical to the v2-final tag); version 3 adds
    the signature layer (the "tint" section, a "tint_gain" per wall and "game_version": 3) and changes nothing else."""
    if version not in (2, 3):
        raise ValueError(f"unknown game version {version}")
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
    if version == 3:
        spec["game_version"] = 3
        for k, w in spec["walls"].items():
            w["tint_gain"] = TINT_GAIN.get(k, 1.0)
        spec["tint"] = {"amplitude": TINT_AMPLITUDE, "waves": TINT_WAVES, "face_shade": FACE_SHADE}
    return spec


SHIPPED = 2          # the game the site and the default data loader use; change to 3 only after docs/PREREG-v3.md's adoption rule is met


if __name__ == "__main__":
    for v, name in ((2, "spec_v2.json"), (3, "spec_v3.json")):
        spec = build(v)
        for r in spec["map"]:
            assert len(r) == 16, r
        (HERE / name).write_text(json.dumps(spec, indent=1) + "\n")
        print("wrote", HERE / name)
    (HERE / "spec.json").write_text((HERE / f"spec_v{SHIPPED}.json").read_text())
    print("wrote", HERE / "spec.json", f"(= v{SHIPPED})")
