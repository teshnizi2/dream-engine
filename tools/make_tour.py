"""Writes site/js/tour.json: a scripted key sequence (list of [move, strafe, turn]) that walks a loop through
the landmarks: doorway -> amber pillar (circled once) -> north door -> window gallery -> red room -> back to the hall.
A simple waypoint follower on the real game, so the sequence is physically valid (collisions, turn rate).

    python tools/make_tour.py
"""
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from game import raycaster as rc  # noqa: E402

WAYPOINTS = [
    (5.45, 7.5),                                              # through the blue doorway, pillar dead ahead
    (6.5, 6.4), (8.5, 6.4), (8.55, 8.6), (6.5, 8.6), (6.45, 6.55),   # circle the amber pillar
    (6.5, 4.6), (6.5, 2.5), (6.5, 1.5),                       # north door, up the corridor to the gallery
    (11.5, 1.5), (14.5, 1.5),                                 # east along the window gallery
    (14.5, 4.5), (14.5, 6.5), (13.4, 7.6),                    # down into the red room
    (13.5, 9.5), (13.5, 11.5), (12.5, 12.5),                  # its south door, into the ochre room
    (10.5, 12.5), (9.5, 12.5), (5.5, 12.5),                   # west along the south corridor
    (3.5, 12.5), (2.5, 12.2),                                 # purple room
    (2.5, 10.5), (2.5, 8.2), (2.5, 7.5),                      # north up the green west corridor
    (5.45, 7.5),                                              # east through the doorway, back to the pillar
]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def build():
    w = rc.World()
    st = rc.state_from_spec_start(w.spec)
    actions, wp = [], 0
    turn_step = math.radians(w.spec["turn_deg"])
    stuck = 0
    while wp < len(WAYPOINTS) and len(actions) < 900:
        tx, ty = WAYPOINTS[wp]
        dx, dy = tx - st.x[0], ty - st.y[0]
        dist = math.hypot(dx, dy)
        if dist < 0.28:
            wp += 1
            continue
        err = wrap(math.atan2(dy, dx) - math.atan2(st.dy[0], st.dx[0]))
        turn = rc.TURN_NONE
        if err > turn_step * 0.55:
            turn = rc.TURN_RIGHT
        elif err < -turn_step * 0.55:
            turn = rc.TURN_LEFT
        move = rc.MOVE_FWD if abs(err) < math.radians(38) else rc.MOVE_NONE
        a = np.array([[move, rc.STRAFE_NONE, turn]], dtype=np.int64)
        before = (st.x[0], st.y[0])
        st = w.step(st, a)
        actions.append([int(move), 0, int(turn)])
        stuck = stuck + 1 if (move == rc.MOVE_FWD and math.hypot(st.x[0] - before[0], st.y[0] - before[1]) < 1e-9) else 0
        if stuck > 25:
            raise RuntimeError(f"stuck near waypoint {wp} {WAYPOINTS[wp]} at ({st.x[0]:.2f},{st.y[0]:.2f})")
    return actions


if __name__ == "__main__":
    acts = build()
    out = ROOT / "site" / "js" / "tour.json"
    out.write_text(json.dumps({"description": "scripted key sequence: [move, strafe, turn] per tick", "actions": acts}))
    print(f"wrote {out} with {len(acts)} ticks ({len(acts) / 24:.1f} s at 24 fps, {len(acts) / 30:.1f} s at 30 fps)")
