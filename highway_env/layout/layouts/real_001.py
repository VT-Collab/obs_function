"""Real scene 001: real Waymo-derived road network (nocturne), human + robot.

Source: tfrecord-00961-of-01000_344.json.

HUMAN_ROUTE / ROBOT_ROUTE were hand-clicked with pick_route.py (saved in
pick_route_sessions/real_001.json), then cleaned up in two steps:

  1. densify(): hand clicks are 10-27m apart, versus ~1m between frames in
     a real vehicle recording. Left as-is, snapping would cut every corner
     between clicks since there's nothing there to pull onto the curve.
     This linearly interpolates extra points every ~2m first.

  2. match_to_road(): projects the densified points onto the real mapped
     lane graph, but -- unlike plain per-point snapping -- assigns whole
     RUNS of points to one lane (sticky, so a path running ambiguously
     between two close parallel lanes doesn't flicker between them frame
     to frame) and then outputs that lane's OWN vertices over the matched
     run, not the projected click points. Since a real lane's own geometry
     is already near-noise-free, everything *within* a run comes out clean
     by construction. Only the jumps between runs (actual lane changes/
     turns) can still show a real heading change, which is genuine road
     structure, not an artifact.

See _real_scene.py for both functions and why the simpler snap_to_road()
(fine for a real vehicle's own recorded trajectory, which doesn't wander
between lanes) wasn't enough for a hand-clicked path.
"""
import json

from _real_scene import SESSIONS_DIR, build_road_from, densify, match_to_road

SCENE_PATH = "/Users/mishafu/Desktop/obs_function/nocturne/data/formatted_json_v2_no_tl_train/tfrecord-00961-of-01000_344.json"
SESSION_PATH = SESSIONS_DIR / "real_001.json"


def build_road():
    return build_road_from(SCENE_PATH)


with open(SESSION_PATH) as _f:
    _session = json.load(_f)

HUMAN_ROUTE = match_to_road(SCENE_PATH, densify(_session["human"], spacing=2.0))
ROBOT_ROUTE = match_to_road(SCENE_PATH, densify(_session["robot"], spacing=2.0))
