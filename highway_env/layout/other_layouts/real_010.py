"""Real scene 010: real Waymo-derived road network (nocturne), plus the
actual recorded trajectories of two real vehicles that stay close for a
sustained stretch (close_frac=0.703 of the recording within 20m).

Source: tfrecord-00793-of-01000_270.json, objects[1]/objects[57].
See _real_scene.py for how the import works.
"""
from _real_scene import build_road_from, trajectories_from

SCENE_PATH = "/Users/mishafu/Desktop/obs_function/nocturne/data/formatted_json_v2_no_tl_train/tfrecord-00793-of-01000_270.json"
HUMAN_VEHICLE_IDX, ROBOT_VEHICLE_IDX = 1, 57


def build_road():
    return build_road_from(SCENE_PATH)


HUMAN_ROUTE, ROBOT_ROUTE = trajectories_from(SCENE_PATH, HUMAN_VEHICLE_IDX, ROBOT_VEHICLE_IDX)
