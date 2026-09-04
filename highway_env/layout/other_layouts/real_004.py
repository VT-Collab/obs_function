"""Real scene 004: real Waymo-derived road network (nocturne), plus the
actual recorded trajectories of two real vehicles that stay close for a
sustained stretch (close_frac=1.0 of the recording within 20m).

Source: tfrecord-00125-of-01000_372.json, objects[20]/objects[96].
See _real_scene.py for how the import works.
"""
from _real_scene import build_road_from, trajectories_from

SCENE_PATH = "/Users/mishafu/Desktop/obs_function/nocturne/data/formatted_json_v2_no_tl_train/tfrecord-00125-of-01000_372.json"
HUMAN_VEHICLE_IDX, ROBOT_VEHICLE_IDX = 20, 96


def build_road():
    return build_road_from(SCENE_PATH)


HUMAN_ROUTE, ROBOT_ROUTE = trajectories_from(SCENE_PATH, HUMAN_VEHICLE_IDX, ROBOT_VEHICLE_IDX)
