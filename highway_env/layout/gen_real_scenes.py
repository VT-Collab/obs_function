"""Scan nocturne's real training scenes, find the ones worth looking at, and
write thin layout files for the best ones into layouts/.

    python gen_real_scenes.py                 # scan, rank, write top N
    python gen_real_scenes.py --n 30           # write 30 instead of the default
    python gen_real_scenes.py --replace 3 7 12 # drop these numbers, pull in
                                                # the next-best unused scenes
                                                # to replace them

For each scene file: count road segments (complexity proxy) and scan every
pair of moving vehicles for the one with the longest sustained proximity
(the real-world version of "don't separate"). Scenes are ranked by
(has a decent pair, then by road count) and the results are cached in
real_scenes_manifest.json so a --replace run doesn't have to re-scan
everything -- it just pulls the next-best candidates that weren't already
written out.

Written layouts are named real_NNN.py and follow real_scene.py's original
pattern exactly (see _real_scene.py): build_road_from(path) imports the raw
road geometry as PolyLaneFixedWidth lanes, trajectories_from(path, i, j)
pulls the two vehicles' actual recorded (x, y) points as HUMAN_ROUTE/
ROBOT_ROUTE.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAYOUTS_DIR = HERE / "layouts"
MANIFEST_PATH = HERE / "real_scenes_manifest.json"
DATA_DIR = HERE.parents[1] / "nocturne" / "data" / "formatted_json_v2_no_tl_train"

MIN_VALID_PTS = 60
MIN_MOVEMENT = 20.0
CLOSE_THRESHOLD = 20.0
MIN_CLOSE_FRAC = 0.55
MIN_ROADS = 10


def analyze_scene(path):
    """Best (human_idx, robot_idx, close_frac, n_roads) for one scene, or None."""
    try:
        scene = json.loads(path.read_text())
        n_roads = sum(1 for r in scene["roads"] if r["type"] in ("lane", "road_edge"))
    except Exception:
        return None
    if n_roads < MIN_ROADS:
        return None

    vehicles = []
    for i, o in enumerate(scene.get("objects", [])):
        if o["type"] != "vehicle":
            continue
        pts = np.array([[p["x"], p["y"]] for p, ok in zip(o["position"], o["valid"]) if ok])
        if len(pts) < MIN_VALID_PTS or np.linalg.norm(pts[-1] - pts[0]) < MIN_MOVEMENT:
            continue
        vehicles.append((i, pts))

    best = None
    for (i1, p1), (i2, p2) in itertools.combinations(vehicles, 2):
        n = min(len(p1), len(p2))
        if n < MIN_VALID_PTS:
            continue
        d = np.linalg.norm(p1[:n] - p2[:n], axis=1)
        close_frac = float((d < CLOSE_THRESHOLD).mean())
        if best is None or close_frac > best[2]:
            best = (i1, i2, close_frac)
    if best is None or best[2] < MIN_CLOSE_FRAC:
        return None
    return {"human_idx": best[0], "robot_idx": best[1], "close_frac": round(best[2], 3), "n_roads": n_roads}


def scan(sample_every=4, limit=None):
    paths = sorted(DATA_DIR.glob("*.json"))[::sample_every]
    if limit:
        paths = paths[:limit]
    candidates = []
    for i, path in enumerate(paths):
        result = analyze_scene(path)
        if result:
            result["path"] = str(path)
            candidates.append(result)
        if (i + 1) % 50 == 0:
            print(f"  scanned {i + 1}/{len(paths)}, {len(candidates)} candidates so far", file=sys.stderr)
    candidates.sort(key=lambda c: (c["n_roads"], c["close_frac"]), reverse=True)
    return candidates


def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"candidates": [], "written": []}  # written: [{num, path, human_idx, robot_idx}]


def save_manifest(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


TEMPLATE = '''"""Real scene {num}: real Waymo-derived road network (nocturne), plus the
actual recorded trajectories of two real vehicles that stay close for a
sustained stretch (close_frac={close_frac} of the recording within {threshold:.0f}m).

Source: {path_name}, objects[{human_idx}]/objects[{robot_idx}].
See _real_scene.py for how the import works.
"""
from _real_scene import build_road_from, trajectories_from

SCENE_PATH = "{path}"
HUMAN_VEHICLE_IDX, ROBOT_VEHICLE_IDX = {human_idx}, {robot_idx}


def build_road():
    return build_road_from(SCENE_PATH)


HUMAN_ROUTE, ROBOT_ROUTE = trajectories_from(SCENE_PATH, HUMAN_VEHICLE_IDX, ROBOT_VEHICLE_IDX)
'''


def write_layout(num, candidate):
    path = Path(candidate["path"])
    text = TEMPLATE.format(
        num=f"{num:03d}", path=candidate["path"], path_name=path.name,
        human_idx=candidate["human_idx"], robot_idx=candidate["robot_idx"],
        close_frac=candidate["close_frac"], threshold=CLOSE_THRESHOLD,
    )
    out = LAYOUTS_DIR / f"real_{num:03d}.py"
    out.write_text(text)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=24, help="how many scenes to write out (default 24)")
    parser.add_argument("--sample-every", type=int, default=4, help="scan every Nth file in the data dir")
    parser.add_argument("--limit", type=int, default=None, help="cap how many files get scanned at all")
    parser.add_argument("--replace", type=int, nargs="*", default=None,
                         help="layout numbers to drop and refill with next-best unused candidates")
    args = parser.parse_args()

    manifest = load_manifest()

    if not manifest["candidates"]:
        print(f"scanning {DATA_DIR} ...", file=sys.stderr)
        manifest["candidates"] = scan(sample_every=args.sample_every, limit=args.limit)
        print(f"found {len(manifest['candidates'])} candidates", file=sys.stderr)
        save_manifest(manifest)

    used_paths = {(w["path"], w["human_idx"], w["robot_idx"]) for w in manifest["written"]}

    if args.replace:
        drop = set(args.replace)
        manifest["written"] = [w for w in manifest["written"] if w["num"] not in drop]
        for num in drop:
            f = LAYOUTS_DIR / f"real_{num:03d}.py"
            if f.exists():
                f.unlink()
        used_paths = {(w["path"], w["human_idx"], w["robot_idx"]) for w in manifest["written"]}
        need = len(drop)
    else:
        need = args.n - len(manifest["written"])

    available = [c for c in manifest["candidates"]
                 if (c["path"], c["human_idx"], c["robot_idx"]) not in used_paths]

    existing_nums = {w["num"] for w in manifest["written"]}
    next_num = 1
    written_this_run = []
    for candidate in available:
        if len(written_this_run) >= max(need, 0):
            break
        while next_num in existing_nums:
            next_num += 1
        write_layout(next_num, candidate)
        entry = {"num": next_num, "path": candidate["path"],
                  "human_idx": candidate["human_idx"], "robot_idx": candidate["robot_idx"],
                  "close_frac": candidate["close_frac"], "n_roads": candidate["n_roads"]}
        manifest["written"].append(entry)
        written_this_run.append(entry)
        existing_nums.add(next_num)
        next_num += 1

    save_manifest(manifest)
    print(f"wrote {len(written_this_run)} layouts: {[w['num'] for w in written_this_run]}")
    print(f"total written so far: {len(manifest['written'])}")


if __name__ == "__main__":
    main()
