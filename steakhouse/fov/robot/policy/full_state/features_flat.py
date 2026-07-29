"""
FULL-STATE, FOV-BLIND observation - FLAT (per-kind), for the MLP + PPO pipeline.

Per-kind station aggregation keeps the dimension FIXED across layouts (layouts
differ in how many pots/boards/sinks they have). Design decisions (locked):
  station state  = 3 flags per kind {any_empty, any_in_progress, any_ready}
                   (exact cook/chop/wash timers dropped as low-value)
  robot self     = pos (norm x,y) + orientation (one-hot 4) + held (one-hot 8)
  human          = pos + orientation + LAST ACTION (one-hot 6) + held (one-hot 8)
  proxies        = keep station idle-age (per kind); DROP human distance (now
                   derivable from the raw positions)
  global         = episode fraction, orders remaining

EXCLUDED (the experiment): the human's FOV cone, beliefs, posterior, entropy.

Layout of the 48-dim vector:
  [0:9]    station flags   3 kinds x {empty, in_progress, ready}
  [9:12]   idle-age        per kind, normalised
  [12:14]  robot pos       (x,y) in [0,1]
  [14:18]  robot orient    one-hot N/S/E/W
  [18:26]  robot held      one-hot [none,meat,onion,plate,hot_plate,steak,garnish,dish]
  [26:28]  human pos
  [28:32]  human orient
  [32:38]  human last act  one-hot [N,S,E,W,stay,interact]
  [38:46]  human held      one-hot (same 8 classes)
  [46]     episode fraction
  [47]     orders remaining, normalised
"""
import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction
from fov.robot.policy.baseline.features import STATIONS, station_locs, _station_state

# 8-class held vocabulary (the only things carriable in the steak domain + none).
HELD_VOCAB = ["none", "meat", "onion", "plate", "hot_plate", "steak", "garnish", "dish"]
_HELD_IDX = {n: i for i, n in enumerate(HELD_VOCAB)}
N_HELD = len(HELD_VOCAB)                 # 8
DIRS = list(Direction.ALL_DIRECTIONS)    # 4  (N,S,E,W)
N_DIR = len(DIRS)                        # 4
N_ACT = len(Action.ALL_ACTIONS)          # 6  (N,S,E,W,stay,interact)

_IN_PROGRESS = ("cooking", "chopping", "washing", "occupied")

# 9 station flags + 3 age + (2+4+8) robot + (2+4+6+8) human + 2 global = 48
OBS_DIM_FULL = 3 * 3 + 3 + (2 + N_DIR + N_HELD) + (2 + N_DIR + N_ACT + N_HELD) + 2


def _grid_dims(mdp):
    mtx = mdp.terrain_mtx
    return len(mtx[0]), len(mtx)          # width (cols/x), height (rows/y)


def _held_onehot(player):
    v = np.zeros(N_HELD, dtype=np.float32)
    obj = player.held_object
    name = "none" if obj is None else getattr(obj, "name", "none")
    v[_HELD_IDX.get(name, 0)] = 1.0
    return v


def _dir_onehot(orient):
    v = np.zeros(N_DIR, dtype=np.float32)
    try:
        v[DIRS.index(tuple(orient))] = 1.0
    except ValueError:
        pass
    return v


def _act_onehot(action):
    v = np.zeros(N_ACT, dtype=np.float32)
    if action is not None:
        i = Action.ACTION_TO_INDEX.get(action)
        if i is not None:
            v[i] = 1.0
    return v


def _norm_pos(pos, w, h):
    return [pos[0] / max(1, w - 1), pos[1] / max(1, h - 1)]


def extract_full_flat(mdp, state, human_index=1, robot_index=0, t=0, horizon=260,
                      age=None, last_human_action=None):
    locs = station_locs(mdp)
    w, h = _grid_dims(mdp)
    p = []

    # [0:9] per-kind station flags (multi-station safe: OR across cells)
    for kind in STATIONS:
        words = [_station_state(mdp, state, l) for l in locs[kind]]
        p.append(1.0 if any(x == "empty" for x in words) else 0.0)
        p.append(1.0 if any(x in _IN_PROGRESS for x in words) else 0.0)
        p.append(1.0 if any(x == "ready" for x in words) else 0.0)

    # [9:12] idle-age per kind (kept proxy)
    for kind in STATIONS:
        p.append(min(1.0, age.get(kind, 0) / 40.0) if age else 0.0)

    # [12:26] robot self-state
    rp = state.players[robot_index]
    p += _norm_pos(rp.position, w, h)
    p += list(_dir_onehot(rp.orientation))
    p += list(_held_onehot(rp))

    # [26:46] human observable signals (NO beliefs / FOV)
    hp = state.players[human_index]
    p += _norm_pos(hp.position, w, h)
    p += list(_dir_onehot(hp.orientation))
    p += list(_act_onehot(last_human_action))
    p += list(_held_onehot(hp))

    # [46:48] global
    p.append(min(1.0, t / max(1, horizon)))
    p.append(min(1.0, len(state.order_list or []) / 4.0))

    obs = np.asarray(p, dtype=np.float32)
    assert obs.shape[0] == OBS_DIM_FULL, (obs.shape[0], OBS_DIM_FULL)
    return obs
