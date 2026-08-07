"""
FULL-STATE, FOV-BLIND observation - GRID (exact per-object), for a small CNN.

Unlike the flat vector, the grid keeps every object at its own cell, so two pots
are distinct and exact positions are preserved. C sheets, each H x W. Indexed
[channel, row(y), col(x)] to match terrain_mtx[y][x].

Channels (42):
  terrain (static, 8): pot P, board B, sink W, meat-disp M, onion-disp O,
                       plate-disp D, serving S, counter X
  station state (3):   empty / in_progress / ready, lit at each station cell
  players (4):         robot cell, robot facing cell, human cell, human facing
  robot held (8):      one-hot class, broadcast across the map
  human held (8):      one-hot class, broadcast
  human last act (6):  one-hot, broadcast
  idle-age (3):        per kind, broadcast (kept proxy)
  global (2):          episode fraction, orders remaining, broadcast

EXCLUDED (the experiment): the human's FOV cone, beliefs, posterior, entropy.
"""
import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import Action
from steakhouse.fov.robot.policy.old.baseline.features import STATIONS, station_locs, _station_state
from steakhouse.fov.robot.policy.old.full_state.features_flat import (
    _HELD_IDX, N_HELD, N_ACT, _IN_PROGRESS)

TERRAIN = ['P', 'B', 'W', 'M', 'O', 'D', 'S', 'X']
N_TERRAIN = len(TERRAIN)
_TI = {c: i for i, c in enumerate(TERRAIN)}

# 8 + 3 + 4 + 8 + 8 + 6 + 3 + 2 = 42
N_CHANNELS = N_TERRAIN + 3 + 4 + N_HELD + N_HELD + N_ACT + 3 + 2


def grid_dims(mdp):
    mtx = mdp.terrain_mtx
    return len(mtx[0]), len(mtx)          # w, h


def _held_idx(player):
    o = player.held_object
    n = "none" if o is None else getattr(o, "name", "none")
    return _HELD_IDX.get(n, 0)


def extract_full_grid(mdp, state, human_index=1, robot_index=0, t=0, horizon=260,
                      age=None, last_human_action=None):
    mtx = mdp.terrain_mtx
    h = len(mtx); w = len(mtx[0])
    x = np.zeros((N_CHANNELS, h, w), dtype=np.float32)
    ch = 0

    # terrain (static)
    for r in range(h):
        row = mtx[r]
        for c in range(w):
            k = row[c]
            if k in _TI:
                x[ch + _TI[k], r, c] = 1.0
    ch += N_TERRAIN

    # station dynamic state, lit at the station cell
    locs = station_locs(mdp)
    for kind in STATIONS:
        for (cx, cy) in locs[kind]:
            stt = _station_state(mdp, state, (cx, cy))
            if stt == "empty":
                x[ch + 0, cy, cx] = 1.0
            elif stt in _IN_PROGRESS:
                x[ch + 1, cy, cx] = 1.0
            elif stt == "ready":
                x[ch + 2, cy, cx] = 1.0
    ch += 3

    # players: cell + facing cell
    rp = state.players[robot_index]
    hp = state.players[human_index]

    def setcell(chan, pos):
        px, py = pos
        if 0 <= py < h and 0 <= px < w:
            x[chan, py, px] = 1.0
    setcell(ch + 0, rp.position)
    setcell(ch + 1, (rp.position[0] + rp.orientation[0], rp.position[1] + rp.orientation[1]))
    setcell(ch + 2, hp.position)
    setcell(ch + 3, (hp.position[0] + hp.orientation[0], hp.position[1] + hp.orientation[1]))
    ch += 4

    # held (broadcast one-hot)
    x[ch + _held_idx(rp), :, :] = 1.0; ch += N_HELD
    x[ch + _held_idx(hp), :, :] = 1.0; ch += N_HELD

    # human last action (broadcast one-hot)
    if last_human_action is not None:
        i = Action.ACTION_TO_INDEX.get(last_human_action)
        if i is not None:
            x[ch + i, :, :] = 1.0
    ch += N_ACT

    # idle-age per kind (broadcast)
    for kind in STATIONS:
        x[ch, :, :] = min(1.0, age.get(kind, 0) / 40.0) if age else 0.0
        ch += 1

    # global (broadcast)
    x[ch, :, :] = min(1.0, t / max(1, horizon)); ch += 1
    x[ch, :, :] = min(1.0, len(state.order_list or []) / 4.0); ch += 1

    assert ch == N_CHANNELS, (ch, N_CHANNELS)
    return x
