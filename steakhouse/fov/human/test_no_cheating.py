"""
MISHA NEW CHANGE - does the limited-vision human CHEAT?

The claim under test: the agent may only act on station states it has actually
SEEN through its own field of view, and must act on stale or absent information
when it has not looked recently.

A cheating agent is easy to spot statistically: its beliefs would match ground
truth regardless of FOV. So the decisive measurement is BELIEF ACCURACY vs FOV.

  * If belief-vs-truth agreement is ~equal across FOVs -> the agent is reading
    the true state and the FOV gate is cosmetic. CHEATING.
  * If agreement rises with FOV, and narrow FOVs hold measurably WRONG beliefs
    and act on them -> the limitation is real.

Also checked directly:
  1. Every station state written into a belief was in the cone at that moment.
  2. Beliefs start UNKNOWN (no omniscient initialisation).
  3. The agent demonstrably acts on WRONG beliefs (decisions taken while its
     belief contradicted ground truth).

KNOWN, DELIBERATE non-cheat: the agent knows WHERE stations are (mdp
terrain lookup in reset()). That is spatial memory of a familiar kitchen, not
privileged state - it never reveals whether the pot is full, which is the only
thing its decisions branch on. Stated explicitly so it is not mistaken for an
oversight.

Run with: python -m fov.human.test_no_cheating [layout] [n_seeds]
"""
import random
import sys

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from overcooked_ai_py.agents.agent import GreedySteakHumanModel
from fov.human.agent.limited_vision_human import LimitedVisionSteakHuman, UNKNOWN
from fov.human.planning.steak_planner import SteakMotionPlanner

FOVS = [30, 60, 90, 120, 180, 360]
N_STEPS = 260


def audit(layout, fov, seed):
    mdp = SteakHouseGridworld.from_layout_name(layout, start_order_list=['steak'] * 4)
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)
    np.random.seed(seed)
    random.seed(seed)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=N_STEPS + 10)
    planner = SteakMotionPlanner(mdp, mlp)
    human = LimitedVisionSteakHuman(mdp, fov, planner, agent_index=1)
    robot = GreedySteakHumanModel(mlp)
    robot.set_agent_index(0)

    n_orders0 = len(env.state.order_list) if env.state.order_list else 0

    # (2) beliefs must start UNKNOWN
    start_unknown = all(b.value == UNKNOWN for b in human.beliefs.values())

    n_cmp = n_match = n_unknown = 0
    n_acted_on_wrong = 0
    illegal_writes = 0

    for _ in range(N_STEPS):
        s = env.state
        try:
            a_r, _ = robot.action(s)
        except Exception:
            a_r = Action.STAY

        human.observe(s)

        # (1) every non-UNKNOWN belief must either be in the cone right now, or
        # be a remembered value from when it WAS in the cone (i.e. never a fresh
        # read of an unseen station).
        for loc, b in human.beliefs.items():
            # beliefs is keyed by station (x, y) AND by the ROBOT sentinel, which
            # is a string naming an entity rather than a tile. Only stations are
            # auditable here - visible()/_true_state_of() both expect a tile.
            if not isinstance(loc, tuple):
                continue
            if b.value == UNKNOWN:
                continue
            if b.seen_at == human.t and not human.visible(s, loc):
                illegal_writes += 1   # written this tick without being visible

        # belief vs ground truth, over every station
        for loc, b in human.beliefs.items():
            if not isinstance(loc, tuple):
                continue
            truth = human._true_state_of(s, loc)
            n_cmp += 1
            if b.value == UNKNOWN:
                n_unknown += 1
            elif b.value == truth:
                n_match += 1

        sub = human.decide(s)
        # (3) is it acting on a belief that contradicts reality?
        #
        # Explicit subtask -> station map. The first version matched station
        # names as SUBSTRINGS of subtask names, so 'pickup_meat'/'drop_meat'
        # never matched 'pot' and the only subtasks that did match were the
        # check_* ones - which are UNKNOWN by definition and skipped. It
        # therefore always reported 0 regardless of what the agent did.
        SUB2STATION = {
            "pickup_meat": "pot", "drop_meat": "pot", "check_pot": "pot",
            "pickup_steak": "pot",
            "pickup_onion": "board", "drop_onion": "board", "chop_onion": "board",
            "check_board": "board", "pickup_garnish": "board",
            "pickup_plate": "sink", "drop_plate": "sink", "check_sink": "sink",
            "heat_washed_plate": "sink", "pickup_washed_plate": "sink",
        }
        kind = SUB2STATION.get(sub)
        if kind:
            bel = human.believed(kind)
            truths = [human._true_state_of(s, l) for l in human.stations[kind]]
            if bel != UNKNOWN and bel not in truths:
                n_acted_on_wrong += 1

        try:
            a_h, _ = human.action(s)
        except Exception:
            a_h = Action.STAY
        _, _, done, _ = env.step((a_r, a_h))
        if done:
            break

    # MISHA NEW CHANGE - with locations now DISCOVERED rather than given,
    # belief_acc alone is misleading: an agent that has found almost nothing
    # holds a handful of freshly-seen beliefs and scores a trivial 1.000. Report
    # how much of the world it actually mapped alongside it.
    # agent no longer precomputes the grid; use its own known extent
    n_cells = human._w * human._h
    n_stations_found = sum(len(v) for v in human.stations.values())
    n_stations_total = sum(len(mdp.terrain_pos_dict.get(t, []))
                           for t in ('P', 'B', 'W', 'M', 'O', 'D', 'S', 'X'))
    known = n_cmp - n_unknown
    return dict(fov=fov, seed=seed, start_unknown=start_unknown,
                seen_frac=len(human.seen_cells) / max(1, n_cells),
                found_frac=n_stations_found / max(1, n_stations_total),
                explore_steps=human.n_explore,
                illegal_writes=illegal_writes,
                belief_acc=(n_match / known) if known else 0.0,
                unknown_frac=n_unknown / max(1, n_cmp),
                acted_on_wrong=n_acted_on_wrong, delivered=human.n_delivered,
                team=(n_orders0 - (len(env.state.order_list) if env.state.order_list else 0)))


def main():
    layout = sys.argv[1] if len(sys.argv) > 1 else "steak_island"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"CHEAT AUDIT  layout={layout}  seeds={n_seeds}\n")
    print(f"{'fov':>4} {'map_seen':>9} {'stations':>9} {'explore':>8} {'belief_acc':>11} "
          f"{'wrong_acts':>11} {'illegal':>8} {'deliv':>6}")
    rows = {}
    for fov in FOVS:
        rs = [audit(layout, fov, s) for s in range(n_seeds)]
        rows[fov] = rs
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        print(f"{fov:>4} {m('seen_frac')*100:>8.1f}% {m('found_frac')*100:>8.1f}% "
              f"{m('explore_steps'):>8.0f} {m('belief_acc'):>11.3f} "
              f"{m('acted_on_wrong'):>11.1f} {sum(r['illegal_writes'] for r in rs):>8} "
              f"{m('delivered'):>6.1f}", flush=True)

    acc = {f: sum(r["belief_acc"] for r in rows[f]) / len(rows[f]) for f in FOVS}
    illegal = sum(r["illegal_writes"] for f in FOVS for r in rows[f])
    start_ok = all(r["start_unknown"] for f in FOVS for r in rows[f])
    wrong = {f: sum(r["acted_on_wrong"] for r in rows[f]) / len(rows[f]) for f in FOVS}

    print("\n--- verdicts ---")
    print(f"1. no illegal belief writes (state read outside FOV) : "
          f"{'PASS' if illegal == 0 else 'FAIL (' + str(illegal) + ')'}")
    print(f"2. beliefs start UNKNOWN (no omniscient init)        : "
          f"{'PASS' if start_ok else 'FAIL'}")
    print(f"3. narrow FOV holds WRONG beliefs and acts on them   : "
          f"{'PASS' if wrong[FOVS[0]] > 0 else 'FAIL'}  "
          f"(fov{FOVS[0]}={wrong[FOVS[0]]:.1f} vs fov360={wrong[360]:.1f} decisions)")
    seen = {f: sum(r['seen_frac'] for r in rows[f]) / len(rows[f]) for f in FOVS}
    found = {f: sum(r['found_frac'] for r in rows[f]) / len(rows[f]) for f in FOVS}
    team = {f: sum(r['team'] for r in rows[f]) / len(rows[f]) for f in FOVS}
    print(f"4. MAP COVERAGE increases with FOV                   : "
          f"{'PASS' if seen[360] > seen[FOVS[0]] else 'FAIL'}  "
          f"(fov{FOVS[0]}={seen[FOVS[0]]*100:.0f}% -> fov360={seen[360]*100:.0f}%)")
    print(f"5. STATIONS FOUND increases with FOV                 : "
          f"{'PASS' if found[360] > found[FOVS[0]] else 'FAIL'}  "
          f"(fov{FOVS[0]}={found[FOVS[0]]*100:.0f}% -> fov360={found[360]*100:.0f}%)")
    print(f"6. TEAM delivers at every FOV (team win is a win)    : "
          f"{'PASS' if all(team[f] >= 1.0 for f in FOVS) else 'FAIL'}  "
          f"(fov{FOVS[0]}={team[FOVS[0]]:.1f} -> fov360={team[360]:.1f})")
    print("\nIf the agent were cheating, belief accuracy would be ~1.0 at EVERY "
          "fov and\nnothing would ever be UNKNOWN.")


if __name__ == "__main__":
    main()
