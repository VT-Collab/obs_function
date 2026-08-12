"""SEE-IT-HAPPEN vs EASY-TO-REACH: which stash actually gets collected sooner?

    python -m robot.filter.dev.handoff_ab --layouts back_bar,pantry --seeds 0-2

A PROTOTYPE. It changes nothing in core/ -- it only reads the filter and the human
model and measures. The question it answers is the one the cost model currently
answers by assumption:

  VISIBLE   put it where their cone lands soonest after the press. They learn about
            it immediately, but the counter may be a long walk from them.
  NEAR      put it where they can collect it fastest once they know, even if they
            are not looking there now. Short walk, late discovery.
  FILTER    whatever the shipped cost model picks, so the model can be scored
            against both extremes rather than against nothing.

`_collect` in core/qmdp.py assumes VISIBLE-then-walk is the whole story:
pickup = first sight at or after the press, plus the walk. That is a MODEL. This
measures the REALISED pickup by simulating each choice forward against the actual
LimitedVisionHuman and watching for the item to leave the counter into their hands.

WHAT WOULD FALSIFY THE MODEL. If NEAR beats VISIBLE often, then discovery is not
gated on the cone the way the model says -- the human's explore rule finds things
anyway -- and weighting sight is buying error. If VISIBLE wins nearly always, the
model is right and the only question left is its arithmetic. If it SPLITS by
layout, then the choice is layout-dependent and no single rule is correct, which is
the answer that would justify keeping both terms and letting the search arbitrate.

HOW A TRIAL RUNS. From the decision state, the robot beelines to the candidate
counter, presses once, then STAYS for the rest of the trial -- it must not pick the
item back up, or the trial measures the robot tidying up after itself. The human
plays normally throughout. The trial ends when the human is holding the item, or at
`--patience` ticks.
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.environ.get("STEAK_ROOT", os.path.dirname(ROOT)))
sys.path.insert(0, ROOT)

import overcooked_ai_py                                            # noqa: E402
overcooked_ai_py.LAYOUTS_DIR = os.path.join(ROOT, "layout", "layouts")

from overcooked_ai_py.mdp.actions import Action                    # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv      # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld  # noqa: E402
from common import geometry as geo                                 # noqa: E402
from common.views import TruthView                                 # noqa: E402
from human.limited_vision_human import (LimitedVisionHuman,        # noqa: E402
                                        FORGET_HORIZON)
from robot.filter.core.qmdp import fast_clone_human                # noqa: E402
from robot.methods import make_robot                               # noqa: E402

LAYOUTS = ["back_bar", "banquet_pass", "butchery", "chefs_table", "divide", "pantry"]
SHAREABLE = {"washed_plate", "garnish", "steak_dish", "garnish_dish", "dish", "meat",
             "onion", "plate", "steak"}


def trial(mdp, state, human, cell, item, robot_idx, patience):
    """Robot walks to `cell`, presses, then stays. Tick the human takes `item`.

    Returns (pickup tick or None, tick of the press or None). The robot STAYS after
    pressing so the trial cannot be won by the robot retrieving its own stash.
    """
    hi = 1 - robot_idx
    sim = state.deepcopy()
    h = fast_clone_human(human)
    walk = set(TruthView(mdp, state).walkable) | {tuple(sim.players[robot_idx].position)}
    pressed = None
    for t in range(patience):
        me = sim.players[robot_idx]
        pos, orient = tuple(me.position), tuple(me.orientation)
        if pressed is None:
            mv, arrived = geo.step_towards(walk, pos, orient, cell)
            ar = Action.INTERACT if arrived else (mv or Action.STAY)
        else:
            ar = Action.STAY
        ah, _ = h.action(sim)
        sim, _, _, _ = mdp.get_state_transition(sim, (ar, ah) if robot_idx == 0
                                                else (ah, ar))
        if pressed is None and ar == Action.INTERACT:
            pressed = t + 1
        if pressed is not None:
            held = sim.players[hi].held_object
            if held is not None and held.name == item:
                return t + 1, pressed
        if mdp.is_terminal(sim):
            break
    return None, pressed


def rank_candidates(mdp, state, human, fov, cells, robot_idx, horizon):
    """(visible-soonest, nearest-to-human) from a forecast of the human alone.

    Both are computed the way the cost model would, so the comparison is between
    the model's two competing intuitions rather than between two arbitrary cells.
    """
    hi = 1 - robot_idx
    sim = state.deepcopy()
    h = fast_clone_human(human)
    terrain = mdp.terrain_mtx
    sight, poss = {}, []
    for k in range(horizon):
        hp = sim.players[hi]
        poss.append((tuple(hp.position), tuple(hp.orientation)))
        vis = geo.visible_cells(terrain, tuple(hp.position), tuple(hp.orientation), fov)
        for c in cells:
            if c not in sight and c in vis:
                sight[c] = k
        ah, _ = h.action(sim)
        sim, _, _, _ = mdp.get_state_transition(
            sim, (Action.STAY, ah) if robot_idx == 0 else (ah, Action.STAY))
        if mdp.is_terminal(sim):
            break
    walk = set(TruthView(mdp, state).walkable)
    hp0 = tuple(state.players[hi].position)
    field = geo.dist_field(walk | {hp0}, hp0)
    dist = {c: geo.path_len_in(field, walk | {hp0}, c) for c in cells}
    reach = [c for c in cells if dist.get(c) is not None]
    vis_best = min((c for c in reach if c in sight),
                   key=lambda c: (sight[c], dist[c]), default=None)
    near_best = min(reach, key=lambda c: (dist[c], sight.get(c, 10 ** 6)),
                    default=None)
    return vis_best, near_best, sight, dist


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layouts", default="back_bar,pantry")
    p.add_argument("--fov", type=int, default=60)
    p.add_argument("--seeds", default="0-1")
    p.add_argument("--robot", default="qmdp-greedy")
    p.add_argument("--patience", type=int, default=120,
                   help="ticks a trial waits for the human to collect")
    p.add_argument("--max-decisions", type=int, default=6,
                   help="stash decisions sampled per episode")
    p.add_argument("--horizon", type=int, default=200)
    a = p.parse_args(argv)

    lays = a.layouts.split(",") if a.layouts != "all" else LAYOUTS
    seeds = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi_ = part.split("-")
            seeds += list(range(int(lo), int(hi_) + 1))
        else:
            seeds.append(int(part))

    tally = collections.Counter()
    rows = []
    for lay in lays:
        for seed in seeds:
            mdp = SteakHouseGridworld.from_layout_name(lay)
            env = OvercookedEnv.from_mdp(mdp, horizon=a.horizon, info_level=0)
            env.reset()
            human = LimitedVisionHuman(mdp, a.fov, agent_index=1,
                                       forget_horizon=FORGET_HORIZON, seed=seed)
            bot, drv = make_robot(a.robot, mdp, 0, 1, seed)
            n = 0
            done = False
            while not done and n < a.max_decisions:
                st = env.state
                r_act, info = bot.action(st)
                me = st.players[0]
                item = me.held_object.name if me.held_object else None
                plan = info.get("plan")
                if item in SHAREABLE and plan and plan[0] == "stash":
                    cells = sorted(TruthView(mdp, st).free_counters())
                    fov = bot._map_cone()[0]
                    if fov is not None and cells:
                        vis, near, sight, dist = rank_candidates(
                            mdp, st, bot.post.shadows[fov], fov, cells, 0, 60)
                        pick = plan[1]
                        got = {}
                        for lbl, c in (("VISIBLE", vis), ("NEAR", near),
                                       ("FILTER", pick)):
                            if c is None:
                                got[lbl] = (None, None, None)
                                continue
                            pu, pr = trial(mdp, st, bot.post.shadows[fov], c,
                                           item, 0, a.patience)
                            got[lbl] = (c, pu, pr)
                        n += 1
                        rows.append((lay, seed, env.t, item, sight, dist, got))
                        best = [k for k in ("VISIBLE", "NEAR", "FILTER")
                                if got[k][1] is not None]
                        if best:
                            w = min(best, key=lambda k: got[k][1])
                            tally[w] += 1
                        else:
                            tally["none collected"] += 1
                ha, _ = human.action(st)
                if drv is not None:
                    drv.update(st, ha)
                nxt, _, done, _ = env.step((r_act, ha))
                done = done or mdp.is_terminal(nxt)

    print("SEE-IT-HAPPEN vs EASY-TO-REACH, realised pickup ticks (lower better)")
    print("patience %d, fov %d, robot %s\n" % (a.patience, a.fov, a.robot))
    print("%-12s %4s %4s %-12s | %-22s %-22s %-22s"
          % ("layout", "seed", "t", "item", "VISIBLE cell/see/got",
             "NEAR cell/dist/got", "FILTER cell/got"))
    for lay, seed, t, item, sight, dist, got in rows:
        v, nr, f = got["VISIBLE"], got["NEAR"], got["FILTER"]
        print("%-12s %4d %4d %-12s | %-22s %-22s %-22s"
              % (lay, seed, t, item,
                 "%s see%s got%s" % (v[0], sight.get(v[0]), v[1]),
                 "%s d%s got%s" % (nr[0], dist.get(nr[0]), nr[1]),
                 "%s got%s" % (f[0], f[1])))
    print("\nwins (earliest realised pickup, 3-way; ties go to the first listed):")
    for k, c in tally.most_common():
        print("   %-16s %d" % (k, c))

    # THE HEAD-TO-HEAD IS THE CLAIM THAT MATTERS. A 3-way tally hides ties and
    # scores a one-tick loss the same as never being collected at all, so it cannot
    # answer "does the shipped model ever do WORSE than picking the most visible
    # counter". This does, and it counts a failure to collect as the horizon so
    # never-collected is the worst possible outcome rather than a missing row.
    def val(x, patience):
        return patience if x is None else x

    for other in ("VISIBLE", "NEAR"):
        w = l = t_ = 0
        worst = []
        for lay, seed, t, item, sight, dist, got in rows:
            f = val(got["FILTER"][1], a.patience)
            o = val(got[other][1], a.patience)
            if f < o:
                w += 1
            elif f > o:
                l += 1
                worst.append((f - o, lay, seed, t, got["FILTER"], got[other]))
            else:
                t_ += 1
        print("\nFILTER vs %s   better %d   equal %d   WORSE %d   (of %d)"
              % (other, w, t_, l, len(rows)))
        for d, lay, seed, t, fg, og in sorted(worst, reverse=True)[:8]:
            print("   WORSE by %2d ticks  %-12s seed%d t%-3d  filter %s got%s   %s %s got%s"
                  % (d, lay, seed, t, fg[0], fg[1], other, og[0], og[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
