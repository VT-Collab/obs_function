"""THE SCREENING PROBE.  Does this layout give a limited FOV anything to do?

Run this BEFORE building a planner or a module on top of a layout. It answers
one question -- is there a persistent gap between what the human knows and what
is true -- and if the answer is no, nothing built on top can possibly work,
because every mechanism the module has operates on that gap.

    python probe.py

===========================================================================
WHY THIS FILE EXISTS AT ALL
===========================================================================
Three module architectures have now returned null on gc00/gc03/gc04/gc06, and
the reason was never the module: on a 3x3 convex room the human re-observes
every station within ~4 ticks, FORGET_HORIZON is 12, so K_t == s essentially
always. `h_wasted = 0` on every episode ever run there. The gap the method
operates on did not exist, so there was nothing to infer and nothing to act on.

That is a property of the LAYOUT and it is measurable in seconds without a
robot, a checkpoint or a cost function. Measuring it first is the difference
between iterating on the environment and iterating on a module that was never
the problem.

===========================================================================
THE TWO HALVES
===========================================================================
STATIC (geometry only, no simulation)
    legs        pairwise walking distance between stations. The longest leg must
                exceed FORGET_HORIZON = 12 or beliefs never decay in transit and
                the human effectively has perfect memory.
    blind       for each timed station, the fraction of walkable cells from
                which a straight line to it crosses a wall. This is ignorance
                that survives at EVERY cone width, including 360, because
                _visible_cone short-circuits to True at 360 and only _clear_los
                can still block. gc00 scores 0.00 here by construction.

DYNAMIC (human alone, robot plays STAY)
    gap         mean number of facts the human has wrong, per tick
    gap_timed   ...restricted to the three stations that carry a hidden clock
                (grill / board / sink). These are the only facts that change
                without anyone touching them, so they are the only ones a human
                can be wrong about through no fault of their own -- and they are
                exactly what an unseen-assistance opportunity is made of.
    persist     mean number of CONSECUTIVE ticks a fact stays wrong. A gap that
                closes itself next tick is an attention lag, not ignorance. This
                is the number that separates a real blind spot from gc00's
                two-tick refresh, and it should be read before `gap`.
    solo        completion time with no robot at all. The FOV cost is
                solo(30) - solo(360); if that is ~0 the cone is cosmetic.

===========================================================================
WHAT A LAYOUT HAS TO SCORE TO BE WORTH BUILDING ON
===========================================================================
    max leg   >= 12      (FORGET_HORIZON) or memory never decays
    blind     >= ~0.3    on at least one timed station, or there is no
                         structural ignorance and we are back to cone-only
    persist   >= ~5      or the gap closes before anyone could act on it
    gap_timed >  0       or there is literally no unseen assistance to give

These are screening thresholds, not tuned constants -- nothing downstream reads
them. They are written down so the decision to keep or discard a grid is made
against a stated bar instead of by eye.
"""

import argparse
import collections
import json

import _paths  # noqa: F401   MUST be first
import layouts
from env import (Kitchen, make_human, ROBOT_INDEX, HUMAN_INDEX, DNF_PENALTY,
                 HORIZON, N_ORDERS)

from overcooked_ai_py.mdp.overcooked_mdp import Action                # noqa: E402
from fov.human.agent.limited_vision_human import (                    # noqa: E402
    TERRAIN_KIND, FORGET_HORIZON, SIGHT_RADIUS)

#the three stations with a hidden clock. Everything else in this kitchen only
#changes when somebody touches it, and a human who was standing there when it
#happened is not ignorant of it.
TIMED = ("pot", "board", "sink")

#screening bar, see the header. Read, never optimised against.
BAR = dict(leg=FORGET_HORIZON, blind=0.30, persist=5.0)


# =========================================================================
# 1. STATIC GEOMETRY
# =========================================================================
def walkable(mtx):
    """{(x, y)} for every floor tile. mtx is indexed [y][x] -- the single most
    common bug in this codebase, and it does not raise, it just transposes your
    kitchen."""
    return {(x, y) for y, row in enumerate(mtx)
            for x, c in enumerate(row) if c == ' '}


def stations(mtx):
    """{(x, y): kind} for every tile that is a station we care about."""
    out = {}
    for y, row in enumerate(mtx):
        for x, c in enumerate(row):
            kind = TERRAIN_KIND.get(c, "")
            if kind and kind != "counter":
                out[(x, y)] = kind
    return out


def bfs(start_cells, free):
    """Steps from the nearest of `start_cells` to every reachable floor tile."""
    dist = {c: 0 for c in start_cells if c in free}
    q = collections.deque(dist)
    while q:
        c = q.popleft()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if n in free and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return dist


def approach_cells(loc, free):
    """The floor tiles you can stand on to use the station at `loc`."""
    return [(loc[0] + d[0], loc[1] + d[1]) for d in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if (loc[0] + d[0], loc[1] + d[1]) in free]


def static_report(kitchen, human):
    """Geometry only. `human` is used solely for its own _clear_los, so the
    occlusion we report is the one the agent will actually experience rather
    than a reimplementation that might disagree with it."""
    mtx = kitchen.mdp.terrain_mtx
    free = walkable(mtx)
    st = stations(mtx)

    #--- legs: walking distance between every pair of stations ---------------
    legs = {}
    unreachable = []
    for loc, kind in st.items():
        src = approach_cells(loc, free)
        if not src:
            unreachable.append((loc, kind))
            continue
        d = bfs(src, free)
        for loc2, kind2 in st.items():
            if loc2 == loc:
                continue
            tgt = approach_cells(loc2, free)
            reach = [d[c] for c in tgt if c in d]
            if reach:
                legs[(kind, kind2)] = min(reach)

    #--- blind: fraction of standing positions with no line of sight ---------
    #Cone ignored on purpose. This isolates STRUCTURAL ignorance -- the part a
    #wall creates, which no amount of turning your head fixes and which survives
    #at fov=360 where the cone test short-circuits to True.
    blind = {}
    for loc, kind in st.items():
        if kind not in TIMED:
            continue
        n_blocked = sum(1 for c in free
                        if not human._clear_los(c, loc)
                        or max(abs(c[0] - loc[0]), abs(c[1] - loc[1])) > SIGHT_RADIUS)
        blind[kind] = n_blocked / max(len(free), 1)

    #--- connectivity: one component, or the human can get stranded ----------
    comps = 0
    seen = set()
    for c in free:
        if c in seen:
            continue
        comps += 1
        seen |= set(bfs([c], free))

    return dict(
        w=kitchen.mdp.shape[0], h=kitchen.mdp.shape[1],
        n_free=len(free), n_stations=len(st), components=comps,
        unreachable=unreachable,
        max_leg=max(legs.values()) if legs else 0,
        mean_leg=round(sum(legs.values()) / max(len(legs), 1), 1),
        blind={k: round(v, 3) for k, v in blind.items()},
        blind_max=round(max(blind.values()), 3) if blind else 0.0,
    )


# =========================================================================
# 2. THE EPISTEMIC GAP
# =========================================================================
def wrong_set(human, state):
    """Which facts the human currently has WRONG.

    Same computation as filter/cost_function.wrong_beliefs, run on the real
    human instead of a shadow -- both expose .beliefs, .ROBOT, .agent_index and
    ._true_state_of, so the code is identical. Written out here rather than
    imported so this package does not depend on the filter package.

    The try/except goes around ONE entry with `continue`, never around the whole
    loop: a caller computing `before - after` would read an empty set as "every
    fact just got fixed", the largest possible credit, handed out because
    something threw.
    """
    wrong = set()
    for loc, bel in human.beliefs.items():
        try:
            if loc == human.ROBOT:
                rp = state.players[1 - human.agent_index]
                truth = rp.held_object.name if rp.held_object else "none"
            else:
                truth = human._true_state_of(state, loc)
        except Exception:
            continue
        if bel.value != truth:
            wrong.add(loc)
    return wrong


def kind_of(kitchen, loc):
    """Station kind at a belief key, or '' for the special ROBOT key."""
    try:
        return TERRAIN_KIND.get(kitchen.mdp.terrain_mtx[loc[1]][loc[0]], "")
    except Exception:
        return ""


def solo_episode(kitchen, fov, seed, occlude=True, horizon=None,
                 familiar=True):
    """The human alone; the robot plays STAY every tick.

    No robot at all is the right control here -- we are measuring a property of
    the LAYOUT and the human, and any robot would contaminate it by revealing
    things. STAY rather than removing the player because the mdp expects two.
    """
    horizon = horizon or kitchen.horizon
    kitchen.reset()
    human = make_human(kitchen.mdp, fov, seed, occlude=occlude,
                       familiar=familiar)

    deliveries, ticks = 0, []
    gap, gap_timed = [], []
    #open[loc] = tick at which this fact most recently BECAME wrong
    open_runs, run_lengths = {}, []

    while kitchen.t < horizon and not kitchen.mdp.is_terminal(kitchen.state):
        s = kitchen.state
        h, _info = human.action(s)          # _info["subtask"] is ground truth: never read

        w = wrong_set(human, s)
        gap.append(len(w))
        gap_timed.append(sum(1 for loc in w if kind_of(kitchen, loc) in TIMED))

        #persistence bookkeeping: opened runs that just closed get recorded
        for loc in list(open_runs):
            if loc not in w:
                run_lengths.append(kitchen.t - open_runs.pop(loc))
        for loc in w:
            open_runs.setdefault(loc, kitchen.t)

        sparse, done = kitchen.step(Action.STAY, h)
        if sparse > 0:
            n = int(round(sparse / float(kitchen.mdp.delivery_reward)))
            for _ in range(max(1, n)):
                deliveries += 1
                ticks.append(kitchen.t)
        if done:
            break

    #runs still open at the end of the episode count at their current length --
    #dropping them would systematically discard the LONGEST blind spots, which
    #are precisely the ones we are looking for.
    for loc, t0 in open_runs.items():
        run_lengths.append(kitchen.t - t0)

    finished = len(ticks) >= (kitchen.n_orders - 1)
    completion = ticks[-1] if finished else horizon + DNF_PENALTY
    n = max(len(gap), 1)
    return dict(
        fov=fov, seed=seed, deliveries=deliveries, steps=kitchen.t,
        completion=completion, finished=finished,
        gap=round(sum(gap) / n, 2),
        gap_timed=round(sum(gap_timed) / n, 2),
        persist=round(sum(run_lengths) / max(len(run_lengths), 1), 1),
        persist_max=max(run_lengths) if run_lengths else 0,
        h_wasted=human.n_wasted_commits, h_explore=human.n_explore,
        h_checks=human.n_checks, h_abandoned=human.n_abandoned,
    )


# =========================================================================
# 3. REPORT
# =========================================================================
def run(layout_names, fovs, episodes, occlude=True, horizon=HORIZON,
        n_orders=N_ORDERS, familiar=True):
    out = {}
    for name in layout_names:
        kitchen = Kitchen(name, n_orders, horizon)
        probe_human = make_human(kitchen.mdp, 90, 0, occlude=occlude,
                                 familiar=familiar)
        stat = static_report(kitchen, probe_human)

        print("\n" + "=" * 78)
        print("%s   %dx%d, %d floor tiles, %d stations, %d component(s)"
              % (name, stat["w"], stat["h"], stat["n_free"], stat["n_stations"],
                 stat["components"]))
        if stat["unreachable"]:
            print("  !! UNREACHABLE STATIONS: %s" % (stat["unreachable"],))
        if stat["components"] != 1:
            print("  !! GRID IS NOT CONNECTED -- the human can be stranded")
        print("  legs: max %d (bar %d)  mean %.1f"
              % (stat["max_leg"], BAR["leg"], stat["mean_leg"]))
        print("  blind (fraction of standing cells with no line of sight): %s"
              % ", ".join("%s %.2f" % (k, v) for k, v in sorted(stat["blind"].items())))
        verdict = ("PASS" if stat["max_leg"] >= BAR["leg"]
                   and stat["blind_max"] >= BAR["blind"] else "FAIL")
        print("  static verdict: %s" % verdict)

        print("  %-5s %5s %5s %6s %7s %8s %7s %7s %7s"
              % ("fov", "compl", "deliv", "gap", "gapTIME", "persist", "pmax",
                 "wasted", "explor"))
        rows = []
        for fov in fovs:
            per = [solo_episode(kitchen, fov, s, occlude, horizon, familiar)
                   for s in range(episodes)]
            avg = {k: (sum(r[k] for r in per) / len(per))
                   for k in ("completion", "deliveries", "gap", "gap_timed",
                             "persist", "persist_max", "h_wasted", "h_explore")}
            rows.append(dict(layout=name, fov=fov, **{k: round(v, 2)
                                                      for k, v in avg.items()}))
            print("  %-5d %5.0f %5.1f %6.2f %7.2f %8.1f %7.0f %7.1f %7.1f"
                  % (fov, avg["completion"], avg["deliveries"], avg["gap"],
                     avg["gap_timed"], avg["persist"], avg["persist_max"],
                     avg["h_wasted"], avg["h_explore"]))

        cost = rows[0]["completion"] - rows[-1]["completion"]
        print("  FOV cost  solo(%d) - solo(%d) = %+.0f steps"
              % (fovs[0], fovs[-1], cost))
        out[name] = dict(static=stat, dynamic=rows, fov_cost=cost)
    return out


def main(argv=None):
    p = argparse.ArgumentParser("layout screening: is there an epistemic gap?")
    p.add_argument("--layouts", type=str, default=",".join(layouts.GRIDS))
    p.add_argument("--fovs", type=str, default="30,90,180,360")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--n_orders", type=int, default=N_ORDERS)
    p.add_argument("--no_occlude", action="store_true",
                   help="ablation: walls block movement but not sight")
    p.add_argument("--stranger", action="store_true",
                   help="ablation: human must DISCOVER station locations too")
    p.add_argument("--out", type=str, default="probe.json")
    args = p.parse_args(argv)

    print("staged:", layouts.stage())
    print("FORGET_HORIZON=%d  SIGHT_RADIUS=%d  occlude=%s  familiar=%s"
          % (FORGET_HORIZON, SIGHT_RADIUS, not args.no_occlude, not args.stranger))

    names = [x for x in args.layouts.split(",") if x.strip()]
    fovs = [int(x) for x in args.fovs.split(",") if x.strip()]
    res = run(names, fovs, args.episodes, not args.no_occlude, args.horizon,
              args.n_orders, not args.stranger)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
