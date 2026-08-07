"""MODULE OPPORTUNITY.  Not "can a robot help" -- "can THIS module help".

    python opportunity.py

===========================================================================
THE DISTINCTION THAT MATTERS, AND THE ONE I GOT WRONG FIRST
===========================================================================
The capacity sweep showed the human alone DNFs at 5 orders while a competent
robot finishes in ~360 ticks. That is 126 ticks of headroom -- for ANY robot.
It says nothing about the module, which does not replace the baseline; it
re-ranks the baseline's own candidates using partner-shaped terms.

So the quantity to measure is not "how much does a robot help" but "how often
does a situation arise that the MODULE's mechanisms can act on at all". If that
count is near zero, the module is a no-op no matter how good the cost function
is -- which is precisely how three previous architectures returned null.

The warning was already visible and I nearly walked past it: `h_wasted = 0` on
every episode of every layout. Zero completed-and-pointless trips means zero
redundancy events, which means two of the module's three mechanisms have nothing
to fire on.

===========================================================================
WHAT THE MODULE CAN ACTUALLY DO, AND WHAT EACH NEEDS TO EXIST
===========================================================================
    #1 be seen holding/doing X      needs the human to be ABOUT to duplicate X
    #2 place things where seen      needs a placement that changes their plan
    #3 not do what they are doing   needs OVERLAP in what the two are doing

All three are made of the same raw material: a moment where the two agents are
aimed at the same work, plus a fact about whether the human can see it. So:

    overlap         ticks where robot and human are engaged on the SAME station
                    line. No overlap -> nothing to deconflict, nothing to signal.
    hidden_overlap  overlap ticks where the human CANNOT see the robot.
                    <-- THE MODULE'S OPPORTUNITY, and the number to read first.
                    Overlap they can see resolves itself: their own model
                    yields. Overlap they cannot see is the case where both
                    agents spend the effort and one trip was for nothing.
    seen_overlap    overlap they CAN see. Free wins the module does not need to
                    earn -- and if this dwarfs hidden_overlap, the layout is
                    handing the human the answer and the module is redundant.
    true_wasted     the human completed a fetch/drop that the TRUE world state
                    made pointless. Counted against truth, not against belief,
                    because the human model's own n_wasted_commits excludes the
                    UNKNOWN case -- a human who has FORGOTTEN is not counted as
                    wasting a trip, which is exactly the case occlusion creates.
                    That exclusion is why h_wasted reads 0 here.
    stale_ready     ticks where a timed station is READY in truth and the human
                    believes otherwise. The raw material for unseen assistance.

A layout is worth building a module on when hidden_overlap and true_wasted are
comfortably non-zero. Everything else is decoration.
"""

import argparse
import json

import _paths  # noqa: F401   MUST be first
import layouts
from env import Kitchen, make_human, ROBOT_INDEX, HUMAN_INDEX, DNF_PENALTY
from planner import make_planner

from fov.human.agent.limited_vision_human import (                    # noqa: E402
    WASTED_IF_OCCUPIED, TERRAIN_KIND, ROBOT_HELD_FETCH, UNKNOWN)

TIMED = ("pot", "board", "sink")


def line_of(planner, subtask):
    """The CONTESTED station kind for a subtask, or "".

    Not `target_kind` alone: that returns where the agent WALKS. For pickup_meat
    it returns the meat dispenser, and two agents at a dispenser duplicate
    nothing -- it is infinite and uncontested. The station that makes the errand
    redundant is the POT, because a full pot is what makes the trip pointless.
    WASTED_IF_OCCUPIED is the human's own table for exactly that; target_kind is
    the fallback for the three subtasks it does not list, where the walk-to
    station IS the contested one.
    """
    if not subtask:
        return ""
    return WASTED_IF_OCCUPIED.get(subtask) or planner.target_kind(subtask)


def engaged_line(agent_obj, mdp, state, idx, planner):
    """Which contested station line an agent is on: what it CARRIES takes
    precedence, else the station it is FACING."""
    p = state.players[idx]
    held = p.held_object.name if p.held_object else None
    if held:
        carried = line_of(planner, ROBOT_HELD_FETCH.get(held, ""))
        if carried:
            return carried
    fx, fy = p.position[0] + p.orientation[0], p.position[1] + p.orientation[1]
    if fx < 0 or fy < 0:
        return ""
    try:
        kind = TERRAIN_KIND.get(mdp.terrain_mtx[fy][fx], "")
    except Exception:
        return ""
    return kind if kind in TIMED else ""


def true_state_of(human, state, loc):
    """Ground truth at a station, via the human's own parser so we cannot
    disagree with it about what 'ready' means."""
    try:
        return human._true_state_of(state, loc)
    except Exception:
        return UNKNOWN


def run_one(layout, fov, seed, robot_kind, n_orders, horizon):
    k = Kitchen(layout, n_orders=n_orders, horizon=horizon)
    k.reset()
    human = make_human(k.mdp, fov, seed)
    robot = make_planner(robot_kind, k.mdp, seed)
    hp = human.planner

    c = dict(overlap=0, hidden_overlap=0, seen_overlap=0, true_wasted=0,
             stale_ready=0, ticks=0, deliveries=0)
    ticks = []

    while k.t < horizon and not k.mdp.is_terminal(k.state):
        s = k.state
        a, _ = robot.action(s)
        h_act, h_info = human.action(s)          # ground-truth label, DIAGNOSTIC ONLY
        c["ticks"] += 1

        # ---- overlap, and whether it is hidden ---------------------------
        #The human's line comes from _current -- what they are DOING this tick,
        #not _sampled, what they would LIKE to do. An intention they cannot act
        #on is not a plan worth protecting.
        h_line = line_of(hp, human._current)
        r_line = engaged_line(robot, k.mdp, s, ROBOT_INDEX, hp)
        if h_line and r_line and h_line == r_line:
            c["overlap"] += 1
            rp = s.players[ROBOT_INDEX]
            if human.visible(s, rp.position):
                c["seen_overlap"] += 1
            else:
                c["hidden_overlap"] += 1

        # ---- stale readiness: unseen-assistance raw material --------------
        for kind in TIMED:
            for loc in human.stations.get(kind, []):
                truth = true_state_of(human, s, loc)
                bel = human.beliefs.get(loc)
                if truth == "ready" and (bel is None or bel.value != "ready"):
                    c["stale_ready"] += 1

        # ---- a trip that TRUTH made pointless -----------------------------
        #Checked on arrival, against the real world rather than the belief that
        #sent them. This is the number the human's own counter cannot report.
        sub = human._current
        wk = WASTED_IF_OCCUPIED.get(sub or "")
        if wk and human.prev_chosen_subtask == sub:
            locs = human.stations.get(wk, [])
            if locs and all(true_state_of(human, s, l) not in ("empty", UNKNOWN)
                            for l in locs):
                near = any(abs(s.players[HUMAN_INDEX].position[0] - l[0])
                           + abs(s.players[HUMAN_INDEX].position[1] - l[1]) <= 1
                           for l in locs)
                if near:
                    c["true_wasted"] += 1

        sparse, done = k.step(a, h_act)
        if sparse > 0:
            for _ in range(max(1, int(round(sparse / float(k.mdp.delivery_reward))))):
                c["deliveries"] += 1
                ticks.append(k.t)
        if done:
            break

    finished = c["deliveries"] >= (n_orders - 1)
    c["completion"] = ticks[-1] if finished else horizon + DNF_PENALTY
    c["finished"] = finished
    c["h_abandoned"] = human.n_abandoned
    c["h_wasted"] = human.n_wasted_commits
    return c


def main(argv=None):
    p = argparse.ArgumentParser("how often can the MODULE do anything?")
    p.add_argument("--layouts", type=str, default=",".join(layouts.GRIDS))
    p.add_argument("--fovs", type=str, default="30,90,180,360")
    p.add_argument("--robots", type=str, default="blind,react")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--n_orders", type=int, default=6)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--out", type=str, default="opportunity.json")
    args = p.parse_args(argv)

    layouts.stage()
    names = [x for x in args.layouts.split(",") if x.strip()]
    fovs = [int(x) for x in args.fovs.split(",") if x.strip()]
    robots = [x for x in args.robots.split(",") if x.strip()]

    print("n_orders=%d (=%d deliverable)  horizon=%d  %d episodes"
          % (args.n_orders, args.n_orders - 1, args.horizon, args.episodes))
    print("hidden_overlap is THE number: overlap the human cannot see.\n")

    out = []
    for name in names:
        print("=" * 82)
        print(name)
        print("  %-6s %-5s %6s %8s %8s %8s %9s %8s %6s"
              % ("robot", "fov", "compl", "overlap", "HIDDEN", "seen",
                 "trueWast", "staleRdy", "aband"))
        for rk in robots:
            for fov in fovs:
                per = [run_one(name, fov, s, rk, args.n_orders, args.horizon)
                       for s in range(args.episodes)]
                m = {k: sum(r[k] for r in per) / len(per) for k in
                     ("completion", "overlap", "hidden_overlap", "seen_overlap",
                      "true_wasted", "stale_ready", "h_abandoned")}
                out.append(dict(layout=name, robot=rk, fov=fov, **m))
                print("  %-6s %-5d %6.0f %8.1f %8.1f %8.1f %9.1f %8.1f %6.1f"
                      % (rk, fov, m["completion"], m["overlap"],
                         m["hidden_overlap"], m["seen_overlap"],
                         m["true_wasted"], m["stale_ready"], m["h_abandoned"]))
        print()

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
