"""Checks for the VALUE FUNCTION and the degrade-to-baseline guarantee.

    python -m robot.filter.tests.test_value_tail

PARITY is the one that must never go red. The whole claim of this layer is that
it can improve on its nominal policy and cannot scramble it, and the only reason
that is true is that the baseline's own action is always a candidate and always
wins ties. qmdp-base collapses the search to that single candidate, so its
action trace has to equal its baseline's tick for tick -- and it has to do so
while running the real rollout machinery, or it is checking nothing.

The baseline it is checked against is `handoff`, NOT `handoff`, because
that is the rung the filter actually wraps: the layer consumes a distribution
over sub-tasks and only a -stoch wrapper has a real one. Parity therefore also
proves the rng streams line up -- the wrapper draws from its own seeded stream,
and the filter must not perturb it by calling action() twice or sampling again.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.environ.get("STEAK_ROOT", os.path.dirname(ROOT)))
sys.path.insert(0, ROOT)

import overcooked_ai_py                                            # noqa: E402
overcooked_ai_py.LAYOUTS_DIR = os.path.join(ROOT, "layout", "layouts")

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv      # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld  # noqa: E402
from human.limited_vision_human import LimitedVisionHuman          # noqa: E402
from robot.filter.core.qmdp import clone_human, norm_entropy    # noqa: E402
from robot.filter.core.value_tail import tail_ticks, _geo            # noqa: E402
from common import geometry as geo                              # noqa: E402
from overcooked_ai_py.mdp.actions import Action                 # noqa: E402
from common.views import TruthView                              # noqa: E402
from robot.methods import make_robot                               # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def trace(layout, fov, seed, method, horizon):
    """[(robot_action, human_action)] for one episode."""
    mdp = SteakHouseGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env.reset()
    human = LimitedVisionHuman(mdp, fov, agent_index=1, seed=seed)
    robot, post = make_robot(method, mdp, 0, 1, seed)
    out, done = [], False
    while not done:
        s = env.state
        ra, _ = robot.action(s)
        ha, _ = human.action(s)
        if post is not None:
            post.update(s, ha)
        out.append((str(ra), str(ha)))
        nxt, _, done, _ = env.step((ra, ha))
        done = done or mdp.is_terminal(nxt)
    return out


def test_parity(layouts, fovs, seeds, horizon=60):
    for lay in layouts:
        for fov in fovs:
            for sd in seeds:
                a = trace(lay, fov, sd, "handoff", horizon)
                b = trace(lay, fov, sd, "qmdp-base", horizon)
                same = a == b
                first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                check("parity %s fov%d s%d" % (lay, fov, sd), same,
                      "" if same else "diverges at tick %s: %s vs %s"
                      % (first, a[first], b[first]))


def test_determinism(layout="divide", fov=60, seed=0, horizon=25):
    a = trace(layout, fov, seed, "qmdp", horizon)
    b = trace(layout, fov, seed, "qmdp", horizon)
    check("determinism: same inputs, same trace", a == b)


def test_clone_is_faithful(layout="divide", fov=60, seed=0, warm=12):
    """A clone taken mid-episode must pick the same next action as its original.

    This is the honesty check for the rollout: if the clone's first simulated
    action differs from what that shadow would really do, every rollout starts
    with a systematic error. It only holds because the clone carries `t`,
    `last_subtask` and the rng along with the view -- copying beliefs alone, as
    the superseded subtask-level filter did, fails this.
    """
    mdp = SteakHouseGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=warm + 5, info_level=0)
    env.reset()
    human = LimitedVisionHuman(mdp, fov, agent_index=1, seed=seed)
    robot, post = make_robot("handoff", mdp, 0, 1, seed)
    for _ in range(warm):
        s = env.state
        ra, _ = robot.action(s)
        ha, _ = human.action(s)
        env.step((ra, ha))

    s = env.state
    c = clone_human(human)
    check("clone carries t", c.t == human.t, "%s vs %s" % (c.t, human.t))
    check("clone carries last_subtask", c.last_subtask == human.last_subtask)
    a_clone, _ = c.action(s)
    a_real, _ = human.action(s)
    check("clone predicts the same next action", a_clone == a_real,
          "%s vs %s" % (a_clone, a_real))
    check("clone shares the mdp (no deep copy of the world)", c.mdp is human.mdp)


def test_entropy():
    flat = {30: .2, 60: .2, 90: .2, 180: .2, 360: .2}
    sharp = {30: .96, 60: .01, 90: .01, 180: .01, 360: .01}
    check("norm_entropy: flat is 1", abs(norm_entropy(flat) - 1.0) < 1e-9)
    check("norm_entropy: sharp is near 0", norm_entropy(sharp) < 0.25)


def test_tail_prices_the_stash(layout="back_bar", fov=90, warm=25):
    """A stash on a pass counter the human KNOWS about must be the cheapest.

    This is the property the whole layer rests on, and it has to be emergent:
    nothing anywhere says "prefer visible counters". If this ordering ever
    inverts, the tail has stopped pricing geography and no amount of controller
    tuning will put the behaviour back.
    """
    from overcooked_ai_py.mdp.overcooked_mdp import ObjectState
    mdp = SteakHouseGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=warm + 5, info_level=0)
    env.reset()
    h = LimitedVisionHuman(mdp, fov, agent_index=1, seed=0)
    for _ in range(warm):
        ha, _ = h.action(env.state)
        env.step(((0, 0), ha))
    s = env.state
    g = _geo(mdp, s)
    P = set(g.passes)
    free = list(TruthView(mdp, s).free_counters())

    def price(c):
        t = s.deepcopy()
        t.objects[c] = ObjectState(9999, "washed_plate", c)
        return tail_ticks(mdp, t, human_view=h.view, human_index=1)

    known = [c for c in free if c in P and h.view._fresh(c) is not None]
    blind = [c for c in free if c in P and h.view._fresh(c) is None]
    away = [c for c in free if c not in P]
    if not (known and away):
        check("tail: stash pricing (needs all three kinds)", True, "skipped")
        return
    mk = min(price(c) for c in known)
    ma = min(price(c) for c in away)
    check("tail: a KNOWN pass counter beats a non-pass one", mk <= ma,
          "%.1f vs %.1f" % (mk, ma))
    if blind:
        mb = min(price(c) for c in blind)
        check("tail: a KNOWN pass counter beats a blind-spot one", mk <= mb,
              "%.1f vs %.1f" % (mk, mb))


def test_C_ranks_the_greedy_step(layouts=("butchery", "divide"), fov=60, ticks=18):
    """C must cost a step AWAY from the target above the step towards it.

    The unit test that found the real bug. With the head running to a fixed T
    this scored 14-59% -- at chance, and on divide actively backwards -- because
    a fixed t_end throws away the one tick the action choice controls, leaving it
    to be recovered from a tail that resolves at 4-8 ticks. Terminating the head
    at the first INTERACT put it at 75-81%. If this drops back to chance, the
    head/tail composition has broken again and every downstream number is void.
    """
    moves = [a for a in Action.MOTION_ACTIONS if a != Action.STAY]
    for lay in layouts:
        mdp = SteakHouseGridworld.from_layout_name(lay)
        env = OvercookedEnv.from_mdp(mdp, horizon=ticks + 2, info_level=0)
        env.reset()
        h = LimitedVisionHuman(mdp, fov, agent_index=1, seed=0)
        bot, post = make_robot("qmdp-greedy", mdp, 0, 1, 0)
        wins = tot = 0
        for _ in range(ticks):
            s = env.state
            ranked, cones = bot.baseline.rank_subtasks(s), bot._cones()
            if ranked and cones:
                me = s.players[0]
                pos, orient = tuple(me.position), tuple(me.orientation)
                walk = bot.walkable(s)
                jobs, _ = bot._jobs(s, ranked, pos, orient, walk)
                if jobs:
                    v, cs = jobs[0]
                    g = cs[0]
                    mv, arr = geo.step_towards(walk, pos, orient, g)
                    if mv and not arr and (-mv[0], -mv[1]) in moves:
                        opp = (-mv[0], -mv[1])
                        cb = sum(w * bot._rollout(s, g, f, mv) for f, w in cones.items())
                        co = sum(w * bot._rollout(s, g, f, opp) for f, w in cones.items())
                        tot += 1
                        wins += cb < co
            ra, _ = bot.action(s)
            ha, _ = h.action(s)
            nxt, _, done, _ = env.step((ra, ha))
            if done or mdp.is_terminal(nxt):
                break
        if tot:
            check("C ranks the greedy step above stepping away (%s)" % lay,
                  wins >= 0.6 * tot, "%d/%d" % (wins, tot))


if __name__ == "__main__":
    LAYOUTS = ["divide", "butchery", "pantry", "back_bar"]
    test_entropy()
    test_clone_is_faithful()
    test_determinism()
    test_tail_prices_the_stash()
    test_C_ranks_the_greedy_step()
    test_parity(LAYOUTS, [30, 90, 360], [0, 1])
    print("\n%d failure(s)" % len(FAILS))
    for f in FAILS:
        print("  " + f)
    sys.exit(1 if FAILS else 0)
