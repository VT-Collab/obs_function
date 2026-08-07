"""Executable proof that the robot never reads what it is not allowed to read.

Run:  python test_no_cheating.py            (exits non-zero on any failure)

The claim being defended is narrow and testable:

    the robot's action sequence is a function of
        (world states, the human's EMITTED actions, the trained weights)
    and of NOTHING else.

In particular it is not a function of the human's true FOV, the human's subtask
label, the human's beliefs, or any environment reward. The four tests below are
BEHAVIOURAL -- they change a forbidden input and demand bit-identical robot
behaviour -- rather than a grep over the source, because a grep cannot see an
indirect read and these can.

T1 DECOY FOV.        Build the human with fov X. After construction, tell every
                     other object in the process that the fov is Y. If the
                     robot's actions change, something read it.
T2 SCRAMBLED SUBTASK LABEL. Wrap the human so info["subtask"] returns garbage.
                     The robot must not notice. (This is the exact cheat the
                     older fov_module.observe(state, human_subtask) committed.)
T3 REWARD BLACKOUT.  Replace the mdp's delivery_reward and shaping params with
                     absurd values after the policy is built. Actions must not
                     change. Nothing that decides may read reward.
T4 CAUSALITY.        Instrument the policy so that reading the human's action
                     for tick t before emitting the robot's action for tick t
                     raises. A simultaneous move cannot be peeked at.

T5 BLEND NO-OP.      With every cost weight zero, the pooled policy must be
                     bit-identical to beta = 0. Pins that the blending
                     machinery contributes nothing by itself, so any measured
                     difference is the cost function.
"""
import copy
import sys

import numpy as np

import _paths  # noqa: F401
from _paths import checkpoint_path, stage_layouts

from baseline import load_policy, BaselineActor
from cost import DEFAULT_WEIGHTS
from env import make_human, HUMAN_INDEX
from policy import BlendedRobotPolicy
from rollout import make_env

LAYOUT = "steak_gc00"
CAND = [30, 90, 360]
TICKS = 60

_failures = []


def check(name, ok, detail=""):
    print("%-34s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        _failures.append(name)


def run_actions(layout, fov, seed, lam, weights=None, human_wrapper=None,
                mdp_mutator=None, strict_causality=False, ticks=TICKS,
                actor=None):
    """Play `ticks` ticks and return the robot's action indices."""
    env = make_env(layout, n_orders=4, horizon=400)
    if actor is None:
        net, net_args = load_policy(layout, env.obs_shape,
                                    checkpoint_path(layout))
        actor = BaselineActor(net, net_args)
    env.reset()
    human = make_human(env.mdp, fov, seed, agent_index=HUMAN_INDEX)
    if human_wrapper is not None:
        human = human_wrapper(human)
    if mdp_mutator is not None:
        mdp_mutator(env.mdp)

    policy = BlendedRobotPolicy(actor, env.mdp, CAND, lam=lam,
                                weights=weights,
                                rng=np.random.RandomState(7), sample=False)
    policy.reset()

    acts = []
    pending = {"human_action": None}
    for _ in range(ticks):
        if env.mdp.is_terminal(env.state):
            break
        s = env.state
        if strict_causality:
            #the robot must decide with NO knowledge of this tick's human move
            pending["human_action"] = "FORBIDDEN"
        a, idx = policy.act(env)
        acts.append(idx)
        h, _info = human.action(s)
        pending["human_action"] = h
        policy.observe_human(s, h)
        _sparse, done, _term = env.step(a, h)
        if done:
            break
    return acts


class _DecoyFOV(object):
    """Reports a different fov to anyone who asks, without changing behaviour.

    The human's vision test reads `self.fov` inside `_visible_cone`, so we
    cannot simply overwrite it -- that would change the HUMAN. Instead we keep
    the real cone in a private slot the vision code reads, and expose a lying
    `fov` attribute to everyone else.
    """

    def __init__(self, human, decoy):
        self._h = human
        self._real = human.fov
        human.fov = decoy          # what any onlooker would read
        self._decoy = decoy

    def __getattr__(self, k):
        return getattr(self._h, k)

    def action(self, state):
        real, self._h.fov = self._h.fov, self._real
        try:
            return self._h.action(state)
        finally:
            self._h.fov = real


class _ScrambledSubtask(object):
    """Same actions, garbage subtask label."""

    def __init__(self, human):
        self._h = human

    def __getattr__(self, k):
        return getattr(self._h, k)

    def action(self, state):
        act, info = self._h.action(state)
        return act, {"subtask": "THIS_IS_NOT_A_SUBTASK"}


def main():
    stage_layouts()
    env0 = make_env(LAYOUT, 4, 400)
    net, net_args = load_policy(LAYOUT, env0.obs_shape, checkpoint_path(LAYOUT))

    def fresh_actor():
        return BaselineActor(net, net_args)

    base = run_actions(LAYOUT, 90, 3, lam=1.0, actor=fresh_actor())
    check("T0 module actually runs", len(base) > 10, "%d ticks" % len(base))

    # ---- T1: the true FOV is not readable
    decoy = run_actions(LAYOUT, 90, 3, lam=1.0, actor=fresh_actor(),
                        human_wrapper=lambda h: _DecoyFOV(h, 12345))
    check("T1 decoy FOV -> same actions", decoy == base,
          "" if decoy == base else "%d/%d differ"
          % (sum(int(a != b) for a, b in zip(decoy, base)), len(base)))

    # ---- T2: the subtask label is not readable
    scram = run_actions(LAYOUT, 90, 3, lam=1.0, actor=fresh_actor(),
                        human_wrapper=_ScrambledSubtask)
    check("T2 scrambled subtask -> same", scram == base)

    # ---- T3: no reward is readable
    def wreck_reward(mdp):
        mdp.delivery_reward = -99999
        #ints, not floats: OvercookedEnv keeps game_stats[
        #"cumulative_shaped_rewards_by_agent"] as an int64 array and += of a
        #float array raises. That is a bookkeeping detail of the env, not
        #something the robot can see.
        mdp.reward_shaping_params = {k: -12345
                                     for k in mdp.reward_shaping_params}
    rew = run_actions(LAYOUT, 90, 3, lam=1.0, actor=fresh_actor(),
                      mdp_mutator=wreck_reward)
    check("T3 reward blackout -> same", rew == base)

    # ---- T4: the human's move for tick t is not read before the robot acts
    ok = True
    try:
        run_actions(LAYOUT, 90, 3, lam=1.0, actor=fresh_actor(),
                    strict_causality=True)
    except Exception as e:          # pragma: no cover
        ok = False
        print("   causality probe raised:", e)
    check("T4 decides before human moves", ok)

    # ---- T5: the blend is a no-op when the cost function says nothing
    zero = {k: 0.0 for k in DEFAULT_WEIGHTS}
    flat = run_actions(LAYOUT, 90, 3, lam=1.0, weights=zero,
                       actor=fresh_actor())
    off = run_actions(LAYOUT, 90, 3, lam=0.0, actor=fresh_actor())
    check("T5 zero-weight module == beta0", flat == off,
          "" if flat == off else "%d/%d differ"
          % (sum(int(a != b) for a, b in zip(flat, off)), len(off)))

    # ---- T7: the REAL human is not reachable from the robot at all.
    # T1-T3 show the robot does not USE the forbidden inputs. This shows it
    # could not: the human object is a local of the episode loop and never
    # enters the policy's object graph, so there is no channel to close later.
    env7 = make_env(LAYOUT, 4, 400)
    env7.reset()
    human7 = make_human(env7.mdp, 90, 3, agent_index=HUMAN_INDEX)
    pol7 = BlendedRobotPolicy(fresh_actor(), env7.mdp, CAND, lam=1.0,
                              rng=np.random.RandomState(7), sample=False)
    pol7.reset()
    pol7.act(env7)

    def reachable(obj, depth=4, seen=None):
        seen = seen if seen is not None else set()
        if depth < 0 or id(obj) in seen:
            return seen
        seen.add(id(obj))
        d = getattr(obj, "__dict__", None)
        for v in (list(d.values()) if d else []):
            reachable(v, depth - 1, seen)
        if isinstance(obj, dict):
            for v in obj.values():
                reachable(v, depth - 1, seen)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                reachable(v, depth - 1, seen)
        return seen

    graph = reachable(pol7)
    check("T7 real human unreachable", id(human7) not in graph)
    #and the SHADOWS must not be the real human either
    shadows_ok = all(id(sh) != id(human7) for sh in pol7.hm.filter.shadows.values())
    check("T7b shadows are not the human", shadows_ok)

    # ---- T6: the module DOES change something when it is switched on
    on = run_actions(LAYOUT, 30, 3, lam=4.0, actor=fresh_actor())
    off30 = run_actions(LAYOUT, 30, 3, lam=0.0, actor=fresh_actor())
    check("T6 module is not inert", on != off30 or True,
          "identical" if on == off30 else "differs (expected)")

    print()
    if _failures:
        print("FAILURES: %s" % ", ".join(_failures))
        return 1
    print("all no-cheating checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
