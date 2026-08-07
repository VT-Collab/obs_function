"""One episode of robot + limited-vision human, and the metrics off it.

===========================================================================
THE TICK ORDER IS THE EXPERIMENT
===========================================================================
    s = env.state
    a = policy.act(env)            robot decides.  sees s, and human actions
                                   h_0..h_{t-1}.  NOT h_t.
    h, info = human.action(s)      human decides.  `info` carries the true
                                   subtask label and is DISCARDED here -- it is
                                   the single most tempting cheat in this
                                   codebase and the reason the old
                                   fov_module.py's `observe(state,
                                   human_subtask)` signature was abandoned.
    policy.observe_human(s, h)     the filter consumes (s, h). An action is a
                                   physical event; watching your partner move
                                   is not privileged information.
    env.step(a, h)                 simultaneous move resolves.

===========================================================================
METRICS
===========================================================================
The environment's reward function is NOT available to the robot, but it is
what we are measuring, so it is read here and only here:

  deliveries        orders served this episode. `is_terminal` fires at
                    len(order_list) <= 1, so with n_orders=4 the ceiling is 3.
  completion_time   the tick the LAST order was served, or the DNF penalty
                    (horizon + 100) if the team never finished. Same convention
                    fov/robot/policy/old/RESULTS.md used, so the two sets of
                    numbers are directly comparable.
  t_first/t_second/t_third   per-delivery timestamps, which separate "slower"
                    from "stalled at the last step".

Human-side diagnostics (n_wasted_commits, n_explore, ...) come straight off the
human agent's own counters. They explain WHY a condition won; no score depends
on them.
"""
import time

import numpy as np

import _paths  # noqa: F401

from env import SteakHumanRobotEnv, make_human, HUMAN_INDEX, ROBOT_INDEX

DNF_PENALTY = 100


def play_episode(env, actor, policy_factory, fov, seed, temperature=0.5,
                 max_ticks=None, collect_trace=False):
    """Run one episode. `policy_factory(mdp)` builds the robot for this run."""
    env.reset()
    mdp = env.mdp
    human = make_human(mdp, fov, seed, temperature=temperature,
                       agent_index=HUMAN_INDEX)
    policy = policy_factory(mdp)
    policy.reset()

    horizon = env.horizon if max_ticks is None else min(env.horizon, max_ticks)
    deliveries = 0
    delivery_ticks = []
    t0 = time.time()
    fov_map_hist = []
    #what the robot actually did, by action index. A module whose "win" is
    #really "the robot learned to stand still" would show up here and nowhere
    #else, so it is worth the six integers.
    action_hist = [0] * 6

    while env.t < horizon and not mdp.is_terminal(env.state):
        s = env.state

        robot_action, _idx = policy.act(env)
        action_hist[_idx] += 1

        #the human's own decision. info["subtask"] is the ground-truth label --
        #never read.
        human_action, _info = human.action(s)

        policy.observe_human(s, human_action)

        sparse, done, _terminal = env.step(robot_action, human_action)
        if sparse > 0:
            n = int(round(sparse / float(mdp.delivery_reward)))
            for _ in range(max(1, n)):
                deliveries += 1
                delivery_ticks.append(env.t)
        if env.t % 20 == 0:
            fov_map_hist.append(policy.hm.map_fov())
        if done:
            break

    finished = len(delivery_ticks) >= (env.n_orders - 1)
    completion = (delivery_ticks[-1] if finished else horizon + DNF_PENALTY)

    out = {
        "layout": env.layout,
        "fov": fov,
        "seed": seed,
        "deliveries": deliveries,
        "steps": int(env.t),
        "completion_time": int(completion),
        "finished": bool(finished),
        "t_delivery": [int(x) for x in delivery_ticks],
        "wall_s": round(time.time() - t0, 2),
        # -- human-side diagnostics (explanatory only)
        "h_wasted": int(human.n_wasted_commits),
        "h_explore": int(human.n_explore),
        "h_checks": int(human.n_checks),
        "h_abandoned": int(human.n_abandoned),
        "h_delivered": int(human.n_delivered),
        # -- filter diagnostics. p_true is scored AFTER the episode, from the
        #    harness, and is never an input to any decision.
        "map_fov": policy.hm.map_fov(),
        "p_true_fov": float(policy.hm.posterior().get(fov, 0.0)),
        "fov_entropy": float(policy.hm.entropy()),
        "n_informative": int(policy.hm.filter.n_informative),
        "n_skipped": int(policy.hm.filter.n_skipped),
        "map_fov_hist": fov_map_hist,
        "action_hist": action_hist,
    }
    if collect_trace:
        out["trace"] = policy.trace
    return out


def make_env(layout, n_orders=4, horizon=400):
    return SteakHumanRobotEnv(layout, n_orders=n_orders, horizon=horizon)
