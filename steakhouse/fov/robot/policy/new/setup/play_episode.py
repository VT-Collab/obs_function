"""THE EXPERIMENT.  One arm ladder, same human, same seeds, paired throughout.

    python play_episode.py --layout setup_island --fovs 30,90,360 --episodes 5

===========================================================================
THE LADDER, AND WHAT EACH RUNG ISOLATES
===========================================================================
    noop      no robot at all                 floor
    random    a body doing arbitrary things   floor
    blind     hand-written planner, partner-blind          THE BASELINE
    react     ...+ reactive deconfliction (position/held)  absorbs "a partner
                                                           model helps"
    fov360    ...+ module, posterior PINNED to 360         absorbs "a partner
                                                           model with subtask
                                                           inference helps"
    module    ...+ module, real inferred posterior         <-- THE CLAIM
    oracle    ...+ module, posterior pinned to the TRUE    ceiling
              cone

Each rung adds exactly one thing, so each gap is attributable:

    blind  -> react     value of reacting to where the partner IS
    react  -> fov360    value of modelling what the partner is DOING
    fov360 -> module    VALUE OF INFERRING WHAT THEY CAN SEE   <-- the paper
    module -> oracle    inference error

The fov360 rung exists because of a measured hole in the old campaign: pinning
the cone to a constant performed as well as the inferred posterior
(old/module/QMDP/RESULTS.md section 6e), so "a model of the partner helps" was
the safe claim and "inferring the partner's cone helps" was not established.
This arm makes that comparison the headline rather than a footnote.

Both floors are reported on every table because section 6b measured a uniform
random robot BEATING the trained policy, and section 5 measured scale-matched
noise winning 65.9% of paired cells. A win rate quoted without them means
nothing.

===========================================================================
PAIRED BY CONSTRUCTION
===========================================================================
Every (fov, seed) is played once per arm, and the arms share the SAME human
seed, the SAME physics and the SAME action-selection rule. make_human seeds the
GLOBAL stream before constructing, so every arm starts from an identical draw
sequence and diverges only because the world diverged. Report the paired
difference per seed, never two independent means.

    FREE CONTROL: --k 1. The top-1 of the baseline's subtask distribution is
    exactly what the baseline plays, so module and blind must come out
    BIT-IDENTICAL. If they ever stop being identical, something reads state it
    should not.

===========================================================================
COLLISIONS ARE OFF, ALWAYS
===========================================================================
Kitchen.__init__ disables them unconditionally, so bumping into your partner is
free in every arm. That is what makes a measured win INFORMATIONAL rather than
physical: the robot cannot score by getting out of the way, only by not
blindsiding. These layouts have narrow corridors, so the blocking channel would
be large if it were on.
"""

import argparse
import json
import time

import _paths  # noqa: F401   MUST be first
import layouts
import qmdp
import cost_function
from env import Kitchen, make_human, ROBOT_INDEX, HUMAN_INDEX, DNF_PENALTY
from planner import make_planner
from fov_filter import FOVFilter, CANDIDATE_FOVS
from qmdp import QMDPModule, sync_shadows

from overcooked_ai_py.mdp.overcooked_mdp import Action                  # noqa: E402
from fov.human.agent.limited_vision_human import SAMPLING_SUBTASKS      # noqa: E402

ARMS = ("noop", "random", "blind", "react", "fov360", "module", "oracle")
#arms that attach the module; the rest are bare planners
MODULE_ARMS = ("fov360", "module", "oracle")


class Robot:
    """One arm's robot. Owns the planner and, for module arms, the filter."""

    def __init__(self, mdp, arm, seed, true_fov, k, depth, alpha,
                 candidate_fovs=CANDIDATE_FOVS):
        self.arm = arm
        self.mdp = mdp
        kind = "react" if arm == "react" else (
            arm if arm in ("noop", "random") else "blind")
        self.planner = make_planner(kind, mdp, seed)
        self.use_module = arm in MODULE_ARMS
        self.n_opinion = self.n_override = self.n_ticks = 0
        self.n_choice = 0

        #THE ONLY PLACE true_fov IS ALLOWED TO REACH THE ROBOT, and only on the
        #oracle arm, which exists precisely to be an upper bound.
        pin = None
        if arm == "fov360":
            pin = 360
        elif arm == "oracle":
            pin = true_fov

        self.filter = None
        self.module = None
        if self.use_module:
            self.filter = FOVFilter(mdp, candidate_fovs,
                                    human_agent_index=HUMAN_INDEX, pin=pin)
            self.module = QMDPModule(mdp, self.planner, k=k, depth=depth,
                                     alpha=alpha)

    def act(self, state):
        """-> Action. Called BEFORE the human has moved this tick.

        ORDER MATTERS. p_base is built BEFORE the planner acts, because
        subtask_distribution reads the commitment the planner is about to
        advance. Then the planner takes its own tick normally -- so its
        bookkeeping is never touched by the module -- and only the PRIMITIVE we
        emit differs. That is what makes a deviation a one-step deviation.
        """
        self.n_ticks += 1
        if not self.use_module:
            a, _info = self.planner.action(state)
            return a

        p_base = self.module.base_action_probs(state)
        idx = self.module.choose(state, p_base, self.filter)

        #the baseline still takes its own tick, on its own logic
        a_base, _info = self.planner.action(state)

        #diagnostics: n_live is how often the baseline was uncertain AT ALL,
        #which upper-bounds how often the module can ever speak
        if (self.module.last.get("n_live") or 0) > 1:
            self.n_choice += 1
        q = self.module.last.get("q") or {}
        if len(set(q.values())) > 1:
            self.n_opinion += 1

        if idx is None:
            return a_base                  #deferred: bit-identical to blind
        self.n_override += 1
        return Action.INDEX_TO_ACTION[int(idx)]

    def observe_human(self, state, human_action):
        """The tick's evidence. Called AFTER the human has moved."""
        if self.filter is not None:
            self.filter.update(state, human_action)


def play_one(layout, arm, fov, seed, n_orders, horizon, k, depth, alpha,
             candidate_fovs=CANDIDATE_FOVS, temperature=0.5):
    """One episode. THE TICK ORDER IS THE EXPERIMENT."""
    kitchen = Kitchen(layout, n_orders, horizon)
    kitchen.reset()
    human = make_human(kitchen.mdp, fov, seed, temperature=temperature)
    robot = Robot(kitchen.mdp, arm, seed, fov, k, depth, alpha, candidate_fovs)

    deliveries, ticks = 0, []
    t0 = time.time()
    while kitchen.t < horizon and not kitchen.mdp.is_terminal(kitchen.state):
        s = kitchen.state
        a = robot.act(s)                    # robot sees s and h_0..h_{t-1}
        h, _info = human.action(s)           # _info["subtask"] NEVER read
        robot.observe_human(s, h)            # only now is the evidence admissible
        sparse, done = kitchen.step(a, h)
        if sparse > 0:
            for _ in range(max(1, int(round(sparse / float(kitchen.mdp.delivery_reward))))):
                deliveries += 1
                ticks.append(kitchen.t)
        if done:
            break

    #the mdp declares terminal after n_orders-1 deliveries, so that is the bar
    finished = deliveries >= (n_orders - 1)
    completion = ticks[-1] if finished else horizon + DNF_PENALTY
    row = dict(layout=layout, arm=arm, fov=fov, seed=seed,
               deliveries=deliveries, steps=kitchen.t,
               completion=completion, finished=finished,
               n_opinion=robot.n_opinion, n_override=robot.n_override,
               n_choice=robot.n_choice,
               n_ticks=robot.n_ticks, wall_s=round(time.time() - t0, 1),
               h_wasted=human.n_wasted_commits, h_abandoned=human.n_abandoned,
               h_explore=human.n_explore)
    if robot.filter is not None:
        #scored AFTER the fact by the harness; the true fov reaches nothing on
        #the robot's side of the tick loop except the oracle's pin
        row["map_fov"] = robot.filter.map_fov()
        row["p_true_fov"] = float(robot.filter.posterior().get(fov, 0.0))
    return row


def main(argv=None):
    p = argparse.ArgumentParser("the arm ladder")
    p.add_argument("--layout", type=str, default="setup_island")
    p.add_argument("--arms", type=str, default=",".join(ARMS))
    p.add_argument("--fovs", type=str, default="30,90,360")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--n_orders", type=int, default=6)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--w_credit", type=float, default=1.0)
    p.add_argument("--w_cover", type=float, default=1.0)
    p.add_argument("--tau_effort", type=float, default=None)
    p.add_argument("--beta", type=float, default=None,
                   help="weight on the partner terms vs log p_base")
    p.add_argument("--out", type=str, default="arms.jsonl")
    args = p.parse_args(argv)

    layouts.stage()
    cost_function.W_CREDIT = args.w_credit
    cost_function.W_COVER = args.w_cover
    if args.tau_effort is not None:
        qmdp.TAU_EFFORT = args.tau_effort
    if args.beta is not None:
        qmdp.BETA = args.beta

    arms = [a for a in args.arms.split(",") if a.strip()]
    fovs = [int(x) for x in args.fovs.split(",") if x.strip()]
    print("[setup] %s arms=%s fovs=%s eps=%d N=%d horizon=%d k=%d depth=%d"
          % (args.layout, arms, fovs, args.episodes, args.n_orders - 1,
             args.horizon, args.k, args.depth), flush=True)

    rows = []
    for fov in fovs:
        for arm in arms:
            for seed in range(args.episodes):
                r = play_one(args.layout, arm, fov, seed, args.n_orders,
                             args.horizon, args.k, args.depth, args.alpha)
                rows.append(r)
                print("  fov=%-4d %-7s seed=%d del=%d comp=%-4d op=%3d/%-4d ovr=%3d %5.1fs"
                      % (fov, arm, seed, r["deliveries"], r["completion"],
                         r["n_opinion"], r["n_ticks"], r["n_override"],
                         r["wall_s"]), flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    table(rows, arms, fovs)
    return 0


def table(rows, arms, fovs):
    print("\n%-6s %-8s %8s %8s %8s %8s %7s"
          % ("fov", "arm", "compl", "deliv", "finish", "vs blind", "opinion"))
    print("-" * 62)
    for fov in fovs:
        base = [r for r in rows if r["fov"] == fov and r["arm"] == "blind"]
        bmean = sum(r["completion"] for r in base) / len(base) if base else 0
        for arm in arms:
            sel = [r for r in rows if r["fov"] == fov and r["arm"] == arm]
            if not sel:
                continue
            c = sum(r["completion"] for r in sel) / len(sel)
            d = sum(r["deliveries"] for r in sel) / len(sel)
            f = sum(1 for r in sel if r["finished"]) / len(sel)
            op = sum(r["n_opinion"] for r in sel) / max(sum(r["n_ticks"] for r in sel), 1)
            print("%-6d %-8s %8.1f %8.2f %8.2f %+8.1f %6.0f%%"
                  % (fov, arm, c, d, f, c - bmean, 100 * op))
    print("\nnegative 'vs blind' = finished SOONER than the baseline.")


if __name__ == "__main__":
    raise SystemExit(main())
