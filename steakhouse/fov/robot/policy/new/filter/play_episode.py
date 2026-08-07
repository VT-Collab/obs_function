"""
THE EXPERIMENT.  Same checkpoint, same human, same seeds -- module on vs off.

    python play_episode.py --layout steak_gc00 --fovs 30,90,360 --episodes 5

===========================================================================
PAIRED BY CONSTRUCTION
===========================================================================
Every (fov, seed) is played once per arm, and the arms share the SAME frozen
weights, the SAME human seed, the SAME physics and the SAME action-selection
rule. The filter is BUILT AND RUN IN BOTH ARMS -- only whether its output is
consulted differs -- so the two conditions have identical structure and the
difference is the cost function and nothing else.

Report the paired difference per seed, never two independent means.

    FREE CONTROL: run with --k 1. The top-1 of p_base is exactly what the
    baseline plays, so the two arms must come out BIT-IDENTICAL. If they ever
    stop being identical, something reads state it should not.

===========================================================================
COLLISIONS ARE OFF, ALWAYS
===========================================================================
Kitchen.__init__ calls disable_collisions() unconditionally, so bumping into
your partner is free in every arm and every run. That is what makes a measured
win INFORMATIONAL rather than physical: the robot cannot score by getting out of
the way, only by not blindsiding. Say it once in the write-up; there is no flag.

===========================================================================
BOTH ARMS TAKE THE ARGMAX, AND THAT IS A REAL CHOICE
===========================================================================
The module is deterministic by construction -- it argmaxes over the baseline's
top-k -- so for the comparison to be paired the baseline arm argmaxes too.

The cost, from CARC_RUNS.md section 6: gc01, gc05 and cram2 DEADLOCK under
argmax and need sampling; gc07 is the reverse. So this harness is only honest on
layouts whose specialist delivers greedily -- gc00, gc04, cram, gs00, gc03,
gc06, api. Run it on gc01 and you measure the deadlock, not the module.

The old package sampled both arms instead, but sampling breaks the k=1 identity
above, which is the one control that costs nothing. Argmax keeps it.

===========================================================================
WHAT BELONGS NEXT TO ANY WIN   (old/module/QMDP/RESULTS.md section 6b)
===========================================================================
On the old checkpoints a UNIFORM-RANDOM robot BEAT the trained self-play policy
paired with this human -- 2.946 deliveries vs 2.814, 206 ticks vs 251. And
scale-matched NOISE in place of the module won 62-66% of paired cells.

So "beats the baseline" is a low bar, and two floors belong beside every result:
    baseline.py --robot random     a body doing arbitrary things
    baseline.py --robot none       the human with no robot at all
This file measures the module against the baseline. On its own it does not tell
you the module is worth having.

===========================================================================
THE KNOBS, AND WHAT EACH TRADES
===========================================================================
  --k        top-k of p_base the module may re-rank. 1 = the baseline exactly.
             Bigger = more freedom to deviate, more compute, more risk of
             trading away task competence.
  --depth    rollout length. Longer sees consequences that have not happened
             yet; it is also the dominant cost, and the shadows' beliefs go
             stale the further out you go.
  --alpha    never take an action with p_base < alpha * p_base[best]. 0 lets the
             module pick any of the k; 1 lets it pick nothing.
  --fovs     the HUMAN's true cone. One arm-pair per value.
  --cand     the filter's hypothesis set. Need not contain the true fov.
  --certain / --n_certain
             the cone-pruning gate: roll every cone until one owns `certain` of
             the mass, then roll only the top `n_certain`.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

import baseline  # noqa: F401   sys.path shim -- import before anything below
from baseline import (Kitchen, Actor, make_human, resolve_device, stage_layouts,
                      save, HUMAN_INDEX, DNF_PENALTY)

from overcooked_ai_py.mdp.overcooked_mdp import Action          # noqa: E402
from inference import SamplingBayesFOVInference                 # noqa: E402
import qmdp                                                     # noqa: E402
import cost_function                                            # noqa: E402
from qmdp import QMDPModule, sync_shadows                       # noqa: E402

#the filter's hypothesis set. The same six the human library was validated on.
CANDIDATE_FOVS = [30, 60, 90, 120, 180, 360]


class Robot:
    """Frozen baseline, optionally re-ranked by the QMDP module.

    The filter is built here in BOTH arms. That is the control's whole job: the
    two conditions must differ in exactly one thing.
    """

    def __init__(self, mdp, actor, use_module, candidate_fovs=CANDIDATE_FOVS,
                 k=3, depth=5, alpha=0.5, horizon=400, seed=0):
        self.mdp, self.actor, self.use_module = mdp, actor, use_module
        self.candidate_fovs = list(candidate_fovs)
        self.k, self.depth, self.alpha = k, depth, alpha
        self.horizon, self.seed = horizon, seed
        #ticks where the candidates actually scored differently. THE number to
        #read first: if it is ~0 the module is a no-op and any difference in the
        #results is noise.
        self.n_opinion = 0
        self.n_ticks = 0
        self.reset()

    def reset(self):
        """New episode: wipe the GRU and REBUILD the filter.

        The shadows carry per-episode knowledge -- discovered cells, beliefs,
        decay clock. Reusing them would hand the model a map it earned in a
        different episode.
        """
        self.actor.reset()
        #mlp=None is correct: the shadows route with their own BFS over seen
        #floor, and a real planner would hand them the whole map.
        self.filter = SamplingBayesFOVInference(
            self.mdp, None, self.candidate_fovs, human_agent_index=HUMAN_INDEX)
        self.module = QMDPModule(self.mdp, self.actor, k=self.k,
                                 depth=self.depth, alpha=self.alpha,
                                 horizon=self.horizon)

    def act(self, kitchen):
        """-> (Action, index). Called BEFORE the human has moved this tick."""
        p = self.actor.probs(kitchen.robot_obs())
        self.n_ticks += 1
        if self.use_module:
            idx = self.module.choose(kitchen.state, p, self.filter, seed=self.seed)
            q = self.module.last.get("q") or {}
            if len(set(q.values())) > 1:
                self.n_opinion += 1
        else:
            #the shadows still perceive, so both arms do identical work
            sync_shadows(self.filter, kitchen.state)
            idx = int(np.argmax(p))
        return Action.INDEX_TO_ACTION[int(idx)], int(idx)

    def observe_human(self, state, human_action):
        """The tick's evidence. Called AFTER the human has moved."""
        self.filter.update(state, human_action)


def play_one(kitchen, actor, fov, seed, use_module, k=3, depth=5, alpha=0.5,
             candidate_fovs=CANDIDATE_FOVS, temperature=0.5):
    """One episode. THE TICK ORDER IS THE EXPERIMENT:

        s = kitchen.state
        a = robot.act(kitchen)      robot decides on s, sees h_0..h_{t-1}
        h, info = human.action(s)   human decides on s. info["subtask"] is the
                                    ground-truth label -- DISCARDED here
        robot.observe_human(s, h)   only now is the evidence admissible
        kitchen.step(a, h)          simultaneous move resolves
    """
    kitchen.reset()
    human = make_human(kitchen.mdp, fov, seed, temperature=temperature)
    robot = Robot(kitchen.mdp, actor, use_module, candidate_fovs, k, depth,
                  alpha, kitchen.horizon, seed)

    deliveries, delivery_ticks, action_hist = 0, [], [0] * 6
    t0 = time.time()

    while kitchen.t < kitchen.horizon and not kitchen.mdp.is_terminal(kitchen.state):
        s = kitchen.state
        a, idx = robot.act(kitchen)
        action_hist[idx] += 1
        h, _info = human.action(s)          # _info["subtask"] NEVER read
        robot.observe_human(s, h)
        sparse, done = kitchen.step(a, h)
        if sparse > 0:
            n = int(round(sparse / float(kitchen.mdp.delivery_reward)))
            for _ in range(max(1, n)):
                deliveries += 1
                delivery_ticks.append(kitchen.t)
        if done:
            break

    finished = len(delivery_ticks) >= (kitchen.n_orders - 1)
    completion = (delivery_ticks[-1] if finished
                  else kitchen.horizon + DNF_PENALTY)

    return dict(
        layout=kitchen.layout, fov=fov, seed=seed, module=bool(use_module),
        k=k, depth=depth, alpha=alpha,
        deliveries=int(deliveries), steps=int(kitchen.t),
        completion_time=int(completion), finished=bool(finished),
        t_delivery=[int(x) for x in delivery_ticks], action_hist=action_hist,
        wall_s=round(time.time() - t0, 2),
        # -- human-side diagnostics, explanatory only
        h_wasted=int(human.n_wasted_commits), h_explore=int(human.n_explore),
        h_checks=int(human.n_checks), h_abandoned=int(human.n_abandoned),
        h_delivered=int(human.n_delivered),
        # -- module diagnostics. n_opinion is the one to read first.
        n_opinion=int(robot.n_opinion), n_ticks=int(robot.n_ticks),
        # -- filter diagnostics, SCORED AFTER THE FACT from the harness. The
        #    true fov reaches nothing on the robot's side of the tick loop.
        map_fov=robot.filter.map_fov(),
        p_true_fov=float(robot.filter.posterior().get(fov, 0.0)),
        fov_entropy=float(robot.filter.entropy()),
        n_informative=int(robot.filter.n_informative),
        n_skipped=int(robot.filter.n_skipped),
    )


def run(layout, fovs, episodes, seed0=0, k=3, depth=5, alpha=0.5,
        candidate_fovs=CANDIDATE_FOVS, horizon=400, n_orders=4, device=None,
        quiet=False, ckpt=None, algo="local", pseed=1):
    """Both arms over the same seeds. Seed-major, so a run killed halfway still
    holds COMPLETE pairs rather than one finished arm and one empty."""
    kitchen = Kitchen(layout, n_orders, horizon)
    actor = Actor(layout, kitchen.obs_shape, ckpt_path=ckpt, device=device)
    rows = []
    for fov in fovs:
        for seed in range(seed0, seed0 + episodes):
            for use_module in (False, True):
                row = play_one(kitchen, actor, fov, seed, use_module, k, depth,
                               alpha, candidate_fovs)
                #provenance: which trained policy produced this row
                row["algo"], row["pseed"] = algo, pseed
                rows.append(row)
                if not quiet:
                    print("  fov=%-4s seed=%-3d %-8s del=%d t=%-3d comp=%-3d "
                          "opinion=%3d/%-3d %4.1fs"
                          % (fov, seed, "MODULE" if use_module else "baseline",
                             row["deliveries"], row["steps"],
                             row["completion_time"], row["n_opinion"],
                             row["n_ticks"], row["wall_s"]), flush=True)
    return rows


def compare(rows, title=""):
    """The paired table: per FOV, baseline vs module, matched on seed."""
    fovs = sorted({r["fov"] for r in rows})
    print("\n" + (title or "module vs baseline")
          + "    (collisions off, argmax both arms)")
    print("%-5s %4s %15s %15s %8s %8s %13s"
          % ("fov", "n", "deliveries b/m", "completion b/m", "dcomp", "ddel",
             "win/tie/loss"))
    print("-" * 76)
    out = {}
    for fov in fovs:
        base = {r["seed"]: r for r in rows if r["fov"] == fov and not r["module"]}
        mod = {r["seed"]: r for r in rows if r["fov"] == fov and r["module"]}
        seeds = sorted(set(base) & set(mod))
        if not seeds:
            continue
        dc = [mod[s]["completion_time"] - base[s]["completion_time"] for s in seeds]
        dd = [mod[s]["deliveries"] - base[s]["deliveries"] for s in seeds]
        #a WIN is strictly faster with NO loss of deliveries -- the same rule
        #RESULTS.md used, so these numbers are comparable to it
        win = sum(1 for i in range(len(seeds)) if dc[i] < 0 and dd[i] >= 0)
        loss = sum(1 for i in range(len(seeds)) if dc[i] > 0 or dd[i] < 0)
        tie = len(seeds) - win - loss
        out[fov] = dict(n=len(seeds), d_comp=float(np.mean(dc)),
                        d_del=float(np.mean(dd)), win=win, tie=tie, loss=loss)
        print("%-5s %4d %7.2f /%6.2f %7.1f /%6.1f %8.1f %8.2f %13s"
              % (fov, len(seeds),
                 np.mean([base[s]["deliveries"] for s in seeds]),
                 np.mean([mod[s]["deliveries"] for s in seeds]),
                 np.mean([base[s]["completion_time"] for s in seeds]),
                 np.mean([mod[s]["completion_time"] for s in seeds]),
                 np.mean(dc), np.mean(dd), "%d/%d/%d" % (win, tie, loss)))

    op = [r["n_opinion"] / max(r["n_ticks"], 1) for r in rows if r["module"]]
    print("\nmodule had an opinion on %.0f%% of ticks (mean over module episodes)"
          % (100 * np.mean(op) if op else 0))
    print("negative dcomp = module finished SOONER. A win is strictly faster "
          "with no deliveries lost.")
    return out


def main(argv=None):
    p = argparse.ArgumentParser("module vs baseline, paired on (fov, seed)")
    p.add_argument("--layout", type=str, default="steak_gc00")
    p.add_argument("--fovs", type=str, default="30,90,360")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--cost", choices=["clash", "preference"], default="clash",
                   help="clash = cost_function.py (info gain + plan compatibility); "
                        "preference = preference_cost_function.py (maximise the "
                        "partner's own PRIORITY -- reads task value, oracle arm)")
    p.add_argument("--w_credit", type=float, default=1.0,
                   help="weight on the information term; 0 = penalty only")
    p.add_argument("--cand", type=str, default="30,60,90,120,180,360")
    p.add_argument("--certain", type=float, default=None,
                   help="override qmdp.CERTAIN, the cone-mass pruning gate")
    p.add_argument("--n_certain", type=int, default=None,
                   help="override qmdp.N_CERTAIN, cones rolled once certain")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--n_orders", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="module.jsonl")
    #--- which trained policy to load -------------------------------------
    #ZSC wave-1 checkpoints live at
    #   <root>/<algo>/<layout>_seed<S>/sp_<layout>.pt
    #and the seed ranges differ per algo: SP is 5-15, E3T and SP-eps are 1-5.
    #`local` falls back to baseline.find_checkpoint (the old seed-1 specialists).
    p.add_argument("--algo", type=str, default="local",
                   choices=["local", "sp", "e3t", "sp_eps"])
    p.add_argument("--pseed", type=int, default=1, help="POLICY seed, not episode seed")
    p.add_argument("--ckpt_root", type=str,
                   default=os.path.expanduser("/scratch1/%s/steakhouse_zsc"
                                              % os.environ.get("USER", "mishafu")))
    args = p.parse_args(argv)

    ckpt = None
    if args.algo != "local":
        ckpt = os.path.join(args.ckpt_root, args.algo,
                            "%s_seed%d" % (args.layout, args.pseed),
                            "sp_%s.pt" % args.layout)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)

    stage_layouts()
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    #module-level knobs, overridden here so a sweep needs no code edits
    if args.certain is not None:
        qmdp.CERTAIN = args.certain
    if args.n_certain is not None:
        qmdp.N_CERTAIN = args.n_certain
    cost_function.W_CREDIT = args.w_credit
    if args.cost == "preference":
        import preference_cost_function
        qmdp.COST = preference_cost_function

    fovs = [int(x) for x in args.fovs.split(",") if x.strip()]
    cand = [int(x) for x in args.cand.split(",") if x.strip()]
    print("[setup] %s fovs=%s eps=%d k=%d depth=%d alpha=%.2f cand=%s "
          "certain=%.2f/%d w_credit=%.2f device=%s"
          % (args.layout, fovs, args.episodes, args.k, args.depth, args.alpha,
             cand, qmdp.CERTAIN, qmdp.N_CERTAIN, args.w_credit, device), flush=True)

    rows = run(args.layout, fovs, args.episodes, args.seed0, args.k, args.depth,
               args.alpha, cand, args.horizon, args.n_orders, device,
               ckpt=ckpt, algo=args.algo, pseed=args.pseed)
    save(rows, args.out)
    compare(rows, title="%s %s seed%d  k=%d depth=%d alpha=%.2f"
            % (args.layout, args.algo, args.pseed, args.k, args.depth, args.alpha))
    return 0


if __name__ == "__main__":
    sys.exit(main())
