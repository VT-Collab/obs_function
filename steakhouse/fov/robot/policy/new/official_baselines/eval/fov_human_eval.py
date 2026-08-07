"""
THE HEADLINE MEASUREMENT: a baseline trained with ZERO FOVHuman exposure,
played against a LimitedVisionSteakHuman.

Nothing else in the repo does this. SP/eval_checkpoints.py plays a checkpoint
against ITSELF -- both chefs driven by the same network -- which is a self-play
score and says nothing about a limited-FOV partner. This pairs a trained policy
with the actual human agent and counts deliveries.

    python fov_human_eval.py --layout steak_gc00 --fov 90 --episodes 10
    python fov_human_eval.py --layout steak_gc00 --all_fovs --out rows.jsonl

===========================================================================
WHAT IS AND IS NOT ALLOWED TO REACH THE ROBOT
===========================================================================
Copied deliberately from filter/baseline.py, because the whole point of this
file is to produce rows that subtract cleanly against that one.

MAY      the full world state (these are fully-observable agents -- that is what
         build_full_state hands the network), the human's emitted ACTION each
         tick (a physical event anyone in the kitchen can watch), and the
         trained weights.

MAY NOT  the human's true FOV (--fov configures the HUMAN and is passed to
         nothing on the robot side), the human's subtask label -- `info` from
         human.action() carries the ground-truth subtask and is DISCARDED on
         the spot -- and any environment reward.

The baseline robot here does not even consume the human's action; it is a
frozen policy reading world state. The MAY list is written from the filter's
point of view so that when the filter arm is added, the two arms differ in the
module and in nothing else.

===========================================================================
THE SEAT AND THE ACTION RULE ARE FIXED, ON PURPOSE
===========================================================================
    robot = player 0, human = player 1        matches filter/baseline.py
    actions SAMPLED from the policy's distribution, not argmax

Sampling is what ZSC-Eval evaluates with, and CARC_RUNS.md section 6 measured
argmax DEADLOCKING three of these layouts (gc01, gc05, cram2 all score 40.00
greedy and 60.00 sampled) -- a greedy policy walks into a loop that sampling
escapes. `--mode argmax` exists so the deadlock can be reproduced, but the
table is sampled.

===========================================================================
EPISODE SEEDS ARE SHARED ACROSS POLICIES
===========================================================================
Episode i uses seed0 + i for EVERY policy in the sweep. The human's private RNG
is seeded off the global `random` stream inside its reset(), so seeding here
means every policy faces the identical sequence of human draws. Without that,
an 11-seed SP mean and a 5-seed E3T mean would differ partly because they met
different humans.

===========================================================================
COMPLETION TIME IS THE METRIC. REWARD IS SATURATED.
===========================================================================
Measured on steak_gc00 (3 episodes, old seed-1 SP):

    random robot + fov30 human   ->  reward 60.00, h_delivered 3.00

The FOV human SOLO-SOLVES the small layouts. It is a competent scripted agent,
the mdp terminates at one order left so only 3 deliveries are needed, and 400
ticks is far more than enough. Every arm pins at the 60.00 ceiling and the
reward column measures nothing.

So the headline number is COMPLETION TIME -- the tick of the last delivery, or
horizon + DNF_PENALTY if the team never finished. Same primary metric
filter/baseline.py uses, for the same reason. Reward and finish-rate are still
reported, because a DNF has to be visible somewhere and an average that quietly
folds one in is a lie.

If you ever need reward to have headroom, raise the horizon rather than
n_orders: observation channel 22 is orders-remaining NORMALIZED, so changing
n_orders at eval time shifts an input the policy was trained on.

===========================================================================
WHY TWO UNTRAINED FLOORS ARE IN THE TABLE
===========================================================================
Because "SP finishes in 270 ticks at fov90" means nothing on its own -- the
human would have finished anyway. The floors bracket what "the robot did not
help" looks like:

    random   moves constantly, so it is always getting out of its own way, but
             also interacts at chance and occasionally does something useful
    noop     always STAY. contributes nothing at all -- but it also parks in
             one cell forever, which on a tight kitchen can BLOCK a corridor

Neither is a clean "no robot" condition, and there is no such condition in a
two-player mdp. Together they bound it. First measurement on gc00 already shows
the trained policy losing to BOTH at fov30-90 and beating both at fov180-360:
a self-play robot helps a partner that can see and gets in the way of one that
cannot.
"""

import argparse
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


# =========================================================================
# PATHS. official_baselines is written to be run with ITSELF on sys.path
# (utils.features / algorithm.rMAPPOPolicy are top level).
# =========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(HERE)
#  .../fov/robot/policy/new/official_baselines  ->  5 up is steakhouse/
STEAKHOUSE = os.path.abspath(os.path.join(BASELINES, "..", "..", "..", "..", ".."))
for _p in (STEAKHOUSE, BASELINES, os.path.dirname(STEAKHOUSE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from overcooked_ai_py.mdp.overcooked_mdp import (                   # noqa: E402
    SteakHouseGridworld, Action, BASE_REW_SHAPING_PARAMS)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv       # noqa: E402

from utils.features import build_full_state, N_LAYERS               # noqa: E402
from algorithm.rMAPPOPolicy import R_MAPPOPolicy                    # noqa: E402

from fov.human.agent.limited_vision_human import LimitedVisionSteakHuman  # noqa: E402
from fov.human.planning.steak_planner import SteakMotionPlanner     # noqa: E402


ROBOT_INDEX = 0
HUMAN_INDEX = 1
N_ACTIONS = 6
HORIZON = 400
N_ORDERS = 4
#the six the human library was validated on
FOVS = [30, 60, 90, 120, 180, 360]
#the tick a team that never finished is scored at, so a DNF is strictly worse
#than any real completion time and the arms stay comparable. Same constant and
#same meaning as filter/baseline.py.
DNF_PENALTY = 100

SCRATCH = "/scratch1/%s" % os.environ.get("USER", "mishafu")
#<algo> -> the directory holding <layout>_seed<S>/. "sp_seed1" is the OLD
#pre-ZSC run, kept reachable so the new numbers can be checked against the
#table in CARC_RUNS.md.
ALGO_ROOTS = {
    "sp":       os.path.join(SCRATCH, "steakhouse_zsc", "sp"),
    "e3t":      os.path.join(SCRATCH, "steakhouse_zsc", "e3t"),
    "sp_eps":   os.path.join(SCRATCH, "steakhouse_zsc", "sp_eps"),
    "sp_old":   os.path.join(SCRATCH, "steakhouse_sp", "specialist"),
}
ALGO_SEEDS = {
    "sp":     list(range(5, 16)),     # 11 seeds, ZSC's range
    "e3t":    list(range(1, 6)),
    "sp_eps": list(range(1, 6)),
    "sp_old": [1],
    "random": [0],                    # no weights; the seed only names the row
    "noop":   [0],
}


def stage_layouts():
    """cp -n fov/layouts_final/layouts/*.layout -> overcooked_ai_py/data/layouts/

    from_layout_name() only reads overcooked_ai_py/data/layouts/, and the
    validated library lives elsewhere. NEVER overwrite an existing file.
    """
    src = os.path.join(STEAKHOUSE, "fov", "layouts_final", "layouts")
    dst = os.path.join(STEAKHOUSE, "overcooked_ai_py", "data", "layouts")
    if not os.path.isdir(src) or not os.path.isdir(dst):
        return 0
    n = 0
    for name in os.listdir(src):
        if not name.endswith(".layout"):
            continue
        target = os.path.join(dst, name)
        if not os.path.exists(target):
            with open(os.path.join(src, name)) as f:
                body = f.read()
            with open(target, "w") as f:
                f.write(body)
            n += 1
    return n


def find_checkpoint(algo, layout, seed, tag="final"):
    """Absolute path of one policy's weights, or None.

    NEVER globs. Every run directory also contains smoke/<same names>, written
    by the 800-step smoke test -- identical filenames, identical file sizes,
    junk weights. CARC_RUNS.md section 5 documents that trap; the only defence
    is naming the exact path.
    """
    root = ALGO_ROOTS.get(algo)
    if root is None:
        return None
    d = os.path.join(root, f"{layout}_seed{seed}")
    if not os.path.isdir(d):
        return None

    if tag == "final":
        for name in ("final.pt", f"sp_{layout}.pt"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return None

    #init / mid -- the FCP pool tags, chosen by SCORE from progress.jsonl
    from utils.ckpt import select_pool_checkpoints
    pool = select_pool_checkpoints(d)
    return pool.get(tag)


# =========================================================================
# THE TWO PLAYERS
# =========================================================================
class Actor:
    """A frozen policy: obs -> 6 probabilities, carrying the GRU forward.

    The recurrent state is the easy thing to get wrong. It must be threaded
    through the whole episode or this is simply not the policy that trained.
    """

    def __init__(self, ckpt_path, obs_shape):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved = ck.get("args") or {}
        args = SimpleNamespace(
            hidden_size=saved.get("hidden_size", 64),
            recurrent_N=saved.get("recurrent_N", 1),
            lr=saved.get("lr", 5e-4),
            critic_lr=saved.get("critic_lr", 5e-4),
            opti_eps=saved.get("opti_eps", 1e-5))
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N

        self.policy = R_MAPPOPolicy(args, obs_shape, obs_shape, act_dim=N_ACTIONS)
        self.policy.actor.load_state_dict(ck["actor"])
        if "critic" in ck:
            self.policy.critic.load_state_dict(ck["critic"])
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.episode = ck.get("episode", "?")
        self.reset()

    def reset(self):
        self.rnn = torch.zeros(1, self.recurrent_N, self.hidden_size)

    def probs(self, obs_np):
        """(1, 23, W, H) -> np.ndarray (6,), sums to 1.

        Stops one step before the sample, at ACTLayer's Categorical, so these
        ARE the numbers policy.act() would have sampled from.
        """
        with torch.no_grad():
            obs = torch.from_numpy(obs_np).float()
            masks = torch.ones(1, 1)
            x = self.policy.actor.base(obs)
            x, self.rnn = self.policy.actor.rnn(x, self.rnn, masks)
            distri = self.policy.actor.act.distri(x)
        return distri.probs.numpy().reshape(-1)


class RandomActor:
    """Floor 1. Uniform over the 6 primitives, no weights, no memory."""

    episode = "-"

    def reset(self):
        pass

    def probs(self, obs_np):
        return np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float64)


class NoopActor:
    """Floor 2. Always STAY -- contributes nothing whatsoever.

    Not a clean "no robot" condition: the body is still on the board and still
    occupies a cell, so on a tight kitchen it can wall off a corridor for the
    whole episode. That is exactly why it is reported ALONGSIDE random rather
    than instead of it -- between them they bracket what "the robot did not
    help" costs.
    """

    episode = "-"

    def reset(self):
        pass

    def probs(self, obs_np):
        p = np.zeros(N_ACTIONS, dtype=np.float64)
        p[Action.ACTION_TO_INDEX[Action.STAY]] = 1.0
        return p


#the rows that need no checkpoint. defined after the classes exist.
UNTRAINED = {"random": RandomActor, "noop": NoopActor}


def make_human(mdp, fov, temperature=0.5):
    """A LimitedVisionSteakHuman whose draws are reproducible.

    Its private RNG is `random.Random(random.random())`, seeded inside reset()
    off the GLOBAL random stream -- so random.seed(...) BEFORE this call is
    what makes an episode replayable. planner mlp=None is correct: the human
    routes with its own BFS over seen floor, and a real planner would hand it
    the whole map, which is the cheat the entire human library exists to avoid.
    """
    planner = SteakMotionPlanner(mdp, None)
    return LimitedVisionSteakHuman(mdp, fov, planner, agent_index=HUMAN_INDEX,
                                   temperature=temperature)


def choose(p, rng, mode="sample"):
    if mode == "argmax":
        return int(np.argmax(p))
    #renormalize: float error in the network output can leave the sum a hair
    #off 1.0, which np.random.choice rejects outright
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    return int(rng.choice(N_ACTIONS, p=p))


# =========================================================================
# ONE EPISODE
# =========================================================================
def play_episode(mdp, actor, fov, seed, horizon=HORIZON, n_orders=N_ORDERS,
                 mode="sample", temperature=0.5):
    """Robot (player 0) = the frozen policy. Human (player 1) = FOV human.

    TICK ORDER -- identical to filter/play_episode.py, so the rows are paired:

        s = env.state
        a = choose(actor.probs(obs(s)))     robot decides on s
        h, info = human.action(s)           human decides on s. info holds the
                                            GROUND-TRUTH subtask -- discarded here
        env.step((a, h))                    simultaneous resolve
    """
    #seeds the human's private RNG (via reset) AND the robot's sampler, so the
    #whole episode replays
    random.seed(seed)
    rng = np.random.RandomState(seed)

    #+10 so OUR horizon check ends the episode and the mdp's own terminal
    #(order list run down) stays the "finished early" signal
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=horizon + 10)
    human = make_human(mdp, fov, temperature)
    actor.reset()

    obs_shape_ok = (N_LAYERS, mdp.shape[0], mdp.shape[1])
    deliveries, t_delivery = 0, []
    action_hist = np.zeros(N_ACTIONS, dtype=int)
    steps = 0

    for t in range(horizon):
        state = env.state

        obs = build_full_state(mdp, state, agent_index=ROBOT_INDEX,
                               t=env.t, horizon=horizon)[None, ...]
        assert obs.shape[1:] == obs_shape_ok, obs.shape
        idx = choose(actor.probs(obs), rng, mode)
        robot_action = Action.INDEX_TO_ACTION[idx]
        action_hist[idx] += 1

        #`info` carries the ground-truth subtask label. It is the single most
        #tempting cheat in this codebase. Discarded on the spot.
        human_action, _human_info = human.action(state)

        _, sparse, done, _ = env.step((robot_action, human_action))
        steps = t + 1

        if sparse > 0:
            n = int(round(sparse / mdp.delivery_reward))
            deliveries += n
            t_delivery.extend([steps] * n)

        if done:
            break

    #the mdp terminates at ONE order left, so n_orders=4 caps deliveries at 3
    finished = len(t_delivery) >= n_orders - 1
    completion = t_delivery[-1] if finished else horizon + DNF_PENALTY

    return {
        "fov": fov, "seed": seed, "mode": mode,
        "deliveries": deliveries,
        "reward": deliveries * mdp.delivery_reward,
        "steps": steps,
        "completion_time": completion,
        "finished": bool(finished),
        "t_delivery": t_delivery,
        "action_hist": action_hist.tolist(),
        #the human's own diagnostics. n_explore is the dominant FOV signal --
        #a narrow cone spends its opening just discovering where things are.
        "h_wasted": human.n_wasted_commits,
        "h_explore": human.n_explore,
        "h_checks": human.n_checks,
        "h_abandoned": human.n_abandoned,
        "h_delivered": human.n_delivered,
    }


# =========================================================================
# SWEEP
# =========================================================================
def disable_collisions(mdp):
    """Let the two agents pass through and stand on each other.

    ===================================================================
    WHY THIS EXISTS
    ===================================================================
    The mdp freezes BOTH players for a tick whenever their moves would put
    them on the same cell or swap them (overcooked_mdp._handle_collisions).
    Measured on gc00/gc06, 10 episodes each:

        robot     fov      collisions/ep   rate    finish
        random     30            45.0      0.201    1.00
        sp         30           145.1      0.412    0.60
        noop       30           378.0      0.945    0.00

    A trained self-play robot triggers 2-3.6x more collisions than a random
    one, and each costs the HUMAN a move as well as the robot. At fov30 on
    gc06 the pair is frozen on 39% of all ticks. The collision rate is also
    HIGHER at narrow fov (0.412 vs 0.308) because a blind human walks into
    the robot instead of around it.

    So "the robot obstructs the human" and "the human cannot see" are
    entangled in every number the collisions-on arm produces. This flag
    separates them.

    ===================================================================
    WHAT IT COSTS -- SAY THIS OUT LOUD IN THE PAPER
    ===================================================================
    * The policies were TRAINED with collisions. Evaluating without them puts
      every policy off-distribution. It hits all arms equally, so the ranking
      stays fair, but the absolute numbers are not "how this policy performs".
    * Two agents can occupy one cell, which is not standard Overcooked and not
      what ZSC-Eval reports.
    * TRAINING IS UNAFFECTED. SP/E3T/FCP are defined by their training
      procedure and that is untouched -- same mdp, same collisions, same
      ZSC-Eval hyperparameters. Only the evaluation environment changes.
    * EVERY ARM MUST USE THE SAME SETTING, the filter included. Baselines
      without collisions and the method with them is not a comparison.

    The upside for the FOV claim: with collisions off, "the module just
    learned to get out of the way" stops being a possible explanation, so any
    gain has to come from actually inferring what the partner can see.

    Instance-level override, so it affects only the mdp object handed in and
    overcooked_mdp.py is never edited.
    """
    mdp._handle_collisions = lambda old_positions, new_positions: new_positions
    return mdp


def build_mdp(layout, n_orders=N_ORDERS):
    """start_order_list MUST be a real list. Every .layout file declares it as
    the STRING 'steak, steak, steak', so len() counts characters and
    deliver_dish compares 'steak' == 's' -- orders are never consumed and NO
    delivery reward ever fires. CARC_RUNS.md section 4, trap 1.

    rew_shaping_params is passed explicitly for the same reason: this fork maps
    None -> NO_REW_SHAPING_PARAMS (all zeros) where ZSC maps it to real values.
    Nothing here reads reward, but it is part of the transition, so eval and
    training must match.
    """
    mdp = SteakHouseGridworld.from_layout_name(
        layout, start_order_list=["steak"] * n_orders,
        rew_shaping_params=dict(BASE_REW_SHAPING_PARAMS))
    assert not isinstance(mdp.start_order_list, str), \
        "start_order_list came back a STRING -- no delivery will ever fire"
    return mdp


def policies_for(layout, algos, obs_shape, tag="final"):
    """-> [(algo, seed, actor, ckpt_path_or_None)], skipping what is missing."""
    out = []
    for algo in algos:
        if algo in UNTRAINED:
            out.append((algo, 0, UNTRAINED[algo](), None))
            continue
        for seed in ALGO_SEEDS.get(algo, []):
            path = find_checkpoint(algo, layout, seed, tag)
            if path is None:
                continue
            try:
                out.append((algo, seed, Actor(path, obs_shape), path))
            except Exception as e:                       # shape mismatch, etc.
                print(f"  !! {algo} seed{seed}: {type(e).__name__}: {e}",
                      flush=True)
    return out


def sweep(layout, algos, fovs, episodes, seed0=0, mode="sample",
          temperature=0.5, n_orders=N_ORDERS, horizon=HORIZON, tag="final",
          collisions=True):
    mdp = build_mdp(layout, n_orders)
    if not collisions:
        disable_collisions(mdp)
    obs_shape = (N_LAYERS, mdp.shape[0], mdp.shape[1])

    pols = policies_for(layout, algos, obs_shape, tag)
    have = {}
    for algo, _s, _a, _p in pols:
        have[algo] = have.get(algo, 0) + 1
    print(f"[{layout}] grid={mdp.shape} policies={len(pols)} {have}", flush=True)
    if not pols:
        return []

    rows, start = [], time.time()
    for fov in fovs:
        for algo, seed, actor, path in pols:
            for i in range(episodes):
                #episode seed depends on i ONLY -- every policy meets the same
                #sequence of humans, so the comparison is paired
                r = play_episode(mdp, actor, fov, seed0 + i, horizon, n_orders,
                                 mode, temperature)
                r.update(layout=layout, algo=algo, policy_seed=seed,
                         ckpt=os.path.basename(path) if path else "random",
                         ckpt_episode=actor.episode, grid=list(mdp.shape),
                         #stamped on every row so a collisions-on and a
                         #collisions-off file can never be silently pooled
                         collisions=bool(collisions))
                rows.append(r)
        done = [r for r in rows if r["fov"] == fov]
        print(f"  fov {fov:3d}  n={len(done):4d}  "
              f"mean_reward {np.mean([r['reward'] for r in done]):6.2f}  "
              f"({time.time() - start:.0f}s)", flush=True)
    return rows


def summarize(rows):
    """(algo, fov) -> mean reward / deliveries / finish rate, +/- std over the
    POLICY SEEDS. The std is the number the whole 11-seed re-run exists for:
    it is across independently trained policies, not across episodes."""
    out = {}
    keys = sorted({(r["algo"], r["fov"]) for r in rows})
    for algo, fov in keys:
        sub = [r for r in rows if r["algo"] == algo and r["fov"] == fov]
        per_seed = {}
        for r in sub:
            per_seed.setdefault(r["policy_seed"], []).append(r["reward"])
        seed_means = [float(np.mean(v)) for v in per_seed.values()]
        out[(algo, fov)] = {
            "n_episodes": len(sub),
            "n_seeds": len(per_seed),
            "reward_mean": float(np.mean([r["reward"] for r in sub])),
            "reward_std_over_seeds": float(np.std(seed_means)) if len(seed_means) > 1 else 0.0,
            "deliveries_mean": float(np.mean([r["deliveries"] for r in sub])),
            "finish_rate": float(np.mean([r["finished"] for r in sub])),
            "completion_mean": float(np.mean([r["completion_time"] for r in sub])),
            "h_explore_mean": float(np.mean([r["h_explore"] for r in sub])),
            "h_wasted_mean": float(np.mean([r["h_wasted"] for r in sub])),
        }
    return out


#untrained rows are the floor, so they print first and the trained rows are
#read against them
_ORDER = {"noop": 0, "random": 1, "sp_old": 2, "sp": 3, "sp_eps": 4, "e3t": 5}


def print_table(rows):
    """Two blocks. Completion time is the headline; reward is the sanity check.

    Reward SATURATES at 60.00 on the small layouts -- the FOV human solves them
    alone -- so a reward-only table would show every arm tied and hide the fact
    that a self-play robot slows a narrow-cone partner down by 40%. Both are
    printed so that saturation is visible rather than assumed.
    """
    summ = summarize(rows)
    fovs = sorted({f for _a, f in summ})
    algos = sorted({a for a, _f in summ}, key=lambda a: (_ORDER.get(a, 9), a))

    print("\n" + "=" * 86)
    print("COMPLETION TIME, ticks, LOWER IS BETTER  (DNF scored at horizon+%d)"
          % DNF_PENALTY)
    print("mean +/- std ACROSS POLICY SEEDS. (f=..) is the finish rate.")
    print("=" * 86)
    print(f"{'algo':<9}{'seeds':>6}  " + "".join(f"{'fov' + str(f):>13}" for f in fovs))
    for algo in algos:
        cells, n_seeds = [], 0
        for f in fovs:
            s = summ.get((algo, f))
            if s is None:
                cells.append(f"{'-':>13}")
                continue
            n_seeds = s["n_seeds"]
            cells.append(f"{s['completion_mean']:6.1f}(f{s['finish_rate']:.2f})")
        print(f"{algo:<9}{n_seeds:>6}  " + "".join(cells))

    print("\n" + "-" * 86)
    print("team reward (max 60 = 3 deliveries). SATURATES -- see the docstring.")
    print("-" * 86)
    print(f"{'algo':<9}{'seeds':>6}  " + "".join(f"{'fov' + str(f):>13}" for f in fovs))
    for algo in algos:
        cells, n_seeds = [], 0
        for f in fovs:
            s = summ.get((algo, f))
            if s is None:
                cells.append(f"{'-':>13}")
                continue
            n_seeds = s["n_seeds"]
            cells.append(f"{s['reward_mean']:7.2f}±{s['reward_std_over_seeds']:<5.2f}")
        print(f"{algo:<9}{n_seeds:>6}  " + "".join(cells))
    print("=" * 86)


def save(rows, path):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[saved] {len(rows)} rows -> {path}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser("play a trained baseline against the FOV human")
    p.add_argument("--layout", type=str, default="steak_gc00")
    p.add_argument("--algos", type=str, default="noop,random,sp,sp_eps,e3t",
                   help="comma separated: " + ",".join(list(ALGO_ROOTS) + ["random", "noop"]))
    p.add_argument("--fov", type=int, default=None)
    p.add_argument("--all_fovs", action="store_true")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--mode", type=str, default="sample",
                   choices=["sample", "argmax"])
    p.add_argument("--temperature", type=float, default=0.5,
                   help="the HUMAN's subtask sampler, not the robot's")
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--n_orders", type=int, default=N_ORDERS)
    p.add_argument("--tag", type=str, default="final",
                   choices=["init", "mid", "final"])
    p.add_argument("--no_collisions", dest="collisions", action="store_false",
                   default=True,
                   help="agents pass through each other. Isolates partner "
                        "observability from physical obstruction -- read "
                        "disable_collisions() before using this in a paper.")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    n = stage_layouts()
    if n:
        print(f"[setup] staged {n} layout files", flush=True)
    #the bottleneck is the python env loop and the human's per-tick BFS, never
    #the 5x5 conv -- extra torch threads only add contention
    torch.set_num_threads(1)

    fovs = FOVS if (args.all_fovs or args.fov is None) else [args.fov]
    if not args.collisions:
        print("[setup] COLLISIONS DISABLED -- agents pass through each other. "
              "Policies are off-distribution; every arm must match.", flush=True)
    rows = sweep(args.layout, [a for a in args.algos.split(",") if a],
                 fovs, args.episodes, args.seed0, args.mode,
                 args.temperature, args.n_orders, args.horizon, args.tag,
                 args.collisions)
    if not rows:
        print("no policies found -- has training finished?")
        return 1
    print_table(rows)
    save(rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
