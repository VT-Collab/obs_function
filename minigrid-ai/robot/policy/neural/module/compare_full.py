# ═══════════════════════════════════════════════════════════════════════════
# robot/policy/neural/module/ - the FULL comparison table.
#
# compare_module.py answers "does the FOV module beat the baseline it wraps".
# This answers the bigger question: where do BOTH of them sit relative to the
# human alone and to the hand-coded assistants, broken down by the human's true
# field of view.
#
#   python -m robot.policy.neural.module.compare_full                 # 50 seeds
#   python -m robot.policy.neural.module.compare_full --seeds 100
#
# Every condition sees IDENTICAL layouts: the same (seed, fov) pairs, and
# eval_three_way.run_episode rebuilds the env from the seed each time.
#
# CONDITIONS, in the order they appear:
#   no_assist    the human alone. The floor - anything below this is harmful.
#   static_120   hand-coded, assumes every human has a 120 FOV. Mismatch at 60/180.
#   dynamic      hand-coded, infers FOV from the human's moves. The strong
#                hand-written reference.
#   baseline     the frozen no_fov rec_ppo policy. Sees the raw grid and the
#                human's body, has NO notion of what they can see, so it re-says
#                things constantly.
#   module       the FOV module wrapping that same frozen baseline. This is the
#                comparison the project exists to make.
#
# METRICS
#   succ         % of episodes where the human reached the goal
#   adj          adjusted steps = raw steps + n_assists, so a reveal is priced at
#                exactly one step of the human's time (eval_three_way:201). Lower
#                is better. Reported over ALL episodes.
#   adj_s        the same, over SUCCESSFUL episodes only. Read this one when the
#                success rates differ, because a condition that fails fast can
#                post a flatteringly low `adj` by never spending the steps.
#   rev          mean reveals per episode - the over-talking measure.
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import argparse, os, statistics as st, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

import torch

import robot.policy.deterministic.eval_three_way as e3w
from robot.policy.deterministic.eval_three_way import run_episode
from robot.policy.deterministic.no_assist import NoAssist
from robot.policy.deterministic.static_assist import StaticAssist
from robot.policy.deterministic.dynamic_assist import DynamicAssist
from robot.policy.neural.module.fov_module import FovModule
from robot.policy.neural.module.compare_module import BaselineRobot, CKPT

# must match the env the baseline was trained in, or the policy sees OOD input
e3w.RANDOM_WALLS = True
e3w.MAX_STEPS = 190


def evaluate(make_robot, seeds, fovs):
    """One robot instance per FOV, reused across seeds - run_episode resets it.
    Mirrors compare_module.evaluate so the two scripts stay comparable."""
    torch.manual_seed(0)          # policies act greedily; pin RNG for exactness
    per = []
    for fov in fovs:
        robot = make_robot()
        for s in seeds:
            per.append((fov, run_episode(s, fov, robot)))
    return per


def _rows(per, fov=None):
    return [r for f, r in per if fov is None or f == fov]


def succ(per, fov=None):
    x = _rows(per, fov)
    return 100.0 * sum(1 for r in x if r["success"]) / len(x) if x else float("nan")


def adj(per, fov=None):
    x = _rows(per, fov)
    return st.mean([r["adjusted_steps"] for r in x]) if x else float("nan")


def adj_success(per, fov=None):
    x = [r for r in _rows(per, fov) if r["success"]]
    return st.mean([r["adjusted_steps"] for r in x]) if x else float("nan")


def rev(per, fov=None):
    x = _rows(per, fov)
    return st.mean([r["n_assists"] for r in x]) if x else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--fovs", type=int, nargs="+", default=[60, 120, 180])
    p.add_argument("--patience", type=int, default=1,
                   help="K for the hand-coded assistants; 1 matches the module's CandidateFinder")
    p.add_argument("--conf-switch", type=float, default=0.25)
    p.add_argument("--module-comm-cost", type=float, default=0.02,
                   help="the price the MODULE uses in its own scoring (lam*cost term). This is "
                        "a decision threshold, not the env reward - raising it makes the module "
                        "quieter. Default 0.02 preserves previously reported module behaviour.")
    p.add_argument("--skip-handcoded", action="store_true",
                   help="baseline and module only (much faster)")
    a = p.parse_args()
    seeds = list(range(a.seeds))
    K = a.patience

    if not os.path.exists(CKPT):
        sys.exit(f"missing baseline checkpoint: {CKPT}\n"
                 f"Train one first:  python -m robot.policy.neural.baseline.no_fov.train "
                 f"--method rec_ppo --seed 1\nthen copy it to that path.")

    conditions = []
    if not a.skip_handcoded:
        conditions += [
            ("no_assist",  lambda: NoAssist()),
            ("static_120", lambda: StaticAssist(assumed_fov=120, patience=K)),
            ("dynamic",    lambda: DynamicAssist(patience=K)),
        ]
    conditions += [
        ("baseline", lambda: BaselineRobot()),
        ("module",   lambda: FovModule(CKPT, method="rec_ppo",
                                       comm_cost=a.module_comm_cost,
                                       conf_switch=a.conf_switch)),
    ]

    print(f"baseline ckpt : {os.path.relpath(CKPT)}")
    print(f"module        : conf_switch={a.conf_switch} comm_cost={a.module_comm_cost}")
    print(f"n             : {a.seeds} seeds x {len(a.fovs)} FOV "
          f"= {a.seeds*len(a.fovs)} episodes per condition, identical layouts\n", flush=True)

    results = {}
    for name, mk in conditions:
        results[name] = evaluate(mk, seeds, a.fovs)
        print(f"  {name} done", flush=True)

    fs = a.fovs
    W = 11 + 8 * len(fs) + 8 + 3 + 9 * len(fs) + 8 + 7

    # ── success ──────────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("SUCCESS %  (higher is better)")
    print("-" * W)
    print(f"{'':<11}" + "".join(f"{'fov'+str(f):>8}" for f in fs) + f"{'ALL':>8}")
    for name, _ in conditions:
        per = results[name]
        print(f"{name:<11}" + "".join(f"{succ(per,f):>8.1f}" for f in fs) + f"{succ(per):>8.1f}")

    # ── adjusted steps ───────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("ADJUSTED STEPS = raw steps + reveals  (lower is better)")
    print("  'all' counts failures at the 190 cap; 'succ-only' is the honest speed number")
    print("-" * W)
    print(f"{'':<11}" + "".join(f"{'fov'+str(f):>8}" for f in fs)
          + f"{'ALL':>8}   " + "".join(f"{'s'+str(f):>8}" for f in fs) + f"{'S-ALL':>8}")
    for name, _ in conditions:
        per = results[name]
        print(f"{name:<11}" + "".join(f"{adj(per,f):>8.1f}" for f in fs)
              + f"{adj(per):>8.1f}   "
              + "".join(f"{adj_success(per,f):>8.1f}" for f in fs)
              + f"{adj_success(per):>8.1f}")

    # ── reveals ──────────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("REVEALS PER EPISODE  (the over-talking measure)")
    print("-" * W)
    print(f"{'':<11}" + "".join(f"{'fov'+str(f):>8}" for f in fs) + f"{'ALL':>8}")
    for name, _ in conditions:
        per = results[name]
        print(f"{name:<11}" + "".join(f"{rev(per,f):>8.1f}" for f in fs) + f"{rev(per):>8.1f}")

    # ── the headline delta ───────────────────────────────────────────────────
    bp, mp = results["baseline"], results["module"]
    print("\n" + "=" * W)
    print("MODULE - BASELINE")
    print("-" * W)
    print("  success %   " + "  ".join(f"fov{f}: {succ(mp,f)-succ(bp,f):+6.1f}" for f in fs)
          + f"   ALL: {succ(mp)-succ(bp):+.1f}")
    print("  adj steps   " + "  ".join(f"fov{f}: {adj(mp,f)-adj(bp,f):+6.1f}" for f in fs)
          + f"   ALL: {adj(mp)-adj(bp):+.1f}")
    print("  reveals     " + "  ".join(f"fov{f}: {rev(mp,f)-rev(bp,f):+6.1f}" for f in fs)
          + f"   ALL: {rev(mp)-rev(bp):+.1f}")
    print()


if __name__ == "__main__":
    main()
