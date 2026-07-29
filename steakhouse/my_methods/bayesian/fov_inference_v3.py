"""
MISHA NEW CHANGE - Bayesian FOV inference on the v3 layouts, i.e. the ones that
carry REAL (RNG-free) FOV-driven subtask divergence.

This is the payoff step. fov_divergence_search_v2.py finds layouts where three
FOV hypotheses genuinely choose different subtasks at a fixed random seed, and
writes them to fov/layouts_v3/. This script points SteakBayesFOVInference (the
minigrid-style log-space filter) at them and measures how accurately it recovers
the human's true FOV from watching their actions online.

EVERYTHING HERE IS PINNED TO WHAT THE SEARCH VALIDATED, because each of these
was a bug that silently destroyed the signal at some point:

  * kb_update_delay is read from the layout header and applied to BOTH the
    ground-truth human and every shadow. At the project default of 0 a single
    frame of contact commits a fact to memory, every FOV learns the same things,
    and the posterior provably cannot move (58/60 searched layouts had exactly
    zero FOV effect). Mismatching it between truth and shadows would be worse
    still - the likelihood would be wrong for the true hypothesis too.
  * The human is the PLAIN SteakLimitVisionHumanModel. StickySubtaskHumanModel
    stalls it ~20 steps in having touched 1-2 subtasks.
  * The teammate is a full-vision GreedySteakHumanModel that actually works the
    task, so the world keeps changing. With v1's park-and-hide robot the world
    state froze and there was nothing to be ignorant about.
  * np.random is reseeded per episode, because auto_unstuck (agent.py:449) makes
    the simulation nondeterministic and would otherwise swamp the comparison.

Run with: python -m my_methods.bayesian.fov_inference_v3 [n_workers] [mode] [n_seeds]
"""
import ast
import multiprocessing as mp
import os
import random
import re
import sys

import numpy as np

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import Action
from overcooked_ai_py.agents.agent import GreedySteakHumanModel, SteakLimitVisionHumanModel
from my_methods.bayesian.fov_subtask_divergence_test import build_mdp_and_mlp
from my_methods.bayesian.fov_bayes_filter import SteakBayesFOVInference

N_STEPS = 200
LAYOUTS_V3_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fov", "layouts_v3")


def parse_v3_layout(path):
    # errors="replace": a single stray non-UTF-8 byte in one layout file was
    # enough to abort an entire 306-trial CARC job before it ran a single
    # episode (UnicodeDecodeError from f.read()). Everything this parser reads -
    # the comment header and the grid - is ASCII, so replacing an undecodable
    # byte is strictly better than losing the batch.
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    def grab(pattern, cast=int, default=None):
        m = re.search(pattern, content)
        return cast(m.group(1)) if m else default

    grid = [ln.strip() for ln in
            re.search(r'"grid":\s*"""(.*?)"""', content, re.DOTALL).group(1).strip().split("\n")]
    fov = re.search(r"# FOV triple \(deg\): (\d+), (\d+), (\d+)", content)
    return dict(
        grid=grid,
        fov_triple=tuple(int(x) for x in fov.groups()),
        kb_update_delay=grab(r"kb_update_delay:\s*(\d+)", int, 2),
        robot_mode=grab(r"# teammate:\s*(\w+)", str, "work"),
        order_list=ast.literal_eval(re.search(r'"start_order_list":\s*(\[[^\]]*\])', content).group(1)),
        cook_time=grab(r'"cook_time":\s*(\d+)'),
        chop_time=grab(r"'chop_time':\s*(\d+)"),
        wash_time=grab(r"'wash_time':\s*(\d+)"),
        num_items_for_steak=grab(r"'num_items_for_steak':\s*(\d+)", int, 1),
    )


def run_trial(args):
    layout_file, true_fov, mode, seed = args
    name = os.path.splitext(os.path.basename(layout_file))[0]
    base = dict(layout=name, true_fov=true_fov, mode=mode, seed=seed)
    try:
        cfg = parse_v3_layout(layout_file)
        mdp, mlp = build_mdp_and_mlp(name, cfg["grid"], order_list=cfg["order_list"],
                                     cook_time=cfg["cook_time"], chop_time=cfg["chop_time"],
                                     wash_time=cfg["wash_time"],
                                     num_items_for_steak=cfg["num_items_for_steak"])
    except Exception as e:
        return dict(base, error=f"build failed: {type(e).__name__}: {e}")

    delay = cfg["kb_update_delay"]
    np.random.seed(seed)
    random.seed(seed)
    setup_state = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=400).state.deepcopy()
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=400)

    human = SteakLimitVisionHumanModel(mlp, setup_state, vision_limit=True,
                                       vision_bound=true_fov, kb_update_delay=delay, debug=False)
    human.set_agent_index(1)
    human.init_knowledge_base(setup_state)

    teammate = GreedySteakHumanModel(mlp)
    teammate.set_agent_index(0)

    inf = SteakBayesFOVInference(mlp, setup_state, candidate_fovs=cfg["fov_triple"],
                                 human_agent_index=1, likelihood=mode,
                                 agent_cls=SteakLimitVisionHumanModel,
                                 initial_kb="omniscient", kb_update_delay=delay)

    uniform = 1.0 / len(cfg["fov_triple"])
    estimates, consec = [], 0
    informative = False
    for _ in range(N_STEPS):
        state = env.state
        try:
            a_robot, _ = teammate.action(state)
        except Exception:
            a_robot = Action.STAY
        try:
            a_human, _ = human.action(state)
            consec = 0
        except Exception:
            a_human = Action.STAY
            consec += 1
            if consec >= 8:
                break
        inf.update(state, a_human)
        estimates.append(inf.map_fov())
        post = inf.posterior()
        if max(abs(post[f] - uniform) for f in cfg["fov_triple"]) > 0.01:
            informative = True
        _, _, done, _ = env.step((a_robot, a_human))
        if done:
            break

    if not estimates:
        return dict(base, error="no steps")
    n = len(estimates)
    half = n // 2
    post = inf.posterior()
    return dict(base, fov_triple=cfg["fov_triple"], kb_update_delay=delay, steps=n,
                accuracy=sum(1 for e in estimates if e == true_fov) / n,
                late_accuracy=sum(1 for e in estimates[half:] if e == true_fov) / max(1, n - half),
                final_correct=estimates[-1] == true_fov,
                informative=informative,
                p_true=post[true_fov], entropy=inf.entropy(),
                divergence=(inf.n_goal_divergent_steps / inf.n_steps_seen
                            if inf.n_steps_seen else 0.0))


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    modes = (["greedy", "boltzmann"] if len(sys.argv) <= 2 or sys.argv[2] == "all"
             else [sys.argv[2]])
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    if not os.path.isdir(LAYOUTS_V3_DIR):
        print(f"no {LAYOUTS_V3_DIR} - run fov_divergence_search_v2.py first")
        return
    # Skip macOS AppleDouble resource forks ("._name.layout"). macOS tar bundles
    # extended attributes into these; they end in .layout but are binary, and one
    # of them aborted an entire 306-trial job with UnicodeDecodeError.
    files = sorted(os.path.join(LAYOUTS_V3_DIR, f)
                   for f in os.listdir(LAYOUTS_V3_DIR)
                   if f.endswith(".layout") and not f.startswith("._"))
    if not files:
        print(f"{LAYOUTS_V3_DIR} is empty - run fov_divergence_search_v2.py first")
        return

    jobs = []
    for path in files:
        cfg = parse_v3_layout(path)
        for true_fov in cfg["fov_triple"]:
            for mode in modes:
                for seed in range(n_seeds):
                    jobs.append((path, true_fov, mode, seed))

    print(f"{len(files)} v3 layouts x 3 FOVs x {len(modes)} mode(s) x {n_seeds} seeds "
          f"= {len(jobs)} trials on {n_workers} workers", flush=True)

    results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_trial, jobs)):
            results.append(r)
            if "error" in r:
                print(f"[{i+1}/{len(jobs)}] {r['layout']} fov={r['true_fov']} "
                      f"ERROR {r['error']}", flush=True)
            else:
                print(f"[{i+1}/{len(jobs)}] {r['layout']} fov={r['true_fov']:>3} "
                      f"{r['mode']:<9} s{r['seed']} acc={r['accuracy']:.2f} "
                      f"late={r['late_accuracy']:.2f} final={r['final_correct']} "
                      f"P(true)={r['p_true']:.3f} H={r['entropy']:.2f} "
                      f"info={r['informative']} steps={r['steps']}", flush=True)

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\nno valid trials")
        return
    print(f"\n=== SUMMARY ({len(valid)}/{len(jobs)}) ===")
    for mode in modes:
        rs = [r for r in valid if r["mode"] == mode]
        if not rs:
            continue
        n = len(rs)
        inf_rs = [r for r in rs if r["informative"]]
        print(f"\n--- mode={mode} ({n} trials) ---")
        print(f"  mean per-step accuracy : {sum(r['accuracy'] for r in rs)/n:.3f}")
        print(f"  mean LATE accuracy     : {sum(r['late_accuracy'] for r in rs)/n:.3f}")
        print(f"  final-estimate correct : {sum(1 for r in rs if r['final_correct'])/n:.3f}")
        print(f"  informative episodes   : {len(inf_rs)}/{n}")
        if inf_rs:
            print(f"  final correct | informative: "
                  f"{sum(1 for r in inf_rs if r['final_correct'])/len(inf_rs):.3f}")
        print(f"  mean P(true fov)       : {sum(r['p_true'] for r in rs)/n:.3f}")
        print(f"  mean entropy           : {sum(r['entropy'] for r in rs)/n:.3f} "
              f"(uniform over 3 = 1.099)")

    print("\nper-layout late accuracy (best first):")
    by_layout = {}
    for r in valid:
        by_layout.setdefault(r["layout"], []).append(r)
    for lay, rs in sorted(by_layout.items(),
                          key=lambda kv: -sum(r["late_accuracy"] for r in kv[1]) / len(kv[1])):
        acc = sum(r["late_accuracy"] for r in rs) / len(rs)
        fin = sum(1 for r in rs if r["final_correct"]) / len(rs)
        print(f"  {lay:<24} late={acc:.3f} final={fin:.3f} "
              f"kbd={rs[0].get('kb_update_delay')} fov={rs[0].get('fov_triple')}")


if __name__ == "__main__":
    main()
