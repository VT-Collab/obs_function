"""
MISHA NEW CHANGE - Bayesian FOV inference accuracy across all 23 validated
sustained-divergence layouts (fov/layouts/fov_search_rank01..23.layout, see
fov_search_results.md). For each layout, for each of its 3 FOV candidates as
TRUE ground truth: run the real StickySubtaskHumanModel at that FOV + the same
lightweight scripted robot, watched online by SteakBayesFOVInference
(fov_bayes_filter.py - the minigrid-style log-space filter). Reports accuracy
(fraction of steps the MAP estimate matches ground truth), final-belief
correctness, and how fast/confidently it converges.

MISHA NEW CHANGE: now sweeps the LIKELIHOOD MODE as an experimental variable
rather than hardcoding one. "greedy" is the minigrid-faithful epsilon-greedy
scoring against the hypothesis planner's best action; "boltzmann" is the softmax
over one-step-ahead motion costs. The previous session switched to boltzmann on
a hunch and never measured it - this runs both over the same trials so the
comparison is real. Pass a mode name as argv[2] to run just one.

Reuses the already-built MediumLevelPlanner caches (fov_search_rank<NN>_am.pkl)
- no rebuilding needed, this should be fast.

Run with: python -m my_methods.bayesian.fov_inference_batch [n_workers] [mode] [max_rank]
"""
import os
import re
import sys
import multiprocessing as mp
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import Action
from my_methods.bayesian.fov_subtask_divergence_test import build_mdp_and_mlp
from my_methods.bayesian.sticky_subtask_human import StickySubtaskHumanModel
from my_methods.bayesian.fov_sustained_batch import robot_next_action
from my_methods.bayesian.fov_bayes_filter import SteakBayesFOVInference, apply_initial_kb
from overcooked_ai_py.agents.agent import GreedySteakHumanModel

# Overridable so the local smoke test (which only has rank01's planner cache)
# can run a handful of steps instead of a full episode.
N_STEPS = int(os.environ.get("FOV_N_STEPS", 60))
MODES = ["greedy", "boltzmann"]

# MISHA NEW CHANGE - which teammate the watched human is paired with.
#
#   "hide" - the original scripted robot from the layout search: grab the meat,
#            walk to hide_pos, then STAY forever. Diagnosed as fatal for
#            inference: with the robot parked and holding the meat, the world
#            state freezes (knowledge-base key stuck at "0.-1.-1.meat" for 118
#            consecutive steps on rank01) and the human livelocks against it.
#            A frozen world contains nothing for the FOV hypotheses to disagree
#            about, so all three keep identical knowledge and the posterior can
#            never move off the prior. See diagnose_shadow_divergence.py.
#
#   "work" - a full-vision GreedySteakHumanModel that actually performs the
#            steak workflow. It continuously changes pot / chopping board / sink
#            state, which is exactly the information a narrow-FOV human misses
#            and a wide-FOV human sees. THAT asymmetry is what makes the FOV
#            hypotheses choose different subtasks, which is what the Bayes
#            update needs in order to have any evidence at all.
ROBOT_MODE = os.environ.get("FOV_ROBOT", "work")

# What the human (and each hypothesis) knows at t=0: "fov" | "empty" | "omniscient".
INITIAL_KB = os.environ.get("FOV_INITIAL_KB", "fov")
LAYOUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fov", "layouts")

# rank -> hide_pos (not stored in the .layout file itself - it's the robot's
# script target, not part of the physical layout) - carried over from
# curate_layouts.py's WINNERS table.
HIDE_POS = {
    1: (10, 3), 2: (7, 4), 3: (11, 4), 4: (8, 6), 5: (4, 4), 6: (9, 7), 7: (9, 3),
    8: (7, 4), 9: (9, 5), 10: (9, 3), 11: (2, 5), 12: (7, 6), 13: (9, 5), 14: (4, 3),
    15: (12, 3), 16: (3, 5), 17: (6, 5), 18: (7, 3), 19: (8, 4), 20: (10, 5),
    21: (10, 6), 22: (2, 4), 23: (5, 2),
}


def parse_layout_file(path):
    with open(path) as f:
        content = f.read()
    grid_match = re.search(r'"grid":\s*"""(.*?)"""', content, re.DOTALL)
    grid = [line.strip() for line in grid_match.group(1).strip().split("\n")]
    fov_match = re.search(r"# FOV triple \(deg\): (\d+), (\d+), (\d+)", content)
    fov_triple = tuple(int(x) for x in fov_match.groups())
    m_pos = None
    robot_start = None
    human_pos = None
    for y, row in enumerate(grid):
        for x, c in enumerate(row):
            if c == 'M':
                m_pos = (x, y)
            elif c == '1':
                robot_start = (x, y)
            elif c == '2':
                human_pos = (x, y)
    order_list_match = re.search(r'"start_order_list":\s*(\[[^\]]*\])', content)
    order_list = eval(order_list_match.group(1))
    cook_time = int(re.search(r'"cook_time":\s*(\d+)', content).group(1))
    chop_time = int(re.search(r"'chop_time':\s*(\d+)", content).group(1))
    wash_time = int(re.search(r"'wash_time':\s*(\d+)", content).group(1))
    num_items = int(re.search(r"'num_items_for_steak':\s*(\d+)", content).group(1))
    return dict(grid=grid, fov_triple=fov_triple, m_pos=m_pos, robot_start=robot_start,
                human_pos=human_pos, order_list=order_list, cook_time=cook_time,
                chop_time=chop_time, wash_time=wash_time, num_items_for_steak=num_items)


def run_inference(args):
    rank, true_fov, mode = args
    name = f"fov_search_rank{rank:02d}"
    path = os.path.join(LAYOUTS_DIR, f"{name}.layout")
    cfg = parse_layout_file(path)
    hide_pos = HIDE_POS[rank]
    robot_idx, human_idx = 0, 1
    base = dict(rank=rank, true_fov=true_fov, mode=mode)

    try:
        mdp, mlp = build_mdp_and_mlp(name, cfg["grid"], order_list=cfg["order_list"],
                                      cook_time=cfg["cook_time"], chop_time=cfg["chop_time"],
                                      wash_time=cfg["wash_time"], num_items_for_steak=cfg["num_items_for_steak"])
    except Exception as e:
        return dict(base, error=f"build failed: {type(e).__name__}: {e}")

    setup_env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=300)
    setup_state = setup_env.state.deepcopy()

    sim_env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=300)
    real_human = StickySubtaskHumanModel(mlp, setup_state, vision_limit=True,
                                         vision_bound=true_fov, debug=False)
    real_human.set_agent_index(human_idx)
    # Same initial-knowledge rule as the hypotheses - if ground truth started
    # omniscient while the shadows started FOV-limited, the likelihood model
    # would be misspecified for every hypothesis including the true one.
    apply_initial_kb(real_human, setup_state, INITIAL_KB)

    robot = None
    if ROBOT_MODE == "work":
        robot = GreedySteakHumanModel(mlp)
        robot.set_agent_index(robot_idx)

    inf = SteakBayesFOVInference(mlp, setup_state, candidate_fovs=cfg["fov_triple"],
                                 human_agent_index=human_idx, likelihood=mode,
                                 initial_kb=INITIAL_KB)

    estimates = []
    entropies = []
    try:
        for t in range(N_STEPS):
            state = sim_env.state
            if robot is not None:
                try:
                    a_robot, _ = robot.action(state)
                except Exception:
                    a_robot = Action.STAY
            else:
                a_robot = robot_next_action(mlp, state, robot_idx, cfg["m_pos"], hide_pos)
            try:
                a_human, _ = real_human.action(state)
            except AssertionError:
                break
            # ORDERING: update BEFORE env.step, so every hypothesis sees the same
            # state the real human acted on.
            inf.update(state, a_human)
            estimates.append(inf.map_fov())
            entropies.append(inf.entropy())
            joint = [Action.STAY, Action.STAY]
            joint[robot_idx], joint[human_idx] = a_robot, a_human
            _, _, done, _ = sim_env.step(tuple(joint))
            if done:
                break
    except Exception as e:
        return dict(base, error=f"sim failed: {type(e).__name__}: {e}")

    if not estimates:
        return dict(base, error="no steps completed")

    correct = sum(1 for e in estimates if e == true_fov)
    accuracy = correct / len(estimates)
    final_estimate = estimates[-1]
    first_correct = next((i for i, e in enumerate(estimates) if e == true_fov), None)
    stable_after_first = (first_correct is not None and
                          all(e == true_fov for e in estimates[first_correct:]))
    # Accuracy over the LATTER HALF - a filter that needs a while to converge is
    # fine; one that never settles is not. This is the number that matters most.
    half = len(estimates) // 2
    late_accuracy = (sum(1 for e in estimates[half:] if e == true_fov) / len(estimates[half:])
                     if estimates[half:] else 0)
    post = inf.posterior()
    goal_divergence_rate = (inf.n_goal_divergent_steps / inf.n_steps_seen
                            if inf.n_steps_seen else 0)
    return dict(base, fov_triple=cfg["fov_triple"], n_steps=len(estimates),
                accuracy=accuracy, late_accuracy=late_accuracy,
                final_correct=(final_estimate == true_fov), final_estimate=final_estimate,
                first_correct_step=first_correct, stable_after_first=stable_after_first,
                final_belief=[round(post[f], 4) for f in inf.candidate_fovs],
                final_entropy=entropies[-1], p_true=round(post[true_fov], 4),
                goal_divergent_steps=inf.n_goal_divergent_steps,
                goal_divergence_rate=goal_divergence_rate, n_crashes=inf.n_crashes)


def summarise(results, jobs_n):
    valid = [r for r in results if "error" not in r]
    print(f"\n=== SUMMARY ({len(valid)}/{jobs_n} valid) ===", flush=True)
    if not valid:
        return
    for mode in MODES:
        rs = [r for r in valid if r["mode"] == mode]
        if not rs:
            continue
        n = len(rs)
        print(f"\n--- mode={mode} ({n} trials) ---")
        print(f"  mean per-step accuracy : {sum(r['accuracy'] for r in rs)/n:.3f}")
        print(f"  mean LATE-half accuracy: {sum(r['late_accuracy'] for r in rs)/n:.3f}")
        print(f"  final-estimate correct : {sum(1 for r in rs if r['final_correct'])/n:.3f} "
              f"({sum(1 for r in rs if r['final_correct'])}/{n})")
        print(f"  stable-after-first     : {sum(1 for r in rs if r['stable_after_first'])/n:.3f}")
        print(f"  mean P(true fov)       : {sum(r['p_true'] for r in rs)/n:.3f}")
        print(f"  mean final entropy     : {sum(r['final_entropy'] for r in rs)/n:.3f} "
              f"(uniform over 3 = 1.099)")
        print(f"  mean goal-divergence   : {sum(r['goal_divergence_rate'] for r in rs)/n:.3f}")
        print(f"  trials w/ >=1 divergent: {sum(1 for r in rs if r['goal_divergent_steps'] > 0)}/{n}")
        print(f"  total shadow crashes   : {sum(r['n_crashes'] for r in rs)}")

    print("\nper-layout breakdown (late-half accuracy per true-FOV):")
    by_rank = {}
    for r in valid:
        by_rank.setdefault(r["rank"], []).append(r)
    for rank in sorted(by_rank):
        for mode in MODES:
            rs = sorted([r for r in by_rank[rank] if r["mode"] == mode], key=lambda r: r["true_fov"])
            if not rs:
                continue
            accs = [f"{r['true_fov']}:{r['late_accuracy']:.2f}" for r in rs]
            print(f"  rank{rank:02d} {mode:<9} fov_triple={rs[0]['fov_triple']} "
                  f"late_acc=[{', '.join(accs)}] div={rs[0]['goal_divergence_rate']:.2f}")


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    # argv[2]: a single mode name, or "all"/omitted for the full sweep.
    modes = MODES if len(sys.argv) <= 2 or sys.argv[2] == "all" else [sys.argv[2]]
    max_rank = int(sys.argv[3]) if len(sys.argv) > 3 else 23

    jobs = []
    for rank in range(1, max_rank + 1):
        path = os.path.join(LAYOUTS_DIR, f"fov_search_rank{rank:02d}.layout")
        cfg = parse_layout_file(path)
        for true_fov in cfg["fov_triple"]:
            for mode in modes:
                jobs.append((rank, true_fov, mode))

    print(f"running {len(jobs)} inference trials "
          f"({max_rank} layouts x 3 FOVs x {len(modes)} modes) across {n_workers} workers, "
          f"{N_STEPS} steps each...", flush=True)

    results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_inference, jobs)):
            results.append(r)
            if "error" in r:
                print(f"[{i+1}/{len(jobs)}] rank={r['rank']:02d} fov={r['true_fov']:>3} "
                      f"{r['mode']:<9} ERROR: {r['error']}", flush=True)
            else:
                print(f"[{i+1}/{len(jobs)}] rank={r['rank']:02d} fov={r['true_fov']:>3} "
                      f"{r['mode']:<9} acc={r['accuracy']:.2f} late={r['late_accuracy']:.2f} "
                      f"final_ok={r['final_correct']} first={r['first_correct_step']} "
                      f"P(true)={r['p_true']:.3f} H={r['final_entropy']:.2f} "
                      f"div={r['goal_divergence_rate']:.2f} belief={r['final_belief']}", flush=True)

    summarise(results, len(jobs))


if __name__ == "__main__":
    main()
