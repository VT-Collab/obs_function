"""
MISHA NEW CHANGE - build and test MANY genuinely different layouts (varied
room sizes, station placements, robot/human positions) IN PARALLEL, each
with a RANDOMLY SAMPLED FOV triple (not fixed to 30/90/180). Each worker
process builds its own MediumLevelPlanner (~3-7 min, unavoidable per unique
layout) but since they all run CONCURRENTLY across local cores, wall-clock
time for the whole batch stays close to that of ONE build, not N builds.

A trial passes if all 3 pairwise subtask-sequence disagreements (over 60
steps, both agents acting every step, robot self-replanning to a hide spot
after grabbing meat) are nontrivial and nobody gets stuck.

Run with: python -m my_methods.bayesian.fov_parallel_layout_search [n_workers]
"""
import os
import sys
import random
import multiprocessing as mp
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import Action
from my_methods.bayesian.fov_subtask_divergence_test import build_grid, build_mdp_and_mlp
from my_methods.bayesian.sticky_subtask_human import StickySubtaskHumanModel
from my_methods.bayesian.fov_sustained_batch import robot_next_action

N_STEPS = 120


def approach_tiles(pos):
    x, y = pos
    return {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)}


def make_layout(idx, rng):
    """Genuinely different room each time: random size, random station
    placement, random human/robot/M/hide positions (all collision-checked)."""
    # MISHA NEW CHANGE: capped back to 15-17x9-11 - MediumLevelPlanner build
    # time scales badly non-linearly with room size (a 20x11 room took 28+
    # min locally earlier this session vs 3 min for 15x9, far worse than the
    # ~60% area increase suggests), so larger rooms were starving worker
    # slots for a very long time and tanking overall batch throughput.
    width = rng.choice([15, 16, 17])
    height = rng.choice([9, 10, 11])
    ix_lo, ix_hi = 2, width - 3
    iy_lo, iy_hi = 2, height - 3

    def rand_pos():
        return (rng.randint(ix_lo, ix_hi), rng.randint(iy_lo, iy_hi))

    used = set()

    def unique_pos():
        while True:
            p = rand_pos()
            if p not in used:
                used.add(p)
                return p

    human_pos = unique_pos()
    robot_start = unique_pos()
    while robot_start == human_pos:
        used.discard(robot_start)
        robot_start = unique_pos()
    m_dy = rng.choice([-1, 1])
    m_pos = (robot_start[0], robot_start[1] + 1)  # south of robot_start - robot faces it by default
    used.add(m_pos)

    hide_pos = unique_pos()
    tries = 0
    while hide_pos in approach_tiles(m_pos) or hide_pos == m_pos:
        used.discard(hide_pos)
        hide_pos = unique_pos()
        tries += 1
        if tries > 50:
            break

    other_stations = ['O', 'D', 'P', 'B', 'W', 'S']
    features = {robot_start: '1', human_pos: '2', m_pos: 'M'}
    for sym in other_stations:
        p = unique_pos()
        features[p] = sym

    # MISHA NEW CHANGE: much wider FOV sampling (10-180 in steps of 2, not
    # 15-180 in steps of 5) for real variety in candidate spread/spacing.
    fov_triple = tuple(sorted(rng.sample(range(10, 181, 2), 3)))
    # vary order list (length), and now also game-mechanic settings, not just
    # one fixed ['steak','steak'] / cook_time=15 / chop_time=5 / wash_time=5.
    n_orders = rng.choice([1, 2, 3, 4, 5])
    order_list = ['steak'] * n_orders
    cook_time = rng.choice([8, 10, 12, 15, 18, 22, 25])
    chop_time = rng.choice([2, 3, 4, 5, 6, 8])
    wash_time = rng.choice([2, 3, 4, 5, 6, 8])
    # MISHA NEW CHANGE: num_items_for_steak=2 is suspected to be the source
    # of an IndexError seen in an earlier run (untested code path in the
    # underlying game logic) - dropped back to the validated default only.
    num_items_for_steak = 1
    # MISHA NEW CHANGE: include a per-job prefix (SLURM_JOB_ID, or a random
    # fallback locally) so concurrently-running search jobs never collide on
    # the same cached-planner filename for a given idx.
    job_prefix = os.environ.get("SLURM_JOB_ID", "local")
    name = f"fov_parallel_{job_prefix}_{idx}"
    return (name, build_grid(width, height, features), human_pos, robot_start, m_pos, hide_pos, fov_triple,
            order_list, cook_time, chop_time, wash_time, num_items_for_steak)


def run_trial(config):
    idx, seed = config
    rng = random.Random(seed)
    (name, grid, human_pos, robot_start, m_pos, hide_pos, fov_triple, order_list,
     cook_time, chop_time, wash_time, num_items_for_steak) = make_layout(idx, rng)
    robot_idx, human_idx = 0, 1

    try:
        mdp, mlp = build_mdp_and_mlp(name, grid, order_list=order_list, cook_time=cook_time,
                                      chop_time=chop_time, wash_time=wash_time,
                                      num_items_for_steak=num_items_for_steak)
    except Exception as e:
        return {"idx": idx, "error": f"build failed: {type(e).__name__}: {e}", "passed": False}

    setup_env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=200)
    setup_state = setup_env.state.deepcopy()

    progressions, positions = {}, {}
    try:
        for fov in fov_triple:
            sim_env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=200)
            shadow = StickySubtaskHumanModel(mlp, setup_state, vision_limit=True, vision_bound=fov, debug=False)
            shadow.set_agent_index(human_idx)
            shadow.init_knowledge_base(setup_state)
            subtasks, pos_trace = [], []
            for t in range(N_STEPS):
                state = sim_env.state
                pos_trace.append(state.players[human_idx].position)
                a_robot = robot_next_action(mlp, state, robot_idx, m_pos, hide_pos)
                try:
                    a_human, _ = shadow.action(state)
                except AssertionError:
                    subtasks.append('STUCK')
                    break
                subtasks.append(shadow.prev_chosen_subtask)
                joint = [Action.STAY, Action.STAY]
                joint[robot_idx], joint[human_idx] = a_robot, a_human
                _, _, done, _ = sim_env.step(tuple(joint))
                if done:
                    subtasks.append('DONE')
                    break
            progressions[fov] = subtasks
            positions[fov] = pos_trace
    except Exception as e:
        return {"idx": idx, "error": f"sim failed: {type(e).__name__}: {e}", "passed": False}

    fa, fb, fc = fov_triple
    seq_a, seq_b, seq_c = progressions[fa], progressions[fb], progressions[fc]

    def pair_disagreement(s1, s2):
        n = min(len(s1), len(s2))
        total = sum(1 for x, y in zip(s1, s2) if x != y)
        half = n // 2
        second_half = sum(1 for x, y in zip(s1[half:n], s2[half:n]) if x != y)
        return total, second_half

    ab_total, ab_late = pair_disagreement(seq_a, seq_b)
    bc_total, bc_late = pair_disagreement(seq_b, seq_c)
    ac_total, ac_late = pair_disagreement(seq_a, seq_c)

    all_distinct = len({tuple(seq_a), tuple(seq_b), tuple(seq_c)}) == 3
    any_stuck = any(len(set(positions[fov])) <= 3 for fov in fov_triple)
    # MISHA NEW CHANGE: require disagreement to persist into the LATTER HALF
    # of the episode too, not just accumulate early and then fully
    # reconverge - "genuine divergence across the entire episode", not a
    # front-loaded blip.
    min_frac = max(8, int(N_STEPS * 0.15))
    sustained = min(ab_late, bc_late, ac_late) >= 3
    passed = (all_distinct and not any_stuck and sustained
              and min(ab_total, bc_total, ac_total) >= min_frac)

    return {
        "idx": idx, "name": name, "size": (grid and (len(grid[0]), len(grid))), "order_list": order_list,
        "cook_time": cook_time, "chop_time": chop_time, "wash_time": wash_time,
        "num_items_for_steak": num_items_for_steak,
        "human_pos": human_pos, "robot_start": robot_start, "m_pos": m_pos, "hide_pos": hide_pos,
        "fov_triple": fov_triple, "pairs_total": (ab_total, bc_total, ac_total),
        "pairs_late_half": (ab_late, bc_late, ac_late),
        "all_distinct": all_distinct, "any_stuck": any_stuck, "sustained": sustained, "passed": passed,
        "sequences": progressions,
    }


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    configs = [(i, 1000 + i) for i in range(n_trials)]
    print(f"running {n_trials} genuinely different layouts across {n_workers} parallel workers...", flush=True)

    results = []
    done_count = 0
    with mp.Pool(n_workers) as pool:
        # MISHA NEW CHANGE: stream results AS they finish (imap_unordered)
        # instead of blocking on the whole batch (map), so progress is
        # visible in the .out log live instead of only at the very end.
        for r in pool.imap_unordered(run_trial, configs):
            results.append(r)
            done_count += 1
            if "error" in r:
                print(f"[{done_count}/{n_trials}] [ERROR] idx={r['idx']}: {r['error']}", flush=True)
                continue
            status = "PASS" if r["passed"] else "fail"
            print(f"[{done_count}/{n_trials}] [{status}] idx={r['idx']} size={r['size']} fov={r['fov_triple']} "
                  f"orders={r['order_list']} cook={r['cook_time']} chop={r['chop_time']} wash={r['wash_time']} "
                  f"hide_pos={r['hide_pos']} pairs_total(ab,bc,ac)={r['pairs_total']} "
                  f"pairs_late_half={r['pairs_late_half']} any_stuck={r['any_stuck']}", flush=True)

    passed = [r for r in results if r.get("passed")]
    print(f"\n=== {len(passed)}/{n_trials} layouts PASSED (genuine 3-way divergence, not stuck) ===\n", flush=True)

    print("\n=== PASSING LAYOUT DETAILS ===", flush=True)
    for r in passed:
        print(f"\nidx={r['idx']} name={r['name']} size={r['size']} human={r['human_pos']} "
              f"robot_start={r['robot_start']} M={r['m_pos']} hide_pos={r['hide_pos']} fov_triple={r['fov_triple']} "
              f"pairs_total={r['pairs_total']} pairs_late_half={r['pairs_late_half']}", flush=True)
        for fov, seq in r["sequences"].items():
            print(f"  FOV={fov}: {seq}", flush=True)


if __name__ == "__main__":
    main()
