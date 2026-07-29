"""
MISHA NEW CHANGE - validate the DESIGNED layout set and write the survivors.

This replaces random search. Random search over 768 wall-embedded layouts gave
0/681 passing, 39/681 with ANY FOV effect, mean real divergence 0.1 - because
the region where a station can sit and make two FOV hypotheses disagree is only
2-9 cells out of ~90 on a 15x9 grid, so it essentially never happens by chance.

fov_designed_layouts.py instead SOLVES that geometry and places the pot and
chopping board into the differential regions by construction. Validated on the
first one built (15x9, anchor (3,6) facing north, triple (30,90,180)):

    fov= 30  DONE  129 steps  5 stalls  12 subtasks  [... pickup_onion ...]
    fov= 90  DONE  129 steps  0 stalls  12 subtasks  [... pickup_onion ...]
    fov=180  DONE  129 steps  0 stalls  12 subtasks  [... pickup_garnish ...]
    real divergence  30v90 = 2 (late 2),  30v180 = 2 (late 2),  90v180 = 0

i.e. every FOV completes the orders, the full-vision control never stalls, and
the narrow hypotheses choose pickup_onion where full vision chooses
pickup_garnish - a different subtask in a different order, surviving phase
correction and sustained into the late half.

This script runs that check across every enumerated design and keeps the ones
that clear both gates.

Run with: python -m my_methods.bayesian.validate_designed [n_workers] [n_seeds] [max_n]
"""
import multiprocessing as mp
import os
import sys

from my_methods.bayesian.fov_designed_layouts import enumerate_designed
from my_methods.bayesian.fov_subtask_divergence_test import build_mdp_and_mlp
from my_methods.bayesian.fov_divergence_search_v2 import (
    rollout, pair_disagreement, discard_planner_cache, DEAD,
    MAX_CONTROL_STALL_FRAC, MIN_CONTROL_SUBTASKS,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fov", "layouts_designed")
ORDERS, COOK, CHOP, WASH = ['steak'] * 3, 10, 3, 3
KB_DELAY = 2
# At least this many of the 3 FOV pairs must show real divergence. The first
# validated design separated 2 of 3 (the mid/wide pair shared visibility of the
# board), which is still enough for the filter to discriminate, so 2 is the bar.
MIN_SEPARATING_PAIRS = 2


def check(args):
    cfg_in, n_seeds = args
    name = f"fovdes_{cfg_in['name']}"
    t = cfg_in["fov_triple"]
    cfg = dict(grid=cfg_in["grid"], fov_triple=t, order_list=ORDERS, cook_time=COOK,
               chop_time=CHOP, wash_time=WASH, num_items_for_steak=1,
               robot_mode="work", human_mode="plain", kb_update_delay=KB_DELAY,
               m_pos=(0, 0), hide_pos=(0, 0), name=name)
    try:
        mdp, mlp = build_mdp_and_mlp(name, cfg["grid"], order_list=ORDERS, cook_time=COOK,
                                     chop_time=CHOP, wash_time=WASH, num_items_for_steak=1)
    except Exception as e:
        return dict(name=name, error=f"build: {type(e).__name__}: {e}")
    try:
        rolls = {s: {f: rollout(mdp, mlp, cfg, f, s) for f in t} for s in range(n_seeds)}
        ctrl = {s: rollout(mdp, mlp, cfg, 180, s) for s in range(n_seeds)}
    except Exception as e:
        discard_planner_cache(name)
        return dict(name=name, error=f"sim: {type(e).__name__}: {e}")
    discard_planner_cache(name)

    # Full-observability gate: 180deg must finish, barely stall, work the task.
    ctrl_done = sum(1 for s in range(n_seeds) if ctrl[s][0] and ctrl[s][0][-1] == 'DONE')
    ctrl_stall = max(ctrl[s][2] / max(1, len(ctrl[s][0])) for s in range(n_seeds))
    ctrl_sub = min(len({x for x in ctrl[s][0] if x not in DEAD}) for s in range(n_seeds))
    full_ok = (ctrl_done == n_seeds and ctrl_stall <= MAX_CONTROL_STALL_FRAC
               and ctrl_sub >= MIN_CONTROL_SUBTASKS)

    # Real (phase-corrected) divergence per pair, worst seed.
    pair_scores = []
    for i in range(3):
        for j in range(i + 1, 3):
            worst = min(pair_disagreement(rolls[s][t[i]][0], rolls[s][t[j]][0])[2]
                        for s in range(n_seeds))
            worst_late = min(pair_disagreement(rolls[s][t[i]][0], rolls[s][t[j]][0])[3]
                             for s in range(n_seeds))
            pair_scores.append((worst, worst_late))
    n_sep = sum(1 for p, _ in pair_scores if p > 0)
    n_sep_late = sum(1 for _, l in pair_scores if l > 0)

    all_r = [rolls[s][f] for s in range(n_seeds) for f in t]
    n_done = sum(1 for r in all_r if r[0] and r[0][-1] == 'DONE')
    passes = full_ok and n_sep >= MIN_SEPARATING_PAIRS and n_sep_late >= 1

    if passes:
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            grid_str = "\n                ".join(cfg["grid"])
            with open(os.path.join(OUT_DIR, f"{name}.layout"), "w") as fh:
                fh.write(
                    f"# MISHA NEW CHANGE - DESIGNED layout (not random search).\n"
                    f"# The pot and chopping board are placed BY CONSTRUCTION into the cells\n"
                    f"# visible to a wider FOV but not a narrower one from the human's working\n"
                    f"# pose, so vision provably changes what the human KNOWS about the two\n"
                    f"# stations the teammate keeps changing. Random search could not find this:\n"
                    f"# the target region is 2-9 cells of ~90, and 768 random layouts gave a mean\n"
                    f"# real divergence of 0.1.\n"
                    f"#\n"
                    f"# room={cfg_in['room']} anchor={cfg_in['anchor']} facing={cfg_in['ori']}\n"
                    f"# pot={cfg_in['pot_pos']} board={cfg_in['board_pos']}\n"
                    f"# differential cells: narrow|mid={cfg_in['n_diff_nm']} mid|wide={cfg_in['n_diff_mw']}\n"
                    f"# FOV triple (deg): {t[0]}, {t[1]}, {t[2]}\n"
                    f"# kb_update_delay: {KB_DELAY}   <- REQUIRED; at 0 every FOV learns the same\n"
                    f"#   things and inference is impossible.\n"
                    f"# teammate: work (full-vision GreedySteakHumanModel)\n"
                    f"# human model: plain SteakLimitVisionHumanModel\n"
                    f"#\n"
                    f"# FULL-OBSERVABILITY CONTROL (180 deg): {ctrl_done}/{n_seeds} DONE, "
                    f"{ctrl_stall*100:.1f}% stalled, {ctrl_sub} subtasks\n"
                    f"# real phase-corrected divergence per FOV pair (worst seed): "
                    f"{[p for p, _ in pair_scores]}\n"
                    f"# ... late half: {[l for _, l in pair_scores]}\n"
                    f"# rollouts completing the task: {n_done}/{len(all_r)}\n"
                    "{\n"
                    f'    "grid":  """{grid_str}""",\n'
                    f'    "start_order_list": {ORDERS},\n'
                    f'    "cook_time": {COOK},\n'
                    '    "delivery_reward": 20,\n'
                    "    'num_items_for_steak': 1,\n"
                    f"    'chop_time': {CHOP},\n"
                    f"    'wash_time': {WASH},\n"
                    '    "rew_shaping_params": None\n'
                    "}\n")
        except Exception:
            pass

    return dict(name=name, fov=t, room=cfg_in["room"], anchor=cfg_in["anchor"],
                ori=cfg_in["ori"], pairs=[p for p, _ in pair_scores],
                pairs_late=[l for _, l in pair_scores], n_sep=n_sep,
                ctrl_done=ctrl_done, ctrl_stall=ctrl_stall, ctrl_sub=ctrl_sub,
                full_ok=full_ok, n_done=n_done, n_rolls=len(all_r), passes=passes)


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    max_n = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    designs = enumerate_designed(max_n)
    print(f"validating {len(designs)} DESIGNED layouts, {n_seeds} seeds, {n_workers} workers",
          flush=True)
    print(f"gates: >={MIN_SEPARATING_PAIRS}/3 FOV pairs with real divergence (>=1 late), "
          f"AND full-FOV control DONE on every seed with <={MAX_CONTROL_STALL_FRAC*100:.0f}% "
          f"stalls and >={MIN_CONTROL_SUBTASKS} subtasks\n", flush=True)

    results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(check, [(d, n_seeds) for d in designs])):
            results.append(r)
            if "error" in r:
                print(f"[{i+1}/{len(designs)}] {r['name'][:40]:<40} ERR {r['error'][:40]}", flush=True)
            else:
                print(f"[{i+1}/{len(designs)}] {r['name'][:40]:<40} "
                      f"pairs={r['pairs']} late={r['pairs_late']} "
                      f"ctrl={r['ctrl_done']}done,{r['ctrl_stall']*100:.0f}%,{r['ctrl_sub']}sub "
                      f"done={r['n_done']}/{r['n_rolls']} "
                      f"{'PASS' if r['passes'] else 'fail'}", flush=True)

    valid = [r for r in results if "error" not in r]
    keep = [r for r in valid if r["passes"]]
    print(f"\n=== {len(keep)}/{len(valid)} DESIGNED layouts passed ===")
    if valid:
        print(f"full-FOV control passed on {sum(1 for r in valid if r['full_ok'])}/{len(valid)}")
        print(f">=2 pairs separating on {sum(1 for r in valid if r['n_sep'] >= 2)}/{len(valid)}")
        print(f"all 3 pairs separating on {sum(1 for r in valid if r['n_sep'] == 3)}/{len(valid)}")
    if keep:
        print(f"\nwritten to {OUT_DIR}")
        for r in sorted(keep, key=lambda r: -sum(r["pairs"]))[:25]:
            print(f"  {r['name'][:44]:<44} pairs={r['pairs']} late={r['pairs_late']} "
                  f"ctrl={r['ctrl_stall']*100:.0f}% stall")


if __name__ == "__main__":
    main()
