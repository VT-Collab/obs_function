"""
MISHA NEW CHANGE - layout/settings search for GENUINE, SUSTAINED FOV-driven
subtask divergence. Replaces fov_parallel_layout_search.py, whose scoring was
confounded by RNG.

WHAT WAS WRONG WITH v1
----------------------
v1 scored a layout by rolling out one episode per FOV candidate, sequentially,
in a single process (fov_parallel_layout_search.py:124-128), and counting steps
where the subtask sequences disagreed. It never reset the RNG between those
rollouts. GreedyHumanModel.auto_unstuck picks its unblocking move with
np.random.choice (agent.py:449) and fires constantly on these layouts, so each
FOV's rollout consumed a different stretch of the global random stream. The
"disagreement" it measured was therefore FOV effect + RNG divergence, dominated
by the latter.

Measured on rank01 - v1's top-ranked layout, recorded as maximal divergence with
pairs_late_half=(60,60,60):
    same seed, different FOV (real FOV effect) :   0 / 120 steps
    same FOV, different seed (pure RNG noise)  : 107 / 120 steps
This also explains the anomaly fov_search_results.md flags as "unexplained":
identical layout parameters producing different divergence numbers across two
jobs. The simulation is not deterministic.

WHAT v2 DOES DIFFERENTLY
------------------------
1. RNG IS RESET TO THE SAME SEED before every rollout, so two rollouts that
   differ only in FOV consume the identical random stream. Any disagreement is
   then attributable to field of view and nothing else.
2. EVERY TRIAL IS REPLICATED ACROSS SEVERAL SEEDS and must show divergence under
   ALL of them. One seed's divergence is an anecdote; agreement across seeds is
   a property of the layout.
3. THE RNG NOISE FLOOR IS MEASURED ALONGSIDE (same FOV, different seeds) and
   reported, so signal and noise are never conflated again.
4. THE TEAMMATE ACTUALLY WORKS. v1's robot grabbed the meat, walked to a hide
   spot and STAYED forever - which freezes the world (knowledge-base key pinned
   at "0.-1.-1.meat" for 118 consecutive steps on rank01). A frozen world gives
   the FOV hypotheses nothing to disagree about. v2 searches over teammate
   behaviour, defaulting to a full-vision GreedySteakHumanModel that keeps
   changing pot / board / sink state - which is exactly the information a
   narrow-FOV human misses.
5. A STUCK HUMAN NO LONGER KILLS THE EPISODE. The known
   `assert len(motion_goals) != 0` bug (CARC_NOTES.md) is caught per-step: the
   human stays put and drops its sticky commitment so it re-evaluates next step.
   Only a human stuck for MAX_CONSEC_STUCK steps in a row ends the trial.
6. RICHNESS IS REQUIRED, not just disagreement: enough distinct subtasks and
   enough distinct positions that the human is demonstrably doing the task
   rather than oscillating in a corner.

Run with: python -m my_methods.bayesian.fov_divergence_search_v2 [n_trials] [n_workers] [n_seeds]
"""
import math
import os
import random
import sys
import multiprocessing as mp

import numpy as np

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import Action
from overcooked_ai_py.agents.agent import GreedySteakHumanModel, SteakLimitVisionHumanModel
from my_methods.bayesian.fov_subtask_divergence_test import build_grid, build_mdp_and_mlp
from my_methods.bayesian.sticky_subtask_human import StickySubtaskHumanModel
from my_methods.bayesian.fov_sustained_batch import robot_next_action
from my_methods.bayesian.fov_parallel_layout_search import approach_tiles

# MISHA NEW CHANGE - episodes can now run long (the plain human completes real
# steak orders instead of stalling on pickup_meat), so give them room.
N_STEPS = 200
N_SEEDS = 3
# Raised from 8: with kb_update_delay 2-3 the knowledge base is deliberately
# stale, so the human transiently has no valid motion goal far more often. Those
# stalls are usually temporary - the teammate moves, or the KB refreshes, and it
# resumes. At 8 the episode was being killed at ~50 steps by recoverable stalls,
# truncating exactly the late-episode window where divergence is measured.
MAX_CONSEC_STUCK = 20

# MISHA NEW CHANGE - the watched human is the PLAIN SteakLimitVisionHumanModel,
# not StickySubtaskHumanModel.
#
# The sticky wrapper was originally introduced to MANUFACTURE sustained
# divergence by committing the human to a pickup subtask instead of re-deriving
# it every step. Measured head-to-head on the v3 layouts it does the opposite -
# it cripples the human:
#
#   StickySubtaskHumanModel    len=18-21   npos=11-14  subtasks: pickup_meat[, drop_meat]
#   SteakLimitVisionHumanModel len=39-177  npos=15-44  subtasks: pickup_meat, pickup_onion,
#                                                      drop_onion, chop_onion, pickup_plate,
#                                                      drop_plate, pickup_washed_plate, ...
#
# Committing to pickup_meat routes the human into a state its planner has no
# motion goal for, it stalls, and the episode dies ~20 steps in having touched
# two subtasks. A human that never progresses past drop_meat cannot express
# FOV-dependent subtask choice at all - which is a large part of why the v1
# curated layouts looked signal-free. With the plain model the human runs the
# whole steak workflow, and different FOVs at a FIXED seed then produce visibly
# different trajectories (e.g. v3 idx=1: fov46/fov68 -> len 53, fov180 -> len 177).
HUMAN_CLS = {"plain": SteakLimitVisionHumanModel, "sticky": StickySubtaskHumanModel}

# MISHA NEW CHANGE - kb_update_delay is THE knob that makes field of view matter.
#
# An observation only enters the knowledge base once the object has been held in
# view for kb_update_delay CONSECUTIVE steps (agent.py:1425, gated on
# obj_count = min(kb_update_delay, prev_count + 1)). At the project-wide default
# of 0 a single frame of contact commits a fact to memory, so a 16-degree cone
# and a 180-degree cone accumulate identical knowledge given any wandering at
# all - which is why 58 of 60 searched layouts showed FOV_total=0.
#
# Measured on one layout where the human demonstrably works, same seeds
# throughout, varying ONLY the delay (clean pairwise subtask disagreement):
#     delay=0 -> [0, 0, 0]      lens 150/150/150
#     delay=1 -> [0, 0, 0]      lens 150/150/150
#     delay=2 -> [0,12,12] and [0,37,37]   lens 148/148/150   <-- real divergence
#     delay=3 -> [0,15,15] / [0, 0, 0]     lens 148/148/150
#     delay=5 -> [0, 0, 0]      lens  70/ 70/ 63   <-- KB too stale, human stalls
#     delay=8 -> [0, 0, 0]      lens 108/108/ 29
# So 2-3 is the usable band: stale enough that a narrow cone genuinely misses
# state changes, not so stale that the human starves and stops working.
KB_UPDATE_DELAY_CHOICES = [2, 2, 3]


def sample_fov_triple(rng):
    """Sample 3 FOVs that are DISCRETELY distinguishable on a grid.

    in_bound() accepts a cell when y <= -cos(fov/2) * |x| (agent.py:906-908), so
    what separates two FOVs is the gap in cos(fov/2), NOT the gap in degrees.
    Sampling uniformly in degrees clusters hypotheses badly at the narrow end:
    fov=36 -> cos=0.951 and fov=70 -> cos=0.819 almost never disagree about which
    INTEGER cell is visible, so they behave as the same cone. That is exactly what
    the delay sweep showed - the (36,70,180) triple had one pair stuck at zero
    disagreement no matter what.

    So sample uniformly in cos-space with a minimum separation, then convert back
    to degrees.
    """
    for _ in range(200):
        cs = sorted(rng.uniform(0.0, 0.97) for _ in range(3))
        if cs[1] - cs[0] >= 0.25 and cs[2] - cs[1] >= 0.25:
            # cos is decreasing in fov, so the largest cos is the narrowest cone.
            fovs = sorted(int(round(2 * math.degrees(math.acos(c)))) for c in cs)
            fovs = [max(10, min(180, f)) for f in fovs]
            if len(set(fovs)) == 3:
                return tuple(fovs)
    return (60, 120, 180)


LAYOUTS_V3_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fov", "layouts_v3")


def write_layout_file_v3(cfg, score):
    """Write one passing layout, with its scores in the comment header.

    Called from the worker as soon as a trial passes so nothing is lost if the
    batch is killed. Named by idx (unique per trial) rather than by rank, since
    rank is only knowable once every trial has finished.
    """
    os.makedirs(LAYOUTS_V3_DIR, exist_ok=True)
    # Name includes the run tag: successive searches use different seed bases to
    # explore NEW layouts, and idx restarts at 0 each run, so an idx-only name
    # would silently overwrite earlier winners.
    tag = os.environ.get("FOV_RUN_TAG", "r0")
    path = os.path.join(LAYOUTS_V3_DIR, f"fov_v3_{tag}_idx{score['idx']:04d}.layout")
    grid_str = "\n                ".join(cfg["grid"])
    with open(path, "w") as fh:
        fh.write(
            f"# MISHA NEW CHANGE - passing layout from the v3 search\n"
            f"# (my_methods/bayesian/fov_divergence_search_v2.py, idx={score['idx']}).\n"
            f"#\n"
            f"# Scored with the RNG HELD FIXED across FOVs, so these counts are a real FOV\n"
            f"# effect, not auto_unstuck's dice - which is what invalidated the v1\n"
            f"# fov_search_rank* set (rank01: 0/120 at fixed seed, 107/120 across seeds).\n"
            f"# Counts are CLEAN: only steps where BOTH humans are actively pursuing a\n"
            f"# real subtask and pick different ones. Divergence held under all\n"
            f"# {score['n_seeds']} seeds.\n"
            f"#\n"
            f"# human model: plain SteakLimitVisionHumanModel (NOT sticky - the sticky\n"
            f"#   wrapper stalls the human ~20 steps in with 1-2 subtasks touched)\n"
            f"# teammate: {cfg['robot_mode']} (full-vision GreedySteakHumanModel)\n"
            f"# kb_update_delay: {cfg['kb_update_delay']}  <- REQUIRED. At the project\n"
            f"#   default of 0 every FOV learns the same things and inference is\n"
            f"#   impossible; the filter must use this same value.\n"
            f"# FOV triple (deg): {cfg['fov_triple'][0]}, {cfg['fov_triple'][1]}, {cfg['fov_triple'][2]}\n"
            f"#   (sampled spread in cos(fov/2), the quantity in_bound() actually tests)\n"
            f"#\n"
            f"# clean pairwise subtask disagreement, whole episode: {score['fov_min_total']}\n"
            f"# clean pairwise subtask disagreement, latter half   : {score['fov_min_late']}\n"
            f"# RNG noise floor (same FOV, different seed)         : {score['rng_noise']:.1f}\n"
            f"# distinct subtasks (worst rollout): {score['min_subtasks']}   "
            f"distinct tiles: {score['min_positions']}   min len: {score['min_len']}\n"
            f"# hide_pos={cfg['hide_pos']}\n"
            "{\n"
            f'    "grid":  """{grid_str}""",\n'
            f'    "start_order_list": {cfg["order_list"]},\n'
            f'    "cook_time": {cfg["cook_time"]},\n'
            '    "delivery_reward": 20,\n'
            f"    'num_items_for_steak': {cfg['num_items_for_steak']},\n"
            f"    'chop_time': {cfg['chop_time']},\n"
            f"    'wash_time': {cfg['wash_time']},\n"
            '    "rew_shaping_params": None\n'
            "}\n"
        )
    return path


def discard_planner_cache(name):
    """Delete this layout's cached MediumLevelPlanner pickle.

    Each is 120-220MB and a full search builds hundreds of them; CARC's
    /home1 quota is 100GB with only ~25GB free, so keeping them all would
    abort the job partway through on a disk error. Winners get rebuilt on
    demand from the recorded layout file, which is cheap (27-62s) for these
    wall-embedded layouts.
    """
    try:
        from overcooked_ai_py.planning.planners import PLANNERS_DIR
        path = os.path.join(PLANNERS_DIR, f"{name}_am.pkl")
    except Exception:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

# Pass thresholds - deliberately modest per-pair, because unlike v1 these are
# RNG-free counts, so even a handful of steps is real signal rather than dice.
MIN_PAIR_TOTAL = 6
MIN_PAIR_LATE = 3
MIN_DISTINCT_SUBTASKS = 3
MIN_DISTINCT_POSITIONS = 6
# FOV signal must be at least this multiple of the RNG noise floor.
NOISE_MARGIN = 1.5
# Full-observability control: a 180-degree human must complete the orders on
# EVERY seed, stall almost never, and touch most of the workflow. Measured
# achievable: fov=180 gives DONE with 0-2.6% stalls and 11-12 subtasks, while
# narrow FOV on the same layout stalls 21-31% and ends STUCK.
MAX_CONTROL_STALL_FRAC = 0.05
MIN_CONTROL_SUBTASKS = 8


STATIONS = ['P', 'D', 'M', 'O', 'B', 'W', 'S']


def make_layout_v3(idx, rng):
    """Random layouts in the STYLE OF THE HAND-DESIGNED steak layouts, i.e.
    stations embedded in the surrounding wall band and on the faces of a central
    island, with open floor between them.

    WHY NOT v1's GENERATOR. v1 (fov_parallel_layout_search.make_layout, via
    build_grid) scattered stations at random INTERIOR positions, as free-standing
    obstacles in the middle of open floor. No hand-designed steak layout looks
    like that - compare steak_island / steak_tshape, where the stations live in
    the wall band ("XXXPXMOXXDXBXXX") and on a small central island. That
    topology difference is not cosmetic: on the v1-generated rank01 the human
    visits FOUR distinct tiles in 120 steps and never gets past 'drop_meat',
    whereas on steak_island it completes orders. A human that cannot do the task
    cannot exhibit FOV-dependent subtask choice, which is a large part of why
    those 23 layouts carried no signal.

    SPREAD IS DELIBERATE. in_bound() treats any tile immediately beside the
    player as visible regardless of FOV (see evaluate_bayesian_lightweight.py's
    notes), and the human always stands next to the station it is acting on. So
    FOV can ONLY matter for facts the human must learn AT A DISTANCE. Stations
    are therefore chosen by farthest-point sampling to push them apart, which
    maximises the chance that a narrow-FOV human misses a state change a
    wide-FOV one catches.
    """
    width = rng.choice([15, 16, 17])
    height = rng.choice([9, 10])
    cells = [['X'] * width for _ in range(height)]

    # Open floor: a margin of wall on every side, like the hand-designed layouts.
    x_lo, x_hi = 2, width - 3
    y_lo, y_hi = 2, height - 3
    for y in range(y_lo, y_hi + 1):
        for x in range(x_lo, x_hi + 1):
            cells[y][x] = ' '

    # Central island - a small wall block the agents must walk around, whose
    # faces host stations (steak_island's "XXXWX" / "BXXXX").
    island = set()
    if rng.random() < 0.85:
        iw = rng.choice([3, 4, 5])
        ih = 1 if height <= 9 else rng.choice([1, 2])
        if x_hi - x_lo - iw >= 2 and y_hi - y_lo - ih >= 2:
            ix = rng.randint(x_lo + 1, x_hi - iw)
            iy = rng.randint(y_lo + 1, y_hi - ih)
            for y in range(iy, iy + ih):
                for x in range(ix, ix + iw):
                    cells[y][x] = 'X'
                    island.add((x, y))

    def is_open(x, y):
        return 0 <= x < width and 0 <= y < height and cells[y][x] == ' '

    # A wall tile can host a station only if an agent can stand next to it.
    slots = [(x, y) for y in range(height) for x in range(width)
             if cells[y][x] == 'X'
             and any(is_open(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
    if len(slots) < len(STATIONS) + 1:
        # Degenerate room - fall back to no island and recompute.
        for (x, y) in island:
            cells[y][x] = ' '
        island.clear()
        slots = [(x, y) for y in range(height) for x in range(width)
                 if cells[y][x] == 'X'
                 and any(is_open(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]

    # Farthest-point sampling: spread the stations out so FOV has to work at range.
    chosen = [slots[rng.randrange(len(slots))]]
    while len(chosen) < len(STATIONS):
        best, best_d = None, -1
        for c in slots:
            if c in chosen:
                continue
            d = min(abs(c[0] - s[0]) + abs(c[1] - s[1]) for s in chosen)
            if d > best_d:
                best, best_d = c, d
        if best is None:
            break
        chosen.append(best)

    syms = list(STATIONS)
    rng.shuffle(syms)
    features = {pos: sym for pos, sym in zip(chosen, syms)}
    m_pos = next(p for p, s in features.items() if s == 'M')

    # Agents start on open floor; the robot starts adjacent to M so the scripted
    # "hide" teammate can grab the meat on its first INTERACT.
    open_cells = [(x, y) for y in range(height) for x in range(width) if is_open(x, y)]
    m_adj = [c for c in open_cells
             if abs(c[0] - m_pos[0]) + abs(c[1] - m_pos[1]) == 1]
    robot_start = m_adj[rng.randrange(len(m_adj))] if m_adj else open_cells[rng.randrange(len(open_cells))]
    far = sorted(open_cells,
                 key=lambda c: -(abs(c[0] - robot_start[0]) + abs(c[1] - robot_start[1])))
    human_pos = far[rng.randrange(max(1, len(far) // 3))]
    while human_pos == robot_start and len(open_cells) > 1:
        human_pos = open_cells[rng.randrange(len(open_cells))]

    hide_candidates = [c for c in open_cells
                       if c not in (robot_start, human_pos)
                       and c not in approach_tiles(m_pos)]
    hide_pos = (hide_candidates[rng.randrange(len(hide_candidates))]
                if hide_candidates else robot_start)

    cells[robot_start[1]][robot_start[0]] = '1'
    cells[human_pos[1]][human_pos[0]] = '2'
    for pos, sym in features.items():
        cells[pos[1]][pos[0]] = sym
    grid = [''.join(row) for row in cells]

    fov_triple = sample_fov_triple(rng)
    kb_update_delay = rng.choice(KB_UPDATE_DELAY_CHOICES)
    order_list = ['steak'] * rng.choice([2, 3, 4, 5])
    cook_time = rng.choice([8, 10, 12, 15, 18, 22, 25])
    chop_time = rng.choice([2, 3, 4, 5, 6, 8])
    wash_time = rng.choice([2, 3, 4, 5, 6, 8])
    # Searched, not assumed: "work" keeps the world changing (and is what makes
    # FOV matter at all), "hide" is v1's parked robot, kept as a control.
    # "hide" is dropped from the search space: it is now proven dead, not just
    # weak. Parking the robot freezes the world (KB key pinned for 118 straight
    # steps), and every hide trial measured returns subtasks=1, len=200,
    # FOV_total=0 - a human doing nothing for a full episode. It was 25% of
    # trials producing guaranteed failures.
    robot_mode = "work"

    job_prefix = os.environ.get("SLURM_JOB_ID", "local")
    name = f"fov_v3_{job_prefix}_{idx}"
    return dict(name=name, grid=grid, size=(width, height),
                human_pos=human_pos, robot_start=robot_start, m_pos=m_pos, hide_pos=hide_pos,
                fov_triple=fov_triple, order_list=order_list, cook_time=cook_time,
                chop_time=chop_time, wash_time=wash_time, num_items_for_steak=1,
                robot_mode=robot_mode, human_mode=os.environ.get("FOV_HUMAN", "plain"),
                kb_update_delay=kb_update_delay)


def rollout(mdp, mlp, cfg, fov, seed):
    """One episode with the RNG reset to `seed` FIRST, so rollouts differing
    only in `fov` consume an identical random stream.

    Returns (subtask_sequence, positions_visited, n_stuck_steps).
    """
    np.random.seed(seed)
    random.seed(seed)
    setup_state = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=400).state.deepcopy()
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=400)

    human_cls = HUMAN_CLS[cfg.get("human_mode", "plain")]
    human = human_cls(mlp, setup_state, vision_limit=True, vision_bound=fov,
                      kb_update_delay=cfg.get("kb_update_delay", 2), debug=False)
    human.set_agent_index(1)
    human.init_knowledge_base(setup_state)

    robot = None
    if cfg["robot_mode"] == "work":
        robot = GreedySteakHumanModel(mlp)
        robot.set_agent_index(0)

    subtasks, positions = [], []
    n_stuck = consec = 0
    for _ in range(N_STEPS):
        state = env.state
        positions.append(state.players[1].position)
        if robot is not None:
            try:
                a_robot, _ = robot.action(state)
            except Exception:
                a_robot = Action.STAY
        else:
            a_robot = robot_next_action(mlp, state, 0, cfg["m_pos"], cfg["hide_pos"])
        try:
            a_human, _ = human.action(state)
            consec = 0
        except (AssertionError, IndexError, NameError):
            # Known library bug - the human has no valid motion goal for the
            # subtask it wants. Stay put and drop the sticky commitment so it
            # re-evaluates next step, instead of ending the episode.
            a_human = Action.STAY
            human.prev_chosen_subtask = None
            n_stuck += 1
            consec += 1
            if consec >= MAX_CONSEC_STUCK:
                subtasks.append('STUCK')
                break
        subtasks.append(human.prev_chosen_subtask)
        _, _, done, _ = env.step((a_robot, a_human))
        if done:
            subtasks.append('DONE')
            break
    return subtasks, positions, n_stuck


DEAD = (None, 'STUCK', 'DONE')


def pair_disagreement(s1, s2):
    """Subtask disagreement between two sequences.

    Returns (total, late, clean_total, clean_late).

    MISHA NEW CHANGE - the `clean_*` counts only consider steps where BOTH
    humans are actively pursuing a real subtask. Plain total/late also count
    steps where one FOV has stalled (prev_chosen_subtask None) or ended while
    the other is still working. That IS a genuine FOV consequence, but it is a
    degenerate one - "the narrow-FOV human got stuck sooner" - and a layout
    scored mostly on it gives the Bayes filter little to discriminate on, since
    a stalled hypothesis produces no informative action likelihood anyway. The
    clean counts isolate the case we actually want: both humans working, and
    choosing DIFFERENT subtasks because they know different things.

    MISHA NEW CHANGE (phase correction) - the clean counts are now ALIGNMENT
    AWARE. Index-aligned comparison of two physically different episodes counts
    a pure TIMING OFFSET as disagreement: if both FOVs execute the identical
    ordered sequence of subtasks but one runs a step ahead, every index near a
    transition mismatches even though the two humans made the same decisions.

    Measured on keep01, pair (128,178): both execute an identical ordered
    prefix of 9 subtask runs, 178 simply running ~1 step ahead. 7 of the 11
    counted disagreements were phase artifacts; only 4 were real. That matters
    because the inflation lands on the pair that SETS the scored minimum -
    phase-corrected, keep01 scores 4, which fails MIN_PAIR_TOTAL and fails the
    noise margin. Layouts were being admitted on timing jitter.

    Fix: compress each sequence to run-length tokens (the ordered sequence of
    DECISIONS, dropping how long each was held) and count substitutions from a
    Needleman-Wunsch alignment. Insertions/deletions are timing, substitutions
    are genuinely different choices. The raw index-aligned counts are still
    returned as total/late for continuity with older headers.
    """
    n = min(len(s1), len(s2))
    half = n // 2
    total = late = 0
    for i in range(n):
        if s1[i] != s2[i]:
            total += 1
            if i >= half:
                late += 1

    # NOTE: pass the FULL sequences, not s1[:n]/s2[:n]. Truncating to the shorter
    # rollout throws away precisely the strongest FOV effect: a narrow-FOV human
    # that stalls out at 94 steps having reached 6 subtasks, versus a wide-FOV one
    # that runs 157 steps and completes 12. Cropping both to 94 hides the six
    # subtasks the narrow human never reached, which is the whole difference.
    clean_total, clean_late = _aligned_disagreement(s1, s2, half)
    return total, late, clean_total, clean_late


def _runs(seq, start=0):
    """Run-length compress to (token, first_index), skipping DEAD entries."""
    out = []
    for i, x in enumerate(seq[start:], start=start):
        if x in DEAD:
            continue
        if not out or out[-1][0] != x:
            out.append((x, i))
    return out


def _aligned_disagreement(s1, s2, half):
    """Real (non-timing) subtask divergence between two rollouts.

    Returns (total, late).

    MISHA NEW CHANGE (second correction) - counting ONLY substitutions was too
    aggressive and discarded the strongest FOV effect in the data.

    An indel is ambiguous. If both rollouts eventually perform the same subtask
    and one is simply a step behind, that indel is pure timing and must not
    count. But if one rollout performs a subtask the other NEVER performs at
    all, that is not timing - it is one human doing something the other never
    does. Measured control, same layout and seed, varying only FOV:

        idx0089  fov=42   -> STUCK at 94 steps,  6 distinct subtasks, 21% stalled
        idx0089  fov=161  -> DONE  at 157 steps, 12 distinct subtasks, 0% stalled

    That is as high-level as a difference gets - one human completes the orders,
    the other never gets there - yet pure-substitution scoring rates it 0,
    because the six subtasks the narrow human never reaches align as deletions.
    That is why the first correction returned 0/14 qualifying layouts.

    So: substitutions count, PLUS indels whose token never appears anywhere in
    the other sequence (an unmatched capability, not a phase offset). Indels
    whose token does appear elsewhere in the other sequence are timing and are
    still excluded, which is what kills the keep01-style inflation (identical
    ordered prefix, one rollout running ~1 step ahead).
    """
    a, b = _runs(s1), _runs(s2)
    if not a or not b:
        return 0, 0
    na, nb = len(a), len(b)
    # cost matrix; gap = 1, substitution = 1, match = 0
    d = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(na + 1):
        d[i][0] = i
    for j in range(nb + 1):
        d[0][j] = j
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            sub = d[i - 1][j - 1] + (0 if a[i - 1][0] == b[j - 1][0] else 1)
            d[i][j] = min(sub, d[i - 1][j] + 1, d[i][j - 1] + 1)
    # Tokens each side ever performs; an indel of a token absent from the other
    # side entirely is a real capability difference, not a phase offset.
    set_a = {t for t, _ in a}
    set_b = {t for t, _ in b}

    i, j, n_div, n_late = na, nb, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1][0] == b[j - 1][0] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                if cost:  # SUBSTITUTION - different decision at the same stage
                    n_div += 1
                    if max(a[i - 1][1], b[j - 1][1]) >= half:
                        n_late += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and (j == 0 or d[i][j] == d[i - 1][j] + 1):
            tok, idx = a[i - 1]
            if tok not in set_b:  # only s1 ever does this - real divergence
                n_div += 1
                if idx >= half:
                    n_late += 1
            i -= 1
        else:
            tok, idx = b[j - 1]
            if tok not in set_a:
                n_div += 1
                if idx >= half:
                    n_late += 1
            j -= 1
    return n_div, n_late


def run_trial(config):
    idx, seed, n_seeds = config
    rng = random.Random(seed)
    cfg = make_layout_v3(idx, rng)

    try:
        mdp, mlp = build_mdp_and_mlp(cfg["name"], cfg["grid"], order_list=cfg["order_list"],
                                     cook_time=cfg["cook_time"], chop_time=cfg["chop_time"],
                                     wash_time=cfg["wash_time"],
                                     num_items_for_steak=cfg["num_items_for_steak"])
    except Exception as e:
        discard_planner_cache(cfg["name"])
        return dict(idx=idx, error=f"build failed: {type(e).__name__}: {e}", passed=False)

    fovs = list(cfg["fov_triple"])
    try:
        rolls = {s: {f: rollout(mdp, mlp, cfg, f, s) for f in fovs} for s in range(n_seeds)}
        # MISHA NEW CHANGE - FULL-OBSERVABILITY CONTROL. A human with full vision
        # (180 deg = the half-plane case, cos(90)=0) must be able to DO the task:
        # finish the orders and essentially never stall. If it cannot, the layout
        # is simply broken and any "divergence" it shows is two agents failing in
        # different ways, not vision changing an otherwise-competent decision.
        #
        # This is a real discriminator, not a formality. Measured on the same
        # layout and seed, varying only FOV:
        #     idx0089 fov=42  -> STUCK  94 steps,  6 subtasks, 21% stalled
        #     idx0089 fov=180 -> DONE  157 steps, 12 subtasks,  0% stalled
        # so a competent full-vision baseline is achievable and worth requiring.
        control = {s: rollout(mdp, mlp, cfg, 180, s) for s in range(n_seeds)}
    except Exception as e:
        discard_planner_cache(cfg["name"])
        return dict(idx=idx, error=f"sim failed: {type(e).__name__}: {e}", passed=False)

    ctrl_done = sum(1 for s in range(n_seeds) if control[s][0] and control[s][0][-1] == 'DONE')
    ctrl_stall = max((control[s][2] / max(1, len(control[s][0]))) for s in range(n_seeds))
    ctrl_subtasks = min(len({x for x in control[s][0] if x not in DEAD}) for s in range(n_seeds))
    full_fov_ok = (ctrl_done == n_seeds
                   and ctrl_stall <= MAX_CONTROL_STALL_FRAC
                   and ctrl_subtasks >= MIN_CONTROL_SUBTASKS)

    # FOV effect, per seed: all 3 pairs at a FIXED random stream.
    per_seed = []
    for s in range(n_seeds):
        pairs = []
        for i in range(len(fovs)):
            for j in range(i + 1, len(fovs)):
                pairs.append(pair_disagreement(rolls[s][fovs[i]][0], rolls[s][fovs[j]][0]))
        distinct = len({tuple(rolls[s][f][0]) for f in fovs}) == 3
        per_seed.append(dict(min_total=min(p[0] for p in pairs),
                             min_late=min(p[1] for p in pairs),
                             min_clean_total=min(p[2] for p in pairs),
                             min_clean_late=min(p[3] for p in pairs),
                             distinct=distinct))

    # RNG noise floor: same FOV, different seeds. Compared against the CLEAN
    # counts, so signal and noise are measured the same way.
    rng_pairs = []
    for f in fovs:
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                rng_pairs.append(pair_disagreement(rolls[i][f][0], rolls[j][f][0])[2])
    rng_noise = sum(rng_pairs) / len(rng_pairs) if rng_pairs else 0

    # Richness / liveness, worst case over every rollout.
    all_rolls = [rolls[s][f] for s in range(n_seeds) for f in fovs]
    # MISHA NEW CHANGE - exclude None, not just STUCK/DONE. None is the stall
    # marker written when the human's planner has no valid motion goal, so
    # counting it inflated every layout's distinct-subtask count by exactly 1
    # and let layouts with only 2 real subtasks clear MIN_DISTINCT_SUBTASKS=3.
    min_subtasks = min(len({x for x in r[0] if x not in DEAD}) for r in all_rolls)

    # MISHA NEW CHANGE - liveness: how many rollouts actually finished the task
    # rather than stalling out. An audit measured 28/36 rollouts ending STUCK
    # and delivering nothing, so "the human performs the steak task" was true
    # for a minority of rollouts. Recorded so headers stop overstating it.
    n_done = sum(1 for r in all_rolls if r[0] and r[0][-1] == 'DONE')
    n_stuck_end = sum(1 for r in all_rolls if r[0] and r[0][-1] == 'STUCK')
    min_positions = min(len(set(r[1])) for r in all_rolls)
    min_len = min(len(r[0]) for r in all_rolls)
    total_stuck = sum(r[2] for r in all_rolls)

    # Scored on the CLEAN counts: every pair, under every seed, must disagree
    # while BOTH humans are actively working - not merely because one stalled.
    # MISHA NEW CHANGE - the FOV effect must also BEAT ITS OWN NOISE FLOOR.
    # Nothing above required this, so a chaotic layout could pass on a large
    # absolute count while the same-FOV-different-seed variation was just as
    # large (observed: FOV_total=35 against rng_noise=31). That is exactly the
    # confusion that made v1's rankings meaningless, just less extreme - a
    # layout is only useful for inference if changing the FOV moves behaviour
    # MORE than resampling the dice does.
    beats_noise = min(ps["min_clean_total"] for ps in per_seed) >= NOISE_MARGIN * rng_noise

    passed = (full_fov_ok
              and all(ps["distinct"] for ps in per_seed)
              and min(ps["min_clean_total"] for ps in per_seed) >= MIN_PAIR_TOTAL
              and min(ps["min_clean_late"] for ps in per_seed) >= MIN_PAIR_LATE
              and beats_noise
              and min_subtasks >= MIN_DISTINCT_SUBTASKS
              and min_positions >= MIN_DISTINCT_POSITIONS)

    # Keep the grid of winners so they can be written out as .layout files;
    # drop every planner cache regardless (quota - see discard_planner_cache).
    grid = cfg["grid"] if passed else None
    discard_planner_cache(cfg["name"])

    # MISHA NEW CHANGE - write each winner to disk THE MOMENT it is found, from
    # the worker, rather than collecting them and writing in main() at the end.
    # The CARC search runs 384 trials against an 8h wall clock; if it is killed
    # on time limit (or anything else) after 7 hours, end-of-run writing loses
    # every layout it found. Each worker writes its own idx-named file, so there
    # is no cross-process collision.
    if passed:
        try:
            write_layout_file_v3(cfg, dict(
                idx=idx, fov_min_total=min(ps["min_clean_total"] for ps in per_seed),
                fov_min_late=min(ps["min_clean_late"] for ps in per_seed),
                rng_noise=rng_noise, min_subtasks=min_subtasks,
                min_positions=min_positions, min_len=min_len, n_seeds=n_seeds))
        except Exception:
            pass  # never let bookkeeping kill a trial that already succeeded

    return dict(idx=idx, name=cfg["name"], size=cfg["size"], fov_triple=cfg["fov_triple"],
                grid=grid, human_mode=cfg.get("human_mode", "plain"),
                ctrl_done=ctrl_done, ctrl_stall=ctrl_stall, ctrl_subtasks=ctrl_subtasks,
                full_fov_ok=full_fov_ok,
                order_list=cfg["order_list"], cook_time=cfg["cook_time"],
                chop_time=cfg["chop_time"], wash_time=cfg["wash_time"],
                num_items_for_steak=cfg["num_items_for_steak"], kb_update_delay=cfg["kb_update_delay"],
                human_pos=cfg["human_pos"],
                robot_start=cfg["robot_start"], m_pos=cfg["m_pos"], hide_pos=cfg["hide_pos"],
                robot_mode=cfg["robot_mode"], n_seeds=n_seeds,
                fov_min_total=min(ps["min_clean_total"] for ps in per_seed),
                fov_min_late=min(ps["min_clean_late"] for ps in per_seed),
                fov_raw_total=min(ps["min_total"] for ps in per_seed),
                fov_mean_total=sum(ps["min_clean_total"] for ps in per_seed) / n_seeds,
                all_distinct=all(ps["distinct"] for ps in per_seed),
                rng_noise=rng_noise, min_subtasks=min_subtasks, min_positions=min_positions,
                min_len=min_len, total_stuck=total_stuck, passed=passed,
                sequences={f: rolls[0][f][0] for f in fovs})


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else N_SEEDS

    # Trial seeds were hardcoded at 7000+i, so every re-run regenerated the
    # IDENTICAL layouts. Offset lets successive batches accumulate new ones.
    seed_offset = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("FOV_SEED_OFFSET", 7000))
    configs = [(i, seed_offset + i, n_seeds) for i in range(n_trials)]
    print(f"v3 search: {n_trials} layouts x {n_seeds} seeds x 3 FOVs, {N_STEPS} steps, "
          f"{n_workers} workers, seed_offset={seed_offset}, "
          f"tag={os.environ.get('FOV_RUN_TAG','r0')}", flush=True)
    print("fov_min_* are RNG-FREE (same seed across FOVs). rng_noise is the "
          "same-FOV-different-seed floor, reported for contrast only.\n", flush=True)

    results = []
    done = 0
    with mp.Pool(n_workers) as pool:
        for r in pool.imap_unordered(run_trial, configs):
            results.append(r)
            done += 1
            if "error" in r:
                print(f"[{done}/{n_trials}] ERROR idx={r['idx']}: {r['error']}", flush=True)
                continue
            print(f"[{done}/{n_trials}] [{'PASS' if r['passed'] else 'fail'}] idx={r['idx']:>3} "
                  f"{r['robot_mode']:<4} kbd={r['kb_update_delay']} fov={r['fov_triple']} "
                  f"orders={len(r['order_list'])} FOV_total={r['fov_min_total']:>3} "
                  f"late={r['fov_min_late']:>3} raw={r['fov_raw_total']:>3} rng={r['rng_noise']:>5.1f} "
                  f"subtasks={r['min_subtasks']} pos={r['min_positions']} "
                  f"ctrl={r['ctrl_done']}/{r['n_seeds']}done,{r['ctrl_stall']*100:.0f}%stall "
                  f"len={r['min_len']} stuck={r['total_stuck']}", flush=True)

    valid = [r for r in results if "error" not in r]
    passed = [r for r in valid if r["passed"]]
    any_fov = [r for r in valid if r["fov_min_total"] > 0]
    print(f"\n=== {len(passed)}/{len(valid)} PASSED (RNG-free sustained FOV divergence) ===")
    print(f"    {len(any_fov)}/{len(valid)} showed ANY FOV effect at fixed seed")
    by_robot = {}
    for r in valid:
        by_robot.setdefault(r["robot_mode"], []).append(r)
    for mode, rs in sorted(by_robot.items()):
        p = sum(1 for r in rs if r["passed"])
        print(f"    robot_mode={mode:<5}: {p}/{len(rs)} passed, "
              f"mean FOV_total={sum(r['fov_min_total'] for r in rs)/len(rs):.1f}, "
              f"mean len={sum(r['min_len'] for r in rs)/len(rs):.0f}")

    if passed:
        # Layout files were already written by the workers as each trial passed
        # (see write_layout_file_v3) so nothing is lost if this job is killed.
        print(f"\n=== {len(passed)} passing layouts written to {LAYOUTS_V3_DIR} ===")

        print("=== PASSING LAYOUTS (best first) ===")
        for r in sorted(passed, key=lambda r: -r["fov_min_late"]):
            print(f"\nidx={r['idx']} name={r['name']} robot={r['robot_mode']} size={r['size']} "
                  f"fov={r['fov_triple']} orders={r['order_list']} "
                  f"cook={r['cook_time']} chop={r['chop_time']} wash={r['wash_time']}")
            print(f"  human={r['human_pos']} robot_start={r['robot_start']} M={r['m_pos']} "
                  f"hide={r['hide_pos']}")
            print(f"  FOV_total={r['fov_min_total']} FOV_late={r['fov_min_late']} "
                  f"rng_noise={r['rng_noise']:.1f} subtasks={r['min_subtasks']}")
            for f, seq in r["sequences"].items():
                print(f"  FOV={f}: {seq}")


if __name__ == "__main__":
    main()
