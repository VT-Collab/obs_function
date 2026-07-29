"""
MISHA NEW CHANGE - exploratory/diagnostic only, not part of any pipeline.

Batch-tests multiple compact (~15x9, fast to build locally) layouts for
GENUINE sustained subtask-level FOV divergence: uses StickySubtaskHumanModel
(sticky_subtask_human.py) so a stale belief drives a full redundant round
trip instead of self-correcting in 1 step, and explicitly verifies the
robot's hiding spot does NOT coincide with any of the meat dispenser's
approach tiles (the accidental-blocking case found in fov_meat_dx3_dy2,
where the human got stuck retrying forever instead of genuinely completing
a round trip and only THEN reconverging).

FOV set: (30, 90, 180) instead of (60, 120, 180) - verified via in_bound()
that FOV > 180 is NOT meaningfully wider (the formula is symmetric around
180: FOV=270 behaves identically to FOV=90), so 30/90/180 is the widest
*meaningful* spread achievable, and gives a much bigger 30-vs-180 contrast
than 60-vs-180 did.

Run with: python -m my_methods.bayesian.fov_sustained_batch
"""
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import Action
from my_methods.bayesian.fov_subtask_divergence_test import build_grid, build_mdp_and_mlp, render_layout_image
from my_methods.bayesian.sticky_subtask_human import StickySubtaskHumanModel

FOV_CANDIDATES = (30, 90, 180)
N_FORWARD_STEPS = 40


def approach_tiles(pos):
    x, y = pos
    return {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)}


def robot_next_action(mlp, state, robot_idx, m_pos, hide_pos):
    """MISHA NEW CHANGE: both agents act every step now (no more freezing one
    while the other moves - that's what caused the earlier permanent-block
    bug, since a frozen human is a static obstacle instead of a teammate
    who's also moving through their own task). The robot's move toward
    hide_pos is RE-PLANNED from its actual current position every single
    step, so if the human happens to be in its way at some instant, it just
    self-corrects on the next step instead of executing a stale plan blind."""
    player = state.players[robot_idx]
    if player.held_object is None:
        return Action.INTERACT  # assumes robot starts adjacent+facing m_pos
    if player.position == hide_pos:
        return Action.STAY
    goal = (hide_pos, (0, 1))
    if not mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal):
        return Action.STAY
    action_plan, _, _ = mlp.mp.get_plan(player.pos_and_or, goal)
    return action_plan[0] if action_plan else Action.STAY


def run_layout(name, grid, human_pos, robot_start, m_pos, hide_pos):
    assert hide_pos not in approach_tiles(m_pos), \
        f"{name}: hide_pos {hide_pos} blocks the meat dispenser's approach tile - would cause permanent oscillation, not genuine divergence"
    assert hide_pos != m_pos

    mdp, mlp = build_mdp_and_mlp(name, grid)
    robot_idx, human_idx = 0, 1

    setup_env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=150)
    setup_state = setup_env.state.deepcopy()
    human_ori = setup_state.players[human_idx].orientation
    print(f"\n=== {name} ===")
    print(f"human@{human_pos}/{human_ori}  M@{m_pos}  robot hide_pos target={hide_pos}")

    progressions = {}
    positions = {}
    robot_positions = {}
    for fov in FOV_CANDIDATES:
        sim_env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=150)
        shadow = StickySubtaskHumanModel(mlp, setup_state, vision_limit=True, vision_bound=fov, debug=False)
        shadow.set_agent_index(human_idx)
        shadow.init_knowledge_base(setup_state)

        subtasks, pos_trace, robot_trace = [], [], []
        for t in range(N_FORWARD_STEPS):
            state = sim_env.state
            pos_trace.append(state.players[human_idx].position)
            robot_trace.append(state.players[robot_idx].position)
            a_robot = robot_next_action(mlp, state, robot_idx, m_pos, hide_pos)
            try:
                a_human, _ = shadow.action(state)
            except AssertionError:
                subtasks.append('STUCK')
                break
            subtasks.append(shadow.prev_chosen_subtask)
            joint = [Action.STAY, Action.STAY]
            joint[robot_idx], joint[human_idx] = a_robot, a_human
            sim_env.step(tuple(joint))
        progressions[fov] = subtasks
        positions[fov] = pos_trace
        robot_positions[fov] = robot_trace
        print(f"  FOV={fov:>3}: {subtasks}")
        print(f"    human unique positions: {len(set(pos_trace))}/{len(pos_trace)}  "
              f"robot final pos: {robot_trace[-1] if robot_trace else None} (target {hide_pos})")

    # use the FOV=180 run's final robot position for the schematic image (any FOV works, robot script is FOV-independent)
    img_path = render_layout_image(name, grid, human_pos, human_ori, robot_positions[180][-1])
    print(f"  image -> {img_path}")

    ref = progressions[180]
    disagreement = {fov: sum(1 for a, b in zip(seq, ref) if a != b) for fov, seq in progressions.items()}
    full_diverges = len(set(tuple(s) for s in progressions.values())) > 1
    any_stuck = any(len(set(positions[fov])) <= 3 for fov in FOV_CANDIDATES)
    reached_hide = all(robot_positions[fov][-1] == hide_pos for fov in FOV_CANDIDATES)
    print(f"  --> full-sequence divergence: {full_diverges}  |  disagreement vs FOV=180: {disagreement}  |  "
          f"any human stuck (<=3 unique tiles): {any_stuck}  |  robot reached hide_pos in all runs: {reached_hide}")
    return full_diverges, disagreement, any_stuck


def main():
    layouts = []

    # MISHA NEW CHANGE (v2 - fixed): the first version placed robot_start/M on
    # the SAME row as the human (row 3), so get_plan()'s straight-line path to
    # hide_pos walked directly through the human's own tile and got physically
    # stuck there (two players can't share a tile, and get_plan() has no idea
    # where the other player currently stands - it's a static single-agent
    # plan). Fix: keep every robot-side position on rows >=4, strictly below
    # the human's row 3, so the path never needs to cross the human's tile.
    # Renamed (_v2) so these rebuild fresh instead of loading the old, wrong
    # cached planner under the same layout name.

    # D1: hide_pos offset (-2,1) from human(7,3) - verified (30,90,180) = (False,False,True)
    human_pos, robot_start, m_pos, hide_pos = (7, 3), (9, 5), (9, 6), (5, 4)
    features = {robot_start: '1', human_pos: '2', m_pos: 'M',
                (12, 1): 'O', (2, 1): 'D', (10, 1): 'P', (7, 6): 'B', (1, 7): 'W', (13, 7): 'S'}
    layouts.append(("fov3090180_D1_v2", build_grid(15, 9, features), human_pos, robot_start, m_pos, hide_pos))

    # D2: mirrored (+2,1)
    human_pos, robot_start, m_pos, hide_pos = (7, 3), (5, 5), (5, 6), (9, 4)
    features = {robot_start: '1', human_pos: '2', m_pos: 'M',
                (2, 1): 'O', (12, 1): 'D', (4, 1): 'P', (7, 6): 'B', (13, 7): 'W', (1, 7): 'S'}
    layouts.append(("fov3090180_D2_v2", build_grid(15, 9, features), human_pos, robot_start, m_pos, hide_pos))

    # D3: different disagreement pattern, offset (-4,3) - verified (30,90,180) = (False,True,True)
    human_pos, robot_start, m_pos, hide_pos = (7, 3), (11, 5), (11, 6), (3, 6)
    features = {robot_start: '1', human_pos: '2', m_pos: 'M',
                (2, 1): 'O', (12, 1): 'D', (7, 1): 'P', (12, 4): 'B', (1, 7): 'W', (13, 7): 'S'}
    layouts.append(("fov3090180_D3_v2", build_grid(15, 9, features), human_pos, robot_start, m_pos, hide_pos))

    # D4: mirrored (+4,3)
    human_pos, robot_start, m_pos, hide_pos = (7, 3), (3, 5), (3, 6), (11, 6)
    features = {robot_start: '1', human_pos: '2', m_pos: 'M',
                (12, 1): 'O', (2, 1): 'D', (7, 1): 'P', (2, 4): 'B', (13, 7): 'W', (1, 7): 'S'}
    layouts.append(("fov3090180_D4_v2", build_grid(15, 9, features), human_pos, robot_start, m_pos, hide_pos))

    # D5: MISHA NEW CHANGE - genuinely distinct 3rd layout (not just a mirror
    # of D1/D2), same successful small-offset recipe (dx=3,dy=2, verified
    # (30,90,180)=(False,False,True)). D3/D4's larger offset (dx=4,dy=3) put
    # the robot far enough from the human's OWN redundant-pickup path that
    # even FOV=180 never saw it - once both agents move simultaneously, what
    # matters isn't just the verified static offset from the human's START,
    # it's whether the human's actual wandering path (which depends on where
    # its mistaken belief sends it) ever comes close to where the robot is.
    human_pos, robot_start, m_pos, hide_pos = (7, 3), (3, 5), (3, 6), (10, 5)
    features = {robot_start: '1', human_pos: '2', m_pos: 'M',
                (12, 1): 'O', (2, 1): 'D', (7, 1): 'P', (12, 4): 'B', (1, 7): 'W', (13, 7): 'S'}
    layouts.append(("fov3090180_D5_v2", build_grid(15, 9, features), human_pos, robot_start, m_pos, hide_pos))

    # D6: MISHA NEW CHANGE - D3/D4/D5 all failed (zero divergence, even at
    # FOV=180) despite using a verified-disagreeing offset for hide_pos alone.
    # Checking in_bound() for robot_start/M too (not just hide_pos) revealed
    # why D1/D2 actually worked: their robot_start/M offsets (+-2,+2 / +-2,+3)
    # are visible to ALL 3 FOVs, so every shadow correctly sees the pickup
    # happen at t=0 - the later divergence comes from what happens to that
    # belief as the robot then walks to a hide_pos invisible at 30/90 but not
    # 180 (apparently the KB doesn't just freeze a stale belief forever, it
    # can un-learn/decay as the tracked object goes out of view again - a
    # subtlety this diagnostic didn't fully reverse-engineer). D3/D4/D5's
    # robot_start/M were ALSO already outside the 30/90 cone from the start,
    # so no shadow ever had a confirmed sighting to lose track of. Fix: reuse
    # D1's exact working relative geometry (robot_start visible-to-all,
    # hide_pos invisible-to-30/90) transplanted onto a different absolute
    # layout, instead of a brand new untested offset combination.
    human_pos = (8, 4)
    robot_start = (human_pos[0] + 2, human_pos[1] + 2)   # (10,6) - verified visible to all 3 FOVs
    m_pos = (human_pos[0] + 2, human_pos[1] + 3)          # (10,7) - south of robot_start, robot faces it by default
    hide_pos = (human_pos[0] - 2, human_pos[1] + 1)       # (6,5)  - verified invisible at 30/90, visible at 180
    features = {robot_start: '1', human_pos: '2', m_pos: 'M',
                (12, 1): 'O', (2, 1): 'D', (7, 1): 'P', (2, 4): 'B', (13, 4): 'W', (1, 7): 'S'}
    layouts.append(("fov3090180_D6_v2", build_grid(15, 9, features), human_pos, robot_start, m_pos, hide_pos))

    results = []
    for name, grid, human_pos, robot_start, m_pos, hide_pos in layouts:
        try:
            full_diverges, disagreement, any_stuck = run_layout(name, grid, human_pos, robot_start, m_pos, hide_pos)
            results.append((name, full_diverges, disagreement, any_stuck))
        except Exception as e:
            import traceback
            print(f"\n=== {name} ===\n  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append((name, False, None, None))

    print("\n=== SUMMARY ===")
    clean_passes = 0
    for name, full_diverges, disagreement, any_stuck in results:
        ok = full_diverges and not any_stuck
        if ok:
            clean_passes += 1
        print(f"{name}: full_diverges={full_diverges} any_stuck={any_stuck} disagreement={disagreement} "
              f"{'PASS (genuine sustained divergence)' if ok else 'FAIL'}")
    print(f"\n{clean_passes}/{len(results)} layouts show genuine (non-stuck) sustained subtask divergence")


if __name__ == "__main__":
    main()
