# v1: infer a human driver's gaze direction (head_angle) online from a real
# behavioral signal, not injected noise.
#
# Ground truth gaze is still *synthesized* (heading + slow random wander) --
# no dataset has real gaze. What's real now: two vehicles' actual Waymo
# trajectories (human + a nearby hazard), and the "observation" fed to the
# filter is a reaction signal generated causally through real geometry: did
# the hazard fall inside the human's (synthetic) gaze cone or not. The
# filter's likelihood is the same geometric test run under each candidate
# head_angle, so it's an actual inverse-observation model, not curve-fitting
# to a fabricated number.
#
# view_dist / view_angle are fixed constants, not inferred: with only a
# reaction/no-reaction signal, trying to recover all three FOV parameters at
# once is underdetermined (narrow cone, short range, and wrong head_angle
# all look the same from outside). head_angle is the one that's fast-moving
# and worth tracking online.
import numpy as np
import matplotlib.pyplot as plt

from cfgs.config import PROJECT_PATH, get_default_scenario_dict
from nocturne import Simulation

OUT_DIR = PROJECT_PATH / 'misha'

VIEW_DIST = 40.0                    # meters, fixed
VIEW_ANGLE = 100 * np.pi / 180      # cone width, fixed
PROXIMITY_THRESH = 25.0             # hazard only counts as a "threat" this close
FLIP_NOISE = 0.12                   # P(reaction signal disagrees with true visibility)
N_BINS = 36                         # head_angle grid resolution (10 deg/bin)
MOTION_NOISE_STD = 0.15             # rad/step, how fast gaze can drift in the filter's model


def wrap(angle):
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def record_trajectories(n_steps=80, dt=0.1):
    """Replay real Waymo trajectories for every moving object in the scene."""
    sim = Simulation(
        scenario_path=str(PROJECT_PATH / 'examples' / 'example_scenario.json'),
        config=get_default_scenario_dict(),
    )
    scenario = sim.getScenario()
    moving = scenario.getObjectsThatMoved()
    for obj in moving:
        obj.expert_control = True

    positions = [[] for _ in moving]
    headings = [[] for _ in moving]
    for _ in range(n_steps):
        for i, obj in enumerate(moving):
            positions[i].append(obj.position.numpy().copy())
            headings[i].append(obj.heading)
        sim.step(dt)

    return [np.array(p) for p in positions], [np.array(h) for h in headings]


def pick_human_and_hazard(positions):
    """Human = vehicle 0. Hazard = whichever other vehicle passes closest to it."""
    human_id = 0
    dists = {
        j: np.linalg.norm(positions[j] - positions[human_id], axis=1).min()
        for j in range(len(positions)) if j != human_id
    }
    hazard_id = min(dists, key=dists.get)
    return human_id, hazard_id, dists[hazard_id]


def bearing_and_dist(human_pos, hazard_pos):
    rel = hazard_pos - human_pos
    return np.arctan2(rel[1], rel[0]), np.linalg.norm(rel)


def run(n_steps=80, dt=0.1, seed=0):
    positions, headings = record_trajectories(n_steps, dt)
    human_id, hazard_id, min_dist = pick_human_and_hazard(positions)
    print(f'human=veh{human_id}, hazard=veh{hazard_id}, '
          f'closest approach={min_dist:.1f}m')

    rng = np.random.default_rng(seed)
    ou_theta, ou_sigma = 0.3, 0.15
    gaze_offset = 0.0

    grid = np.linspace(-np.pi, np.pi, N_BINS, endpoint=False)
    belief = np.ones(N_BINS) / N_BINS
    shift_kernel = np.exp(-0.5 * (wrap(grid) / MOTION_NOISE_STD)**2)
    shift_kernel = np.fft.ifftshift(shift_kernel)  # peak at index 0, else predict step rotates the belief
    shift_kernel /= shift_kernel.sum()
    kernel_fft = np.fft.fft(shift_kernel)

    log = {k: [] for k in
           ['true_gaze', 'est_gaze', 'true_visible', 'reaction', 'belief_visible_prob']}

    for t in range(n_steps):
        bearing, dist = bearing_and_dist(positions[human_id][t], positions[hazard_id][t])
        is_threat = dist <= min(VIEW_DIST, PROXIMITY_THRESH)

        gaze_offset += -ou_theta * gaze_offset * dt + ou_sigma * rng.normal() * np.sqrt(dt)
        gaze = wrap(headings[human_id][t] + gaze_offset)

        true_visible = is_threat and abs(wrap(bearing - gaze)) <= VIEW_ANGLE / 2
        reaction = true_visible if rng.random() > FLIP_NOISE else (not true_visible)

        # --- predict: blur belief by how much gaze could plausibly drift ---
        belief = np.real(np.fft.ifft(np.fft.fft(belief) * kernel_fft))
        belief = np.clip(belief, 0, None)
        belief /= belief.sum()

        # robot's *prior* (pre-evidence) belief about whether human sees the hazard --
        # this is the causal, non-cheating version of "would assistance have triggered"
        if is_threat:
            prior_visible_mask = np.abs(wrap(bearing - grid)) <= VIEW_ANGLE / 2
        else:
            prior_visible_mask = np.zeros(N_BINS, dtype=bool)
        belief_visible_prob = belief[prior_visible_mask].sum()

        # --- update: reweight by whether each hypothesis predicts the observed reaction ---
        likelihood = np.where(prior_visible_mask == reaction, 1 - FLIP_NOISE, FLIP_NOISE)
        belief *= likelihood
        belief /= belief.sum()

        est = np.arctan2((belief * np.sin(grid)).sum(), (belief * np.cos(grid)).sum())

        log['true_gaze'].append(gaze)
        log['est_gaze'].append(est)
        log['true_visible'].append(true_visible)
        log['reaction'].append(reaction)
        log['belief_visible_prob'].append(belief_visible_prob)

    return {k: np.array(v) for k, v in log.items()}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    log = run()

    rmse = np.sqrt(np.mean(wrap(log['est_gaze'] - log['true_gaze'])**2))
    # would the robot's (causal, pre-evidence) belief have correctly called
    # "human can see the hazard" this frame?
    belief_call = log['belief_visible_prob'] > 0.5
    visibility_acc = (belief_call == log['true_visible']).mean()
    print(f'gaze RMSE: {np.degrees(rmse):.1f} deg')
    print(f'threat frames: {log["true_visible"].sum()}/{len(log["true_visible"])}')
    print(f'causal visibility-belief accuracy: {visibility_acc:.0%}')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(np.degrees(log['true_gaze']), label='true gaze (synthetic ground truth)')
    ax1.plot(np.degrees(log['est_gaze']), label='Bayes filter estimate')
    ax1.set_ylabel('head_angle (deg)')
    ax1.legend()

    t = np.arange(len(log['reaction']))
    ax2.fill_between(t, 0, log['true_visible'].astype(float), step='mid', alpha=0.3,
                      label='hazard actually in cone (ground truth)')
    ax2.plot(t, log['belief_visible_prob'], label="robot's causal belief P(human sees hazard)")
    ax2.scatter(t[log['reaction']], np.full(log['reaction'].sum(), 1.05), marker='|',
                color='red', label='observed reaction signal')
    ax2.set_ylim(-0.1, 1.15)
    ax2.set_xlabel('timestep')
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    out_path = OUT_DIR / 'bayes_filter_v1.png'
    plt.savefig(out_path)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
