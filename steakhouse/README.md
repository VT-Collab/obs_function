# Bayesian Field of View Inference for Human-AI Collaboration in Steakhouse

This project extends the Steakhouse-AI multi-human cooking environment with a real-time Bayesian inference system that estimates a human collaborator's field of view (FOV) from their observed behavior, enabling the AI partner to adapt its planning accordingly.

## Overview

This repository contains two main components:

| **Component** | **Purpose** |
|---|---|
| `overcooked_ai_py/` | Steakhouse game engine — MDP, environment, agents, and planner |
| `my_methods/` | Bayesian FOV inference and learned baseline models |

## Introduction

In human-AI collaborative cooking tasks, a human player may operate with a limited field of view — they can only see objects and teammates within a cone of vision. If the AI partner can infer how wide or narrow that FOV is, it can better anticipate what the human knows and plan complementary actions.

This project implements a **real-time Bayesian posterior** over three FOV hypotheses — **90°, 120°, and 179°** — updated every game timestep from the human's current action and game state observation.

### Key Features

- **Bayesian FOV Inference**: At each timestep, the system updates a probability distribution over FOV hypotheses using a KNN-weighted likelihood against a database of human trajectories.
- **Adaptive AI Partner**: The estimated FOV is fed back to the AI agent each step so it can model what the human can and cannot see.
- **Multiple Baselines**: MLP and LSTM baselines for comparison against the Bayesian approach.
- **Steakhouse Game Environment**: A multi-player cooking game with grills, chopping boards, sinks, and pots.

---

## Installation

Set up a conda environment with Python 3.8:

```bash
conda create -n steakhouse_ai python=3.8
conda activate steakhouse_ai
```

Clone the repo and install:

```bash
git clone <repo-url>
cd steakhouse
pip install -e .
pip install moviepy
```

---

## Running the Game

### Play a Layout

```bash
python scripts/multi_player_user_study.py --participant_id 0 --layout "Overcooked2_2-4" --record_video
python scripts/multi_player_user_study.py --participant_id 1 --layout "Overcooked2_1-2" --record_video
python scripts/multi_player_user_study.py --participant_id 2 --layout "Overcooked1_1-4" --record_video
```

Add `--num_players 3` for three players. Adjust game length with `--total_time 300`. Logs are saved to `src/user_study/log/<participant_id>/`.

### Run Bayesian FOV Inference

```bash
cd my_methods/bayesian/
python infer_bayesian.py --layout Overcooked2_2-4 --fov 120
```

This runs multiple permuted episodes on the specified layout, updating the FOV posterior each timestep and logging estimated vs. ground-truth FOV to pickle files.

---

## Game Overview

### Objective

Fulfill and deliver displayed orders sequentially. Points are awarded as follows:

- **10 pts** — correct dish, correct position in order list
- **5 pts** — correct dish, wrong position in order list
- **0 pts** — dish not in order list

### Recipes

| Dish | Ingredients |
|---|---|
| Steak dish | meat + clean plate |
| Steak with garnish | meat + chopped onions + clean plate |
| Chicken dish | boiled chicken + clean plate |
| Chicken with garnish | boiled chicken + chopped onions + clean plate |

Steak takes **8 seconds** to cook; chicken takes **10 seconds**. Cooking times can be adjusted in the `.layout` files under `data/layout/`.

### Equipment

- **Grill**: Cook meat or chicken. States: empty / cooking / ready.
- **Chopping board**: Chop onions (2 interactions). States: empty / full / ready.
- **Sink**: Clean dirty plates (3 interactions). States: empty / full / ready.
- **Pot**: Boil chicken. States: empty / cooking / ready.

### Keyboard Controls

| Chef | Move | Interact |
|---|---|---|
| Blue-hat | Arrow keys | Space |
| Green-hat | WASD | F |
| Red-hat | IJKL | Semicolon |

---

## Bayesian FOV Inference

### Method

At each timestep, the system maintains a posterior distribution `prior` over three FOV hypotheses (90°, 120°, 179°). Given the current game observation and the human's action, the update is:

```
posterior = 0.8 * prior + 0.2 * normalize(likelihood * prior / evidence)
```

The **likelihood** is computed via KNN: the current 65-dim observation vector is compared (by inverse Euclidean distance) against all recorded training observations, and similarity scores are summed per FOV class. This soft update dampens noise while still shifting the posterior toward the correct FOV class over time.

The **estimated FOV** (`argmax(posterior)`) is set on the AI agent every step via `agent1.set_estimated_vision_bound(estimated_fov)`.

### Observation Vector (65-dim)

```
[p0_x, p0_y, p0_ori(2), p0_held(10), p1_x, p1_y, p1_ori(2), p1_held(10),
 p0_action, p1_action, chop_boards(10), grills(10), sink(5), pots(10)]
```

Held object and equipment states are one-hot encoded. See `my_methods/bayesian/dataset.py` for details.

### Training Data

Pickle files in `data/fov_traj/baseline/` named `log_<fov>_<player>_<layout>_dedup.pkl`. Training excludes data from the layout currently being played (cross-layout generalization).

---

## Code Structure

### Steakhouse Game Engine (`overcooked_ai_py/`)

```
overcooked_ai_py/
├── mdp/
│   ├── steakhouse_mdp.py       # Main steakhouse game logic (extends Overcooked MDP)
│   ├── steakhouse_env.py       # Environment classes
│   ├── overcooked_mdp.py       # Base Overcooked game logic
│   ├── overcooked_env.py       # Base environment classes
│   ├── actions.py              # Agent actions
│   └── graphics.py             # Rendering
├── agents/
│   ├── steak_agent.py          # SteakLimitVisionHumanModel, SteakGreedyHumanModel
│   └── agent.py                # Base agent classes
├── planning/
│   ├── steak_planner.py        # Steakhouse medium-level action manager
│   ├── planners.py             # Near-optimal agent planning logic
│   └── search.py               # A* search and shortest path
├── steakhouse_ai_rl/           # RL training scripts (PPO, LSTM, latent learning)
└── visualization/              # Pygame and state visualizer
```

### Inference Methods (`my_methods/`)

```
my_methods/
├── bayesian/
│   ├── infer_bayesian.py       # Main game loop with real-time Bayesian FOV update
│   ├── bayes_inference.py      # Pure functions: process_bayes_knn, bayes_inference
│   └── dataset.py              # obs_to_list, obs_list_to_1D_vec (65-dim encoder)
├── baseline_action_pred/       # MLP: predict action from obs (combo)
├── baseline_heuristic_obs/     # MLP: predict from heuristic observation features
├── baseline_implicit_obs/      # LSTM: predict from implicit observation sequence
├── lstm_fov/                   # LSTM trained directly on FOV classification
└── vae/                        # VAE-based observation reconstruction approach
```

### Scripts (`scripts/`)

```
scripts/
├── collect_human_trajectory.py    # Record human play trajectories
├── collect_soa_data.py            # State-of-art data collection
├── bayes_inference.py             # Standalone Bayesian inference (exact-match version)
├── infer_fov.py                   # FOV inference runner
├── plot_steak_analysis.py         # Analysis plots
├── combine_json.py / combine_pkl.py  # Data merging utilities
└── human_study.py                 # Human study runner
```

### Data (`data/`)

```
data/
├── layout/         # .layout files defining kitchen environments
│   ├── Overcooked2_2-4.layout
│   ├── Overcooked2_2-5.layout
│   ├── Overcooked1_1-4.layout
│   └── ...
├── config/
│   └── kitchen_config.json     # Visualization config
└── fov_traj/
    └── baseline/               # Deduplicated trajectory pickles per (fov, layout)
```

---

## Original Overcooked-AI
This project builds on Overcooked-AI. The steakhouse variant adds new dish types, equipment (grill, sink, chopping board, pot), and limited-vision agent modeling on top of the original cooperative cooking framework.


It also builds on the FOV-aware planner: https://github.com/SophieHsu/FOV-aware-planner/tree/main

---

## Contact

For questions, please reach out to: [mishafu@usc.edu](mailto:mishafu@usc.edu)
