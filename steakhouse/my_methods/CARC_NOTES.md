# CARC workflow notes

Reference for working with USC CARC (Discovery cluster) on this project. Local repo root
is `/Users/mishafu/Desktop/steakhouse`; remote mirror lives at `/home1/mishafu/steakhouse`
(`~/steakhouse` once SSH'd in).

## Connecting

```
ssh mishafu@discovery.usc.edu
```

Passwordless — an SSH key (`~/.ssh/id_ed25519`) is already authorized via `ssh-copy-id`.
If this ever stops working (new machine, key rotated), re-run:
```
ssh-copy-id -i ~/.ssh/id_ed25519.pub mishafu@discovery.usc.edu
```
(needs the CARC password once, interactively — never paste that password into a chat).

If DNS fails to resolve `discovery.usc.edu`, it's a network/VPN issue on the local end,
not a CARC problem — check VPN status.

## SLURM account

Use **`biyik_1173`** (LiraLab). `biyik_1165` also works if needed. Do **not** use
`robinjia_1822` — that's a storage-only allocation (`/project2/robinjia_1822`); `mishafu`
is not a compute member of it and `sbatch` will reject jobs against it.

Check current associations any time with:
```
ssh mishafu@discovery.usc.edu 'bash -l -c "sacctmgr show assoc where user=mishafu"'
```

## Syncing code

Passwordless SSH means these can be run directly, no manual copy-paste needed.

**Quick update** (just the FOV/bayesian code, after editing locally):
```
rsync -avz my_methods/bayesian/ mishafu@discovery.usc.edu:~/steakhouse/my_methods/bayesian/ --exclude='__pycache__'
```

**Minimal full setup** (source + layouts only, no cached planner pickles - what's actually
needed to run anything in `my_methods/bayesian/`):
```
rsync -avz --relative \
  overcooked_ai_py/__init__.py overcooked_ai_py/agents overcooked_ai_py/mdp \
  overcooked_ai_py/planning overcooked_ai_py/helpers.py overcooked_ai_py/static.py \
  overcooked_ai_py/utils.py overcooked_ai_py/configs overcooked_ai_py/data/layouts \
  my_methods/bayesian \
  --exclude='__pycache__' \
  mishafu@discovery.usc.edu:~/steakhouse/
```

**Full mirror** (everything, ~16GB, only needed once / rarely - takes a few minutes):
```
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='.DS_Store' \
  ./ mishafu@discovery.usc.edu:~/steakhouse/
```

There is no live auto-sync — editing locally does not update CARC until one of the above
runs. SSHFS (VSCode extension) was attempted for a live mount and hit unresolved issues;
on-demand rsync is the working path.

## Environment on CARC

Conda env `steakhouse-ai` (Python 3.8.20) already exists on CARC, built from
`steakhouse-ai-requirements.txt` (a `pip freeze` export of the local env - regenerate with
`/Users/mishafu/miniconda3/bin/conda env export -n steakhouse-ai --from-history` for the
conda-tracked packages, and `.../envs/steakhouse-ai/bin/python -m pip freeze` for everything
else, if it ever needs rebuilding).

**Important:** `module` command needs a login shell to exist at all - always either
`ssh ... 'bash -l -c "..."'` for one-off commands, or `#!/bin/bash -l` as the sbatch shebang
(not plain `#!/bin/bash`), or `module load` silently does nothing and everything runs
against the wrong Python.

**Important:** the CARC login node (`discovery1`) kills/throttles heavy processes run
directly on it - this is why pip installing 171 packages there died silently with no error.
Anything nontrivial (installs, the QMDP build, real experiments) must go through `sbatch`,
not run interactively on the login node.

## Running jobs

`my_methods/bayesian/run_carc.sbatch` does everything in one job: installs deps, builds the
`steak_island` QMDP planner if not already cached, then runs the static-vs-dynamic
comparison. Account and partition are already baked in.

```
ssh mishafu@discovery.usc.edu 'cd ~/steakhouse/my_methods/bayesian && sbatch run_carc.sbatch'
```

Monitor:
```
ssh mishafu@discovery.usc.edu 'squeue --me'
ssh mishafu@discovery.usc.edu 'sacct -j <jobid>'
ssh mishafu@discovery.usc.edu 'tail -f ~/steakhouse/my_methods/bayesian/steak_island_qmdp_<jobid>.out'
```
(`.out`/`.err` files land in the same dir the job was submitted from, named
`<job-name>_<jobid>.out`/`.err`.)

## File map

Local and remote paths mirror each other exactly (`~/steakhouse/...` on CARC =
`/Users/mishafu/Desktop/steakhouse/...` locally). Everything relevant lives in
`my_methods/bayesian/`:

| File | What it is |
|---|---|
| `bayesian_inference.py` | `FOVBayesFilter` - the exact Bayesian FOV filter (3 candidate FOVs, no approximation) |
| `evaluate_bayesian.py` | Accuracy test using the real paper-style QMDP teammate, stuck on the original small layout - found 0/30 episodes ever produced real evidence there |
| `evaluate_bayesian_lightweight.py` | Accuracy test using a cheap non-QMDP teammate, works across many layouts - found `steak_island` gives 88.3% accuracy, 100% of episodes informative |
| `fov_divergence_scan.py` | Fast geometry-only layout scanner (no QMDP planner needed) - useful for ruling out bad layouts, but turned out to be a weak predictor of real behavior; always confirm with `evaluate_bayesian_lightweight.py` before trusting a layout |
| `build_qmdp_steak_island.py` | Builds (or loads if cached) the real, expensive `SteakKnowledgeBasePlanner` for `steak_island` - meant to run on CARC only |
| `qmdp_bayesian_vs_static.py` | The actual experiment: robot teammate with a frozen "assumes full vision" belief (static) vs. one fed live FOV estimates from `FOVBayesFilter` (dynamic), compared on task completion speed/reward |
| `run_carc.sbatch` | SLURM job script chaining install -> build -> compare |
| `baysian_inference.py` | Empty stub, leftover typo'd filename - safe to delete, unused |

## Known gotchas

- **`SteakLimitVisionHumanModel.ml_action()` can crash** (`assert len(motion_goals) != 0`)
  on longer/unusual order sequences - a pre-existing library bug, not something to fix here.
  `evaluate_bayesian_lightweight.py` and `evaluate_bayesian.py` both catch `AssertionError`
  around the full per-step block (human action + teammate action + filter step) and just
  end the episode early when it happens.
- **`SteakHumanSubtaskQMDPPlanner.optimal_plan_cost()`** (`planners.py:6772`) is broken -
  references an undefined `subtask` variable and has no `return`. This is the fallback path
  used when the QMDP planner's bounded A* search (`compute_V`, 25-step / 0.01s budget) hits
  its limit without finding a state where all orders are delivered. If that path ever
  actually fires (long order queues, complex layouts), it raises `NameError`, which our
  existing `except AssertionError` guards do **not** catch. Worth widening those guards
  before running longer/bigger experiments.
- **Layout caching gotcha:** any grid built via `SteakHouseGridworld.from_grid(...)` (not
  `from_layout_name(...)`) gets the generic name `"unnamed_layout"` regardless of its actual
  content. Never change the grid string used for a layout that already has cached planner
  pickles under that generic name - it'll silently load a planner built for a *different*
  grid. Named layouts (`from_layout_name("steak_island")` etc.) get proper unique cache
  filenames and don't have this problem.
- **CARC quota:** `/home1/mishafu` has a 100GB limit (was ~28GB used before this project's
  sync). Check with `myquota` on CARC if storage errors show up.
