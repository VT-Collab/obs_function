# Steakhouse self-play baselines — complete reference

**Status: FINAL. All training finished 2026-08-03. All checkpoints evaluated
by actual play.**

Written so that a person or session with **zero prior context** can use these
baselines correctly without re-deriving anything and without picking up a
broken or mismatched checkpoint.

---

# ⚠️ AT A GLANCE — READ THIS BOX FIRST

```
=============================================================================
  EVERYTHING HERE IS  SEED 1.  THERE IS NO OTHER SEED.
=============================================================================
  * ONE training run per layout. seed = 1. That is all that exists.
  * ONE checkpoint per layout, and it is the LAST one, not the best one.
  * No seed 2, no seed 3, no averaging, no error bars, no variance estimate.
  * ZSC-Eval runs 11 seeds per layout (5..15). This is 1 of 1.
    -> A "0.00 FAILED" result here is ONE SEED'S outcome. It is weak
       evidence. It does NOT prove a layout is unlearnable.
    -> A "60.00 SOLVED" result here is also ONE SEED. Do not put an error
       bar on it in a paper. Run more seeds first.
=============================================================================
```

### The file you want

```
/scratch1/mishafu/steakhouse_sp/specialist/<LAYOUT>_seed1/sp_<LAYOUT>.pt
                                                    ^^^^^
                                        always seed1 -- nothing else exists
```

### The 11 layouts that actually work (copy-paste)

```
steak_gc00  steak_gc01  steak_gc03  steak_gc04  steak_gc05
steak_gc06  steak_gc07  steak_cram  steak_cram2 steak_gs00
steak_api
```

Plus 2 partial: `steak_none_3` (22.00) and `steak_gs02` (20.00).

### The 12 that do NOT work — all score exactly 0.00

```
steak_mid_1   steak_mid_2   steak_side_3  steak_side_4
steak_parrallel  steak_tshape  steak_gs03  steak_gs04
steak_gs05    steak_gs07    steak_gs08    steak_gs09
```

### Three sentences of context

1. **Specialists work, the generalist does not.** Use a specialist.
2. **A checkpoint only fits the layout it trained on** — `Linear(32*W*H, 64)`.
   Get `obs_shape` from the env, never hardcode it.
3. **`smoke/` subdirectories contain identically-named junk checkpoints.**
   Never glob. Use the exact path above.

---

## CONTENTS

1. [What you have (read this first)](#1-what-you-have-read-this-first)
2. [The one mistake that will cost you a day](#2-the-one-mistake-that-will-cost-you-a-day)
3. [Where everything lives on CARC](#3-where-everything-lives-on-carc)
4. [Environment traps that silently destroy runs](#4-environment-traps-that-silently-destroy-runs)
5. [**WHICH FILE EXACTLY — getting the right checkpoint**](#5-which-file-exactly--getting-the-right-checkpoint)
6. [Results: the specialist baselines](#6-results-the-specialist-baselines)
7. [Analysis: why 12 layouts failed](#7-analysis-why-12-layouts-failed)
8. [The generalist run and why it failed](#8-the-generalist-run-and-why-it-failed)
9. [How to load and use a checkpoint](#9-how-to-load-and-use-a-checkpoint)
10. [How to verify checkpoints yourself](#10-how-to-verify-checkpoints-yourself)
11. [Dense reward shaping (added on top of the MDP)](#11-dense-reward-shaping-added-on-top-of-the-mdp)
12. [Config provenance vs ZSC-Eval](#12-config-provenance-vs-zsc-eval)
13. [How to rerun / retrain](#13-how-to-rerun--retrain)
14. [Reading a training log line](#14-reading-a-training-log-line)
15. [File map](#15-file-map)
16. [Job IDs and gotchas](#16-job-ids-and-gotchas)

---

## 1. WHAT YOU HAVE (READ THIS FIRST)

Two kinds of policy were trained. **Only one kind works.**

### SPECIALISTS — USE THESE

25 separate policies, **one per layout**, each trained only on its own kitchen.
No padding, no weights shared between layouts.

* **13 of 25 deliver food.**
* **11 of those are essentially solved** (>=56 out of a max 60).
* The other 12 score a hard 0.00 and are not usable.
* **All of this is SEED 1 only** — one run per layout, no repeats. See the
  box at the top and [§5](#5-which-file-exactly--getting-the-right-checkpoint).

### GENERALIST — DO NOT USE

One policy trained across all 25 layouts at once, every grid padded onto a
shared 15x11 canvas.

* **It scored 0.00 on every single layout, for all 500 episodes.**
* It never delivered once — including on layouts where the specialist scores a
  perfect 60.
* The checkpoint exists and loads fine. **It is still useless.** Do not report
  it as a baseline. See [§8](#8-the-generalist-run-and-why-it-failed): it is a
  legitimate negative result, not a bug to fix.

### The short version

> **If you need a working steakhouse baseline policy, take a SPECIALIST
> checkpoint from a "SOLVED" row in [§6](#6-results-the-specialist-baselines), using the
> exact path from [§5](#5-which-file-exactly--getting-the-right-checkpoint),
> and build the network with that layout's own `obs_shape`.**

---

## 2. THE ONE MISTAKE THAT WILL COST YOU A DAY

**A specialist checkpoint only works on the layout it was trained on.**
They are not interchangeable. Not even a little.

`CNNBase` ends in `Linear(32 * W * H, 64)`. The weight shape is determined by
that layout's grid size:

```
sp_steak_gc00.pt   trained on 5x5    ->  ONLY loads into a (23, 5, 5) policy
sp_steak_mid_1.pt  trained on 15x10  ->  ONLY loads into a (23, 15, 10) policy
```

Load the wrong one and you get a shape error if you are lucky, or silent
nonsense if you built the network with the wrong `obs_shape` first.

**Never hardcode the shape. Ask the environment:**

```python
env = SteakSelfPlayEnv(LAYOUT, n_orders=4, horizon=400, seed=0)
obs_shape = env.obs_shape          # correct by construction, always
```

**Sanity check by file size** — it scales with `W*H`:

| grid | example | file size |
|---|---|---|
| 5x5 | `sp_steak_gc00.pt` | 1.0 MB |
| 9x9 | `sp_steak_cram.pt` | 1.9 MB |
| 15x10 | `sp_steak_api.pt` | 3.1 MB |
| 15x11 | `sp_tier1.pt` (generalist) | 3.3 MB |

If a checkpoint's size does not match its layout's grid, you have the wrong
file.

> The **generalist** is the one exception — trained on a padded 15x11 canvas,
> it expects `(23, 15, 11)` for *every* layout. But it does not work, so this
> does not help you.

---

## 3. WHERE EVERYTHING LIVES ON CARC

| what | where |
|---|---|
| login node | `ssh discovery.usc.edu` — **requires USC VPN** |
| **if DNS fails while VPN is up** | ssh the IP directly. `nslookup discovery.usc.edu` (was `10.72.0.13`). ssh caches stale DNS; the IP connects immediately. |
| repo | `~/steakhouse` = `/home1/mishafu/steakhouse` |
| this code | `~/steakhouse/fov/robot/policy/new/official_baselines/` |
| training logs | `~/steakhouse/sp_specialist_10789712_<0-24>.out` |
| generalist log | `~/steakhouse/sp_generalist_10789711.out` |
| eval logs | `~/steakhouse/ckeval_<jobid>.out` |
| **SPECIALIST checkpoints** | `/scratch1/mishafu/steakhouse_sp/specialist/<layout>_seed1/sp_<layout>.pt` — **read [§5](#5-which-file-exactly--getting-the-right-checkpoint), there are identically-named decoys** |
| generalist checkpoint | `/scratch1/mishafu/steakhouse_sp/generalist/tier1_seed1/sp_tier1.pt` |
| conda env | `steakhouse-ai` — python 3.8.20, torch 2.4.1+cu121, numpy 1.24.3 |
| slurm | account `biyik_1173`, partition `main` |

### Three operational facts

**`/scratch1` is purge-eligible.** Copy anything you need long-term somewhere
durable. Checkpoints go there because the `home1` quota is tight — do not
"fix" that by writing to home.

**Everything is CPU.** `torch.cuda.is_available()` is `False` on these compute
nodes. That is fine and intentional — see [§12](#12-config-provenance-vs-zsc-eval).

**Never run python work on the login node.** numpy segfaults there:

```
Importing the numpy C-extensions failed ...
  PyCapsule_Import could not import module "datetime"
Segmentation fault (core dumped)
```

Use `sbatch`, or an interactive allocation:

```bash
srun -A biyik_1173 -p main -c 4 --mem 16G -t 00:20:00 --pty bash
```

### Layout staging — required, and easy to miss

`SteakHouseGridworld.from_layout_name()` **only** reads
`overcooked_ai_py/data/layouts/`. The validated layout library lives in
`fov/layouts_final/layouts/` and is **not** copied there by default.

Every sbatch script here runs this first:

```bash
cp -n fov/layouts_final/layouts/*.layout overcooked_ai_py/data/layouts/
```

"layout not found" from a fresh shell means you skipped this.

---

## 4. ENVIRONMENT TRAPS THAT SILENTLY DESTROY RUNS

Both are already handled inside `utils/env_wrapper.py`. **Do not undo them.**
Each one cost an entire wasted training run to discover.

### Trap 1 — `start_order_list` arrives as a STRING

Every `.layout` file declares:

```
"start_order_list": 'steak, steak, steak',
```

That is a **string**, not a list. Consequences:

* `len(...)` counts **characters** — 19, not 3
* `deliver_dish` does `order_list[0]`, which on a string is the character
  `'s'`, so `'steak' == 's'` is `False`
* **orders are never consumed and no delivery reward ever fires**

`features.py` now asserts against this. Always construct the MDP with
`start_order_list=["steak"] * n_orders`.

### Trap 2 — `rew_shaping_params: None` means ZERO shaping in this fork

```
this repo,  overcooked_mdp.py:672
    self.reward_shaping_params = NO_REW_SHAPING_PARAMS   if None   <- all zeros
ZSC-Eval,   overcooked_mdp.py:980
    self.reward_shaping_params = BASE_REW_SHAPING_PARAMS if None   <- real values
```

**The same line with opposite defaults.** ZSC-Eval gets shaping for free; this
fork silently gives all zeros. The full failure cascade:

```
reward is identically 0
  -> returns 0
  -> critic learns V = 0            (vloss pinned at 0.000)
  -> advantages are float noise off V (~1e-9)
  -> buffer normalizes them by their own std
  -> 1e-9 / 1e-9 = UNIT-SCALE noise, at full gradient strength
  -> policy trains hard on nothing

observed: clip_frac 0.45, entropy 1.79 -> 1.11 in 30 episodes
```

Two fixes are in place:

1. `env_wrapper.py` passes `rew_shaping_params=BASE_REW_SHAPING_PARAMS`
   explicitly, plus an assert that fires if shaping is ever all-zero again.
2. `buffer.py` guards `adv.std() < 1e-6` and emits zeros rather than
   amplifying rounding error.

---

## 5. WHICH FILE EXACTLY — GETTING THE RIGHT CHECKPOINT

### There is exactly ONE checkpoint per layout, per seed

```
/scratch1/mishafu/steakhouse_sp/specialist/<layout>_seed<SEED>/sp_<layout>.pt
```

Concretely, for `steak_gc00` at seed 1 — this is THE file:

```
/scratch1/mishafu/steakhouse_sp/specialist/steak_gc00_seed1/sp_steak_gc00.pt
```

### EVERYTHING IS SEED 1

**`seed1` is the only seed that exists.** Every number in this document — every
60.00, every 0.00 — comes from exactly one training run per layout.

```
specialist/steak_gc00_seed1/     <- exists
specialist/steak_gc00_seed2/     <- DOES NOT EXIST
specialist/steak_gc00_seed3/     <- DOES NOT EXIST
```

What that means in practice:

* **No variance estimate.** You cannot put an error bar on any of these
  numbers. There is nothing to average.
* **A 0.00 is one seed's failure**, not proof a layout is unlearnable. Seeds
  matter a lot in sparse-reward RL — ZSC-Eval runs **11 per layout**
  (`seed_begin=5`, `seed_max=15`) precisely because single runs are noisy.
* **A 60.00 is also one seed.** It is a real, played, verified 60.00 — but it
  is n=1.

To add seeds (nothing gets clobbered, each writes its own directory):

```bash
SEED=2 sbatch fov/robot/policy/new/official_baselines/SP/run_specialists.sbatch
SEED=3 sbatch .../run_specialists.sbatch
# -> /scratch1/$USER/steakhouse_sp/specialist/<layout>_seed2/sp_<layout>.pt
```

### It is the LAST checkpoint, not the BEST one

`self_play.py` saves every `--save_interval 25` episodes to **the same
filename**, overwriting. There is no best-checkpoint tracking, no
`sp_<layout>_ep450.pt`, no `best.pt`. What survives is whatever the run wrote
last.

**Did that cost anything?** Checked, and effectively no. Peak vs final training
`sparse_ret`, for every layout that ever delivered:

```
layout          peak (episode)   final    gap
steak_gc00      60.00 (ep 490)   60.00    0.00
steak_gc07      59.60 (ep 490)   59.60    0.00
steak_gs00      59.20 (ep 490)   59.20    0.00
steak_none_3    12.40 (ep 490)   12.40    0.00
steak_gc06      60.00 (ep 430)   59.60    0.40
steak_gc01      60.00 (ep 420)   59.20    0.80
steak_gc04      58.40 (ep 480)   57.20    1.20
steak_gc03      60.00 (ep 470)   58.80    1.20
steak_cram2     58.80 (ep 480)   57.20    1.60
steak_api       59.20 (ep 470)   54.80    4.40
steak_cram      58.40 (ep 470)   50.40    8.00
```

Every peak is inside the last 80 episodes, and checkpoints save every 25, so
the saved weights are always near the peak. More importantly the *training*
number is a noisy running average over sampled rollouts — the **played eval**
is the authority, and there the final checkpoints of `cram` and `api` both
score a perfect **60.00**. So the peak/final gap in the log is noise, not
degradation.

> If you later want true best-checkpoint saving, that is a small change in
> `self_play.py` — track the best `sparse_ret` and write a second file.

### TRAP 1 — `smoke/` holds an IDENTICALLY NAMED junk checkpoint

Every layout directory contains **two** files with the **same basename**:

```
specialist/steak_gc00_seed1/sp_steak_gc00.pt          <- THE REAL ONE (ep 499)
specialist/steak_gc00_seed1/smoke/sp_steak_gc00.pt    <- JUNK (800-step smoke test)
```

They are **the same file size** (1012402 bytes for `gc00`), because the network
architecture is identical — only the weights differ. Size will not save you.
Only the path and the modification time distinguish them:

```
Aug 1 22:34   smoke/sp_steak_gc00.pt     <- written during the smoke test
Aug 2 03:15   sp_steak_gc00.pt           <- written by the real run
```

**Never glob for `**/sp_<layout>.pt`.** Always give the exact path with no
`smoke` in it. If you are unsure, check `ck["episode"]` — the real one is
`499` (or `475` for `steak_mid_2`); a smoke checkpoint is `3`.

### TRAP 2 — two STALE directories from the pre-bugfix runs

At the TOP level of `steakhouse_sp/`, outside `specialist/` and `generalist/`:

```
/scratch1/mishafu/steakhouse_sp/steak_side_2_seed1/    Aug 1 21:48-21:49
/scratch1/mishafu/steakhouse_sp/tier1_seed1/           Aug 1 21:59
```

These are from the **first aborted runs, before the reward-shaping bug in
[§4](#4-environment-traps-that-silently-destroy-runs) was found**. They were
trained on a reward that was identically zero. They are worthless and were
never evaluated. `steak_side_2` is not even a Tier 1 layout.

**Ignore them, or delete them:**

```bash
rm -rf /scratch1/$USER/steakhouse_sp/steak_side_2_seed1        /scratch1/$USER/steakhouse_sp/tier1_seed1
```

### The complete file inventory

56 `.pt` files under `steakhouse_sp/`:

| count | path | use it? |
|---:|---|---|
| **25** | `specialist/<layout>_seed1/sp_<layout>.pt` | **YES — these are the baselines** |
| 25 | `specialist/<layout>_seed1/smoke/sp_<layout>.pt` | no — 800-step smoke tests |
| 1 | `generalist/tier1_seed1/sp_tier1.pt` | no — scored 0.00 everywhere ([§8](#8-the-generalist-run-and-why-it-failed)) |
| 1 | `generalist/tier1_seed1/smoke/sp_tier1.pt` | no |
| 2 | `steak_side_2_seed1/...` | no — stale, pre-bugfix |
| 2 | `tier1_seed1/...` | no — stale, pre-bugfix |

### One-liner to list only the real ones

```bash
ls /scratch1/$USER/steakhouse_sp/specialist/*/sp_*.pt
```

That glob excludes `smoke/` (it is one level deeper) and both stale
directories (they are not under `specialist/`). 25 files, one per layout.

---

## 6. RESULTS: THE SPECIALIST BASELINES

### Scoring

`delivery_reward = 20`. `is_terminal` fires at `len(order_list) <= 1`, so with
`n_orders=4` you can deliver **3**.

> **60.00 = perfect episode. 40.00 = 2 of 3 orders. 20.00 = 1 of 3.**

### Run parameters

> **SEED 1 ONLY.** Every number in the table below is from a single training
> run per layout. No repeats, no averaging. See the box at the top of this
> document.

Job array `10789712`, **seed 1 (the only seed)**, `num_env_steps 1e7` = **500 episodes**,
`episode_length 400`, `horizon 400`, `n_rollout_threads 50`, `n_orders 4`.

24 of 25 tasks COMPLETED at ep 490. `steak_mid_2` hit the 24h wall (TIMEOUT)
at ep 490, so its checkpoint is ep 475; every other checkpoint is ep 499.

### The table

`train` = last logged `sparse_ret` during training (averaged over sampled
rollouts). `det` / `stoch` = 10 played episodes with the **final** checkpoint,
argmax vs sampled (eval jobs `10820032` / `10820033`).

| layout | grid | cells | train | **det** | **stoch** | 1st deliver | verdict |
|--------|------|------:|------:|--------:|----------:|-----------:|---------|
| steak_gc04 | 7×7 | 49 | 57.20 | **60.00** | **60.00** | ep 130 | **SOLVED** |
| steak_cram | 9×9 | 81 | 50.40 | **60.00** | **60.00** | ep 300 | **SOLVED** |
| steak_gs00 | 6×5 | 30 | 59.20 | **60.00** | **60.00** | ep 230 | **SOLVED** |
| steak_gc03 | 7×6 | 42 | 58.80 | **60.00** | **60.00** | ep 160 | **SOLVED** |
| steak_gc00 | 5×5 | 25 | 60.00 | **60.00** | 58.00 | ep 80 | **SOLVED** |
| steak_gc06 | 5×7 | 35 | 59.60 | **60.00** | 58.00 | ep 90 | **SOLVED** |
| steak_api | 15×10 | 150 | 54.80 | **60.00** | 56.00 | ep 240 | **SOLVED** |
| steak_gc01 | 6×5 | 30 | 59.20 | 40.00 | **60.00** | ep 120 | **SOLVED** — use stochastic |
| steak_gc05 | 8×7 | 56 | 56.40 | 40.00 | **60.00** | ep 90 | **SOLVED** — use stochastic |
| steak_cram2 | 13×5 | 65 | 57.20 | 40.00 | **56.00** | ep 80 | **SOLVED** — use stochastic |
| steak_gc07 | 9×6 | 54 | 59.60 | **60.00** | 48.00 | ep 120 | **SOLVED** — use argmax, high variance |
| steak_none_3 | 15×10 | 150 | 12.40 | 20.00 | **22.00** | ep 460 | PARTIAL — late bloomer |
| steak_gs02 | 8×6 | 48 | 0.40 | **20.00** | 2.00 | ep 460 | MARGINAL — argmax only |
| steak_mid_1 | 15×10 | 150 | 0.00 | 0.00 | 0.00 | never | FAILED (committed) |
| steak_gs03 | 9×7 | 63 | 0.00 | 0.00 | 0.00 | never | FAILED (committed) |
| steak_gs05 | 8×8 | 64 | 0.00 | 0.00 | 0.00 | never | FAILED (committed) |
| steak_gs08 | 7×9 | 63 | 0.00 | 0.00 | 0.00 | never | FAILED (committed) |
| steak_gs04 | 10×7 | 70 | 0.00 | 0.00 | 0.00 | never | FAILED (committed) |
| steak_gs09 | 11×9 | 99 | 0.00 | 0.00 | 0.00 | never | FAILED (committed) |
| steak_side_3 | 15×10 | 150 | 0.00 | 0.00 | 0.00 | never | FAILED (undecided) |
| steak_side_4 | 15×10 | 150 | 0.00 | 0.00 | 0.00 | never | FAILED (undecided) |
| steak_parrallel | 15×10 | 150 | 0.00 | 0.00 | 0.00 | never | FAILED (undecided) |
| steak_tshape | 15×10 | 150 | 0.00 | 0.00 | 0.00 | never | FAILED (undecided) |
| steak_gs07 | 12×8 | 96 | 0.00 | 0.00 | 0.00 | never | FAILED (undecided) |
| steak_mid_2 | 15×11 | 165 | 0.00 | 0.00 | 0.00 | never | FAILED (undecided, TIMEOUT) |

Four layouts — `gc04`, `cram`, `gs00`, `gc03` — scored 60.00 in **both** modes
with `worst` = 60 across 10 episodes. Not luck.

### ALWAYS SAY WHICH ACTION MODE YOU REPORT

The two modes disagree, in both directions:

```
steak_gc01    det 40.00  ->  stoch 60.00     argmax DEADLOCKS the policy
steak_gc05    det 40.00  ->  stoch 60.00     argmax DEADLOCKS the policy
steak_cram2   det 40.00  ->  stoch 56.00     argmax DEADLOCKS the policy
steak_gc07    det 60.00  ->  stoch 48.00     fragile when sampled (worst run 0)
steak_gs02    det 20.00  ->  stoch  2.00     only works greedily
```

A greedy (argmax) policy can walk into a loop that sampling escapes — that is
exactly what `gc01`, `gc05` and `cram2` are doing, and they are fully solved
once sampled. Neither mode dominates. Pick one, state it, be consistent.

### Which layouts succeeded: TOPOLOGY, not size

| family | what it is | result |
|---|---|---|
| `gc*` | generated **clustered** | **8 / 8 solved** |
| `cram*` | hand-made contention | **2 / 2 solved** |
| `gs*` | generated **spread** | 2 / 8 delivered (`gs00` 60.00, `gs02` 20.00) |
| large hand-designed | `api` `mid_1` `side_3` `side_4` `parrallel` `tshape` `none_3` | 2 / 7 (`api` solved, `none_3` partial) |

**Grid size is not the variable.** `steak_api` is 150 cells and scores a
perfect 60; `steak_gs03` is 63 cells and scores 0. Clustered kitchens put the
stations close together, so the pan → board → sink → serve chain is short
enough to stumble into. Spread kitchens turn the same chain into a long walk
and the agent never closes it.

> This lines up with `fov/layouts_final/LAYOUTS.md`, whose headline finding is
> that **contention layouts drive robot influence**. The layouts best for the
> FOV research are the same ones self-play can actually solve. `steak_gc00` is
> INFL 28.5 *and* a perfect baseline.

---

## 7. ANALYSIS: WHY 12 LAYOUTS FAILED

### There are TWO failure modes and they need DIFFERENT fixes

Split the 12 zeros by final policy entropy (`ln 6 = 1.79` means uniform /
undecided; `0` means fully committed):

**A. COMMITTED — entropy 0.80 to 0.98**
`mid_1` `gs03` `gs05` `gs08` `gs04` `gs09`

The policy **decided** on a behaviour, and that behaviour does not deliver.
`mid_1`'s value loss climbed `0.68 → 9.35`, meaning it is collecting plenty of
**shaped** reward. It has settled into a local optimum where it farms pickups
and chopping without ever closing the chain to a delivery.

> More training steps alone will probably **not** fix these. They need the
> late-chain shaping strengthened, or shaping annealed away sooner so the true
> delivery objective starts to bite.

**B. UNDECIDED — entropy 1.01 to 1.17**
`side_3` `side_4` `parrallel` `tshape` `gs07` `mid_2`

Never converged on anything at all. These are the ones where **more steps** is
the correct lever.

**Do not treat the 12 zeros as one group.**

### The step budget truncated late learners

```
steak_gs02     first delivery at ep 460 / 500
steak_none_3   first delivery at ep 460 / 500, and STILL CLIMBING at the cutoff
                   ep 470  ->  sparse_ret  4.80
                   ep 490  ->  sparse_ret 12.40
                   final checkpoint (ep 499) evaluated at 20.00
```

Two layouts crossed over in the **last 8%** of the budget, one still improving
when it ran out. For reference `steak_cram` needed 300 episodes and
`steak_api` needed 240.

> **A 0.00 at 1e7 steps is NOT evidence that a layout is unlearnable.**

### Always evaluate the FINAL checkpoint

`steak_none_3` reads **0.00** from its ep-450 checkpoint and **20.00** from its
ep-499 checkpoint — same run. An early eval would have written it off.

---

## 8. THE GENERALIST RUN AND WHY IT FAILED

Job `10789711`.
`/scratch1/mishafu/steakhouse_sp/generalist/tier1_seed1/sp_tier1.pt`

**One policy, all 25 Tier 1 layouts.** Every grid padded onto a shared 15x11
canvas (as large as the biggest layout) and dropped in at a **random offset**
each reset so the policy cannot memorize absolute coordinates. Padded cells are
written as `'X'` counter terrain, **not** zeros — all-zero terrain channels
mean *floor* in `features.py`, which would have told the agent the dead space
outside the kitchen was walkable.

### Result: total failure

**0.00 sparse reward on all 25 layouts, across all 500 episodes. Not one
delivery** — including on `steak_gc00`, where the specialist scores a perfect
60.00 on the identical kitchen with identical code and identical shaping.

### Why — the value loss tells the story

Value loss tracks return magnitude:

```
generalist          vloss  0.25 - 0.63    flat for the entire run
mid_1 specialist    vloss  0.68 -> 9.35   same 15x10 layout class
winning specialists vloss  13 - 26
```

**20-40x less return per episode.** Spread across 25 kitchens it never gets
enough repetitions in any one to master even the shaped subgoals, and it pays
the padding tax on top: `steak_gc00` fills only **15%** of the 15x11 canvas, so
85% of its convolutional work is dead space — at a different offset every
episode. Final entropy stalled at 1.303 versus 0.458–0.877 for the winners: it
never committed to anything.

### This is a legitimate finding, not a bug

It independently confirms that per-layout training is the right approach.

Note also: **this padding approach is not what ZSC-Eval does.** Their
`layout_generator.mdp_gen_fn_from_dict` / `embed_grid` is only ever called from
`overcooked_test.py` and `benchmarking.py` — never from a training script.
`train_sp.sh` takes `layout=$1`, one layout per run. They never needed padding
because their layouts are all 5x4 / 5x5 / 9x5; Tier 1 here spans (5,5)–(15,11).

---

## 9. HOW TO LOAD AND USE A CHECKPOINT

```python
import sys, types, torch
sys.path.insert(0, "fov/robot/policy/new/official_baselines")
from algorithm.rMAPPOPolicy import R_MAPPOPolicy
from utils.env_wrapper import SteakSelfPlayEnv

LAYOUT = "steak_gc00"        # must match the checkpoint filename exactly

# obs_shape MUST come from this layout's env. see section 2.
env = SteakSelfPlayEnv(LAYOUT, n_orders=4, horizon=400, seed=0)
obs_shape = env.obs_shape                     # (23, W, H)

args = types.SimpleNamespace(hidden_size=64, recurrent_N=1,
                             lr=5e-4, critic_lr=5e-4, opti_eps=1e-5)
policy = R_MAPPOPolicy(args, obs_shape, obs_shape, act_dim=6)

ck = torch.load(f"/scratch1/mishafu/steakhouse_sp/specialist/"
                f"{LAYOUT}_seed1/sp_{LAYOUT}.pt",
                map_location="cpu", weights_only=False)
policy.actor.load_state_dict(ck["actor"])
policy.critic.load_state_dict(ck["critic"])
```

The checkpoint dict is `{"actor", "critic", "episode", "args"}`.
**`ck["args"]` holds every hyperparameter that run used**, so you never have to
guess. All are `episode: 499` except `steak_mid_2` (475).

### Rolling out

```python
obs = env.reset()                                  # (2, 23, W, H)
rnn = torch.zeros(env.num_agents, 1, 64)           # (2, L, H)
masks = torch.ones(env.num_agents, 1)              # (2, 1)
total = 0.0
for t in range(400):
    with torch.no_grad():
        actions, _, rnn = policy.actor(
            torch.from_numpy(obs).float(), rnn, masks,
            True)                                  # True = argmax, False = sample
    obs, sparse, shaped, done, truncated = env.step(actions.squeeze(-1).numpy())
    total += sparse
    if done:
        break
```

### Getting the ACTION DISTRIBUTION (for a filter / inference layer)

`SP/self_play.py` exposes `action_probs()`. Use it instead of `policy.act()`
whenever you need all six numbers rather than one sampled choice:

```python
from SP.self_play import action_probs
probs, rnn_states = action_probs(policy, obs, rnn_states, masks)
#   probs (N, 6), each row sums to 1
#   verified: argmax(probs) == policy.act(deterministic=True)
```

It is the identical forward pass (`base -> rnn -> action_out`), just stopped
one step earlier at the `Categorical` instead of collapsing to one integer.
**It advances the GRU memory like any forward pass** — thread `rnn_states`
through your eval loop or the agent has no memory.

### Shapes, in one place

```
N = n_rollout_threads * 2 agents      both chefs are ROWS in one batch
obs         (N, 23, W, H)   float32, ego-centric per agent
rnn_states  (N, 1, 64)      (N, recurrent_N, hidden_size)
masks       (N, 1)          0 = episode just reset -> wipes GRU memory
actions     (N, 1)          int64 in [0,6): N, S, E, W, stay, interact
```

Flatten order is **thread-major**:
`[env0-chef0, env0-chef1, env1-chef0, env1-chef1, ...]`.
Keep it consistent or the two chefs get scrambled.

**Why self-play does not collapse into identical behaviour:** `features.py` is
ego-centric — `agent_index` decides who lands in channels `[0,1]` ("me") and
who lands in `[2,3]` ("them"). Same weights, different input, different
behaviour.

### The 23 observation channels

```
[0]      ego agent position
[1]      ego agent facing cell
[2]      other agent position
[3]      other agent facing cell
[4-10]   object planes: meat, onion, plate, washed_plate, steak, garnish, dish
[11-18]  terrain masks, one per TERRAIN entry: P B W M O D S X
[19]     station status   0 empty / 0.5 in progress / 1 ready
[20]     station timer    elapsed / total, in [0,1]
[21]     time left in episode, normalized
[22]     orders remaining, normalized
```

Note `' '` (walkable floor) has **no** terrain plane — it is the implicit
"all eight terrain channels are zero" category.

---

## 10. HOW TO VERIFY CHECKPOINTS YOURSELF

**A checkpoint that loads cleanly can still be a policy that never delivers.**
All 25 load fine; 12 are useless. Loading is not verification — playing is.

```bash
# all 25 layouts, argmax, 5 episodes each
sbatch fov/robot/policy/new/official_baselines/SP/run_eval_checkpoints.sbatch

# specific layout, more episodes, sampled actions
LAYOUTS=steak_gc00 EPISODES=20 MODE=stochastic \
    sbatch .../SP/run_eval_checkpoints.sbatch
```

Prints mean / best / worst delivery reward per layout. **Must run on a compute
node** — see [§3](#3-where-everything-lives-on-carc).

> **`--export` gotcha:** SLURM splits `--export` on commas, so
> `--export=ALL,LAYOUTS=a,b,EPISODES=10` silently passes only `LAYOUTS=a` and
> treats `b` and `EPISODES=10` as separate variables. Pass **one layout per
> job**, or edit the default list inside the sbatch script.

---

## 11. DENSE REWARD SHAPING (ADDED ON TOP OF THE MDP)

The MDP itself is **not** modified. All of this lives in
`utils/env_wrapper.py::_DenseShaper`.

### The problem

This fork fires only **two** shaped rewards, both at the pan:
`PLACEMENT_IN_POT_REW` and `COOKING_STEAK_REW`. Chopping, washing, plating,
garnishing and dish pickup pay **nothing** — `PLATE_PICKUP_REWARD` is commented
out in the fork and the rest were never wired at all.

That is a dense signal for 2 steps of an 8-step chain
(meat → pan → cook → onion → chop → plate → wash → plate steak → garnish →
deliver).

### The solution

`_DenseShaper` snapshots state before and after each step and pays out the
events the MDP forgot:

| event | reward |
|---|---|
| pick up meat / onion / plate | 1.0 |
| onion → chopping board, plate → sink | 3.0 |
| each chop / wash tick | 1.0 |
| carry off a finished steak / washed_plate / garnish | 3.0 |
| assemble the dish | 5.0 |

Magnitudes mirror ZSC-Eval's `BASE_REW_SHAPING_PARAMS`. Measured effect on a
**random** policy over 3 episodes:

```
steak_gc00    before  3.0 reward /  1 nonzero tick   after  75.0 / 67
steak_cram    before  0.0 /  0                       after  21.0 / 19
steak_mid_1   before  0.0 /  0                       after   6.0 /  6
```

### Two deliberate design decisions

**Counters are ignored.** Dropping an onion on a counter is not progress; only
placing it on a chopping board is. Only station locations count.

**Every event type is capped at `n_orders * 3` per episode.** Without a cap the
agent finds the pick-up / put-down loop and farms it forever instead of
cooking — the classic shaping exploit. It would show up as beautiful reward
curves attached to a completely useless policy.

---

## 12. CONFIG PROVENANCE VS ZSC-EVAL

Diffed against `ZSC-Eval/zsceval/config.py`, `overcooked_config.py`, and
`scripts/overcooked/shell/train_sp.sh`.

### Matching (verified)

`hidden_size 64` · `recurrent_N 1` · `data_chunk_length 10` · `gamma .99` ·
`gae_lambda .95` · `clip_param .2` · `max_grad_norm 10` · `opti_eps 1e-5` ·
action-layer `gain 0.01` · orthogonal init · clipped value loss ·
huber loss (δ=10) · `lr` / `critic_lr 5e-4` · `value_loss_coef 1` ·
`ppo_epoch 15` · `num_mini_batch 1` · `episode_length 400` ·
`n_rollout_threads 50` · entropy annealed `0.2 → 0.01` over 1e7 steps ·
`reward_shaping_horizon 1e8` (10x the run, so shaping never really turns off —
ZSC-Eval does the same) · proper time limits via `bad_masks` ·
CNN `32,3,1,1  64,3,1,1  32,3,1,1`.

GAE recursion and the clipped surrogate were verified line-for-line identical.

### Deliberately different or missing

* **`use_valuenorm`** — ZSC-Eval defaults it True; not implemented here. Less
  critical now that shaping no longer anneals to zero, so the reward scale is
  stable. Still the most defensible algorithmic thing to add next.
* **1 seed** vs ZSC-Eval's **11** (`seed_begin=5`, `seed_max=15`). This is the
  biggest methodological gap in the whole setup. A 0.00 on a single seed is
  weak evidence a layout is unlearnable, and a 60.00 on a single seed has no
  error bar. Everything in this document is n=1.

### Performance notes

**`VecSteakEnv` steps its environments SEQUENTIALLY.** Extra `--cpus-per-task`
only feeds torch's conv threads, not the env loop — asking for 32 cpus does
**not** make the rollout 32x faster. Real parallelism would mean writing a
`SubprocVecEnv` in `utils/env_wrapper.py`, not changing a slurm flag.

Measured throughput: ~500 fps specialist, ~250 fps generalist (bigger canvas),
~120 fps on the largest 15x11 grid.

**GPU is not worth it.** The bottleneck is `envs.step()` — pure python,
CPU-bound, un-batchable. The conv stack on these grid sizes is microseconds
either way.

---

## 13. HOW TO RERUN / RETRAIN

```bash
ssh discovery.usc.edu                     # or the IP if DNS is stale
cd ~/steakhouse

# 25 specialists, one per layout, 10 concurrent
sbatch fov/robot/policy/new/official_baselines/SP/run_specialists.sbatch

# the generalist (known to fail; only rerun if changing the approach)
sbatch fov/robot/policy/new/official_baselines/SP/run_generalist.sbatch

# a different seed writes to a different SAVE_DIR, so nothing is clobbered
SEED=2 sbatch .../run_specialists.sbatch
```

### Syncing code from a laptop

`rsync` over this link fails ("unexpected end of file"). Use tar over ssh:

```bash
cd fov/robot/policy/new
tar czf - --exclude='._*' --exclude='__pycache__' official_baselines \
  | ssh discovery.usc.edu 'cd ~/steakhouse/fov/robot/policy/new && tar xzf -'
ssh discovery.usc.edu 'find ~/steakhouse -name "._*" -delete'   # macOS junk
```

### Recommended next experiments, in priority order

1. **More steps for the UNDECIDED failures** — `side_3` `side_4` `parrallel`
   `tshape` `gs07` `mid_2`. Use `--num_env_steps 30000000`. Directly supported
   by the late-bloomer evidence in [§7](#7-analysis-why-12-layouts-failed).
   Note the 24h wall: `mid_2` already timed out at 1e7, so raise `--time` or
   lower `n_rollout_threads`.
2. **Different shaping for the COMMITTED failures** — `mid_1` `gs03` `gs05`
   `gs08` `gs04` `gs09`. These are stuck farming shaped reward. Strengthen the
   late-chain rewards in `_DenseShaper`, or shorten `reward_shaping_horizon` so
   the true objective takes over sooner.
3. **More seeds — arguably the highest-value thing on this list.** Everything
   documented here is `seed1`, n=1 per layout. ZSC-Eval runs 11 per layout.
   Until there are 3+ seeds you cannot tell a real failure from an unlucky
   one, and you cannot report variance. Cheap to run:
   `for s in 2 3 4 5; do SEED=$s sbatch .../run_specialists.sbatch; done`
4. **`use_valuenorm`** — the one remaining algorithmic gap.
5. **Curriculum** (initialize big layouts from a small-layout checkpoint) —
   currently blocked by the 1:1 shape binding in
   [§2](#2-the-one-mistake-that-will-cost-you-a-day). Would need adaptive
   pooling in `cnn.py` to make checkpoints size-agnostic.

---

## 14. READING A TRAINING LOG LINE

```
ep 490/500 | steps 9,820,000 | fps 592 | sparse_ret 60.00 | vloss 16.910
  | ploss -0.0193 | ent 0.458 | clip 0.097 | shape_c 0.90 | ent_c 0.014
```

| field | how to read it |
|---|---|
| `sparse_ret` | **the only number that matters.** 60.00 = perfect. Real deliveries. |
| `ent` | policy entropy. `1.79` = uniform over 6 actions (`ln 6`). Falling to ~0.5 means it committed. Stuck at 1.79 = never learned. Near 0 very early = collapsed. |
| `clip` | fraction of the batch the PPO clip bit. Healthy 0.05–0.20. Pinned near 0.45 means the policy is moving far too fast — that is what the all-zero-reward bug looked like. |
| `vloss` | tracks **return magnitude**, so higher is *good* here. Winners hit 13–26. A run stuck near 0.3 is collecting almost no reward. |
| `ploss` | policy loss. Sign and magnitude are not very diagnostic alone. |
| `shape_c` | shaping coefficient. Should stay near 1.0 — the horizon is 10x the run. |
| `ent_c` | current entropy coefficient, annealing 0.2 → 0.01. |
| `per-layout:` | multi-layout runs only. Per-kitchen breakdown, so the aggregate cannot hide which ones are failing. |

---

## 15. FILE MAP

| file | what it does |
|---|---|
| `SP/self_play.py` | the runner — PPO update, training loop, argparse, `action_probs()` |
| `SP/run_specialists.sbatch` | **25 policies, one per layout** (`--array=0-24%10`). This produced the working baselines. |
| `SP/run_generalist.sbatch` | 1 policy, all 25 layouts, padded canvas. **Known to fail** — see §8. |
| `SP/run_self_play.sbatch` | older combined script the two above derive from. Prefer the standalone ones. |
| `SP/eval_checkpoints.py` | load + **play** checkpoints, report real delivery reward |
| `SP/run_eval_checkpoints.sbatch` | run the above on a compute node |
| `SP/CARC_RUNS.md` | this document |
| `utils/env_wrapper.py` | env, multi-layout pool, padding, `_DenseShaper`, both MDP traps handled |
| `utils/buffer.py` | GAE + `bad_masks`, chunked recurrent generator, advantage-noise guard |
| `utils/features.py` | the 23-channel observation builder |
| `utils/cnn.py` | conv trunk → `Linear(32*W*H, 64)`. **Source of the 1:1 shape binding.** |
| `utils/rnn.py` | GRU layer, handles both rollout and training branches |
| `utils/act.py` | Categorical action head, `distri()` for raw distributions |
| `algorithm/r_actor_critic.py` | `R_Actor`, `R_Critic` |
| `algorithm/rMAPPOPolicy.py` | policy container + the two optimizers |

---

## 16. JOB IDS AND GOTCHAS

| job | what |
|---|---|
| `10789711` | generalist training (failed — §7) |
| `10789712` | specialist array, tasks 0–24, **seed 1** (the real baselines) |
| `10820032` | final eval, argmax, 10 episodes, all 25 |
| `10820033` | final eval, sampled, 10 episodes, all 25 |

### Array task index → layout

The array order is **INFL-ranked** from `fov/layouts_final/LAYOUTS.md` (highest
robot-influence first), so task 0 is `steak_gc00` (INFL 28.5) and task 24 is
`steak_gs05` (INFL 8.5):

```
 0 gc00    1 gc06    2 gc04    3 cram2      4 cram
 5 gs00    6 mid_1   7 gc03    8 gc05       9 side_3
10 gs07   11 side_4 12 api    13 gc07      14 gc01
15 gs03   16 gs04   17 parrallel  18 gs02  19 gs08
20 mid_2  21 gs09   22 none_3 23 tshape    24 gs05
```

`steak_gs01` is deliberately excluded from Tier 1 — it fails the team-win gate
at fov30 in the layout battery.

### SLURM assigns running array tasks their own job IDs

Seeing extra `sp_specialist` job IDs in `squeue` is **normal** and does not mean
duplicate submissions. For example `10789724` is really `10789712_11`. Confirm
with:

```bash
sacct -j <mystery-id> --format=JobID
```

### Other things that bit us

* `--export` splits on commas (see [§10](#10-how-to-verify-checkpoints-yourself))
* `--num_env_steps` is `type=int` in argparse — pass `10000000`, **not** `1e7`.
  `--reward_shaping_horizon` and `--entropy_coef_horizon` are floats and do
  accept `1e8` / `1e7`.
* macOS `tar` writes `._*` AppleDouble files; delete them after any sync or
  python will try to import them.
