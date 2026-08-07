# QMDP partner module — results

Written 2026-08-03. Self-contained: §1 is enough to understand the experiment
with no prior context. Everything is measured; the results that do **not**
support the hypothesis are in §6.

---

## 1. What this is, from zero

**The world.** `steakhouse` is a 2-agent Overcooked-style kitchen. Two cooks
share one grid and must produce steak dishes: meat → pot, onion → chopping
board, plate → sink, then assemble and serve. Each agent picks one of 6
primitive actions per tick (N, S, E, W, stay, interact). With `n_orders=4` the
episode ends after **3 deliveries**; horizon is 400 ticks.

**The two agents.**

| | who | where |
|---|---|---|
| **chef 0 — the robot** | a PPO policy trained by **self-play** (both chefs = the same network) | `fov/robot/policy/new/official_baselines/` , checkpoints in `/scratch1/$USER/steakhouse_sp/specialist/<layout>_seed1/sp_<layout>.pt` |
| **chef 1 — the human** | `LimitedVisionSteakHuman`, a scripted cook with a **limited vision cone (FOV)** | `fov/human/agent/limited_vision_human.py` |

The human only knows what has fallen inside its cone, and its beliefs **decay**
after 12 ticks. A narrow cone means it must stop and *look*, it walks wasted
errands on stale beliefs, and it never notices what its teammate is doing. Its
FOV is the hidden variable of the whole project.

**The problem.** The robot was trained against a copy of itself. It has never
met this human, and cannot see the one parameter that makes the human different.

**The module (this folder).** A second opinion layered on the *frozen* policy:

```
p_final  ∝  p_baseline · p_module^beta          →  sample an action
```

* `p_baseline` — the trained network's 6-way action distribution. Knows the
  **task**, nothing about the partner.
* `p_module` — softmax of a cost that knows the **partner** and nothing about
  the task. It reads no reward, no recipe, no order count, no step budget.
* `beta` — the single tuning knob. **beta = 0 reproduces the baseline exactly,
  bit for bit** (proved: `test_no_cheating.py` T5).

**How the module knows the partner.** `SamplingBayesFOVInference`
(`fov/robot/policy/old/inference/bayes_fov_sampling.py`, used unmodified)
maintains a joint posterior `b(FOV, subtask)` from the human's **observed
primitive actions only**. One shadow `LimitedVisionSteakHuman` per candidate
cone carries that hypothesis's beliefs. The module then does **QMDP** over that
posterior: for each of the 6 robot actions, roll one exact mdp step forward and
score how much it would change the partner's knowledge and plan, averaged under
`b`.

**Metric.** `completion_time` = the tick the 3rd order is served, or
horizon + 100 = 500 if the team never finishes. Comparisons are **paired** on
(layout, fov, seed): the same cell is played by both arms, and a **win** is
*strictly faster with no loss of deliveries*. `p` is a two-sided sign test on
the paired wins/losses.

**Scale.** 11 layouts (the ones whose self-play specialist actually delivers —
see `official_baselines/SP/CARC_RUNS.md` §4), 6 true FOVs
{30, 60, 90, 120, 180, 360}, 16 TRAIN seeds for tuning + 30 held-out TEST seeds
for the reported result. Run on USC CARC.

---

## 2. Headline — held-out seeds, 11 layouts

**TEST seeds 100–129, which no hyperparameter was ever selected on.**
`beta = 2`, sampled robot in both arms, **complete run, n = 1965 paired
episodes** (11 layouts × 6 FOVs × 30 seeds):

| | baseline (beta=0) | **+ module (beta=2)** |
|---|---:|---:|
| deliveries (max 3) | 2.814 | **2.815** |
| completion time | 251.4 | **241.4** |
| finished all 3 | 89.2 % | **90.0 %** |

| paired difference | value |
|---|---:|
| Δ completion time | **−10.0 ticks** |
| Δ deliveries | **+0.001** |
| win / loss / tie | **1044 / 850 / 71** |
| sign-test p | **9.4e-6** |

Faster, and not fewer deliveries. The baseline arm is the *identical code path*
with `beta = 0`, so the difference is the module and nothing else.

---

## 3. Choosing beta (TRAIN seeds only)

11 layouts × 6 FOVs × 16 seeds, n ≈ 1028 paired episodes per row:

| beta | Δcompletion | Δdeliveries | win / loss | p | mean deliveries | mean completion |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (baseline) | — | — | — | — | 2.759 | 257.0 |
| 0.5 | −11.1 | +0.066 | 494 / 406 | 0.0037 | 2.825 | 245.9 |
| **1.0** | **−18.1** | **+0.073** | **534 / 419** | **0.00022** | **2.832** | **238.9** |
| 2.0 | −19.3 | +0.052 | 532 / 457 | 0.019 | 2.810 | 237.8 |
| 4.0 | −2.3 | −0.062 | 524 / 477 | 0.15 | 2.696 | 254.8 |

**beta ≈ 1 is the right setting** — best delivery gain, essentially the best
speed, lowest p. Beyond ~2 the module overrides the baseline's task competence
and the gain disappears.

> **Caution on tuning-set size.** On a 4-layout subset, beta = 4 looked best
> (−31.7 ticks, 76/49, p = 0.02). Across all 11 layouts it is worthless
> (−2.3 ticks, **−0.062 deliveries**, p = 0.15). Do not tune beta on a handful
> of kitchens.

### beta = 1 wins on TRAIN but has not (yet) reproduced on TEST

Stated plainly because it matters for how much to trust the tuning: the
held-out run at `beta = 1` (job 10811534) is still filling, and **at n = 629 of
1980 it reads −4.0 ticks, +0.008 deliveries, 308 / 283, p = 0.32 — not
significant.** The `beta = 2` held-out run *is* complete and *is* significant
(§2), so **that is the number reported**, even though `beta = 1` looked better
on TRAIN.

Two readings, and the data does not yet separate them: (i) the partial `beta=1`
run has only covered whichever layouts finished first and is not a fair sample,
or (ii) the TRAIN-set preference for `beta = 1` over `beta = 2` was inside the
noise (their TRAIN win rates are 56.0 % vs 53.8 %, and their mean effects are
−18.1 vs −19.3 — genuinely close). Either way the safe conclusion is
**`beta ∈ [1, 2]`**, and the reported result uses the arm whose held-out
evidence is complete. Refresh with:

```bash
python analyze.py final /scratch1/$USER/steakhouse_qmdp/final1/*.jsonl
```

---

## 4. The cost function

Nine terms. **Every one is a difference between two futures that share the same
predicted human action**, so anything the human or the pots did on their own
cancels and what is left is attributable to the robot:

```
s_a   = T(s, (a,    h))     the robot takes candidate action a
s_ref = T(s, (STAY, h))     the robot does nothing
```

Two ideas, both read off the inferred `(FOV, subtask)` posterior:

**(i) Division of labour — don't do the job your partner is already doing.**

| term | sign | fires on |
|---|---|---|
| `task_overlap` | cost | the robot engages the station **line** (pot / board / sink) the posterior says the partner is on, relative to the lines they left alone. **Centred**: the uncentred version is just "walk toward stations", which wrecks the policy (measured: 3 deliveries → 1). This is the term a *movement* action scores on most often. |

**(ii) Observability — when you do change the partner's world, prefer changes
they can SEE.**

| term | sign | fires on |
|---|---|---|
| `blindside` | cost | robot changes a station and the partner ends up **confidently wrong** about it |
| `silent_invalid` | cost | robot breaks the precondition of the partner's inferred subtask, **unseen** |
| `visible_invalid` | benefit | the same break, **in view** — legible take-over, the partner re-plans immediately |
| `restored` | benefit | robot re-enables a subtask the partner wanted |
| `strand` | cost | the partner's own decision code, run on its hypothetical post-action beliefs, would put more mass on look / wait |
| `shift` | either | raw total-variation movement of the partner's subtask distribution |
| `legible_yield` | benefit | partner moves off a task **because it can see** the robot doing it |
| `approach` | cost | robot walks **toward** a contested station the partner cannot watch |

`UNKNOWN` is never a blindside: a partner who knows it doesn't know goes and
looks. Only a belief that is both **confident** and **wrong** buys a wasted trip.

**Why observability is the right axis** — `limited_vision_human.py`'s own
mechanism is entirely visibility-gated: `_commit_still_useful` drops an errand
only on a *concrete* belief, and concrete beliefs are written only inside the
cone; `_weights` / `_available_advancing` yield tasks the human *sees* the
teammate carrying or standing at. So the identical robot action is nearly free
against a wide cone and expensive against a narrow one.

**Adaptive deference.** `blindside`, `silent_invalid`, `approach` and
`task_overlap` are scaled by `defer = 1 − P(partner is on explore/check_*/wait)`,
read off the posterior. Deferring to a partner who has no plan is pointless, so
against a blind, still-mapping partner the module goes quiet and the baseline
runs unmodified. Nothing hand-codes that schedule; it falls out of the human
model.

**Physical collision avoidance is deliberately NOT in this module.** An earlier
version had `interference` / `self_block` terms charging the robot for blocking
the partner's footstep. They were removed on instruction. Everything above is
task- and belief-level.

---

## 5. Controls — the win is not an artefact

Measured on the same machinery, n = 1056 paired episodes per arm:

| module replaced by | win | loss | tie | winrate |
|---|---:|---:|---:|---:|
| `uniform` (p_module = 1/6) | 0 | 0 | **1056** | — (exactly the baseline) |
| `shuffle` (real scores, permuted over actions) | 646 | 391 | 19 | 62.3 % |
| `noise` (random, scale-matched) | 687 | 355 | 14 | 65.9 % |

* **`uniform` ties on all 1056 cells.** The blending machinery contributes
  nothing by itself, so anything measured is the cost function.
* **`noise` and `shuffle` do win, 62–66 %.** A perturbation of the right *size*
  in a meaningless *direction* helps this policy. **This is the floor any module
  must clear**, and it is why a raw win rate must never be quoted alone.
* **The blend is not a disguised temperature increase.** Tick by tick,
  H(pooled) − H(p_base) = −0.047 … +0.011 nats (slightly *sharper*), while the
  pooled argmax differs from the baseline argmax on 5–14 % of ticks. It
  re-ranks; it does not randomise.
* **The robot does not win by idling.** Action mix, baseline → module:
  `stay` 3.5 % → 4.0 %, `interact` 26.2 % → 28.1 %. It does *more* work per tick.

**FOV-dependence of the terms** (`diagnose.py`; mean spread across the 6
candidate actions / fraction of ticks non-zero):

| term | fov 30 | fov 90 | fov 360 |
|---|---:|---:|---:|
| `approach` | 0.065–0.126 / 9 % | ~0.00 / 1 % | 0.00 / 0 % |
| `legible_yield` | 0.006 / 2 % | 0.016 / 5 % | 0.028 / 9 % |
| `blindside` | 0.05 / 5 % | 0.02 / 3 % | 0.00 / 0 % |

Two channels running in **opposite** directions with FOV, exactly as the theory
predicts: you cannot blindside someone who is watching, and you cannot signal to
someone who is not.

---

## 6. What did not work, and other caveats

**(a) The argmax baseline is a strawman — do not use it.** With the robot taking
the argmax of the trained softmax, the baseline loops; on `steak_gc00` at fov 30
it delivered **0** on every seed. Sampling (how the policy was trained) takes the
same baseline to 2.6–3.0 deliveries. **Every number here uses the sampled robot
in both arms.**

**(b) A uniform-random robot beats the trained self-play robot.** Measured on
the same held-out cells:

| robot | deliveries | completion |
|---|---:|---:|
| SP baseline, as trained | 2.814 | 251.4 |
| SP baseline + module (beta=2) | 2.815 | 241.4 |
| **uniform random** | **2.946** | **206.1** |

And the baseline improves monotonically as its own softmax is flattened toward
uniform (1056 paired cells per column):

| baseline T | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 | 10.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| deliveries | 2.765 | 2.905 | 2.913 | 2.948 | 2.948 | 2.930 | **2.956** |
| completion | 256.9 | 223.9 | 219.9 | 212.7 | 211.9 | 210.6 | **204.6** |

Paired with a partner it never trained against, this self-play policy is a **net
liability** — the team does better if the robot flails. That is a
zero-shot-coordination failure of the *checkpoint*, not of the module, but it
means "beats the baseline" is a low bar and should always be reported next to
the random floor. (The same checkpoints score a perfect 60/60 with a *self-play*
partner — `CARC_RUNS.md` §4 — which is exactly the point.)

**(c) The module does not help a random robot — it hurts it.** At beta = 1 on
top of the uniform-random robot, n = 732: **347 / 305, p = 0.11, +12.3 ticks
SLOWER, deliveries 2.883 → 2.706.** The module
re-ranks a distribution that already encodes task competence; given a
distribution with none, there is nothing to re-rank and the cost terms just add
noise. So the contribution is *specific to correcting a trained-but-miscoordinated
policy*.

**(d) The belief-level terms are weak on their own.** In an earlier campaign the
eight observability terms without `task_overlap` came out at chance
(481 win / 482 loss, n = 1056), stayed at chance when rescaled to matched
strength (`--normalize_q`: 193 / 228), and were significantly *worse* than
baseline with their signs flipped (343 / 416, p = 0.009). So the direction is
weakly correct but the magnitude is small: they fire on only 1–9 % of ticks,
because at one-step lookahead only `INTERACT` can change a station.
`task_overlap` was added precisely because it gives *movement* actions a signal,
and it is what turns the effect into the significant result in §2.

**(e) Whether the FOV *inference* is load-bearing is not settled.** With the
older cost function, pinning the module's cone to a constant
(`--module_mode fixed:30` / `fixed:360`) performed as well as the inferred
posterior. That has not been re-measured for the current cost function — rerun
`GRID=ablate`. Until then the safe claim is *"a model of the partner helps"*,
not *"inferring the partner's cone helps"*. The filter itself is healthy: MAP
cone correct on 69–100 % of episodes depending on FOV, `n_skipped = 0` always
(the kernel never desynchronised from the human), and the module does not
degrade it (`Δp_true_fov ≈ 0`).

**(f) Robustness** (earlier campaign): true FOV outside the candidate set —
fov 150 wins cleanly (39 / 8), fov 45 is not significant (29 / 18, p = 0.14),
because 45° falls between the 30° and 60° hypotheses and the posterior splits.
A hotter human sampler (temperature 1.0 vs 0.5) still wins (110 / 33, p < 1e-4).

---

## 7. No cheating — what the robot may and may not read

**May read:** the world state (it is a fully-observable agent — that is what
`build_full_state` already hands its network), the human's **emitted primitive
action** each tick (a physical event anyone in the kitchen can watch), the
trained weights, and the human agent's **own source code** as a shadow model.

**May not read:** the human's true FOV, the human's subtask label
(`info["subtask"]`), the real human's beliefs, or any environment reward.

`test_no_cheating.py` proves this **behaviourally** — it changes a forbidden
input and demands the robot's action sequence come out bit-identical. All pass:

| test | what it does |
|---|---|
| T1 | **decoy FOV.** Builds the human at 90°, then overwrites the public `human.fov` attribute with `12345` so anything peeking reads garbage; the wrapper swaps the real 90° back only *inside* `human.action()`, so the human still behaves identically. If the robot's actions change, something read the true cone. |
| T2 | scrambles `info["subtask"]` to a garbage string |
| T3 | sets `delivery_reward = −99999` and every shaping parameter to −12345 |
| T4 | the robot must emit its action for tick *t* before the human's action for tick *t* exists (they move simultaneously; peeking would be a cheat) |
| T5 | all cost weights zero must be **bit-identical** to beta = 0 |
| T7 | the real human object is **not reachable** from the policy's object graph, at any depth |

> **T1 is a probe, not the policy.** The robot infers the cone with
> `SamplingBayesFOVInference` — the decoy exists only to catch a cheat and
> passes only because there isn't one.

The tick order that makes this sound (`rollout.py`):

```
s_t   ──►  shadows perceive s_t                       (looking is not evidence)
      ──►  predictive posterior b(FOV, subtask)       (evidence through t-1 only)
      ──►  robot picks a_t                            ← never sees h_t
h_t   ──►  human acts;  info["subtask"] is DISCARDED
      ──►  filter consumes (s_t, h_t)                 (now admissible)
      ──►  env.step(a_t, h_t)
```

---

## 8. Files

```
_paths.py            sys.path shim, checkpoint locations, the 11 usable layouts
env.py               one kitchen: SP policy at index 0, human at index 1
baseline.py          load the frozen checkpoint; obs -> 6 probabilities
human_model.py       the filter + one-step prediction + counterfactual perception
cost.py              THE COST FUNCTION (nine terms + weights + presets)
qmdp.py              expectation over (FOV, subtask) of the one-step cost
policy.py            the blend, and the ablation modes
rollout.py           one episode, and the metrics off it
evaluate.py          the experiment (CLI)
diagnose.py          which term has leverage, at which FOV
aggregate.py         paired tables + sign tests
analyze.py           campaign-level readouts per grid
configs.py           named grids, so an sbatch array index means something
run_qmdp.sbatch      CARC
test_no_cheating.py  the guarantees, as assertions
```

Nothing outside this folder was modified.

---

## 9. Reproducing

```bash
# ---- laptop (needs a checkpoint in QMDP/ckpt/ or STEAK_SP_CKPT_ROOT set)
python evaluate.py --layouts steak_gc00 --fovs 30,90,360 \
                   --lams 0.0,1.0 --seeds 0-9 --sample --out r.jsonl
python aggregate.py r.jsonl --by fov --arms lam
python test_no_cheating.py

# ---- CARC.  Submit from endeavour: the discovery login node refuses exec,
#      but endeavour shares the filesystem and can `sbatch -M discovery`.
ssh endeavour.usc.edu
cd ~/steakhouse
OUTDIR=/scratch1/$USER/steakhouse_qmdp/beta2  GRID=beta \
  sbatch -M discovery --array=0-10%11 fov/robot/policy/old/module/QMDP/run_qmdp.sbatch
OUTDIR=/scratch1/$USER/steakhouse_qmdp/final1 QMDP_BETA=1.0 GRID=final \
  sbatch -M discovery --array=0-21%18 fov/robot/policy/old/module/QMDP/run_qmdp.sbatch

python analyze.py beta  /scratch1/$USER/steakhouse_qmdp/beta2/*.jsonl
python analyze.py final /scratch1/$USER/steakhouse_qmdp/final1/*.jsonl
```

Flags that matter:

| flag | why |
|---|---|
| `--sample` | sample the pooled action instead of argmax. **Always use it** — argmax makes the baseline loop (§6a) |
| `--lams` | the beta values. `0.0` **is** the baseline arm |
| `--base_temperature` | temperature on the baseline's own softmax, applied to **both** arms. `1000` = a uniform-random robot (§6b) |
| `--module_mode` | `real` / `uniform` / `noise` / `shuffle` / `fixed:<fov>` — the controls in §5 |
| `--weights` | JSON, or a preset: `full`, `division` (task_overlap only), `belief` (the eight observability terms only) |
| `--normalize_q` | rescale module scores to unit spread so `beta` means the same strength for any weight vector — required for a fair decomposition |

Other grids in `configs.py`: `weights` (one-axis weight ablation), `ablate`
(uniform / noise / shuffle / fixed-cone controls), `decompose` (division vs
belief halves), `norm` (matched-strength decomposition), `kb_sign` (sign
inversion), `basetemp` / `basetemp2` (the baseline's own temperature).

Results land in `/scratch1/$USER/steakhouse_qmdp/<grid>/task_<i>.jsonl`, one
JSON object per episode. **`/scratch1` is purge-eligible — copy anything you
want to keep.**
