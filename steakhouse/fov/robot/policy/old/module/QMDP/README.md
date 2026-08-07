# QMDP — a partner-model cost on top of a frozen self-play policy

Everything in this folder is new. **Nothing outside this folder is modified.**
The three trees it reads are used as-is:

| what | where | used for |
|---|---|---|
| SP baseline (net, features, checkpoints) | `fov/robot/policy/new/official_baselines/` | the action distribution |
| the limited-vision human | `fov/human/agent/limited_vision_human.py` | the partner, and the shadow model |
| the joint (FOV, subtask) filter | `fov/robot/policy/old/inference/bayes_fov_sampling.py` | the posterior |

---

> **New here? Read `RESULTS.md` §1 — it explains the whole setup from zero,
> then §2 for the result and §6 for the caveats.**

## 1. One paragraph

A PPO robot trained by **self-play** is paired with `LimitedVisionSteakHuman`,
a scripted cook whose **field of view is hidden** and whose beliefs decay. The
robot has never met this partner and cannot see the parameter that makes it
different — and it shows: paired with this human the trained robot is **beaten
by a uniform-random robot**. This module layers a partner-aware second opinion
on the *frozen* policy:

```
p_final  ∝  p_baseline · p_module^beta        beta = 0 IS the baseline, bit for bit
```

`p_module` comes from a cost function that reads **no reward, no recipe, no
order count, no step budget** — only the partner's inferred **field of view**
and **subtask**, from `SamplingBayesFOVInference` running on the human's
observed primitive actions. Held out on **1965 paired episodes** across 11 layouts
and 6 FOVs: **−10.0 ticks, +0.001 deliveries, 1044 win / 850 loss,
p = 9.4e-6.**

Two ideas in the cost:

> **1. Do not do the job your partner is already doing.**
> **2. When you do change your partner's world, prefer changes they can SEE.**

**Physical collision avoidance is deliberately out of scope** — an earlier
version had terms charging the robot for blocking the partner's footstep; they
were removed. Everything here is task- and belief-level.

## 2. Why observability is the right axis

`limited_vision_human.py`'s own mechanism is entirely visibility-gated:

* `_commit_still_useful` drops an errand only on a **concrete** belief, and
  concrete beliefs are written only inside the cone. A human who **sees** the
  robot fill the pot re-plans mid-trip and loses nothing; a human who does not
  walks the whole wasted errand (`n_wasted_commits`).
* `_weights` and `_available_advancing` yield tasks the human **sees** the
  teammate carrying or standing at — again, only in-cone.

So the identical robot action is nearly free against a wide cone and expensive
against a narrow one, and the asymmetry is exactly what the filter estimates.
Measured: `approach` is live on 9 % of ticks at fov 30 and ~0 % at fov 360;
`legible_yield` runs the opposite way, 2 % → 9 %.

## 3. The pipeline, one tick

```
s_t ──► shadows perceive s_t                          human_model.sync_observe
    ──► predictive posterior b(theta, tau)            human_model.predict
    ──► for each of the 6 primitives:                 qmdp.QMDPFOVModule
            s_a   = T(s, (a, h))
            s_ref = T(s, (STAY, h))
            cost(s_a vs s_ref | theta, tau)           cost.ObservableDivergenceCost
        Q(s,a) = -SUM_h P(h) SUM_theta,tau b . cost
    ──► p_mod = softmax(Q)
    ──► log p = log p_base + beta . log p_mod         policy.BlendedRobotPolicy
    ──► argmax (or sample)
h_t ──► filter consumes (s_t, h_t)                    human_model.update
```

**Causality.** The robot decides at `s_t` knowing states `s_0..s_t` and human
actions `h_0..h_{t-1}` — never `h_t`. They move simultaneously. Pinned by
`test_no_cheating.py` T4.

## 4. The cost terms

Every one is a **difference** between two futures that share the same human
action, so anything the human or the pots did on their own cancels and what is
left is attributable to the robot.

    s_a   = T(s, (a, h))       the robot takes candidate action a
    s_ref = T(s, (STAY, h))    the robot does nothing

| term | sign | fires on |
|---|---|---|
| `task_overlap` | cost | robot engages the station **line** (pot/board/sink) the posterior says the partner is on, relative to the lines they left alone. **Centred** — the uncentred version is just "walk toward stations" and wrecks the policy |
| `blindside` | cost | robot changes a station, human ends up confidently **wrong** about it |
| `silent_invalid` | cost | robot breaks the precondition of the partner's inferred subtask, unseen |
| `visible_invalid` | benefit | the same break, **in view** — legible take-over |
| `restored` | benefit | robot re-enables a subtask the partner wanted |
| `strand` | cost | the partner's own policy would put more mass on look/wait |
| `shift` | either | raw TV movement of the partner's subtask distribution |
| `legible_yield` | benefit | partner moves off a task **because it can see** the robot doing it |
| `approach` | cost | robot walks toward a contested station the partner cannot watch |

`UNKNOWN` is never a blindside: a human who knows it does not know goes and
looks. Only a belief that is both **confident** and **wrong** buys a wasted trip.

**Adaptive deference.** `blindside`, `silent_invalid`, `approach` and
`task_overlap` are scaled by `defer = 1 - P(the partner is on
explore/check_*/wait)`, read off the posterior. Deferring to a partner who has
no plan is pointless, so against a blind, still-mapping partner the module goes
quiet and the baseline runs unmodified. Nothing hand-codes that schedule; it
falls out of the human model.

## 5. What is and is not readable

**Readable** — the world state (the robot is fully observable; that is what
`build_full_state` hands its network), the human's emitted **action** each tick
(a physical event anyone in the kitchen can watch), the trained weights, and the
human agent's **own code** as a shadow model.

**Not readable** — the human's true FOV, the human's subtask label
(`info["subtask"]`), the real human's beliefs, and any environment reward.

`test_no_cheating.py` proves this behaviourally rather than by inspection: it
swaps a decoy FOV in, scrambles the subtask label, and wrecks the reward
parameters, and demands the robot's action sequence come out bit-identical.

## 6. Files

```
_paths.py            sys.path shim + checkpoint locations + the 11 usable layouts
env.py               one kitchen: SP policy at index 0, human at index 1
baseline.py          load the frozen checkpoint; obs -> 6 probabilities
human_model.py       filter + one-step prediction + counterfactual perception
cost.py              THE COST FUNCTION
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
RESULTS.md           the numbers
```

## 7. Running it

```bash
# laptop (needs a checkpoint in QMDP/ckpt/ or STEAK_SP_CKPT_ROOT set)
python evaluate.py --layouts steak_gc00 --fovs 30,90,360 \
                   --lams 0.0,1.0 --seeds 0-9 --sample --out r.jsonl
python aggregate.py r.jsonl --by fov --arms lam

# the reference that matters: a UNIFORM-RANDOM robot (see RESULTS.md section 0)
python evaluate.py --layouts steak_gc00 --fovs 30,90,360 --lams 0.0 \
                   --seeds 0-9 --sample --base_temperature 1000 --out r.jsonl

# named weight presets
python evaluate.py ... --weights division   # task_overlap only
python evaluate.py ... --weights belief     # the eight belief-level terms only

# CARC (submit from endeavour: the discovery login node was refusing exec)
QMDP_BETA=1.0 GRID=final sbatch -M discovery --array=0-21%18 run_qmdp.sbatch
```

`--lams 0.0` **is** the baseline arm — the identical code path with the pooling
weight at zero, not a re-implementation — so every comparison is paired on
(layout, fov, seed) and the difference is the module. Verified: with all cost
weights zero the two arms are bit-identical (`test_no_cheating.py` T5), and the
`uniform` ablation ties on all 1056 cells.

Key flags worth knowing:

| flag | why |
|---|---|
| `--sample` | sample the pooled action instead of argmax. **Use it** — argmax makes the baseline loop and inflates the module's margin (RESULTS.md §0a) |
| `--base_temperature` | temperature on the baseline's own softmax, applied to **both** arms. `1000` = a uniform-random robot |
| `--normalize_q` | rescale module scores to unit spread so `beta` means the same strength for any weight vector — required for a fair decomposition |
| `--module_mode` | `real` / `uniform` / `noise` / `shuffle` / `fixed:<fov>` — the controls |
