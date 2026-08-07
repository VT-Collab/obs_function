# ZSC baselines — wave 1

## ✅ TRAINING COMPLETE — 147/147, every SLURM task `COMPLETED`, zero failures

**Status 2026-08-05.** All 7 layouts × (11 SP + 5 E3T + 5 SP-ε) trained and
backed up. 4 of 7 evaluated against the FOV human, in two collision conditions.

```
layout   SP / E3T / SP-eps      evaluated vs FOV human?
gc00     11 / 5 / 5  ✓           YES, both collision arms
gc03     11 / 5 / 5  ✓           YES, both collision arms
gc04     11 / 5 / 5  ✓           YES, both collision arms
gc06     11 / 5 / 5  ✓           YES, both collision arms
cram     11 / 5 / 5  ✓           NOT YET  <- CONTENTION layout \  the two that
cram2    11 / 5 / 5  ✓           NOT YET  <- CONTENTION layout /  matter MOST
gs00     11 / 5 / 5  ✓           NOT YET
```

**The single most useful thing anyone can do next** is run the eval on `cram`,
`cram2`, `gs00` — ~25 min, and it decides whether E3T is genuinely the better
method or only better on clustered kitchens (§5, §7).

> ## 🏆 **[BEST_BASELINES.md](BEST_BASELINES.md) — which checkpoint to load, per layout × fov**
> Best seed, its completion time, median and worst seed, grid size, and the
> human-alone reference. From 8,820 paired episodes (job `10839692`).
> **The seed matters 53× more than the algorithm** — spread across algo means is
> 7.5 ticks, across seed cells 397.

Jobs: `10820346` SP · `10820347` E3T · `10820348` SP-ε ·
`10832597` eval (collisions on) · `10832978` eval (collisions off) ·
`10832704` self-play control · `10832972` collision diagnostic.

Companion to `SP/CARC_RUNS.md`, not a replacement. That document describes the
**seed-1 specialist runs**, which still exist, still work, and are still what
`filter/baseline.py` loads. This one describes the **multi-seed ZSC-parity
re-run**, the two new algorithms beside it, and the FOV-human evaluation.

---

# 0. WHAT THIS PROJECT IS  (start here if you have never seen it)

### The research question

A robot chef works beside a human chef who has a **limited field of view**. The
human cannot see the whole kitchen, so it acts on stale or wrong beliefs — it
walks to a pan that is already full, because from where it is standing it cannot
see that. **Can the robot infer what the human can and cannot see, from the
human's behaviour alone, and act on that inference?**

Three pieces:

| piece | what it is | where |
|---|---|---|
| **the human** | `LimitedVisionSteakHuman` — a **scripted** (not learned) agent with a vision cone, decaying beliefs, and commitment. The thing under study. | `fov/human/agent/limited_vision_human.py` |
| **the baselines** | Robots trained with **zero exposure** to that human. What this document is about. | `fov/robot/policy/new/official_baselines/` |
| **the method** | Misha's own FOV filter: a Bayesian posterior over the human's FOV, estimated from observed actions, used to re-rank the robot's own top-K actions. **Hand-written, do not modify.** | `fov/robot/policy/new/filter/` |

The baselines exist to answer *"is the filter actually necessary?"* If a plain
ZSC method already cooperates fine with a limited-FOV partner, there is no paper.

### The task — an 8-step chain

```
meat → pan → cook → onion → chop → plate → wash → plate steak → garnish → deliver
```

`n_orders=4`, and the MDP terminates at **one order left**, so a team delivers
**3**. `delivery_reward=20`, hence **60.00 = a perfect episode**. Horizon 400
ticks. 6 actions: N, S, E, W, stay, interact.

This is **not** the standard Overcooked soup task — it is longer (8 steps vs 4)
and this fork only pays shaped reward for 2 of the 8, which is why
`utils/env_wrapper.py::_DenseShaper` exists (§8).

### The human, in one paragraph

Scripted, not learned. It sees only what falls inside its FOV cone, writes what
it sees into beliefs stamped with a timestamp, and **forgets** them after
`forget_horizon` ticks. It picks a subtask by softmax over priority
(`temperature=0.5`), **commits** to it, and abandons it only when its cone
*reveals* the errand is pointless — so a wide cone re-plans early while a narrow
cone completes the wasted trip. That asymmetry **is** the FOV effect. It routes
by its own BFS over floor it has actually seen — never a global planner, which
was the single biggest cheat removed from earlier versions. Two opt-in flags
default off: `avoid_robot` (detour around the teammate's seen cell) and
`occlude` (line-of-sight, so the cone cannot see through walls).

### The layouts

`fov/layouts_final/` holds a validated 31-layout library. `LAYOUTS.md` ranks
them by **INFL** — how much *seeing the robot* changes the human's behaviour,
i.e. how much room there is for collaboration to matter. Contention layouts
score highest. **The 7 layouts trained here are the intersection of "high INFL"
and "self-play can actually solve it"**, which turned out to be the same set.

### Sibling documents — read these too

| doc | what it covers |
|---|---|
| `SP/CARC_RUNS.md` | the **OLD** seed-1 specialist runs (25 layouts, n=1). Still accurate about those. This document supersedes its *conclusions*, not its facts. |
| `fov/layouts_final/LAYOUTS.md` | the layout library, the INFL ranking, why these 7 |
| `fov/layouts_final/STATUS.md` | where the human-agent work stands |
| `filter/*.py` docstrings | the method. Hand-written by Misha; **never edit, never port QMDP into it** |

---

# IF YOU READ ONLY ONE THING

1. **147 policies were trained with zero FOVHuman exposure** — SP (11 seeds ×
   7 layouts), E3T (5 × 7), SP-ε (5 × 7). All at ZSC-Eval parity.
2. **Against the FOV human they range from "no better than random" to "much
   worse"**, and most of the "much worse" turned out to be **physical
   collisions**, not bad coordination. Two eval arms exist for exactly this.
3. **With collisions off, E3T is the best arm** on the four clustered layouts.
4. **But E3T collapses in training on `cram`** (33.1±27.3 vs SP's 53.7±7.2), and
   `cram`/`cram2` are the contention layouts the FOV result rests on. **They are
   not evaluated yet.** Do not report a winner until they are.
5. **Report completion time, not reward.** Reward saturates at 60 — the FOV
   human solo-solves the small kitchens.
6. **fov90 and fov120 are the same condition** (98% bit-identical episodes).
   Your FOV axis has 5 levels, not 6.

---

# CONTENTS

| § | what |
|---|---|
| 0 | **What this project is** — the research question, the task, the human, the layouts. Start here if you are new. |
| — | [Headline result + the collision artifact](#-the-headline--and-the-thing-that-almost-fooled-us) |
| — | [The collision decision — read before reporting](#️-the-collision-decision--read-before-reporting-any-number) |
| — | [At a glance: paths, **where checkpoints live**, what is new](#️-at-a-glance) |
| 1 | Why this re-run exists · **seeds do not combine** |
| 2 | What each algorithm is (SP / E3T / SP-ε) |
| 3 | What changed vs the seed-1 config |
| 4 | What each run writes · **how to load a checkpoint** · the shape trap |
| 5 | Training results (self-play score) |
| 6 | The eval harness · the tick order · metrics · results location |
| 7 | What comes next |
| 8 | The environment · **four traps that destroy runs** · dense shaping |
| 9 | How to re-run any of it |
| 10 | File map (new vs unchanged) |
| 11 | Operational notes |
| 12 | Glossary |

---

# 🔴 THE HEADLINE — AND THE THING THAT ALMOST FOOLED US

The first evaluation said: *every baseline is catastrophically worse than a
random robot*. That was **~80% an artifact of physical collisions**, not a
coordination result. Both arms, help-or-hinder vs the random floor, in ticks
(negative = the trained robot helped):

```
                     COLLISIONS ON                    COLLISIONS OFF
              fov30   fov90  fov360   |   fov30   fov90  fov360
gc00  sp     +117.2   +53.8    +4.9   |   +25.6   +26.5    -7.6
gc00  e3t    +134.9   +18.9   -17.0   |    -4.7    +2.7   -18.2
gc03  sp     +124.9   +82.2  +100.6   |    +2.4    +5.6    +2.0
gc03  e3t    +157.2  +123.2  +115.2   |   -16.5    -4.0    -7.9
gc04  sp     +186.3  +168.6  +146.1   |   -16.3    +3.4    -8.0
gc06  sp     +156.9   +95.4   +99.0   |   +20.2    +6.9    +1.5
```

### What the collision rule does

`overcooked_mdp._handle_collisions` freezes **BOTH** players for a tick whenever
their moves would land them on the same cell or swap them. Measured directly
(job `10832972`, 10 episodes each):

```
layout  robot    fov   collisions/ep   rate    reward  finish
gc00    random    30           45.0   0.201     60.0    1.00
gc00    sp        30          145.1   0.412     48.0    0.60
gc06    random    30           22.2   0.108     60.0    1.00
gc06    sp        30          128.1   0.389     46.0    0.70
gc06    noop     any          378.0   0.945      0.0    0.00
```

A trained self-play robot triggers **2–3.6× more collisions than a random one**,
and each one costs the *human* a move too. At fov30 on gc06 the pair is frozen
on 39% of all ticks. The rate is also **higher at narrow FOV** (0.412 vs 0.308)
because a blind human walks *into* the robot instead of around it — so physical
obstruction and the FOV effect are entangled in every collisions-on number.

### With collisions off, E3T is genuinely the best arm

Negative (helping) on **14 of 24** layout×FOV cells, and best-in-column on most:

```
             fov30   fov60   fov90  fov180  fov360
gc00  e3t     -4.7    -3.6    +2.7    -6.1   -18.2
gc03  e3t    -16.5   -71.9    -4.0    -9.7    -7.9
gc04  e3t    -15.3   +14.3   -13.5   -10.5    -7.7
gc06  e3t     -4.7    +6.1    +7.3    -3.1    +1.5
```

This is a real zero-shot-coordination result: the method that trains against a
**lagged, noisy partner** transfers best to a partner it has never seen. SP and
SP-ε hover near the floor; E3T is consistently below it.

### The floors collapse without collisions

`noop` and `random` land within a tick or two of each other, both f=1.00.
`noop`'s catastrophic 0.00 finish rate in the collisions-on arm was **entirely**
a collision artifact — a robot you can walk through cannot wedge a kitchen.

### This is NOT a harness bug — the control was run

Job `10832704` replays the *same checkpoints* through the *same* env
construction, action rule and sampler, partner replaced by a second copy of the
policy:

```
gc00  sp seeds 5,6,7   reward 60.00  finish 1.00
gc00  e3t seeds 1,2    reward 60.00  finish 1.00
gc04  sp seeds 5,6,7   reward 60.00  finish 1.00
```

Perfect with a copy of themselves. That is what makes the FOV-human gap a real
coordination failure rather than a broken evaluation.

### Still not established

* **The 4 evaluated layouts are all `gc*` clustered kitchens** — and those are
  exactly where E3T trains best. On `cram` E3T trains to **33.1±27.3 vs SP's
  53.7±7.2** (§5). The contention layouts could reverse the ranking entirely,
  and they are the ones the FOV story rests on. Do not report E3T as the winner
  until `cram`/`cram2` are evaluated.
* E3T's advantage is 5 seeds against SP's 11. Suggestive, not yet significant.
* Nothing here says the FOV module fixes anything. It says there is a real gap.

---

# ⚖️ THE COLLISION DECISION — READ BEFORE REPORTING ANY NUMBER

Collisions are **on by default**, and `eval/fov_human_eval.py --no_collisions`
turns them off via an instance-level override of `_handle_collisions`
(`overcooked_mdp.py` is never edited).

**Does turning them off break the "we use ZSC methods" claim? No.** SP, E3T and
FCP are defined by their **training procedure**, which is untouched — same MDP,
same collisions, same ZSC-Eval hyperparameters. Only the *evaluation*
environment changes. "Trained with ZSC-Eval's SP/E3T protocols" stays accurate.

**Three things that must be stated if you report the collisions-off arm:**

1. **One line in the methods section.** *"We disable inter-agent collisions at
   evaluation to isolate partner observability from physical obstruction; all
   methods are evaluated identically."*
2. **Every arm gets the same setting — the filter included.** Baselines without
   collisions and your method with them is not a comparison.
3. **The policies are off-distribution.** They trained in a world where bodies
   block. This hits all arms equally so the *ranking* is fair, but the absolute
   numbers are not "how this policy performs" in the trained MDP.

**The argument in favour, which is the strong one:** with collisions on, a
reviewer can say *"your FOV module didn't infer anything, it just learned to get
out of the way."* With collisions off, traffic management is impossible by
construction, so any gain has to come from actually inferring what the partner
can see. The collisions-off arm defends the claim you want to make.

The counter is framing: in an HRI paper a robot walking *through* a human is
odd. Reporting both arms costs ~6 min/layout and settles it.

---

# ⚠️ AT A GLANCE

```
=============================================================================
  OLD RUNS ARE UNTOUCHED.  NOTHING WAS OVERWRITTEN OR DELETED.
=============================================================================
  old (seed 1, 25 layouts)   /scratch1/mishafu/steakhouse_sp/specialist/...
  new (this document)        /scratch1/mishafu/steakhouse_zsc/<algo>/...

  old code   SP/self_play.py            unmodified, byte for byte
  new code   SP/self_play_pop.py   E3T/e3t.py   eval/*
             utils/{ckpt,valuenorm,schedules}.py
=============================================================================
```

| algo | runs | seeds | where | job |
|---|---:|---|---|---|
| **SP** | 77 | 5–15 (11) | `steakhouse_zsc/sp/<layout>_seed<S>/` | `10820346` |
| **E3T** | 35 | 1–5 (5) | `steakhouse_zsc/e3t/<layout>_seed<S>/` | `10820347` |
| **SP-ε** | 35 | 1–5 (5) | `steakhouse_zsc/sp_eps/<layout>_seed<S>/` | `10820348` |

Seven layouts each: `steak_gc00 gc03 gc04 gc06 cram cram2 gs00`.

### WHERE THE CHECKPOINTS LIVE — TWO COPIES, BOTH VERIFIED

```
PRIMARY   /scratch1/mishafu/steakhouse_zsc/<algo>/<layout>_seed<S>/
          purge-eligible. everything reads from here.

BACKUP    /home1/mishafu/steakhouse_zsc_backup/<algo>/<layout>_seed<S>/
          survives a scratch purge. smoke/ excluded. eval rows included.
          verified: 147 final.pt · 147 progress.jsonl · 2387 periodic · 8 eval jsonl
```

**⚠️ `du` LIES ON THIS FILESYSTEM.** VAST does inline compression, so `du -sh`
reports ~127 MB for something that is really **1.96 GB**. Always use
`du -sb --apparent-size` when sizing a copy:

```
147 final.pt          191.5 MB
2387 periodic ckpts  1560.6 MB
apparent total       1954.1 MB      (du -sh says 127M — a 15x under-report)
```

To restore after a purge:
`rsync -a ~/steakhouse_zsc_backup/{sp,e3t,sp_eps} /scratch1/$USER/steakhouse_zsc/`

Nothing from this wave is on the laptop except the **evaluation rows**
(`eval/results/`, 22,080 episodes). The old seed-1 checkpoints are separately
safe — byte-identical copies sit in `fov/robot/policy/old/module/QMDP/ckpt/`.

---

## 1. WHY THIS RE-RUN EXISTS

**No variance.** One run per layout. ZSC-Eval runs 11 (`seed_begin=5`,
`seed_max=15` in `train_sp.sh`). A 0.00 at n=1 is not evidence a layout is
unlearnable and a 60.00 at n=1 has no error bar.

**No checkpoint history.** `self_play.py` saved every `--save_interval` to the
*same filename*, so a finished run left only final weights. **FCP's population
is init/mid/final per seed** — it was not buildable from what existed.

**No score log.** ZSC picks the "mid" checkpoint as the one *scoring half the
final score*, not the one halfway through the clock — and these curves inflect
anywhere from episode 80 (`gc00`) to 460 (`none_3`), so the rules disagree
badly. Applying ZSC's rule needs a recorded score per save.

### Seeds do not combine — each seed is a SEPARATE NETWORK

Eleven seeds on one layout is **eleven `final.pt` files with eleven different
md5s**. They are never averaged. Weights from separately-initialized runs put
unrelated concepts in the same neurons; averaging them yields noise, not a
compromise.

Seeds are not a way to get one better policy. They are a way to measure **how
much the answer depends on training luck**. "Run SP on gc00" returns a
*distribution* of policies; each seed is one sample.

```
gc06   SP   47.6 ± 9.0 (11)
             ↑     ↑    ↑
             |     |    11 separate networks
             |     spread across those 11
             average of their 11 scores
```

At eval each of the 11 plays the human separately, 20 episodes each → 220
episodes per (algo, layout, fov) cell.

**This is exactly what ZSC-Eval does.** `train_sp.sh` is a bash loop launching
one process per seed with the same `--layout_name`; `extract_sp_models.py` then
names each seed's policy `sp5`, `sp6`, … and pulls init/mid/final out of each to
build the FCP population.

The one place seeds are used *together* is a **population** (FCP, MEP, TrajeDi),
and even there they stay distinct frozen networks played one at a time.

---

## 2. WHAT EACH ALGORITHM IS

All three train with **zero FOVHuman exposure** — the requirement the whole wave
exists to satisfy.

### SP — `SP/self_play_pop.py`
Both chefs are the same network. The baseline everything else is measured
against, and simultaneously **FCP stage 1**.

### E3T — `E3T/e3t.py`
Single stage. The partner is a **lagged** copy of the ego
(`θ_p ← (1−τ)θ_p + τθ_ego`, τ=0.1, once per episode) that also takes a
**uniformly random action 25% of the time**.

Self-play fails at zero-shot coordination because its partner is perfectly
predictable, so it converges on a private convention that evaporates against
anyone else. The noise half is what matters here: it is the cheapest stand-in
for *"my partner just did something I could not have predicted"*, which is what
a 90° cone produces when the human walks to a station it cannot see is occupied.

> **This matches ZSC-Eval's E3T exactly** — the published E3T numbers everyone
> cites are this implementation. Verified by grep: the only E3T-specific code in
> their tree is `rMAPPOPolicy_epsilon.py` (the ε mix), `r_mappo_target.py` (the
> soft copy), and one branch in `runner/separated/base_runner.py`.

### SP-ε — same file, `--weights_copy_factor 1.0`
At τ=1.0 the soft copy becomes exact, so the partner **is** the ego and only the
ε-noise remains. No forked file, so the two arms cannot differ anywhere except τ.

**What it answers:** E3T changes two things at once. Result so far — against the
FOV human on the four clustered layouts, SP-ε tracks SP closely while E3T is
consistently better, pointing at the **lagged partner** rather than the noise as
the active ingredient. But in *training* on `cram`, SP-ε (40.6) and E3T (33.1)
both fall far below SP (53.1), so the ε-noise appears actively harmful on tight
kitchens regardless of the lag. 5 seeds; not yet a claim either way.

---

## 3. WHAT CHANGED VS THE SEED-1 CONFIG

Exactly three things. PPO update, GAE, `bad_masks`, the dense shaper, the
23-channel observation, the network, lr, `ppo_epoch`, shaping horizon — all the
same code on the same numbers.

| change | why |
|---|---|
| **`use_valuenorm` ON** | Last algorithmic gap vs ZSC-Eval. Return magnitude grows ~20x over a run (`vloss` 0.68 → 9.35 on mid_1, 13–26 on winners), so one critic lr cannot be right at both ends. `--no_valuenorm` restores the old arithmetic exactly. |
| **entropy anneal → ZSC two-segment** | `0.2 → 0.05` over 5e6, then `0.05 → 0.01`. The old single line sat at 0.105 at the midpoint vs 0.05 — roughly twice the mid-training exploration. All three arms share these numbers so a gap between them is the algorithm, not the schedule. |
| **checkpoint history + progress log** | Section 1. |

> **`utils/buffer.py` was NOT touched.** ValueNorm is wired in by de-normalizing
> at the point of *storage*, so the buffer's GAE still sees real values and is
> the same code that produced the validated runs.

**Known remaining deviation:** ZSC's E3T script puts its entropy knee at 6e6
rather than SP's 5e6. All three arms use 5e6 — a 1e6-step difference (0.05 vs
0.075 at the 5e6 mark) not worth confounding an SP-vs-E3T comparison over.

---

## 4. WHAT EACH RUN WRITES

A finished SP run directory, in full:

```
/scratch1/mishafu/steakhouse_zsc/<algo>/<layout>_seed<S>/
    actor_periodic_{0,25,50,...,475,499}.pt   21 files, episode number in the name
                                              ACTOR ONLY -- no critic key
    final.pt          {actor, critic, value_norm, episode, args}   episode 499
    sp_<layout>.pt    byte-identical copy of final.pt
    progress.jsonl    one line PER EPISODE (500 lines; 250 for E3T/SP-ε)
    smoke/            800-step smoke test. JUNK. NEVER load from here.
```

21 periodic checkpoints per SP run, 11 per E3T/SP-ε run — fewer because those
use 100 rollout threads, so 1e7 env steps is 250 episodes instead of 500.

`sp_<layout>.pt` exists so `eval_checkpoints.py` and
`filter/baseline.py::find_checkpoint` find what they expect with no code change.
Periodic saves are **actor only**: a pool partner is frozen, never asked for a
value, so its critic is dead weight at 21 saves × 147 runs.

`ck["args"]` holds **every hyperparameter that run used**, so nothing has to be
guessed later:

```python
{'seed': 5, 'layouts': 'steak_gc00', 'use_valuenorm': True,
 'entropy_coefs': [0.2, 0.05, 0.01], 'num_env_steps': 10000000,
 'n_rollout_threads': 50, ...}
```

### ⚠️ THE ONE MISTAKE THAT COSTS A DAY

**A checkpoint only loads into the layout it trained on.** `CNNBase` ends in
`Linear(32 * W * H, 64)`, so the weight shape is fixed by that kitchen's grid.
A `gc00` (5×5) checkpoint physically cannot load into `gc04` (7×7).

**Never hardcode the shape — ask the environment:**

```python
mdp = SteakHouseGridworld.from_layout_name(LAYOUT,
          start_order_list=["steak"] * 4,
          rew_shaping_params=dict(BASE_REW_SHAPING_PARAMS))
obs_shape = (23, mdp.shape[0], mdp.shape[1])     # correct by construction
```

Sanity-check by file size — it scales with `W*H` (measured, seed 5):

| layout | grid | `final.pt` | `actor_periodic_*.pt` |
|---|---|---:|---:|
| gc00 | 5×5 | 1,010,275 | 507,582 |
| gs00 | 6×5 | 1,092,195 | 548,542 |
| gc06 | 5×7 | 1,174,115 | 589,502 |
| gc03 | 7×6 | 1,288,803 | 646,846 |
| gc04 | 7×7 | 1,403,491 | 704,190 |
| cram | 9×9 | 1,927,779 | 966,334 |

If a file's size does not match its layout, you have the wrong file.

### How to load one

```python
import sys, types, torch
sys.path.insert(0, "fov/robot/policy/new/official_baselines")
from algorithm.rMAPPOPolicy import R_MAPPOPolicy

ck = torch.load(PATH, map_location="cpu", weights_only=False)
a  = ck["args"]
args = types.SimpleNamespace(hidden_size=a["hidden_size"], recurrent_N=a["recurrent_N"],
                             lr=a["lr"], critic_lr=a["critic_lr"], opti_eps=a["opti_eps"])
policy = R_MAPPOPolicy(args, obs_shape, obs_shape, act_dim=6)
policy.actor.load_state_dict(ck["actor"])
if "critic" in ck:                      # periodic checkpoints are ACTOR ONLY
    policy.critic.load_state_dict(ck["critic"])
policy.actor.eval()
```

Rolling out — **the GRU state must be threaded through the whole episode** or
this is not the policy that trained:

```python
rnn   = torch.zeros(1, args.recurrent_N, args.hidden_size)
masks = torch.ones(1, 1)
actions, _, rnn = policy.actor(torch.from_numpy(obs).float(), rnn, masks,
                               deterministic)   # True=argmax, False=sample
```

For the full distribution instead of one sampled action (what a filter needs),
use `SP.self_play.action_probs` — the identical forward pass stopped one step
earlier at the Categorical. It also advances the GRU, so thread `rnn_states`.
`eval/fov_human_eval.py::Actor` is a worked example of all of this.

### Pointing `filter/baseline.py` at these runs

`filter/baseline.py::CKPT_DIRS` reads the env var **`STEAK_SP_CKPT_ROOT`**
first, then falls back to local dirs. To use a wave-1 checkpoint:

```bash
export STEAK_SP_CKPT_ROOT=/scratch1/$USER/steakhouse_zsc/sp   # <layout>_seed<S>/sp_<layout>.pt
```

Unset, it finds the OLD seed-1 specialists under `steakhouse_sp/specialist`.
Both path shapes work because every run writes the `sp_<layout>.pt` alias.

### Scoring

`delivery_reward = 20`, and `is_terminal` fires at `len(order_list) <= 1`, so
with `n_orders=4` you deliver **3**.

> **60.00 = perfect episode. 40.00 = 2 of 3. 20.00 = 1 of 3.**

### The 23 observation channels

```
[0]      ego agent position          [1]   ego agent facing cell
[2]      other agent position        [3]   other agent facing cell
[4-10]   objects: meat, onion, plate, washed_plate, steak, garnish, dish
[11-18]  terrain masks, one per TERRAIN entry: P B W M O D S X
[19]     station status  0 empty / 0.5 in progress / 1 ready
[20]     station timer   elapsed / total, in [0,1]
[21]     time left in episode, normalized
[22]     orders remaining, normalized
```

`' '` (walkable floor) has **no** plane — it is the implicit "all eight terrain
channels are zero" category. Observations are **ego-centric**: `agent_index`
decides who lands in `[0,1]` vs `[2,3]`, which is what stops self-play from
making both chefs do the identical dance.

### Reading a pool back (for FCP)

```python
from utils.ckpt import select_pool_checkpoints
pool = select_pool_checkpoints("/scratch1/mishafu/steakhouse_zsc/sp/steak_gc00_seed5")
# {"init": ".../actor_periodic_0.pt", "mid": ..., "final": ...}
```

`mid` = the saved checkpoint whose logged score is closest to **half the final
score**, final being the mean of the last 5 logged points (one last point is
noisy — CARC_RUNS §5 measured peak-to-final swings up to 8.0).

---

## 5. TRAINING RESULTS (self-play score, NOT the FOV-human score)

`sparse_ret` at end of training, mean ± std **across seeds**. 60 = perfect.

```
layout              SP                E3T             SP-eps
gc00        59.6±0.4 (11)     59.7±0.2 (5)      59.6±0.5 (5)
gc03        54.2±8.6 (11)     59.6±0.3 (5)      58.9±0.7 (5)
gc04        55.7±7.4 (11)     59.3±0.1 (5)      59.2±0.3 (5)
gc06        47.6±9.0 (11)     59.4±0.7 (5)      59.2±0.8 (5)
cram        53.7±7.2 (11)    33.1±27.3 (5)     40.6±22.1 (5)   <- E3T LOSES badly
cram2       47.3±8.4 (11)     41.9±2.1 (5)      42.2±2.9 (5)   <- E3T loses again
gs00        59.3±0.9 (11)     59.6±0.3 (5)      59.6±0.1 (5)

pooled      SP 53.9 (77)     E3T 53.2 (35)    SP-eps 54.2 (35)
collapsed   0 of 77           2 of 35           1 of 35
```

**FINAL — all 147 runs complete, every cell is the full seed count.**

**The error bars are the point.** `gc06` is 47.6 ± 9.0 across 11 seeds; the old
n=1 table in `CARC_RUNS.md` called it a flat **59.60 SOLVED**. That was a lucky
seed near the top of the range. Same on `gc03` (±8.6) and `gc04` (±7.4).

**No SP run has ever collapsed** — 75/75 finished runs deliver, on every layout.
E3T has 2 collapsed runs and SP-ε 1, all on `cram`.

**E3T is NOT uniformly better, and `cram` is where it breaks.** On the five
clustered layouts it trains tighter than SP (gc04: 59.3±0.1 vs 55.7±7.4). On
`cram` it is **33.1±27.3 against SP's 53.7±7.2** — a worse mean and four times
the spread, made of 2 runs that never learned at all. `cram2` is the same story
milder (41.9 vs 49.0).

`cram` and `cram2` are the hand-built **contention** layouts, the ones
`LAYOUTS.md` ranks highest for robot influence and therefore the ones the FOV
result most depends on. The plausible reading: ε=0.25 random partner actions are
survivable in a roomy kitchen and destructive in a tight one, where a
mistimed random move blocks the only corridor and the shaped-reward chain never
closes. **Worth checking before E3T is reported as the better method** — the
FOV-human eval so far covers only the four clustered layouts where E3T looks
good.

Partner noise stabilizes *training* on open layouts. Whether it helps against
the FOV human is the headline section, and the answer differs by collision
condition.

Regenerate with `scratchpad/status.py` (reads every `progress.jsonl`).

---

## 6. THE EVAL HARNESS — `eval/`

Lives in `official_baselines/eval/`, **not** inside `filter/` — that package is
hand-written and stays that way.

```bash
# one layout, all six FOVs, both floors + every trained seed
python eval/fov_human_eval.py --layout steak_gc00 --all_fovs --episodes 20 --out rows.jsonl
# the whole table, one array task per layout
sbatch eval/run_fov_eval.sbatch                      # collisions ON
COLLISIONS=0 sbatch eval/run_fov_eval.sbatch         # collisions OFF
python eval/aggregate.py '.../eval/*_nocoll.jsonl' --per_layout
```

### The tick order IS the experiment

```
s = env.state
a = choose(actor.probs(obs(s)))     robot decides on s
h, info = human.action(s)           human decides on s. `info` holds the
                                    GROUND-TRUTH SUBTASK -- discarded here
env.step((a, h))                    simultaneous resolve
```

Identical to `filter/play_episode.py`, so the rows are paired and subtractable.

### What the robot MAY and MAY NOT see

| | |
|---|---|
| **MAY** | the full world state (these are fully-observable agents — that is what `build_full_state` hands the network), the human's **emitted action** each tick (a physical event anyone in the kitchen can watch), the trained weights |
| **MAY NOT** | the human's **true FOV** (`--fov` configures the HUMAN and is passed to nothing on the robot side), the human's **subtask label** (`info["subtask"]` — the single most tempting cheat in this codebase), the real human's beliefs, any environment reward |

The baseline robot does not even consume the human's action — it is a frozen
policy reading world state. The MAY list is written from the *filter's* point of
view so that when the filter arm is added, the two differ in the module and in
nothing else.

### Seats, action rule, and pairing

Robot at **index 0**, human at **index 1**, **sampled** actions — matching
`filter/baseline.py` exactly. Sampling is what ZSC-Eval evaluates with, and
CARC_RUNS §6 measured argmax **deadlocking** three layouts (gc01/gc05/cram2 all
score 40.00 greedy, 60.00 sampled). `--mode argmax` reproduces that; the table
is sampled.

Episode *i* uses seed `seed0 + i` for **every** policy. The human's private RNG
is seeded off the global `random` stream inside its `reset()`, so this means an
11-seed SP mean and a 5-seed E3T mean face the **identical sequence of humans**.
Without it the arms would differ partly because they met different partners.

The two collision arms write to **different filenames** (`_coll` / `_nocoll`)
and every row carries a `collisions` field, so they cannot be pooled by
accident.

### ⚠ REWARD IS SATURATED. USE COMPLETION TIME.

A **random** robot beside a **fov30** human scores `reward 60.00,
h_delivered 3.00`. The FOV human solo-solves the small layouts — only 3
deliveries are required and 400 ticks is plenty. Every arm ties at the ceiling.
The headline metric is **completion time** (tick of the last delivery, or
`horizon + 100` for a DNF), same as `filter/baseline.py`.

Need reward headroom? Raise the **horizon**, not `n_orders` — observation
channel 22 is orders-remaining *normalized*, so changing `n_orders` at eval time
shifts an input the policy trained on.

### ⚠ fov90 AND fov120 ARE THE SAME CONDITION — 5 levels, not 6

**98% of fov90 and fov120 episodes are bit-identical** — same completion time,
same action histogram, same human explore count, for the same policy and episode
seed. Confirmed in **both** collision arms, so it is the cone and not the
collision freeze.

`_visible_cone` tests `|rx| <= tan(fov/2) * |ry|` — `1.0*|ry|` at 90°,
`1.732*|ry|` at 120°. On integer cell offsets in a small kitchen almost nothing
falls between those bounds.

* **Do not present fov90 and fov120 as two data points.**
* The axis is **not monotone** either: on gc06 (collisions on) fov180 was worse
  than fov90, 284.9 vs 269.9.
* Layout-dependent — re-check on `cram` (9×9) when it lands.

### Two floors

`random` moves constantly; `noop` always STAYs. **`random` is the reference.**
With collisions ON, `noop` posted 0.00 finish at nearly every FOV — it measures
"the robot is furniture", not "the robot did not help". With collisions OFF the
two floors coincide. `--floor noop` or `--floor best` if you want it anyway.

### Where the results are

Local, in `eval/results/` — 22,080 rows:

```
steak_{gc00,gc03,gc04,gc06}_sample_coll.jsonl      collisions ON,  2760 rows each
steak_{gc00,gc03,gc04,gc06}_sample_nocoll.jsonl    collisions OFF, 2760 rows each
TABLES.txt                                          rendered tables + self-play control
```

Rows are the source of truth; every table is re-derivable without replaying an
episode.

---

## 7. WHAT COMES NEXT, IN ORDER

1. **Re-run both eval arms on all 7 layouts** once `cram`, `cram2`, `gs00`
   finish. Rows files are APPEND-ONLY — delete first or runs mix:
   `rm /scratch1/$USER/steakhouse_zsc/eval/*.jsonl`
2. **Re-check the fov90/fov120 collapse on `cram` (9×9)** — a bigger grid may
   separate the cones, which changes how many FOV levels you can report.
3. **FCP stage 2** — unblocked, and buildable *now* for gc00/gc03/gc04/gc06
   (all have 11 SP seeds). Pool = 4 SP seeds × init/mid/final = 12 frozen
   partners; ego trains against one drawn per episode. ~250 lines. The cost is
   compute: ZSC uses 5e7 steps for pop-12, ~28h, which fits `main`'s 48h wall.
4. **MEP / TrajeDi** — identical to FCP except stage 1 adds a diversity bonus
   (population entropy / trajectory JSD). Stage 2 is shared with FCP.
5. **HSP** — needs an ~18-dim event-count reward vector for the steakhouse chain
   plus a best-response selection pass.
6. **COLE** — iterative graph-based population, 5e7–1.5e8 steps.

**BC / human-proxy is impossible** — no human demonstration data for steakhouse.

---

## 8. THE ENVIRONMENT, AND FOUR THINGS THAT SILENTLY DESTROY A RUN

### Setup

```
conda env    steakhouse-ai   (/home1/mishafu/.conda/envs/steakhouse-ai)
             python 3.8.20 · torch 2.4.1+cu121 · numpy 1.24.3
device       CPU. torch.cuda.is_available() is False on these nodes, and that
             is fine -- the bottleneck is envs.step(), pure python and
             un-batchable. The conv stack on a 5x5 grid is microseconds.
throughput   ~400-600 fps. One run (1e7 env steps) is 6-10h; cram (9x9) 17h+.
```

### Trap 1 — layouts must be STAGED or nothing is found

`SteakHouseGridworld.from_layout_name()` **only** reads
`overcooked_ai_py/data/layouts/`. The validated library lives in
`fov/layouts_final/layouts/` and is **not** copied there by default. Every
sbatch here runs this first, and `eval/fov_human_eval.py::stage_layouts()` does
the same in python:

```bash
cp -n fov/layouts_final/layouts/*.layout overcooked_ai_py/data/layouts/
```

`-n` so concurrent array tasks cannot clobber each other. "layout not found"
from a fresh shell means this was skipped.

### Trap 2 — `start_order_list` arrives as a STRING

Every `.layout` file declares `"start_order_list": 'steak, steak, steak'` — a
**string**, not a list. Then `len()` counts **characters** (19, not 3), and
`deliver_dish` does `order_list[0]`, which on a string is the character `'s'`,
so `'steak' == 's'` is False. **Orders are never consumed and no delivery reward
ever fires.**

Always construct with `start_order_list=["steak"] * n_orders`. `features.py` and
`eval/fov_human_eval.py::build_mdp` both assert against it.

### Trap 3 — `rew_shaping_params: None` means ZERO shaping in this fork

```
this repo,  overcooked_mdp.py:672   None -> NO_REW_SHAPING_PARAMS   <- ALL ZEROS
ZSC-Eval,   overcooked_mdp.py:980   None -> BASE_REW_SHAPING_PARAMS <- real values
```

The same line with opposite defaults. Leave it alone and shaped reward is
identically 0, and the failure is not "slow learning", it is total:

```
reward 0 -> returns 0 -> critic learns V=0 -> advantages are float noise (~1e-9)
-> the buffer normalizes them by their own std -> 1e-9/1e-9 = UNIT-SCALE noise
-> the policy trains at full gradient strength on nothing
observed: vloss 0.000, clip_frac 0.45, entropy 1.79 -> 1.11 in 30 episodes
```

Always pass `rew_shaping_params=dict(BASE_REW_SHAPING_PARAMS)`. `env_wrapper.py`
asserts shaping is not all-zero; `buffer.py` guards `adv.std() < 1e-6`.

### Trap 4 — `smoke/` holds identically-named junk

Every run directory contains `smoke/<same filenames>` from the 800-step smoke
test that gates each job. Same names, same sizes, useless weights.
**Never glob `**/final.pt`.** Give the exact path. If unsure, check
`ck["episode"]` — real SP runs say 499, smoke says 2.

### Dense reward shaping (added ON TOP of the mdp — mdp is not modified)

This fork fires only **two** shaped rewards, both at the pan
(`PLACEMENT_IN_POT_REW`, `COOKING_STEAK_REW`). Chopping, washing, plating,
garnishing and dish pickup pay **nothing** — dense signal for 2 steps of an
8-step chain. `utils/env_wrapper.py::_DenseShaper` diffs consecutive states and
pays the events the mdp forgot:

| event | reward |
|---|---|
| pick up meat / onion / plate | 1.0 |
| onion → chopping board, plate → sink | 3.0 |
| each chop / wash tick | 1.0 |
| carry off a finished steak / washed_plate / garnish | 3.0 |
| assemble the dish | 5.0 |

Magnitudes mirror ZSC-Eval's `BASE_REW_SHAPING_PARAMS`. Two deliberate choices:
**counters are ignored** (dropping an onion on a counter is not progress, only a
chopping board is), and **every event type is capped at `n_orders * 3` per
episode** — without a cap the agent farms the pick-up/put-down loop forever
instead of cooking, which shows up as beautiful reward curves attached to a
useless policy.

### Reading a training log line

```
ep 490/500 | steps 9,820,000 | fps 592 | sparse_ret 60.00 | vloss 16.910
  | ploss -0.0193 | ent 0.458 | clip 0.097 | shape_c 0.90 | ent_c 0.014
```

| field | how to read it |
|---|---|
| `sparse_ret` | **the only number that matters.** Real deliveries. 60 = perfect. |
| `ent` | policy entropy. `1.79` = uniform over 6 actions (`ln 6`). Falling to ~0.5 = committed. Stuck at 1.79 = never learned. |
| `clip` | fraction the PPO clip bit. Healthy 0.05–0.20. Pinned near 0.45 is what the all-zero-reward bug looked like. |
| `vloss` | tracks **return magnitude**, so higher is *good* here. Winners hit 13–26. |
| `shape_c` | shaping coefficient. Stays near 1.0 — the horizon is 10x the run, same as ZSC. |
| `ent_c` | current entropy coefficient, annealing 0.2 → 0.05 → 0.01. |

---

## 9. HOW TO RE-RUN ANY OF IT

```bash
ssh mishafu@10.72.0.13          # NOT discovery.usc.edu -- see §11
cd ~/steakhouse

# --- training (each is a job array; the %N caps concurrency)
sbatch fov/robot/policy/new/official_baselines/SP/run_sp_seeds.sbatch      # 77
sbatch fov/robot/policy/new/official_baselines/E3T/run_e3t.sbatch          # 35
sbatch fov/robot/policy/new/official_baselines/E3T/run_sp_eps.sbatch       # 35

# --- evaluation vs the FOV human (AFTER final.pt exists)
sbatch fov/robot/policy/new/official_baselines/eval/run_fov_eval.sbatch                # collisions ON
COLLISIONS=0 sbatch fov/robot/policy/new/official_baselines/eval/run_fov_eval.sbatch   # collisions OFF

# --- subsets
sbatch --array=0-3 .../run_fov_eval.sbatch          # first 4 layouts only
EPISODES=50 sbatch .../run_fov_eval.sbatch          # more episodes per cell
ENT_ARGS="--entropy_coefs 0.2 0.01 --entropy_coef_horizons 0 1e7" sbatch .../run_sp_seeds.sbatch
```

**Rows files are APPEND-ONLY.** Delete before re-running or two runs silently
mix: `rm /scratch1/$USER/steakhouse_zsc/eval/*.jsonl`

**Nothing gets clobbered by a re-run of training** — each `(layout, seed)` writes
its own directory. A repeated run overwrites only itself.

**Array index → (layout, seed)** is `layout = task_id / n_seeds`,
`seed = seed_begin + task_id % n_seeds`. So SP tasks 0–10 are `gc00` seeds 5–15,
11–21 are `gc03`, … 66–76 are `gs00`. Layouts finish in index order, which is
why `gs00` lands last despite being 6×5 — ordering, not difficulty.

**Checking on it:**

```bash
squeue -u $USER -h -o "%j %T" | sort | uniq -c
ls /scratch1/$USER/steakhouse_zsc/sp/*/final.pt | wc -l          # want 77
sacct -j <jobid> -X -n --format=JobID,State | grep -v COMPLETED  # failures
tail -5 ~/steakhouse/logs/zsc_sp_10820346_0.out
```

---

## 10. FILE MAP

Everything written for this wave. **Nothing existing was modified or deleted.**

| file | what it does |
|---|---|
| `SP/self_play_pop.py` | **NEW.** SP trainer: versioned checkpoints, progress log, valuenorm, piecewise entropy. The runs in this document. |
| `SP/run_sp_seeds.sbatch` | **NEW.** 7 layouts × seeds 5–15 = 77 tasks, `%24` on `biyik_1173`. |
| `E3T/e3t.py` | **NEW.** E3T trainer — lagged ε-noisy partner, ego-only buffer, random seat. `--weights_copy_factor 1.0` makes it SP-ε. |
| `E3T/run_e3t.sbatch` | **NEW.** 7 × seeds 1–5 = 35 tasks, τ=0.1, 100 threads. |
| `E3T/run_sp_eps.sbatch` | **NEW.** Same script, τ=1.0. The ablation. |
| `utils/ckpt.py` | **NEW.** `RunWriter` (periodic saves + progress.jsonl) and `select_pool_checkpoints` (ZSC's init/mid/final rule). |
| `utils/valuenorm.py` | **NEW.** ValueNorm ported from `zsceval/utils/valuenorm.py`, plus a no-op `IdentityValueNorm` for the off path. |
| `utils/schedules.py` | **NEW.** Piecewise-linear entropy anneal, ZSC-style. |
| `eval/fov_human_eval.py` | **NEW.** Plays a checkpoint against `LimitedVisionSteakHuman`. Floors, `--no_collisions`, JSONL rows. |
| `eval/aggregate.py` | **NEW.** Rows → tables + the help-or-hinder delta. |
| `eval/run_fov_eval.sbatch` | **NEW.** One array task per layout. `COLLISIONS=0` for the second arm. |
| `eval/results/*.jsonl` | **NEW.** 22,080 evaluated episodes, `_coll` / `_nocoll`. |
| `eval/results/TABLES.txt` | **NEW.** Both arms + self-play control + collision diagnostic, rendered. |
| `ZSC_BASELINES.md` | **NEW.** This file. |
| — | — |
| `SP/self_play.py` | **UNCHANGED**, byte for byte. Produced the validated seed-1 baselines. |
| `utils/buffer.py` | **UNCHANGED.** ValueNorm is wired in around it, not through it. |
| `utils/{features,cnn,rnn,act,env_wrapper}.py` | **UNCHANGED.** |
| `algorithm/{r_actor_critic,rMAPPOPolicy}.py` | **UNCHANGED.** |
| `SP/CARC_RUNS.md` | **UNCHANGED.** Describes the OLD seed-1 runs. Still correct about them. |

Throwaway analysis scripts live in the session scratchpad, not the repo:
`status.py` (training table), `check_fov.py` (fov-identity test),
`control_selfplay.py` (the self-play control), `collision_diag.py`.

---

## 11. OPERATIONAL NOTES

**`discovery.usc.edu` round-robins and one login node is broken.** `10.72.0.14`
(discovery2) accepts the key then refuses every session —
`shell request failed on channel 0`, which looks like auth failure and is not.
**Use `ssh mishafu@10.72.0.13`** (discovery1).

**`main` allows 48 hours, not 24.** The old 24h wall is what timed
`steak_mid_2` out at ep 490. `oneweek` allows 7 days across 45 nodes — the
escape hatch for FCP-scale runs.

**Two accounts.** SP on `biyik_1173` (24 concurrent), E3T and SP-ε on
`biyik_1165` (8 each) — 40 total, SP gets the most because it is the long pole
that FCP waits on.

**Why 147 jobs.** `7 layouts × (11 + 5 + 5 seeds)`. Per-layout is forced by the
architecture: `CNNBase` ends in `Linear(32*W*H, 64)`, so a gc00 (5×5) checkpoint
physically cannot load into gc04 (7×7). One job = one `(layout, seed, algo)` =
1e7 env steps ≈ 6–10h, because the MDP is pure Python and `VecSteakEnv` steps
its kitchens **sequentially** — extra CPUs do not help, only a `SubprocVecEnv`
rewrite would.

**Array index → layout is `task_id / n_seeds`,** so tasks run in layout order.
`gs00` finishing last is ordering, not difficulty — it is 6×5, smaller than
`gc04`.

**`/scratch1` is purge-eligible — but a verified backup now exists** at
`~/steakhouse_zsc_backup` (1.96 GB apparent, `smoke/` excluded, 147 final.pt +
147 progress.jsonl + 2387 periodic + 8 eval jsonl). See
[At a glance](#️-at-a-glance) for the restore command and the `du`-lies warning.
The old seed-1 runs are separately safe — byte-identical copies live under
`fov/robot/policy/old/module/QMDP/ckpt/`.

**Never run python on the login node** — numpy segfaults. Use `sbatch` or
`srun -A biyik_1173 -p main -c 4 --mem 16G -t 00:20:00 --pty bash`.

**Syncing code:** `rsync` fails over this link; use tar over ssh, then delete
the macOS `._*` files or python will try to import them.

```bash
cd fov/robot/policy/new
tar czf - --exclude='._*' --exclude='__pycache__' official_baselines \
  | ssh mishafu@10.72.0.13 'cd ~/steakhouse/fov/robot/policy/new && tar xzf -'
ssh mishafu@10.72.0.13 'find ~/steakhouse/fov -name "._*" -delete'
```

**BSD sed has no `\|` alternation** — `sed 's/a\|b//'` silently does nothing on
macOS. Use `grep -E` or `perl -pe`.

---

## 12. GLOSSARY

Terms this document leans on. Several mean something specific here.

| term | what it means **in this document** |
|---|---|
| **seed** | One complete, independent training run. Different weight init, different env sampling. `gc00` at 11 seeds = **11 separate networks with 11 different md5s**, never merged. See §1. |
| **arm** | One condition in a comparison. Used two ways: an *algorithm* arm (SP / E3T / SP-ε) and a *collision* arm (collisions on / off). Always check which. |
| **floor** | An untrained robot, there to answer "would the human have been better off alone?" Two of them: `random` (uniform over 6 actions) and `noop` (always STAY). **`random` is the reference**; `noop` is pathological with collisions on. |
| **help or hinder** | `trained arm − floor`, on completion time. **Negative = the robot helped.** The core table. |
| **`sparse_ret`** | Training-log score: real delivery reward, averaged over the last 50 sampled rollouts. **60 = perfect.** The only training number that matters. |
| **completion time** | Tick of the **last** delivery, or `horizon + 100` if the team never finished. The **headline eval metric**, because reward saturates. Lower is better. |
| **DNF** | Did not finish — fewer than 3 deliveries inside the horizon. Scored at `400 + 100 = 500`, **never dropped**, because excluding DNFs would rank a policy that finishes 30% of the time and is fast when it does above one that always finishes. |
| **finish rate `(f0.87)`** | Fraction of episodes reaching 3 deliveries. Printed beside every cell so a mean dragged up by DNFs is visible rather than mysterious. |
| **± in a table** | Standard deviation **across policy seeds**, not across episodes. Each seed's episodes are averaged first, then the spread of those per-seed means is reported. Answers *"how much does the answer depend on which run I happened to train?"* Untrained floors have one "seed", so their ± is 0.00 and means nothing. |
| **specialist** | A policy trained on ONE layout. Forced by the architecture — `Linear(32*W*H, 64)` binds a checkpoint to its grid — and also what ZSC-Eval does. There is no generalist here. |
| **pool / population** | FCP/MEP/TrajeDi stage 1: a set of **frozen** partner networks the ego trains against, one drawn per episode. For FCP pop-12 it is 4 SP seeds × {init, mid, final}. Still never merged. |
| **init / mid / final** | The three checkpoints pulled from each run for a pool. `mid` is chosen **by score** (closest to half the final score), not by clock position — the curves inflect anywhere from episode 80 to 460. |
| **τ (`weights_copy_factor`)** | E3T's partner-lag rate: `θ_p ← (1−τ)θ_p + τθ_ego`, once per episode. **0.1 = E3T** (lagged partner). **1.0 = SP-ε** (partner IS the ego, only ε-noise remains). |
| **ε (`epsilon`)** | Probability the partner takes a uniformly random action instead of its policy's. 0.25, per ZSC-Eval. |
| **FOV** | The human's vision cone in degrees. Configures the **human only** — it never reaches the robot. Six values swept; **fov90 and fov120 are the same condition**, so effectively five. |
| **collisions** | The MDP freezing **both** players for a tick when their moves would collide or swap. On by default; `--no_collisions` disables at eval only. |
| **`_coll` / `_nocoll`** | Filename suffix marking which collision arm a rows file came from. Every row also carries a `collisions` boolean. |
| **smoke** | The 800-step test that gates every training job. Writes into `smoke/` with **identical filenames**. Always junk. Never load from there. |
| **wave 1** | This whole effort: SP + E3T + SP-ε. Wave 2 would be FCP stage 2, then MEP/TrajeDi. |
