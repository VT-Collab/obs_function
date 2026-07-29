# CARC ↔ local: what lives where

**Deliberately NOT synced:** checkpoints (`*.pt`) and training logs stay on CARC only, to
keep local storage free. Code is synced both ways (local is the source of truth; CARC is a
copy pushed by rsync). This file is the bridge — read it first in a new session.

Last updated: 2026-07-28, ~22:40.

---

## 1. Connecting (there is a gotcha)

```bash
ssh mishafu@discovery.usc.edu          # works from a normal terminal
ssh mishafu@10.72.0.13                 # use THIS from an agent/sandboxed shell
```

`discovery.usc.edu` resolves only to private USC addresses (10.72.0.13 / .14), and the
sandboxed shell's resolver cannot see it — `ssh` reports *"Could not resolve hostname"*
even though `nslookup` succeeds. **Connect by IP.** Key-based auth works, no password.
Landing node is `discovery1.hpc.usc.edu`.

`module` is unavailable in non-interactive bash — use `ssh host 'bash -l -s' <<'EOF'`
(login shell) or you get `module: command not found`.

## 2. Layout on CARC

| what | where |
|---|---|
| project | `~/minigrid-ai` (≈59 MB) — **new dir, created this session** |
| conda env | `~/.conda/envs/minigrid` — **new env, created this session** |
| checkpoints | `~/minigrid-ai/robot/policy/neural/baseline/no_fov/checkpoints/sweep/` and `.../sweep2/` |
| training logs | `~/minigrid-ai/logs/nofov_*_<jobid>_<task>.out` / `.err` |
| eval output | `~/minigrid-ai/eval_results/*.tsv` |
| status helper | `~/carc_status.sh` — prints per-task return/entropy/vs_mute |

**Untouched on CARC:** `~/steakhouse`, `~/signal_manifold_project`, and the conda envs
`steakhouse-ai`, `scaffolder`, `scaffolder-gpu`, `manifold_env`.

Env contents: python 3.11, torch 2.13.0+cpu, gymnasium 1.3.0, torch-ac, numpy 2.4.6,
pygame. Versions differ from the local `minigrid` env — that's fine, it was smoke-tested.

```bash
# run anything on CARC
module load conda/25.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate minigrid
cd ~/minigrid-ai
export PYTHONPATH=$PWD/Minigrid:$PWD
export SDL_VIDEODRIVER=dummy      # vendored Minigrid imports pygame even headless
```

## 3. Jobs submitted

| job | script | what varies | status |
|---|---|---|---|
| **10671707** `nofov_sweep` | `carc_sweep.sh` | schedules: warmup, entropy, lr (12 configs c01–c12) | task 12 cancelled by me; rest finishing |
| **10671789** `nofov_rew` | `carc_sweep2.sh` | reward magnitudes: comm_cost, key_bonus (12 configs d01–d12) | running |
| not yet submitted | `carc_eval.sh` | evaluates every checkpoint, greedy + sampled | fire after both sweeps finish |

All at 1M frames, `--method rec_ppo`, 4 CPUs / 12 GB per task.

```bash
squeue -u $USER                                  # what's alive
bash ~/carc_status.sh                            # per-task return/entropy/vs_mute
sacct -j 10671707,10671789 --format=JobID%14,State%12,ExitCode
sbatch carc_eval.sh                              # after sweeps land
cat ~/minigrid-ai/eval_results/*.tsv             # collate results
```

## 4. Pushing code changes local → CARC

```bash
cd /Users/mishafu/Desktop/obs_function/minigrid-ai
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no" \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
  --exclude '*.pt' --exclude 'stale_22ch_checkpoints' --exclude '.DS_Store' \
  ./ mishafu@10.72.0.13:minigrid-ai/
```

macOS ships rsync 2.6.9 (openrsync) — `--info=stats2` and other rsync-3 flags fail.

To pull a single checkpoint back when you actually need one (don't pull them all):

```bash
scp mishafu@10.72.0.13:minigrid-ai/robot/policy/neural/baseline/no_fov/checkpoints/sweep2/d02.pt .
```

---

## 5. Code changed this session (local, already pushed)

| file | change |
|---|---|
| `no_fov/features.py` | **transpose fix** (`x[N_OBJ+2, ax, ay]`, was `[ay, ax]`); 22→16 channels via `OBJ_TYPES`; header rewritten with literal sheet numbers 0–15 |
| `no_fov/my_env_wrapper.py` | added `key_bonus` one-shot milestone + `_correct_key_color()` (grid geometry, validated 60/60). Reward structure otherwise untouched — user's file, do not restructure |
| `no_fov/train.py` | `--time-cost`, `--key-bonus`, `--comm-warmup-frac`, `--entropy-start/--entropy-end`; comm-cost warmup and entropy anneal in the PPO loop; sac import made optional |
| `no_fov/evaluate.py`, `module/fov_module.py` | sac import made optional (`sac.py` is not in the tree) |
| `no_fov/actor_critic.py` | stale `(22,19,19)` comment → `(16,19,19)` |
| new: `carc_sweep.sh`, `carc_sweep2.sh`, `carc_eval.py`, `carc_eval.sh` | cluster scripts |
| local only, deletable | `no_fov/env_wrapper_v2.py`, `no_fov/test_env_wrapper_v2.py` (superseded), `stale_22ch_checkpoints/` (backup of the old 22-ch checkpoints) |

**All old checkpoints are dead** — they were trained at 22 input channels against a
mirrored observation. `stale_22ch_checkpoints/` holds them locally; nothing loads them.

---

## 6. Measurements that took real compute — don't re-derive

| quantity | value |
|---|---|
| mute robot (silent) | success **0.617**, P(reaches correct key) **0.667**, **136.8** steps |
| **mute training return** | `0.667 × key_bonus − 0.067` → key .15→**+0.032**, .30→**+0.132**, .50→**+0.266**, 1.0→**+0.599** |
| no-assist, eval harness | 56.0% success, 139.8 adjusted steps (50 seeds × 3 FOV) |
| `static_120` hand-coded | 75.3% success, 127.9 adj, 6.2 reveals |
| `dynamic` hand-coded | 78.0% success, 125.7 adj, 6.3 reveals |
| assisted human (StaticAssist) | +0.150 success and **−18.1 steps** for **6.3 words** vs mute |
| episode length spread | mean 133.9, **sd 55.9**, range 29–190 |

**Compare training returns against the mute line for that config's `key_bonus`, never
against zero and never across configs** — a bigger `key_bonus` inflates the mute score too.

## 6b. FINAL RESULTS (job 10672642, 50 seeds x 3 FOV, identical layouts)

Baseline = `sweep2/d12.pt` (comm_cost=0.000, key_bonus=0.50, no warmup) — the best of
21 swept checkpoints, copied to `module/checkpoints/rec_ppo_baseline.pt`.

```
SUCCESS %          fov60  fov120  fov180     ALL
no-assist           24.0    64.0    80.0    56.0
baseline (d12)      26.0    76.0    90.0    64.0
static_120          48.0    90.0    88.0    75.3
dynamic             52.0    90.0    92.0    78.0
module              54.0    90.0    90.0    78.0
module(mute)        62.0    90.0    90.0    80.7   <- best

ADJUSTED STEPS     fov60  fov120  fov180     ALL      REVEALS/ep
no-assist          173.8   126.5   119.2   139.8       0.0
baseline (d12)     336.1   248.2   205.6   263.3     128.8
static_120         150.5   114.6   118.7   127.9       6.2
dynamic            147.5   114.7   115.0   125.7       6.3
module             158.0   127.1   129.0   138.1      18.4
module(mute)       137.6   117.6   117.9   124.4       6.3   <- best
```

Deltas vs baseline: module +14.0 pts / −125.3 adj; module(mute) +16.7 pts / −138.9 adj.

**Conclusions**
1. A learned baseline CAN beat no-assist without cheating: d12 is +8.0 pts. But only by
   talking constantly (128.8 reveals), so adjusted steps nearly double.
2. The module does what its header claims: 128.8 → 18.4 reveals (−86%) while *gaining*
   14.0 pts — now demonstrated without the `charge_effective_only` leak.
3. `module(mute)` still beats `module` (80.7 vs 78.0, 6.3 vs 18.4 reveals). The trained
   baseline remains a NET NEGATIVE inside the module. Frame no_fov as an ablation, not a
   foundation.
4. **EVERY config converged to a CONSTANT policy. This is the headline finding.**
   Verified by inspecting action distributions: `d12` and `d04` both emit `key` on
   **100.0% of steps with probability 1.000**; `d02` emits `door` 100%; `d10` `door`
   99.2%. d12 and d04 are different configs yet their evals match to five decimals —
   because they converged to the identical constant.

   Across all 21 checkpoints:
   - 13 → constant `wait` (byte-identical to no-assist)
   - ~6 → constant single reveal, shouted every step (90–150 reveals/ep)
   - 3  → one reveal at episode start, then silent

   **Not one learned a state-dependent policy.** The observation is effectively ignored.
   Notably, state-dependence DOES appear mid-training (an early run showed P(speak)
   ranging 0.281–0.573 across states) and is then lost — PPO collapses to a constant in
   every configuration tried.

   So d12's "+8.0 pts" is NOT learned assistance. It is the effect of broadcasting the
   nearest key's location every timestep — useful information delivered indiscriminately.
   The `module` row is therefore pruning a constant broadcaster, not a learned-but-verbose
   partner, which is why `module(mute)` beats it.

   Next thing to investigate if this is picked up again: why the policy collapses to a
   constant. Candidates - the recurrent encoder's gradient through 190 steps, the
   0.5-Bernoulli terminal reward swamping per-step advantage, or the conv encoder never
   learning useful features because the only reliable signal is action-independent.

## 7. Where the experiment stands

The **old** 22-channel baseline over-talked (~70 reveals/ep) because `charge_effective_only`
made useless words free. That's gone. The **new** baseline under the clean reward
(+1 success − comm − time) reliably converges to **silence**, scoring identically to
no-assist. Prior full comparison (with the mute baseline):

```
                 fov60  fov120  fov180    ALL      adj_ALL   reveals
no-assist         24.0    64.0    80.0    56.0      139.8      0.0
baseline          24.0    64.0    80.0    56.0      140.8      1.0
static_120        48.0    90.0    88.0    75.3      127.9      6.2
dynamic           52.0    90.0    92.0    78.0      125.7      6.3
module            60.0    88.0    86.0    78.0      128.3      7.4
module(mute)      62.0    90.0    90.0    80.7      124.4      6.3   <- best
```

`module(mute)` = FOV logic with **no trained baseline underneath**. It beat `module`,
meaning the trained baseline contributed *negative* value. Goal of the current sweeps:
find a config where the learned baseline is genuinely helpful, so that comparison is fair.

### Constraints the user set (respect these)
- **No cheating**: no FOV, no reads of the human's knowledge base, no action masking, no
  hard-coded assistance logic. A guard on `_resolve` to stop `dead_room`/`empty_room`
  lying was tried and **rejected** — same category as the action mask.
- **Reward numbers and training schedules are fair game.** Structure of
  `my_env_wrapper.py` is not — propose edits, don't restructure it.
- Ideally the module still beats the baseline once the baseline is decent.

### Known trap, unresolved by design
`dead_room` writes into `dead_door_colors` with no truth check. `bayes_agent.py:435` then
makes the human drop that key and `:479` filters it out permanently, so one wrong
`dead_room` on the correct colour makes the episode unwinnable. Measured: an unguarded
speaking policy scored **25.3%** vs **46.7%** for silence. This is why silence is a
rational local optimum, and it is left in on purpose — the policy must learn it.
