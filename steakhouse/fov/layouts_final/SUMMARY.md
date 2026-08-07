# Finalized FOV setup — human agent, properties, and layouts

This directory is the **frozen, validated setup** for the limited-vision steak
human. Everything here was measured against the finalized agent
[`fov/human/agent/limited_vision_human.py`](../human/agent/limited_vision_human.py)
(robot-awareness merged in), not an older prototype. Numbers regenerated
2026-07-26.

- `SUMMARY.md` — this file: the setup, the three properties, the ranked layouts.
- `PASSING.md` — team-win + behavioural-separation table (from `batch_validate`).
- `layouts/` — snapshot copies of all 12 `.layout` files (exact grid + times +
  order list), so this setup is self-contained.

---

## 1. What this setup is for

We are building the **human** half of an assistive team so that a *downstream*
FOV-aware robot can out-coordinate an FOV-unaware one. The robot never gets built
here; the deliverable is a human whose behaviour makes that advantage *possible*.
A robot can only win by knowing the human's field of view if the human:

1. **behaves differently under different FOVs** (something to infer), and
2. **reacts to the robot only when it can actually see it** (something the
   FOV-aware robot can exploit that the unaware one cannot).

## 2. Success metric — team delivery

Success is measured at the **team**: an order delivered by *either* agent counts,
because the human's prep feeding a teammate's delivery is a real team win. More
vision is **not** required to raise the human's *own* delivery count (a greedy
teammate harvests shared stations either way) — the FOV signal lives in the
human's *behaviour*, not its delivery total.

## 3. The finalized human

Single file: `fov/human/agent/limited_vision_human.py`. Key settings:

| setting | value | meaning |
|---|---|---|
| `FORGET_HORIZON` | 12 | a belief expires to UNKNOWN 12 steps after last seen |
| `SIGHT_RADIUS` | 8 | max tiles resolved per look (bounds discovery range) |
| `DEFAULT_TEMPERATURE` | 0.5 | subtask sampler `P(τ) ∝ exp(PRIORITY/T)` |
| `ROBOT_SUPPRESS` | 0.3 | soft down-weight on a fetch the teammate is SEEN doing |
| `STATION_TASKS` | pot/board/sink → tasks | station-yield: tasks dropped when the teammate is SEEN at that station |
| `avoid_robot` | `False` (opt-in) | traffic-avoidance: detour around the teammate's seen cell; **off = byte-identical to base** |
| `occlude` | `False` (opt-in) | line-of-sight: walls block the cone (bigger blind spots, esp. at 360°); **off = base**; only restricts sight so no-cheat-safe |

**Policy.** Empty-handed, the human *samples* among the subtasks it currently
believes are available and task-advancing, then commits until it *observes* the
choice is no longer helpful (`τ_t ~ P(·|o_{0:t})`, closed form in
`subtask_distribution` — which is what lets the robot do **exact** Bayesian FOV
inference, no epsilon).

**Teammate (robot) awareness — two FOV-gated channels.** A *visible* teammate is
reacted to in exactly two no-cheating ways, both firing only on what the human can
currently see and both preserving team-win, divergence, inference and no-cheat
(validated as the only survivors of a 7-config sweep — held-item redundancy,
staging, assembly-suppress, complementary-boost and lookahead-prep were all tested
and rejected as anti-gated / harmful / dead).

*Channel 1 — held-object fetch-suppression (what it's carrying).* When the human
has *seen* the teammate holding an item it soft-suppresses (× `ROBOT_SUPPRESS`,
renormalised — never zeroed) and abandons the matching FETCH:

| teammate seen holding | human fetch made redundant |
|---|---|
| meat | `pickup_meat` |
| onion | `pickup_onion` |
| plate | `pickup_plate` |
| washed_plate | `pickup_washed_plate` |

It does **not** react to a held steak/dish (those act on the human's *own* item)
nor to `chop_onion`/`heat_washed_plate` (progress on the human's *own* station).

*Channel 2 — station-yield (where it's standing).* When the human SEES the teammate
standing at / facing a station (pot/board/sink), it **yields** the tasks that use
that station (`STATION_TASKS`), so it does not compete or collide for a station its
partner is already working — *provided a non-blocked task remains* (soft; never
strands the human). This is a purely POSITIONAL signal the held-object channel
cannot give: a teammate at the *empty* pot about to cook shows nothing in its
hands, but being *seen there* is enough to defer. It is the clearest realization
of "the robot stepping into the human's view matters" — the robot's position alone
reshapes the human's plan — and it is the strongest correctly-gated channel found
(fires more the wider the cone, on all 12 layouts).

Both channels are **FOV-gated** (fire only on `beliefs[ROBOT]` / `_robot_seen`,
written only while the teammate is in the cone, and decaying on `forget_horizon`)
and **soft** (a lone option is never removed, so nothing starves). A *hard* version
of channel 1 once inverted the FOV gradient (steak_island → 0 deliveries); softness
is what keeps the human self-sufficient — see the module docstring's "caution
learned the hard way".

**No cheating.** The agent starts knowing nothing — not even where stations are.
Terrain, station locations, station states, and walkability are learned *only* by
tiles passing `visible()`. Routing is the agent's own BFS over seen floor, not a
full-map planner. Audit: `python -m fov.human.test_no_cheating <layout> <seeds>`.

**Rejected channels & the occlusion frontier.** Beyond the two kept channels, four
more teammate reactions were prototyped and rejected on evidence: *held-item
redundancy* and *staging* (anti-gated / idle-harm), *assembly-suppress* (dead —
robot ~never seen holding a steak), *complementary-boost* / *lookahead-prep*
(hurt divergence), and *belief-transfer* (infer a station's state by watching the
teammate act). Belief-transfer is instructive: it fired but was **~entirely
redundant** (additive writes ≈0/episode), for the same structural reason held-item
and staging failed — **`visible()` has no occlusion**, so seeing the teammate act
at a station means seeing the station too. Adding line-of-sight occlusion to
`visible()` is therefore the single highest-value future change: it would enlarge
blind spots (P2), and it is the one thing that would make the entire "learn by
watching the teammate" family (belief-transfer) genuinely additive **and**
no-cheating (you watched the teammate act; you truly cannot see through a wall). It
is a `visible()` model change and would require re-running this whole gauntlet.

## 4. The three properties (and why the robot needs them)

Measured per layout × FOV over `FOVS = [30, 60, 90, 120, 180, 360]`:

- **P1 — legible divergence.** Behaviour genuinely differs across FOVs (min real,
  phase-corrected subtask divergence over key FOV pairs). This is *what the robot
  infers*. Higher is better.
- **P2 — blind spots grow as FOV narrows.** Mean # of advancing tasks available in
  *truth* but not to the human's FOV-gated belief. This is *the robot's
  non-redundant work*; it must **grow** as the cone narrows.
- **P3 — robot-observation is FOV-gated.** Fraction of ticks the human believes
  the teammate holds a reactable item. This must **grow with FOV** — only then is
  the human's reaction to the robot something a *sighted* human does and a blind
  one cannot, which is exactly the lever an FOV-aware robot exploits.

## 5. Validation (fresh-seed re-verified — see §6a)

| check | result |
|---|---|
| team delivers at every FOV (12 layouts) | **12/12** (min 2.1) |
| ≥3 FOV pairs with real behavioural divergence | **12/12** (minDiv 7–16) |
| no illegal belief writes (state read outside FOV) | **PASS** |
| beliefs start UNKNOWN (no omniscient init) | **PASS** |
| map coverage & stations found increase with FOV | **PASS** |
| exact Bayesian FOV inference (final-correct) | **0.84** mean across ALL 12 layouts (chance 0.167); range 0.67 (side_4) – 0.97 (mid_1); unchanged by station-yield; weak only at 120°/180° aliasing |
| P2 (blind spots ↑ as FOV narrows) | **12/12**, blind@360 = 0.00 everywhere |
| P3 (robot-seen ↑ with FOV) | **robust on 8/12** incl. all 4 recommended; flat/seed-marginal on 4 |

Reproduce: `python -m fov.human.batch_validate 6 3` (team + divergence),
`python -m fov.robot.inference.evaluate_sampling_inference` (inference),
`scratchpad setup_design_sweep.py` (P1/P2/P3).

## 6. Ranked layouts — best to not-so-best

All 12 pass team-win + P2 + behavioural separation. They are ranked by **fitness
for the FOV-aware-robot goal**: large P2 (non-redundant work), a *robust* P3
gradient (FOV-gated robot reaction that clearly rises with the cone), and strong
separation (inferability). Numbers are from an **independent fresh-seed
re-verification** (2 seed sets, 8–16 seeds/FOV, adversarial re-check on any
failure — see §6a). `P2@30` is blind-spot magnitude at the narrowest cone; `sep`
is `batch_validate`'s phase-corrected min-divergence over its best FOV triple;
`ΔP3` = robot-seen(360°) − robot-seen(30°) (bigger = more FOV-gated).

| tier | layout | sep | P2@30 → @360 | P3 30→360 (ΔP3) | P3 | team-win |
|---|---|---|---|---|---|---|
| **BEST** | **steak_parrallel** | 16 | **2.00** → 0.00 | 0.00→0.40 (**+0.40**) | robust | ✓ all FOV |
| **BEST** | **steak_mid_2** | 7 | **1.55** → 0.00 | 0.13→0.59 (**+0.46**) | robust | ✓ all FOV |
| **BEST** | **steak_none_3** | 7 | 1.13 → 0.00 | 0.27→0.53 (+0.26) | robust | ✓ all FOV |
| **BEST** | **steak_side_4** | 7 | 1.05 → 0.00 | 0.19→0.52 (+0.33) | robust | ✓ all FOV |
| good | steak_island2 | 16 | 1.00 → 0.00 | 0.28→0.52 (+0.24) | robust | ✓ all FOV |
| good | steak_test | 13 | 0.96 → 0.00 | 0.31→0.55 (+0.24) | robust | ✓ all FOV |
| good | steak_island | 13 | 0.93 → 0.00 | 0.20→0.62 (+0.42) | robust | ✓ all FOV |
| good | steak_side_2 | 11 | 0.90 → 0.00 | 0.25→0.46 (+0.21) | robust | ✓ all FOV |
| weak | steak_mid_1 | 14 | 1.13 → 0.00 | 0.35→0.35 (~0.00) | flat | ✓ all FOV |
| weak | steak_side_3 | 12 | 0.87 → 0.00 | 0.38→0.42 (+0.04) | flat | ✓ all FOV |
| weak | steak_api | 10 | 0.97 → 0.00 | 0.35→0.37 (~0.00) | flat | ✓ all FOV |
| weak | steak_tshape | 13 | 0.73 → 0.00 | 0.32→0.36 (+0.04) | flat | ✓ all FOV |

**BEST / good** = P3 *robust*: robot-observation clearly rises with the cone
(ΔP3 ≳ +0.2), so a sighted human visibly reacts to the robot and a blind one does
not — exactly the lever the FOV-aware robot exploits. **weak** = P3 *flat*
(ΔP3 ≈ ±0.05): the teammate works where even a narrow cone already looks, so
seeing it is barely FOV-gated. The four *weak* layouts are exactly the ones whose
P3 sign **flips between seed sets** — the adversarial re-check confirmed these are
noise around a near-zero gradient, not a stable inversion — so they are poor
testbeds for the robot-reaction lever, though still valid for P1/P2/team.
`steak_api` separates well (sep 10) but its flat P3 disqualifies it for the goal.

**Recommended set for end-to-end robot experiments: the four BEST layouts**, led
by **steak_parrallel** (the flagship — largest blind-spot gradient P2=2.0, and the
cleanest P3: robot-seen rising from **0.00 at 30°** to 0.40 at 360°).

### 6a. Independent verification (fresh seeds)

Re-measured all 12 layouts twice (seed offsets 0 and 100, 8 property-seeds/FOV;
16-seed adversarial re-check on every property failure). The two runs **agree** on
the robust findings and expose the fragile one:

| property | verdict |
|---|---|
| P2 (blind ↑ as FOV narrows) | **12/12, stable** — blind@360 = 0.00 everywhere, blind@30 = 0.7–2.0 |
| team win at every FOV | **12/12, stable** (min 2.1) |
| exact FOV inference (final-correct) | **0.85** mean, stable (0.843 / 0.852 across runs) |
| P1 / separation (behaviour differs by FOV) | **12/12 positive**, magnitude seed-sensitive; steak_island weakest |
| P3 (robot-seen ↑ with FOV) | **robust on 8/12** (incl. all 4 BEST); **flat/seed-marginal on 4** (mid_1, side_3, api, tshape) |

## 7. The BEST-layout definitions

Legend: `X` counter/wall · `P` pot · `M` meat · `O` onion · `D` dish/plate ·
`B` board · `W` sink · `S` serve · `1` human start · `2` robot start · space =
floor. All use `cook_time=15, chop_time=5, wash_time=5, num_items_for_steak=1,
delivery_reward=20`. The `.layout` file's `start_order_list` is a default; the eval
harnesses override it with N identical `steak` orders (batch_validate N=4,
inference/property-sweep N=8). Full files (all 12) are in `layouts/`.

### steak_parrallel  — flagship (two station clusters + interior block)
```
XXXXXXXXXXXXXXX
XXXPXMOXXDXXXXX
XX           XX
XX           XX
XX  XXBXWXX  XX
XX  XXXXXXX  XX
XX        2  XX
XS    1      XX
XXXXXXXXXXXXXXX
XXXXXXXXXXXXXXX
```

### steak_mid_2  — central board/sink/serve pocket
```
XXXXXXXXXXXXXXX
XXXOXMXXXPXXDXX
XX           XX
XX 2         XX
XX        1  XX
XX   XBWXX   XX
XX   XXSSX   XX
XX           XX
XX           XX
XX           XX
XXXXXXXXXXXXXXX
```

### steak_none_3  — left-wall prep column, right serve bay
```
XXXXXXXXXXXXXXX
XXXBXXPXDXXXXXX
XX        SXXXX
XW 2      SXXXX
XX        XXXXX
XM      1 XXXXX
XO        XXXXX
XX        XXXXX
XXXXXXXXXXXXXXX
XXXXXXXXXXXXXXX
```

### steak_side_4  — top prep row, split serve alcove
```
XXXXXXXXXXXXXXX
XXXBXPXDXMXXXXX
XX        XXXXX
XW 2      XXXXX
XX     1  OXXXX
XXXXXX    XXXXX
XXSSXX    XXXXX
XX        XXXXX
XXXXXXXXXXXXXXX
XXXXXXXXXXXXXXX
```

## 8. Performance: larger vs smaller FOV (greedy FOV-unaware partner)

Measured across all 12 layouts (6 seeds, fixed-horizon throughput = deliveries per
H=300 with a large order list so nothing stalls). Three effects, ranked by how
universal they are:

| effect | how universal | narrow → wide |
|---|---|---|
| **exploration cost** (human steps spent mapping/checking) | **12/12** falls with FOV | up to **300 → 0** steps (parrallel), 139→0 (mid_2) |
| **human own-contribution** (own deliveries / share) | **10/12** rises with FOV | own-share up to **0 → 0.94** (mid_2), 0 → 0.88 (side_4) |
| **team throughput** (total deliveries) | **layout-dependent**, mean **+16%** | 8/12 gain ≥10%; 3/12 flat; tshape **−17%** |

Team throughput narrow→wide, recommended set: parrallel **4.0→5.5 (+36%)**,
mid_2 **4.0→5.1 (+27%)**, none_3 4.4→4.8 (+10%), side_4 4.7→4.5 (−4%).

**Reading:** the FOV effect is dramatic in *who does the work* (a narrow-FOV human
is often dead weight — own≈0, the greedy partner carries it — while a wide-FOV one
does most deliveries) but only weak/inconsistent in *team throughput*, because the
greedy partner reactively substitutes for whatever the blind human drops
(steak_side_4: own-share 0→0.88 yet team flat = perfect substitution). That
flat/modest team gradient under a greedy partner is precisely the **headroom for an
FOV-aware robot**: the narrow human's wasted exploration and dropped work are
absorbed *reactively* today; a robot that knows the human is blind could cover
those blind spots *proactively* and finish faster. Quantifying that aware-vs-unaware
**time** gap is the deferred end-to-end experiment. It is also why success is scored
at the team (team-win) and the FOV signal is read from *behaviour* (P1), not
throughput — throughput alone is not a clean monotonic FOV signal (tshape −17%).
Measure: `scratchpad/perf_vs_fov.py <layout> <seeds> <N> <H>`.
