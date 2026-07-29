# Layout library — full test battery

32 layouts run through the **complete** test battery against the finalized human
([`limited_vision_human.py`](../human/agent/limited_vision_human.py)). Generated
2026-07-27 (4 seeds/battery, 3 seeds/inference, 6 FOVs).

**Result: 31 / 32 pass every test; 25 of those are *influential*; 1 fails.**
All 31 passing `.layout` files are snapshotted in [`layouts/`](layouts/).

## The six tests

| test | pass condition |
|---|---|
| **team-win** | team delivers ≥1 at every FOV |
| **P1** (legible) | cross-FOV subtask divergence (min over key FOV pairs) ≥ 3 |
| **P2** (blind spots) | blind-spot count at 30° > at 360° |
| **no-cheat** | 0 illegal belief writes (never a belief for a non-visible tile) |
| **inference** | exact Bayesian FOV final-correct ≥ 0.5 (chance = 0.167) |
| **INFL** (robot matters) | aware-vs-robot-blind subtask divergence at wide FOV; **influential if ≥ 8** — how much *seeing the robot* changes the human's behaviour, i.e. the collaboration lever |

A layout **PASSES** if the first five hold. **INFL** is the extra property you
asked for — it is *not* a pass/fail gate, it ranks how much the robot genuinely
influences the human (higher = an FOV-aware robot has more to exploit).

## Headline finding: contention drives influence

The most **influential** layouts are the **cramped / clustered** ones (`gc*` =
generated-clustered, `cram*` = hand-made contention): single stations + tight
shared space force the human to react to the robot constantly, and it stays high
even at **full vision** (360°). The least influential are the big **open** rooms
(`steak_island` INFL 2.2). So for the FOV-aware-robot goal, **prefer the
contention layouts.**

## Tier 1 — PASS + influential (robot matters a lot) — recommended set

Ranked by robot-influence. `gs*`/`gc*` are auto-generated (spread / clustered);
the rest are hand-designed. All pass team-win, P1, P2, no-cheat, inference.

| layout | INFL (wide) | P1 | infer | team min | kind |
|---|---|---|---|---|---|
| **steak_gc00** | **28.5** | 22 | 0.83 | 4.5 | 3×3 max-contention |
| **steak_gc06** | **27.0** | 11 | 0.83 | 7.0 | clustered |
| **steak_gc04** | **23.4** | 18 | 0.89 | 5.5 | clustered |
| **steak_cram2** | **23.1** | 15 | 0.83 | 3.8 | packed stations |
| **steak_cram** | **18.5** | 8 | 0.83 | 2.5 | tight room |
| steak_gs00 | 18.2 | 15 | 1.00 | 4.0 | small spread |
| steak_mid_1 | 16.9 | 14 | 0.94 | 2.5 | hand |
| steak_gc03 | 15.6 | 15 | 0.83 | 5.0 | clustered |
| steak_gc05 | 14.5 | 11 | 0.78 | 4.2 | clustered |
| steak_side_3 | 14.2 | 13 | 0.83 | 3.8 | hand |
| steak_gs07 | 13.6 | 13 | 0.83 | 3.0 | spread |
| steak_side_4 | 13.2 | 10 | 0.67 | 4.0 | hand |
| steak_api | 12.8 | 21 | 0.78 | 2.5 | hand |
| steak_gc07 | 12.8 | 5 | 0.78 | 4.8 | clustered |
| steak_gc01 | 12.4 | 18 | 0.89 | 4.2 | clustered |
| steak_gs03 | 11.8 | 12 | 0.89 | 4.5 | spread |
| steak_gs04 | 11.4 | 8 | 0.94 | 4.0 | spread |
| steak_parrallel | 11.2 | 12 | 0.89 | 4.0 | hand (flagship) |
| steak_gs02 | 11.2 | 14 | 0.89 | 5.0 | spread |
| steak_gs08 | 10.9 | 8 | 0.89 | 4.0 | spread |
| steak_mid_2 | 9.6 | 15 | 0.94 | 4.0 | hand |
| steak_gs09 | 9.5 | 12 | 0.89 | 3.5 | spread |
| steak_none_3 | 8.5 | 7 | 0.83 | 4.0 | hand |
| steak_tshape | 8.5 | 12 | 0.89 | 2.5 | hand |
| steak_gs05 | 8.5 | 13 | 0.89 | 3.8 | spread |

## Tier 2 — PASS but low-influence (robot matters little; INFL < 8)

Fully valid testbeds (all five gates pass) but weak for the robot-collaboration
lever — the human barely changes on seeing the robot. Kept in the library; just
not first-choice for FOV-aware-robot experiments.

| layout | INFL (wide) | P1 | infer | team min |
|---|---|---|---|---|
| steak_gs06 | 7.9 | 11 | 0.67 | 3.5 |
| steak_gc02 | 7.9 | 17 | 0.83 | 4.0 |
| steak_island2 | 7.8 | 4 | 0.83 | 3.5 |
| steak_test | 6.2 | 10 | 0.78 | 3.0 |
| steak_side_2 | 4.4 | 7 | 0.78 | 4.2 |
| steak_island | 2.2 | 5 | 0.83 | 3.0 |

## Failed (1) — not in the library snapshot

| layout | failing test | detail |
|---|---|---|
| **steak_gs01** | **team-win** | team = **0.0 at fov30** (the narrow-cone human deadlocks and the pair delivers nothing); 5–6 at every wider FOV. Passes P1 (9), P2, no-cheat, inference (1.00) — it is *only* the fov30 deadlock that fails it. A generated spread layout whose geometry starves the narrow cone; dropped from the library. |

## Performance value of robot-awareness (does the influence pay off?)

Robot-**aware** human vs robot-**blind** human (ignores the teammate), same greedy
partner + seeds, team throughput (deliveries / fixed horizon):

| layout group | team aware | team blind | gain |
|---|---|---|---|
| **contention** (gc00/gc06/cram/cram2/mid_1/side_3) | **5.9** | 5.2 | **+12.5%** |
| spread (parrallel/mid_2/none_3/island/side_2/test) | 4.6 | 4.4 | +4.7% |

Biggest per-layout gains: steak_gc00 **+21%**, steak_gc06 **+21%**, steak_cram +12%,
steak_mid_1 +10%. So the robot-influence is not just behavioural — on contention
layouts it lifts team throughput **12–21%** purely from the human reacting to a
*fixed* greedy robot. That is a **lower bound** on the FOV-aware-robot payoff (an
aware robot that actively manages its visibility should do at least this well),
and it confirms contention layouts as the testbed where seeing the robot matters
most. Definitive proof still needs the end-to-end aware-vs-unaware robot run.
Measure: `scratchpad/perf_value.py <layout>`.

## Reproduce

`python .../scratchpad/fulltest_layout.py <layout> 4 3 12 300` prints the full
JSON verdict for any layout. The generator is
`scratchpad/generate_layouts.py` (`gs*` spread, `gc*` clustered).
