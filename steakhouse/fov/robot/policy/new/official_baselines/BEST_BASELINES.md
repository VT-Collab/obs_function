# 🏆 BEST BASELINE PER LAYOUT — the checkpoint to actually load

Measured against `LimitedVisionSteakHuman`, **8,820 paired episodes**, 
collisions OFF, argmax, 10 episodes per (algo, seed, fov). 
Metric is **completion time** (tick of the 3rd delivery, `500` = DNF) — 
reward saturates at 60 and cannot separate these (§6).

Every cell below is **3.00 deliveries, finish rate 1.00**. The number is speed.

```
path:  /scratch1/$USER/steakhouse_zsc/<algo>/<layout>_seed<S>/sp_<layout>.pt
```

| layout | grid | cells | fov | **BEST** | algo/seed | median seed | worst seed | human ALONE |
|---|---|---:|---:|---:|---|---:|---:|---:|
| gc00 | 5x5 | 25 | 30 | **137.9** | `e3t/4` | 198.9 | 500.0 | 171.6 |
| gc00 | 5x5 | 25 | 60 | **124.7** | `sp/11` | 159.4 | 364.4 | 172.2 |
| gc00 | 5x5 | 25 | 90 | **118.7** | `sp/5` | 148.3 | 278.3 | 140.2 |
| gc00 | 5x5 | 25 | 120 | **118.7** | `sp/5` | 148.3 | 278.3 | 140.2 |
| gc00 | 5x5 | 25 | 180 | **105.7** | `sp/14` | 127.2 | 185.8 | 142.4 |
| gc00 | 5x5 | 25 | 360 | **103.1** | `sp/14` | 122.1 | 242.7 | 139.0 |
| | | | | | | | | |
| gc03 | 7x6 | 42 | 30 | **149.6** | `sp/13` | 179.0 | 231.8 | 204.4 |
| gc03 | 7x6 | 42 | 60 | **152.5** | `e3t/1` | 297.7 | 500.0 | 366.8 |
| gc03 | 7x6 | 42 | 90 | **120.0** | `e3t/3` | 157.2 | 359.0 | 154.4 |
| gc03 | 7x6 | 42 | 120 | **120.0** | `e3t/3` | 157.2 | 359.0 | 154.4 |
| gc03 | 7x6 | 42 | 180 | **120.7** | `e3t/3` | 147.5 | 328.8 | 154.4 |
| gc03 | 7x6 | 42 | 360 | **119.0** | `e3t/3` | 147.4 | 325.0 | 153.0 |
| | | | | | | | | |
| gc04 | 7x7 | 49 | 30 | **131.0** | `sp/9` | 172.0 | 283.4 | 202.4 |
| gc04 | 7x7 | 49 | 60 | **141.7** | `sp/12` | 169.0 | 290.6 | 170.8 |
| gc04 | 7x7 | 49 | 90 | **121.0** | `sp_eps/5` | 140.0 | 163.8 | 156.8 |
| gc04 | 7x7 | 49 | 120 | **121.0** | `sp_eps/5` | 140.0 | 163.8 | 156.8 |
| gc04 | 7x7 | 49 | 180 | **124.5** | `sp_eps/2` | 137.9 | 163.8 | 156.8 |
| gc04 | 7x7 | 49 | 360 | **126.4** | `sp_eps/3` | 131.8 | 267.4 | 156.8 |
| | | | | | | | | |
| gc06 | 5x7 | 35 | 30 | **139.0** | `sp_eps/4` | 177.7 | 357.7 | 180.6 |
| gc06 | 5x7 | 35 | 60 | **136.7** | `e3t/1` | 188.9 | 432.4 | 171.8 |
| gc06 | 5x7 | 35 | 90 | **120.1** | `e3t/4` | 140.2 | 206.5 | 142.0 |
| gc06 | 5x7 | 35 | 120 | **120.1** | `e3t/4` | 140.2 | 206.5 | 142.0 |
| gc06 | 5x7 | 35 | 180 | **117.3** | `sp/8` | 158.9 | 220.5 | 143.2 |
| gc06 | 5x7 | 35 | 360 | **112.6** | `sp/8` | 133.7 | 185.5 | 142.0 |
| | | | | | | | | |
| cram | 9x9 | 81 | 30 | **199.2** | `sp/7` | 445.4 | 500.0 | 254.0 |
| cram | 9x9 | 81 | 60 | **198.6** | `sp/7` | 231.4 | 396.4 | 242.6 |
| cram | 9x9 | 81 | 90 | **167.4** | `sp/8` | 217.5 | 316.7 | 231.4 |
| cram | 9x9 | 81 | 120 | **154.4** | `sp/8` | 238.0 | 323.5 | 238.4 |
| cram | 9x9 | 81 | 180 | **174.6** | `sp_eps/2` | 215.0 | 333.8 | 210.4 |
| cram | 9x9 | 81 | 360 | **178.3** | `sp/7` | 196.5 | 332.5 | 204.6 |
| | | | | | | | | |
| cram2 | 13x5 | 65 | 30 | **170.1** | `sp/15` | 213.8 | 405.7 | 204.4 |
| cram2 | 13x5 | 65 | 60 | **149.7** | `sp/12` | 175.5 | 440.8 | 166.8 |
| cram2 | 13x5 | 65 | 90 | **152.4** | `sp/10` | 172.2 | 222.5 | 167.8 |
| cram2 | 13x5 | 65 | 120 | **152.4** | `sp/10` | 172.2 | 222.5 | 167.8 |
| cram2 | 13x5 | 65 | 180 | **141.9** | `e3t/2` | 164.5 | 208.9 | 166.0 |
| cram2 | 13x5 | 65 | 360 | **136.9** | `sp/13` | 164.7 | 360.3 | 166.0 |
| | | | | | | | | |
| gs00 | 6x5 | 30 | 30 | **174.9** | `sp/13` | 500.0 | 500.0 | 500.0 |
| gs00 | 6x5 | 30 | 60 | **134.4** | `sp/10` | 179.6 | 223.3 | 188.0 |
| gs00 | 6x5 | 30 | 90 | **141.2** | `sp/12` | 174.2 | 263.8 | 173.0 |
| gs00 | 6x5 | 30 | 120 | **138.6** | `sp/12` | 173.7 | 363.9 | 173.0 |
| gs00 | 6x5 | 30 | 180 | **132.4** | `sp/10` | 159.9 | 277.1 | 170.8 |
| gs00 | 6x5 | 30 | 360 | **134.0** | `sp/14` | 153.7 | 247.4 | 169.0 |
| | | | | | | | | |

### What this table says

**1. The seed matters far more than the algorithm.** Spread across algorithm
means is **7.5 ticks**; spread across individual seed cells is **397 ticks**.
Best-in-cell counts are `sp 25 / e3t 11 / sp_eps 6` across 42 layout×fov cells —
and SP wins most simply because it has 11 seeds to E3T's 5. There is no best
*algorithm*; there is a best **checkpoint** per layout.

> This does not contradict §'THE HEADLINE' — that table ranks arms against the
> **random floor** under **sampled** actions. This one ranks raw completion time
> under **argmax**. Reconcile before either goes in a paper.

**2. The median seed is often much worse than the best**, and sometimes DNFs
where the best seed finishes comfortably (`gs00` fov30: best 174.9, median 500.0).
Reporting a single checkpoint reports a lottery ticket.

**3. `random` ≈ `human alone` on every layout** — an untrained robot is worth
nothing, neither helping nor hindering. So any measured effect is attributable
to the policy, which makes the floor clean.

**4. FOV cost is the human's own SEARCH time.** Completion time at fov30 minus
fov360 matches the human's extra `n_checks + n_explore` almost one-for-one on
every layout (gc00 +32 vs +35.5 · gc04 +45 vs +54.8 · cram +46.5 vs +40 ·
gs00 +330 vs +400). The robot cannot look on their behalf and cannot move a
station into their cone, so most of that cost is **structurally unreachable**
by any robot policy.

**5. Grid size is not the difficulty axis — topology is.** `cram2` is 13×5 = 65
cells and easy; `cram` is 9×9 = 81 and the slowest of the seven. Clustered
kitchens keep the 8-step chain short; spread ones turn it into a long walk
(`CARC_RUNS.md` §6 measured the same thing across 25 layouts).

