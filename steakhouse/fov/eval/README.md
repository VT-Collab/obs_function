# fov/eval — reproducible test harnesses

Standalone scripts (persisted from the session scratchpad) that measure the
finalized human on a given layout. All run against
`fov/human/agent/limited_vision_human.py`; conda env **steakhouse-ai** (py3.8),
interpreter `/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python`. Run from
the `steakhouse/` repo root.

| script | what it measures | usage |
|---|---|---|
| `fulltest_layout.py` | **the full battery** — team-win, P1 (divergence), P2 (blind spots), no-cheat, inference, robot-influence — one JSON verdict per layout | `python fov/eval/fulltest_layout.py <layout> 4 3 12 300` |
| `perf_value.py` | robot-**aware** vs robot-**blind** human: team throughput + wasted-work (does reacting to the robot pay off?) | `python fov/eval/perf_value.py <layout> 6 14 300` |
| `influence.py` | robot-in-view **influence**: seen-frac + flip-on-seen (does seeing the robot change the decision, per tick) + team; optional ROBOT_SUPPRESS arg | `python fov/eval/influence.py <layout> 6 12 300 [suppress]` |
| `generate_layouts.py` | regenerate the `steak_gs*` (spread) / `steak_gc*` (clustered/contention) layout library into `data/layouts/` | `python fov/eval/generate_layouts.py` |

Canonical repo harnesses (unchanged, also part of the suite):
- `python -m fov.human.batch_validate 6 3` — team-win + behavioural separation, 12 layouts.
- `python -m fov.robot.inference.evaluate_sampling_inference <layout_csv> 6` — exact FOV inference.
- `python -m fov.human.test_no_cheating <layout> 6` — 6 no-cheating checks.

To full-test the whole library, loop `fulltest_layout.py` over the names in
`fov/layouts_final/layouts/` (or fan out one process per layout).
Occlusion is now a base toggle: `LimitedVisionSteakHuman(..., occlude=True)`.
