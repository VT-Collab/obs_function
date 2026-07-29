# layouts_final — index

Frozen, validated setup for the limited-vision steak human. Regenerated
2026-07-26 against the finalized agent (robot-awareness merged into
[`fov/human/agent/limited_vision_human.py`](../human/agent/limited_vision_human.py)).

**Read [`SUMMARY.md`](SUMMARY.md) first** — it is the authoritative document:
the finalized human and its settings, the success metric (team win), the three
target properties (P1 legible divergence, P2 blind-spots-grow-as-FOV-narrows,
P3 FOV-gated robot observation), the full validation table, and the layouts
ranked best → not-so-best with the BEST-layout grids embedded.

Contents:

| file | what |
|---|---|
| `SUMMARY.md` | setup + properties + ranked layouts + validation (start here) |
| `LAYOUTS.md` | **full 32-layout library**: every layout × the complete test battery (team-win, P1, P2, no-cheat, inference, robot-influence), pass/fail + ranked by how much the robot influences the human |
| `PASSING.md` | older team-win + separation table (12 layouts, `batch_validate`) — superseded by LAYOUTS.md |
| `layouts/*.layout` | snapshot of all **31 passing** layout files (grid + times + orders) |

**No-cheating guarantee (still enforced).** The agent starts knowing nothing —
station locations, states, tile walkability and routes are all learned only by
tiles passing `visible()`; routing is its own BFS over seen floor, never a
full-map planner. Audit: `python -m fov.human.test_no_cheating <layout> <seeds>`
(checks 1–2 are the cheat guarantees; check 3 "acts on wrong belief" is a noisy,
layout-dependent realism proxy — clean on e.g. `steak_mid_2`, structurally ~0 on
`steak_island`, where the narrow cone maps unusually well and forgets instead).

**Recommended set:** the four BEST layouts — `steak_parrallel` (flagship),
`steak_mid_2`, `steak_none_3`, `steak_side_4`. See SUMMARY §6–7.
