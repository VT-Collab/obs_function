# Layouts satisfying BOTH FOV conditions

Validated with the from-scratch limited-vision human
(fov/human/agent/limited_vision_human.py). Divergence is
phase-corrected, so pure timing differences score zero.

| layout | sep pairs | team@30 | team@90 | team@360 | best FOV triple |
|---|---|---|---|---|---|
| steak_parrallel | 12/15 | 3.0 | 3.0 | 3.0 | (30, 120, 180) (min div 16) |
| steak_island2 | 12/15 | 3.0 | 3.0 | 3.0 | (30, 60, 180) (min div 14) |
| steak_mid_1 | 13/15 | 3.0 | 3.0 | 3.0 | (30, 60, 180) (min div 14) |
| steak_side_3 | 13/15 | 3.0 | 3.0 | 3.0 | (30, 60, 360) (min div 13) |
| steak_island | 14/15 | 3.0 | 3.0 | 3.0 | (60, 90, 180) (min div 13) |
| steak_tshape | 14/15 | 3.0 | 2.0 | 2.0 | (30, 120, 180) (min div 13) |
| steak_test | 11/15 | 3.0 | 3.0 | 3.0 | (30, 60, 360) (min div 13) |
| steak_side_2 | 11/15 | 3.0 | 3.0 | 2.7 | (30, 60, 90) (min div 10) |
| steak_api | 13/15 | 3.0 | 3.0 | 3.0 | (60, 90, 360) (min div 10) |
| steak_side_4 | 8/15 | 3.0 | 3.0 | 3.0 | (30, 60, 360) (min div 7) |
| steak_none_3 | 12/15 | 3.0 | 3.0 | 3.0 | (30, 90, 360) (min div 7) |
| steak_mid_2 | 12/15 | 3.0 | 3.0 | 3.0 | (30, 90, 360) (min div 7) |
