"""
full_state/ - the FOV-BLIND baseline given the FULL physical state (everything a
fully-observing teammate could know) EXCEPT the human's private FOV cone/beliefs.

Motivation: the original baseline observation (baseline/features.py, 17 dims) is a
deliberately coarse hand-compression - per-kind 0/.5/1 station states, binary
"is the human holding something", no robot self-state, no orientations. That
makes the FOV module's job easier than it should be. This package gives the
baseline the full state so the honest question can be asked: does inferring the
human's FOV STILL help a partner that already knew everything observable?

Two encodings, measured against each other:
  features_flat.py  per-kind, fixed-dim vector -> the existing MLP + PPO
  features_grid.py  exact per-object C x H x W sheets -> a small CNN (policy_cnn.py)

Excluded from BOTH (the experiment): the human's FOV cone, belief map, posterior,
entropy. Everything else physical/observable is encoded losslessly-ish (exact
timers are dropped as low-value, per design).
"""
