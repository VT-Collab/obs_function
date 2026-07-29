"""
override_v2/ - the FULL-AUTHORITY FOV override experiment, as a self-contained
new set of files. Nothing here modifies the validated pipeline; the frozen
building blocks are imported read-only:

    baseline/features.py      the FOV-blind observation + ACTIONS vocabulary
    baseline/policy.py        the ActorCritic network
    baseline/env_wrapper.py   RobotAssistEnv (the base dynamics + fair planner)
    inference/bayes_fov_sampling.py   the exact Bayesian FOV filter

What is NEW here:
    override.py   FOVOverride (task authority) + ConeReroute (trajectory authority)
    env.py        OverrideEnv - baseline SUGGESTS, module may OVERRIDE anything
    train.py      trains a FOV-blind baseline checkpoint (quiet, fast)
    evaluate.py   baseline vs override, paired per (FOV, seed): delivery + time
    tests.py      mechanism tests (override fires / defers / no crash)

The experiment: does a module that may override the baseline's task AND its path,
using only inferred FOV + entropy, deliver orders in LESS time with NO WORSE
delivery than the frozen FOV-blind baseline it sits on top of?
"""
