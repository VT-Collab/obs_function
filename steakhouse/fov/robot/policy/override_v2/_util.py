"""Small shared helpers for the override experiment."""
import contextlib
import os
import sys


@contextlib.contextmanager
def quiet():
    """Silence the overcooked greedy model's per-step debug prints during a
    rollout. Restores stdout on exit. Use around env.reset()/env.step() loops;
    print your own progress to sys.stderr so it survives."""
    devnull = open(os.devnull, "w")
    old = sys.stdout
    try:
        sys.stdout = devnull
        yield
    finally:
        sys.stdout = old
        devnull.close()
