"""MISHA NEW CHANGE - quick sanity check of the soft-likelihood StickyFOVBayesFilter
on just 2 trials (rank01, true_fov=34 and true_fov=96) before committing to the
full 23-layout rerun."""
from my_methods.bayesian.fov_inference_batch import run_inference

for true_fov in (34, 96, 156):
    r = run_inference((1, true_fov))
    print(r, flush=True)
