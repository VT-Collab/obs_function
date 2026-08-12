"""evaluate.py with the EXPERIMENTAL human redundancy patch installed.

    python -m robot.filter.dev.eval_human_v2 --layouts divide --fovs 90 \
        --seeds 0-4 --methods handoff,qmdp --horizon 400 --out v2.jsonl

Identical to `robot.filter.harness.evaluate` in every respect except that
`common.tasks_containment` is imported first, which makes the HUMAN's redundancy
check count composites (a garnish_dish is a garnish AND a plate). The robot side is
untouched -- the patch routes on BeliefView, and only the human holds one.

Rows are tagged `human_v2: true` so a v1 and a v2 grid can never be pooled by
accident. That is the same hazard that nearly corrupted RESULTS.md section 2c, where
an older grid sat in the same directory under names that NAME_FIX mapped onto the
current ones.

WHAT THE EXPERIMENT IS FOR. The qmdp gain in section 2c is large (+23.7 to +29.7
dishes). Part of it could be the filter exploiting a human that over-produces:
if the partner keeps fetching onions it already has, a robot that hands over the
RIGHT thing looks better than it is. Fixing the human should shrink that part. If
the gain survives, the patch is worth merging and re-measuring properly; if the gain
disappears, the gain was an artefact of a bug in the partner and the patch should be
deleted rather than merged.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.environ.get("STEAK_ROOT", os.path.dirname(ROOT)))
sys.path.insert(0, ROOT)

# BEFORE the harness, so the human's bound name is patched before any robot or
# human object exists.
import common.tasks_containment as _patch                          # noqa: E402,F401

from robot.filter.harness import evaluate                          # noqa: E402

_orig_run = evaluate.run_episode


def run_episode(*args, **kw):
    r = _orig_run(*args, **kw)
    r["human_v2"] = True
    return r


evaluate.run_episode = run_episode

if __name__ == "__main__":
    assert _patch.INSTALLED
    sys.exit(evaluate.main() or 0)
