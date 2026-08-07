"""Import shim. Import this FIRST from every file in this package.

Same job as the block at the top of filter/baseline.py: official_baselines is
written to be run with ITSELF on sys.path (utils.features, algorithm.* and SP.*
are top-level there), and the steakhouse root has to be importable for
overcooked_ai_py and fov.*.

This package does NOT import filter/baseline.py to get the shim. That file pulls
in torch, the checkpoint loader and the whole trained-policy stack, none of which
this package needs -- the baseline here is hand-written. Duplicating eight lines
is cheaper than a dependency on 79KB of unrelated machinery.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
#  .../fov/robot/policy/new/setup  ->  5 up is steakhouse/
STEAKHOUSE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
BASELINES = os.path.join(STEAKHOUSE, "fov", "robot", "policy", "new",
                         "official_baselines")
#We reuse ONE thing from the filter package: the Bayes kernel in its
#inference.py, which fov_filter.py subclasses rather than copies (that is the
#code which measured 69-100% MAP accuracy, and a copy would silently drift from
#it). It imports nothing but overcooked_ai_py and the human model -- in
#particular NOT filter/baseline.py -- so it costs us no torch and no checkpoint
#stack.
#
#DELIBERATELY NOT ON sys.path. filter/ contains qmdp.py, cost_function.py and
#play_episode.py, and this package has files with all three of those names. Any
#sys.path entry for it shadows ours, and the failure is vicious: `import qmdp`
#silently binds the OTHER package's module, so you get an AttributeError about a
#method you are looking straight at. fov_filter.py loads the kernel by FILE PATH
#instead, which cannot collide with anything.
FILTER_DIR = os.path.join(STEAKHOUSE, "fov", "robot", "policy", "new", "filter")

for _p in (STEAKHOUSE, BASELINES, os.path.dirname(STEAKHOUSE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#and make sure OUR directory wins over everything inserted above
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)
