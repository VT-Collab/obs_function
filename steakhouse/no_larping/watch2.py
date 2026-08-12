"""watch.py with the EXPERIMENTAL human redundancy model. Same keys, same HUD.

    python watch2.py --layout chefs_table --fov 360 --robot bayes --fps 4

Every flag watch.py takes, this takes -- it IS watch.py, with
`common/tasks_containment.py` imported first. The only difference on screen is what
the HUMAN decides to do.

TWO experimental pieces, both deletable without touching anything shipped:

  HUMAN   common/tasks_containment.py -- the redundancy check counts COMPOSITES.
  FILTER  robot/filter/core/qmdp2.py -- a plan that leaves the robot with no legal
          sub-task from where it ends pays `stuck_penalty`. Measured motivation: on
          chefs_table fov360 the filter parked the robot at (17,5), a pocket with no
          pot, no sink and no hatch, where its whole ladder is illegal -- 21 idle
          ticks against 1 for bayes driving itself.

WHAT DIFFERS IN THE HUMAN. The redundancy check counts COMPOSITES, not just bare
products. `counters_holding("garnish")` cannot see a `garnish_dish`, so today two
garnish_dishes on a counter stop nothing and the human goes on fetching onions,
chopping them, fetching plates and washing them for garnishes and plates it already
has. A composite is the strongest possible evidence the work is done, because it is
the one form the item can no longer be used for anything else as:

    steak_dish    = a plate and a steak
    garnish_dish  = a plate and a garnish
    dish          = all three

So under this model a garnish_dish stops the garnish chain AND the plate chain.
Constructed check, two garnish_dishes on reachable counters: the current model
blocks 0 verbs, this one blocks 10 -- chop, collect_garnish, get_onion, load_board,
take_onion, and collect_plate, get_plate, load_sink, take_plate, wash.

HUMAN ONLY. `legal_subtasks` is shared, and the patch routes on BeliefView: the
human holds one, every baseline holds a TruthView. So the ROBOT you are watching is
bit-identical to the one watch.py shows, whichever `--robot` you pick, and any
difference you see on screen is the partner's doing.

WHAT TO WATCH FOR. The tell is the human going to EXPLORE earlier than it does under
watch.py. Saturation does not itself cause exploring -- it removes candidates, and
`explore` is what limited_vision_human falls through to when the ranked list comes
back empty -- so "stops fetching onions, starts wandering" is the change working
rather than the human getting lost. If you see it fetch a plate with two
garnish_dishes already sitting on counters, the patch did not fire and that is worth
knowing.

Delete this file and `common/tasks_containment.py` together; neither is merged.
"""
import os
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# BEFORE watch.py, so the human's own bound name is patched before any human or
# robot object exists. limited_vision_human did `from common.tasks import
# legal_subtasks`, so patching the module alone would be a no-op -- install()
# handles both and this import is what runs it.
import common.tasks_containment as _patch                          # noqa: E402

# THE EXPERIMENTAL FILTER, swapped in by name. robot/methods.py's _qmdp() closure
# looks `QMDPFilter` up as a module global at BUILD time, so rebinding it here makes
# every qmdp-* row construct QMDPFilter2 -- no registry edit, and nothing in
# core/qmdp.py is touched. Print the class actually built so the window can never
# silently be showing the shipped filter.
import robot.methods as _methods                                    # noqa: E402
from robot.filter.core.qmdp2 import QMDPFilter2                     # noqa: E402
_methods.QMDPFilter = QMDPFilter2

import watch                                                      # noqa: E402


if __name__ == "__main__":
    assert _patch.INSTALLED, "the human patch did not install"
    args = watch.parse_args()
    print("watch2: EXPERIMENTAL human (composites saturate) + EXPERIMENTAL filter "
          "%s (stranding penalised, margin skipped when the baseline is idle)"
          % QMDPFilter2.__module__.rsplit(".", 1)[-1] + ".py")
    watch.Watcher(args).run()
