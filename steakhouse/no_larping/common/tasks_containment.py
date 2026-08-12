"""TEMPORARY, EXPERIMENTAL: make the HUMAN's redundancy check see composites.

    import common.tasks_containment as _   # patches common.tasks on import

Not merged into `tasks.py` on purpose. It changes what the human decides, so it
moves every number in RESULTS.md, and the whole point of the experiment it was
written for is to find out whether the qmdp gain survives it. If the gain
disappears, this file gets deleted rather than merged.

WHAT IS WRONG WITH THE CURRENT CHECK. Two tables decide when the human stops
producing something:

    _BECOMES      {raw -> what it turns into}, gating the DISPENSER
    _CHAIN_VERBS  {product -> every verb in the chain that makes it}, gating the
                  whole chain through saturated()

Both count the BARE product only. `counters_holding("garnish")` does not see a
`garnish_dish`, so two garnish_dishes on a counter stop nothing: the human goes on
fetching onions, chopping them, fetching plates and washing them, for garnishes and
plates it already has. The composites are exactly the things it can no longer use
for anything else, so they are the strongest possible evidence the work is done.

WHAT A COMPOSITE CONTAINS, which is the whole content of this file:

    steak_dish    = a plate and a steak
    garnish_dish  = a plate and a garnish
    dish          = a plate, a steak and a garnish

So a garnish_dish should stop the garnish chain AND the plate chain -- onion,
chop, collect_garnish, and get_plate, load_sink, wash, collect_plate alike.

HUMAN ONLY, AND THAT IS A REAL DISTINCTION rather than caution. `legal_subtasks` is
shared: the human calls it with a BeliefView and every baseline calls it with a
TruthView. Patching the tables outright would change the BASELINES too, and the
baselines are the control this experiment is measured against -- if they move, the
comparison measures nothing. The wrapper below swaps the tables only for
BeliefView callers, so the robot side of the grid is bit-identical to the grid in
RESULTS.md section 2c.

The swap is safe because everything here is single-threaded and the tables are read
inside the call, never held across it.
"""
import os
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import tasks                                           # noqa: E402
from common.views import BeliefView                                # noqa: E402

# raw -> everything already made that contains it. The dispenser gate counts these.
BECOMES_V2 = {
    "meat": ("steak_dish", "dish"),
    "onion": ("garnish", "garnish_dish", "dish"),
    "plate": ("washed_plate", "steak_dish", "garnish_dish", "dish"),
}

# product -> the chain that makes it. Unchanged from tasks.py; only the COUNTING of
# how much already exists changes, in _saturated_v2 below.
CHAIN_VERBS_V2 = dict(tasks._CHAIN_VERBS)

# product -> composites that already contain one of it.
CONTAINED_IN = {
    "washed_plate": ("steak_dish", "garnish_dish", "dish"),
    "garnish": ("garnish_dish", "dish"),
    "steak_dish": ("dish",),
}

_orig_legal = tasks.legal_subtasks
_orig_saturated = tasks.saturated


def _saturated_v2(view, ok):
    """tasks.saturated, counting composites toward each product's surplus.

    The consume verbs stay legal exactly as the original intends -- take_garnish,
    take_washed_plate and take_steak_dish are absent from _CHAIN_VERBS and this does
    not add them -- so a pile can still drain. That matters more here than before:
    if a garnish_dish blocks the plate chain and nothing can consume the
    garnish_dish, the human freezes instead of finishing, and the eval would read
    that as the change being harmful when it is really a deadlock.
    """
    blocked = set()
    for product, verbs in CHAIN_VERBS_V2.items():
        n = len([c for c in view.counters_holding(product) if ok(c)])
        for comp in CONTAINED_IN.get(product, ()):
            n += len([c for c in view.counters_holding(comp) if ok(c)])
        if n >= tasks.SURPLUS_AT:
            blocked |= verbs
    return blocked


def legal_subtasks(view, held, reachable=None, allow_stash=True):
    """tasks.legal_subtasks, composite-aware for BeliefView callers only."""
    if not isinstance(view, BeliefView):
        return _orig_legal(view, held, reachable, allow_stash)
    b, c, s = tasks._BECOMES, tasks._CHAIN_VERBS, tasks.saturated
    tasks._BECOMES, tasks._CHAIN_VERBS, tasks.saturated = (
        BECOMES_V2, CHAIN_VERBS_V2, _saturated_v2)
    try:
        return _orig_legal(view, held, reachable, allow_stash)
    finally:
        tasks._BECOMES, tasks._CHAIN_VERBS, tasks.saturated = b, c, s


def install():
    """Patch common.tasks AND the human's already-bound reference to it."""
    tasks.legal_subtasks = legal_subtasks
    # limited_vision_human did `from common.tasks import legal_subtasks`, so it
    # holds its own name and patching the module alone would do nothing.
    import human.limited_vision_human as h
    h.legal_subtasks = legal_subtasks
    return True


INSTALLED = install()
