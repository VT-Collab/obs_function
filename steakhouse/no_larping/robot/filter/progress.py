"""Phi(s) -- how far along the recipe is, in units of one delivered dish.

The QMDP rollouts used to score on `sparse` alone, and sparse only ever fires on
a delivery. Forty ticks is not enough to carry a raw onion all the way to the
pass, so every candidate came back 0.0, Q was flat, and the filter degenerated
into whatever its tie-break happened to be. This is the dense signal that makes
the rollouts discriminate.

The env's own `shaped_reward` cannot do that job here. For the steak recipe the
only shaped terms still live in overcooked_mdp are PLACEMENT_IN_POT_REW and
COOKING_STEAK_REW -- chopping, washing, plating, collecting and combining are all
commented out -- and every layout in layout/layouts ships
`rew_shaping_params: None`, so in practice the mdp pays exactly zero shaped
reward. It is still wired into the rollout (it costs one term, and it comes alive
for free if the mdp is ever built with BASE_REW_SHAPING_PARAMS), but the steak
line needs this.

WHAT MAKES IT NEED-AWARE, and why that is the whole point: components are counted
only up to the number of orders still deliverable. Three garnishes when two
dishes remain is worth exactly two garnishes; the fourth onion is worth nothing,
so chopping it earns no potential and a rollout that washes a plate instead
outscores it. The ladder in common/tasks.py has no notion of "enough" -- every
board with an onion on it emits a T_WORK chop forever -- and this is where that
gets corrected, at the level of value rather than legality.

Phi is used as POTENTIAL-BASED shaping, gamma*Phi(s') - Phi(s) (Ng et al. 1999),
not as a bonus. That is deliberate. Telescoped over a rollout the whole term
collapses to gamma^T Phi(s_T) - Phi(s_0), and Phi(s_0) is the SAME for every
candidate because every rollout starts from the same real state -- so it drops
out of the argmax entirely. The ranking is therefore "discounted deliveries, plus
how far along we are when the horizon runs out", and no amount of shuffling an
object back and forth can farm it.
"""
import os
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from common.tasks import POT, BOARD, SINK                          # noqa: E402

# A dish needs three streams to converge, and each of the three runs the same
# three stages: acquire the raw item, load it into its station, wait out the
# timer. The numbers are stage fractions of ONE finished component -- their only
# job is to be strictly increasing, so that every real step of the recipe pays
# something and no step pays more than finishing does.
RAW = 0.20            # meat / onion / plate in hand or on a counter
LOADED = 0.35         # in the pot / on the board / in the sink, timer at zero
COOKED = 1.00         # steak done, garnish chopped, plate washed

W_COMPONENT = 0.25    # a finished component, as a fraction of a delivered dish
W_ASSEMBLY = 0.08     # credit for each merge (plate+steak, plate+garnish)

#steak_dish and garnish_dish each carry two of the three components plus one
#merge; a `dish` carries all three plus both merges. Derived rather than tuned so
#that assembling can never be worth less than the parts it consumes.
W_PARTIAL = 2 * W_COMPONENT + W_ASSEMBLY          # 0.58
W_DISH = 3 * W_COMPONENT + 2 * W_ASSEMBLY         # 0.91

#Streams, keyed by the component they supply.
PLATE, STEAK, GARNISH = "plate", "steak", "garnish"


def orders_remaining(mdp, state):
    """How many more dishes this episode can actually pay for.

    NOT len(order_list). OvercookedGridworld.is_terminal is `len(order_list) <=
    1`, so the final order is never deliverable -- a 3-order layout ends after
    two deliveries. Phi has to agree with that or it keeps promising value for a
    dish the episode will never accept, and (worse) never reaches 0 at the
    terminal state, which is what makes the telescoping above exact.
    """
    ol = state.order_list
    if ol is None:
        return 99                       # unbounded orders: never cap
    if isinstance(ol, str):
        #MALFORMED LAYOUT. A few .layout files give start_order_list as the
        #string 'steak, steak' instead of a list, and the mdp passes it through
        #untouched -- len() then counts characters and order_list[0] is 's', so
        #deliver_dish never matches and the episode pays nothing. Nothing here
        #can fix that, but reading the intent keeps Phi sane on those layouts.
        ol = [o.strip() for o in ol.split(",") if o.strip()]
    return max(0, len(ol) - 1)


def _timer_fraction(obj, full):
    """Where a station's occupant is between LOADED and COOKED."""
    if full <= 0:
        return COOKED
    t = obj.state
    if isinstance(t, (tuple, list)):    # a cooking steak carries ('steak', n, t)
        t = t[-1]
    f = min(1.0, max(0.0, float(t) / float(full)))
    return LOADED + (COOKED - LOADED) * f


def _classify(mdp, state, obj, cell):
    """(stream, fraction) for a loose object, or None if it supplies nothing.

    `cell` is None for a held object -- nothing in hand is mid-timer, so the
    terrain lookup that separates "garnish being chopped on a board" from
    "finished garnish being carried" is only needed for objects on the floorplan.
    """
    n = obj.name
    if n == "meat":
        return STEAK, RAW
    if n == "onion":
        return GARNISH, RAW
    if n == "plate":
        return PLATE, RAW

    terrain = mdp.get_terrain_type_at_pos(cell) if cell is not None else None

    if n == "steak":                                     # only ever on a pot
        if terrain == POT and not mdp.steak_ready_at_location(state, cell):
            return STEAK, _timer_fraction(obj, mdp.steak_cooking_time)
        return STEAK, COOKED
    if n == "garnish":
        if terrain == BOARD and not mdp.garnish_ready_at_location(state, cell):
            return GARNISH, _timer_fraction(obj, mdp.chopping_time)
        return GARNISH, COOKED
    if n == "washed_plate":
        if terrain == SINK and not mdp.plate_washed_at_location(state, cell):
            return PLATE, _timer_fraction(obj, mdp.wash_time)
        return PLATE, COOKED
    return None


def _all_objects(state):
    """[(obj, cell_or_None)] over both pairs of hands and the whole floorplan."""
    out = [(p.held_object, None) for p in state.players if p.held_object]
    out += [(o, c) for c, o in state.objects.items()]
    return out


def potential(mdp, state):
    """Phi(s) in dish-units. 0 at a terminal state, ~0.91 one INTERACT from a
    delivery, and flat in anything the remaining orders do not need."""
    need = orders_remaining(mdp, state)
    if need <= 0:
        return 0.0

    loose = {PLATE: [], STEAK: [], GARNISH: []}
    n_dish = n_steak_dish = n_garnish_dish = 0
    for obj, cell in _all_objects(state):
        if obj.name == "dish":
            n_dish += 1
        elif obj.name == "steak_dish":
            n_steak_dish += 1
        elif obj.name == "garnish_dish":
            n_garnish_dish += 1
        else:
            hit = _classify(mdp, state, obj, cell)
            if hit is not None:
                loose[hit[0]].append(hit[1])

    # ---- allocate assembled objects to orders first ------------------------
    # Each one occupies an order slot and carries its components with it, so the
    # streams below only have to supply what is left over. A fourth steak_dish
    # against two remaining orders is surplus and scores nothing, exactly like a
    # fourth garnish.
    dishes = min(n_dish, need)
    left = need - dishes
    steak_dishes = min(n_steak_dish, left)
    left -= steak_dishes
    garnish_dishes = min(n_garnish_dish, left)
    left -= garnish_dishes

    # every assembled object already holds a plate; a garnish_dish is still
    # missing its steak and a steak_dish is still missing its garnish.
    needed = {PLATE: left,
              STEAK: left + garnish_dishes,
              GARNISH: left + steak_dishes}

    phi = W_DISH * dishes + W_PARTIAL * (steak_dishes + garnish_dishes)
    for stream, k in needed.items():
        if k <= 0:
            continue
        #THE CAP. Best-first, take k, drop the rest on the floor. This single
        #line is what stops the robot chopping a surplus onion: the fourth
        #garnish never enters the sum, so the rollout that produces it scores
        #identically to the rollout that stands still.
        best = sorted(loose[stream], reverse=True)[:k]
        phi += W_COMPONENT * sum(best)
    return phi
