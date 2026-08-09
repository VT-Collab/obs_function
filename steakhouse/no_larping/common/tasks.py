"""The subtask vocabulary and the priority ladder, shared by human and robot.

ONE legality engine, TWO views. The human passes a belief-backed view, the robot
passes a ground-truth view, and the difference between what they do is entirely
the difference between those views. That is the whole experiment: nothing about
the ladder itself is FOV-aware.

A subtask is (verb, target cell). Flat - exactly one level, no hierarchy, no
forced/chosen split. Tiers rank subtasks by WHAT THE ARRIVAL INTERACT PRODUCES,
never by how far away it is; distance only breaks ties inside a tier.

Three things stop the ladder being merely ordered, and all three are here rather
than in the agents, because both agents need them:

  REACHABILITY   `ok(cell)` is threaded through everything, including
                 actionable(). A subtask you cannot walk to is not legal, and
                 gating only the final list is not enough - see actionable().
  TERMINATION    actionable() refuses a pickup with no way to advance the item,
                 which is what kills the stash-and-take-it-back loop.
  SATURATION     saturated() and the surplus caps stop the whole production
                 chain once enough of what it makes is already waiting. The
                 ladder has no other notion of "enough".
"""

# ---------------------------------------------------------------- terrain kinds
POT, BOARD, SINK, MEAT, ONION, PLATE, SERVE, COUNTER = \
    "P", "B", "W", "M", "O", "D", "S", "X"
STATION_CHARS = {POT, BOARD, SINK, MEAT, ONION, PLATE, SERVE, COUNTER}

UNKNOWN = "?"          # belief has expired or was never formed
EMPTY = None           # observed to hold nothing

# --------------------------- the ladder of subtask urgency. smaller number = more urgent
T_DELIVER, T_COMPLETE, T_HALF, T_COLLECT, T_WORK, T_START, T_FETCH, T_STASH, \
    T_EXPLORE = range(1, 10)

TIER_NAME = {1: "DELIVER", 2: "COMPLETE", 3: "HALF", 4: "COLLECT", 5: "WORK",
             6: "START", 7: "FETCH", 8: "STASH", 9: "EXPLORE"}

RAW = ("meat", "onion", "plate")


def actionable(name, view, ok=None):
    """Is there anywhere I could TAKE this object that moves the order forward?

    ARGUMENTS
    ---------
    name  the name of an OBJECT, exactly as the mdp spells it in
          ObjectState.name -- one of: meat, onion, plate (the three raw things a
          dispenser gives you), washed_plate, garnish, steak_dish, garnish_dish
          (the four half-built things), and dish (the finished order). NOT a
          station and NOT a subtask. Read it as "the thing in my hands", or "the
          thing on that counter I am deciding whether to pick up".

    view  a TruthView or a BeliefView, and the caller decides which. This is the
          hinge the whole experiment turns on: the two classes expose an
          identical query surface, so this function cannot tell them apart. Give
          it a TruthView and every station in the kitchen counts; give it a
          BeliefView and only stations this agent has SEEN count, with contents
          that expire. Nothing in here is FOV-aware -- the FOV lives entirely in
          which view you hand it.

    ok    cell -> bool, "can I get there". Defaults to yes-to-everything. The
          human passes a test over its OWN belief map, so "unreachable" means "I
          know of no route", never ground truth: a human who has not yet found
          the wall does not know it is walled off and will walk until it does.

    RETURNS
    -------
    A plain bool. Not a subtask, not a target, not a list -- just "is this thing
    worth having". WHERE to take it is legal_subtasks()'s job; this answers only
    whether the trip is worth starting at all.

    WHY IT EXISTS
    -------------
    It is the rule that makes the ladder TERMINATING rather than merely ordered.
    Without it: the agent holds a steak_dish with no garnish anywhere, has
    nothing to do with it, stashes it on a counter -- and next tick sees a
    perfectly legal "pick up the steak_dish" and picks it straight back up.
    Forever. Every half-built object has that loop available to it, and this one
    test closes all of them: you may only pick a thing up if you believe in a
    way to advance it.

    Reachability has to be part of THIS test and not only of the subtask list,
    for the same reason. Gate only the list and the loop comes back wearing a
    different hat: the human stashes a plate it cannot use, goes empty-handed,
    this function still says a plate is worth having because some sink somewhere
    is believed free, so it fetches another one. Measured on back_bar at fov 30
    before the fix: 142 unbroken ticks of exploring while holding a plate whose
    only sink was in the robot's room -- visible straight through the
    see-through wall, and reachable never.

    Each branch below is one line of the recipe read BACKWARDS: not "what do I do
    next" but "what would have to exist for this thing in my hand to have a
    future".
    """
    ok = ok or (lambda c: True)

    def some(cells):
        """any of these cells reachable? cells come from the view, so they are
        already filtered by what this agent knows."""
        return any(ok(c) for c in cells)

    # A finished order. The only thing left to do with it is serve it, so all it
    # needs is a hatch -- no state, a serving hatch is always willing.
    if name == "dish":
        return some(view.stations(SERVE))

    # Plated steak, still needs its garnish. Two ways to get one: lift it off a
    # board that has finished chopping, or off a counter somebody left one on.
    if name == "steak_dish":
        return some(view.ready(BOARD)) or some(view.counters_holding("garnish"))

    # Garnished plate, still needs its steak, and a steak only comes off a grill
    # that has finished cooking.
    if name == "garnish_dish":
        return some(view.ready(POT))

    # A clean plate is the most useful thing to hold: it can take a steak off a
    # ready grill, take a garnish off a ready board, or collect a garnish
    # somebody has already left out.
    if name == "washed_plate":
        return (some(view.ready(POT)) or some(view.ready(BOARD))
                or some(view.counters_holding("garnish")))

    # Chopped garnish, looking for something to go on: a clean plate waiting in
    # a finished sink, or one already sitting on a counter, or a plated steak.
    if name == "garnish":
        return (some(view.ready(SINK)) or some(view.counters_holding("washed_plate"))
                or some(view.counters_holding("steak_dish")))

    # The three raw things are simpler: each has exactly one machine, and that
    # machine has to be EMPTY, not merely present. A grill already cooking
    # cannot take a second steak, so the meat in your hand has no future until
    # it clears -- which is what sends the agent off to do something else
    # instead of standing at a busy grill.
    if name == "meat":
        return some(view.empty(POT))
    if name == "onion":
        return some(view.empty(BOARD))
    if name == "plate":
        return some(view.empty(SINK))

    # anything else -- there is no such object in this env
    return False


# what advances a raw item, for the handoff clause below
_ADVANCED_BY = {"meat": POT, "onion": BOARD, "plate": SINK}

# What a raw item eventually BECOMES, and how many of that may be sitting on
# counters before fetching more of the raw is pointless. Fetching an onion when
# two chopped garnishes are already waiting to be collected does not advance the
# order, it just consumes the last free counter -- and the ladder would keep
# doing it, because "is the board free" says nothing about whether anybody needs
# another garnish. Stopping at the SOURCE lets the work in progress drain
# instead of stranding it: an item already in hand or already on a station is
# never suppressed, only the decision to start another one.
_BECOMES = {"meat": ("steak_dish",), "onion": ("garnish",),
            "plate": ("washed_plate",)}
SURPLUS_AT = 2

# The WHOLE chain that produces each thing, not just the dispenser at the front
# of it. Capping only the fetch stopped the human making new plates while it
# happily went on washing and collecting more, which lands on the same pile.
# Believing two are already waiting should end plate work outright.
#
# Note what is deliberately NOT in here: take_washed_plate, take_garnish,
# take_steak_dish. Those CONSUME the surplus, so they must stay legal - blocking
# them would freeze the pile in place forever instead of draining it. Same for
# the deliver and combine tiers.
_CHAIN_VERBS = {
    "washed_plate": {"get_plate", "take_plate", "load_sink", "wash",
                     "collect_plate"},
    "garnish":      {"get_onion", "take_onion", "load_board", "chop",
                     "collect_garnish"},
    "steak_dish":   {"get_meat", "take_meat", "load_pot", "plate_steak"},
}


def saturated(view, ok):
    """Verbs to drop because enough of what they make is already waiting.

    Applied to the FINISHED list in legal_subtasks(), after everything else, so
    it is a veto on the whole production chain rather than a rule about any one
    tier. Two counted on reachable counters and every step that leads to a third
    stops - fetch, load, work and collect alike. Capping only the fetch was tried
    first and did nothing: the human stopped making new plates and went on
    washing and collecting the ones already in the sink, which lands on exactly
    the same pile.

    `ok` is the caller's reachability test, so this counts only the surplus this
    agent could actually go and use. A pile in the other room is somebody else's
    problem and must not stop us working. Belief-gated too, when `view` is a
    BeliefView: a narrow cone under-counts the pile and over-produces, which is
    the honest cost of not looking rather than a bug to paper over.
    """
    blocked = set()
    for product, verbs in _CHAIN_VERBS.items():
        if len([c for c in view.counters_holding(product) if ok(c)]) >= SURPLUS_AT:
            blocked |= verbs
    return blocked


def handoffable(name, view, ok):
    """Can't use it myself, but I know somebody who can.

    True when the station that would advance this item is KNOWN and NOT
    reachable, and a counter IS. That belief state -- found it, cannot get to it
    -- is exactly "this is somebody else's job", and rule 3 of the layouts makes
    the follow-through free: every counter the human can reach, the robot can
    reach too, so an ordinary stash IS the handoff.

    Without this the supply chain dies at the wall. still_room puts the plate
    dispenser on the human's side and the sink on the robot's, so once the human
    correctly stops chasing an unreachable sink it stops fetching plates at all
    and that order can never start.

    ONLY for acquiring from a DISPENSER. Applying it to picking things up off
    counters reopens the loop actionable() exists to kill: the human would take
    back the very plate it just handed over, forever.
    """
    ch = _ADVANCED_BY.get(name)
    if not ch or not view.stations(ch):
        return False                       # never found the station: not my idea
    if any(ok(c) for c in view.stations(ch)):
        return False                       # I can reach it, so it is my own job
    if view.counters_holding(name):
        # ONE AT A TIME. There is already one of these waiting to be collected,
        # so fetching another does not help anybody -- it just buries the pass
        # under plates nobody has taken. Without this the human floods every
        # shared counter with the one raw item it can reach a dispenser for and
        # then has nowhere left to stage anything else. Belief-gated like
        # everything else: it only counts the ones it has actually SEEN waiting,
        # so a narrow cone over-supplies more than a wide one, which is the
        # right cost rather than a bug.
        return False
    return any(ok(c) for c in view.free_counters())


def legal_subtasks(view, held, reachable=None, allow_stash=True):
    """[(tier, verb, target_cell)] for everything currently legal AND reachable.

    Judged entirely against `view`. A station never discovered is not on the
    list; a counter whose contents have decayed is not on the list; and a
    station the agent knows of but cannot get to is not on the list either.

    That last one is why `reachable` exists. Legality used to be decided without
    it, and the stash fallback below only fires when nothing else is legal - so
    a human holding a plate, with the only sink walled off in the other room,
    had a "legal" load_sink, never fell through to stash, and walked in circles
    for the rest of the episode. Filtering here means the fallback sees the
    truth, the plate goes down on the pass, and the robot picks it up: the
    handoff the layout exists to force, happening by itself.

    `allow_stash=False` asks for the list WITHOUT that fallback, and exists so a
    caller can run this more than once on tightening assumptions and still get
    an empty answer to mean "nothing to do". A stash offered on the first ask
    makes every ask after it dead code - the human's rank() is built on exactly
    that, and the failure it prevents is spelled out at the stash clause below.
    """
    out = []
    ok = reachable or (lambda c: True)

    def add(tier, verb, cells):
        for c in cells:
            if ok(c):
                out.append((tier, verb, c))

    # ---- hands full -------------------------------------------------------
    if held is not None:
        if held == "dish":
            add(T_DELIVER, "deliver", view.stations(SERVE))

        elif held == "steak_dish":
            add(T_COMPLETE, "add_garnish", view.ready(BOARD))
            add(T_COMPLETE, "combine", view.counters_holding("garnish"))

        elif held == "garnish_dish":
            add(T_COMPLETE, "plate_steak", view.ready(POT))

        elif held == "washed_plate":
            add(T_HALF, "plate_steak", view.ready(POT))
            add(T_HALF, "add_garnish", view.ready(BOARD))
            add(T_HALF, "combine", view.counters_holding("garnish"))

        elif held == "garnish":
            add(T_COMPLETE, "combine", view.counters_holding("steak_dish"))
            add(T_HALF, "collect_plate", view.ready(SINK))
            add(T_HALF, "combine", view.counters_holding("washed_plate"))

        elif held == "meat":
            add(T_START, "load_pot", view.empty(POT))
        elif held == "onion":
            add(T_START, "load_board", view.empty(BOARD))
        elif held == "plate":
            add(T_START, "load_sink", view.empty(SINK))

        # STASH IS THE LAST RESORT, and it must not be offered while a better
        # option is merely unexamined. It used to be added here whenever the
        # strict pass came up empty, which made that pass non-empty, so the
        # caller never tried the optimistic one -- and a human holding a garnish
        # with a stale-but-ready sink two tiles away put the garnish back down,
        # picked it up, put it down, 136 ticks out of 300 on banquet_pass. The
        # caller now asks for stash only after optimism has also failed.
        out = [s for s in out if s[1] not in saturated(view, ok)]
        if not out and allow_stash:
            add(T_STASH, "stash", view.free_counters())
        return out

    # ---- empty handed -----------------------------------------------------
    for name, tier in (("dish", T_COMPLETE),
                       ("steak_dish", T_HALF), ("garnish_dish", T_HALF),
                       ("washed_plate", T_COLLECT), ("garnish", T_COLLECT),
                       ("meat", T_FETCH), ("onion", T_FETCH), ("plate", T_FETCH)):
        if actionable(name, view, ok):
            add(tier, "take_" + name, view.counters_holding(name))

    add(T_COLLECT, "collect_plate", view.ready(SINK))
    add(T_COLLECT, "collect_garnish", view.ready(BOARD))

    add(T_WORK, "chop", view.in_progress(BOARD))
    add(T_WORK, "wash", view.in_progress(SINK))

    # dispensers get the relaxed test: fetch it even if only my partner can use
    # it. Counter pickups above keep the strict one, or we take back what we
    # just handed over.
    for raw, disp in (("meat", MEAT), ("onion", ONION), ("plate", PLATE)):
        if not (actionable(raw, view, ok) or handoffable(raw, view, ok)):
            continue
        # USE THE ONES ALREADY OUT before making more. take_<raw> above already
        # offers any sitting on a counter, and both are T_FETCH, so they compete
        # on distance alone -- and a dispenser is very often nearer than the
        # loose one, so the human kept minting fresh plates and stacking them up
        # while the pile it had already made went untouched. If any is waiting
        # within reach, the dispenser is simply not on the menu.
        if any(ok(c) for c in view.counters_holding(raw)):
            continue
        surplus = sum(len([c for c in view.counters_holding(p) if ok(c)])
                      for p in _BECOMES.get(raw, ()))
        if surplus >= SURPLUS_AT:
            continue                       # enough of these already made
        add(T_FETCH, "get_" + raw, view.stations(disp))

    return [s for s in out if s[1] not in saturated(view, ok)]


# fetch verbs that the robot can make redundant simply by being seen holding the
# same thing. Channel A demotes these WITHIN their tier - never across tiers, and
# never to zero, so the human stays self-sufficient.
FETCH_OF = {"get_meat": "meat", "get_onion": "onion", "get_plate": "plate",
            "take_meat": "meat", "take_onion": "onion", "take_plate": "plate"}
