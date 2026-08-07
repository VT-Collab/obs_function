"""The subtask vocabulary and the priority ladder, shared by human and robot.

ONE legality engine, TWO views. The human passes a belief-backed view, the robot
passes a ground-truth view, and the difference between what they do is entirely
the difference between those views. That is the whole experiment: nothing about
the ladder itself is FOV-aware.

A subtask is (verb, target cell). Flat - exactly one level, no hierarchy, no
forced/chosen split. Tiers rank subtasks by WHAT THE ARRIVAL INTERACT PRODUCES,
never by how far away it is; distance only breaks ties inside a tier.
"""

# ---------------------------------------------------------------- terrain kinds
POT, BOARD, SINK, MEAT, ONION, PLATE, SERVE, COUNTER = \
    "P", "B", "W", "M", "O", "D", "S", "X"
STATION_CHARS = {POT, BOARD, SINK, MEAT, ONION, PLATE, SERVE, COUNTER}

UNKNOWN = "?"          # belief has expired or was never formed
EMPTY = None           # observed to hold nothing

# ------------------------------------------------------------------ the ladder
T_DELIVER, T_COMPLETE, T_HALF, T_COLLECT, T_WORK, T_START, T_FETCH, T_STASH, \
    T_EXPLORE = range(1, 10)

TIER_NAME = {1: "DELIVER", 2: "COMPLETE", 3: "HALF", 4: "COLLECT", 5: "WORK",
             6: "START", 7: "FETCH", 8: "STASH", 9: "EXPLORE"}

RAW = ("meat", "onion", "plate")


def actionable(name, view):
    """Is there a KNOWN way to advance this object?

    This is the rule that makes the ladder terminating rather than merely
    ordered. Without it the agent stashes an unusable item on a counter, sees a
    legal 'pick it up' subtask next tick, and loops forever.
    """
    if name == "dish":
        return bool(view.stations(SERVE))
    if name == "steak_dish":
        return view.any_ready(BOARD) or view.counters_holding("garnish")
    if name == "garnish_dish":
        return view.any_ready(POT)
    if name == "washed_plate":
        return (view.any_ready(POT) or view.any_ready(BOARD)
                or view.counters_holding("garnish"))
    if name == "garnish":
        return (view.any_ready(SINK) or view.counters_holding("washed_plate")
                or view.counters_holding("steak_dish"))
    if name == "meat":
        return view.any_empty(POT)
    if name == "onion":
        return view.any_empty(BOARD)
    if name == "plate":
        return view.any_empty(SINK)
    return False


def legal_subtasks(view, held):
    """[(tier, verb, target_cell)] for everything currently legal.

    Judged entirely against `view`. A station never discovered is not on the
    list; a counter whose contents have decayed is not on the list.
    """
    out = []

    def add(tier, verb, cells):
        for c in cells:
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

        if not out:                              # nothing to do with it: put it down
            add(T_STASH, "stash", view.free_counters())
        return out

    # ---- empty handed -----------------------------------------------------
    for name, tier in (("dish", T_COMPLETE),
                       ("steak_dish", T_HALF), ("garnish_dish", T_HALF),
                       ("washed_plate", T_COLLECT), ("garnish", T_COLLECT),
                       ("meat", T_FETCH), ("onion", T_FETCH), ("plate", T_FETCH)):
        if actionable(name, view):
            add(tier, "take_" + name, view.counters_holding(name))

    add(T_COLLECT, "collect_plate", view.ready(SINK))
    add(T_COLLECT, "collect_garnish", view.ready(BOARD))

    add(T_WORK, "chop", view.in_progress(BOARD))
    add(T_WORK, "wash", view.in_progress(SINK))

    if actionable("meat", view):
        add(T_FETCH, "get_meat", view.stations(MEAT))
    if actionable("onion", view):
        add(T_FETCH, "get_onion", view.stations(ONION))
    if actionable("plate", view):
        add(T_FETCH, "get_plate", view.stations(PLATE))

    return out


# fetch verbs that the robot can make redundant simply by being seen holding the
# same thing. Channel A demotes these WITHIN their tier - never across tiers, and
# never to zero, so the human stays self-sufficient.
FETCH_OF = {"get_meat": "meat", "get_onion": "onion", "get_plate": "plate",
            "take_meat": "meat", "take_onion": "onion", "take_plate": "plate"}
