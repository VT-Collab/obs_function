"""A brand-new, fully deterministic right-of-way rule -- PROTOTYPE, lives
here so nothing in the validated codebase (scene_background.py, build_
scene.py, any real_NNN layout) is touched while it's tested. Every helper
this module needs from scene_background.py is IMPORTED, never copied or
edited.

THE RULE (as specified, not this module's own invention):

  Junction (4-way or 3-way): priority is fixed by WHICH ARM you entered
  from (not where you're going), ranked in a clockwise cycle starting
  from the right (east) arm: right > down > left > up. A 3-way just uses
  whichever 3 of those 4 tiers its own 3 arms actually occupy -- same
  rule, not a separate one.

  Roundabout / merge: the vehicle MERGING IN (entering the ring, or the
  on-ramp joining the highway) always outranks the through/circulating
  traffic already on the main line -- opposite of a real-world roundabout
  (where the ring has priority), a deliberate simplification so a
  merging vehicle is never stuck waiting for a gap that respect_
  priorities-style live negotiation might never actually open.

  A vehicle that's already PAST its own conflict point (mid-exit, e.g. a
  junction's own il{corner}->o{corner} leg, or a roundabout's own
  exit->bendout->farout leg) counts as maximum priority too -- it isn't
  "in" the conflict anymore, so nothing should still be able to hold it.

  THE HIGHER-PRIORITY PARTY NEVER CHECKS PRIORITY AT ALL (only for an
  ordinary same-lane leader, via find_front_vehicle/IDM, same as always)
  -- it drives exactly as if no crossing traffic existed, trusting
  everyone else to yield. The LOWER-priority party checks ONLY against
  candidates that outrank it (never against an equal or lower rank), and
  brakes hard if one is actually about to be a real conflict.

  BOTH checks (who must yield, and the higher-priority party's own last-
  resort safety net below) are built on ONE borrowed, already-validated
  primitive: highway_env's own RegulatedRoad.is_conflict_possible(v1, v2,
  horizon, step) -- predicts both vehicles' future positions at constant
  speed along their own real routes (ControlledVehicle.predict_
  trajectory_constant_speed, which needs a real multi-segment route, see
  scene_background.build_lookahead_route) and checks whether their actual
  physical rectangles would ever overlap within that horizon. This is the
  SAME mechanic RegulatedRoad's own enforce_road_rules() already uses --
  what's different here is WHO checks whom (fixed by rank, not a live
  tie-break) and HOW FAR AHEAD each side looks (see YIELD_HORIZON vs
  OBSTACLE_HORIZON below), not the underlying geometry.

  A LOWER-priority vehicle checks every outranking candidate against a
  GENEROUS horizon (YIELD_HORIZON) -- plenty of lead time to actually
  stop. Once braking, it keeps re-checking every tick from its OWN live
  (possibly now-stationary) position: if the higher-priority vehicle has
  since passed and moved on, the predicted paths stop overlapping and it
  naturally resumes -- if the SAME vehicle is still inbound, or a
  DIFFERENT one now also qualifies, it keeps braking. There's no separate
  "already committed, stop checking" rule needed: a real, live geometric
  check that only ever says yes when a collision is CURRENTLY plausible
  cannot go stale the way a fixed geometric zone flag can, so it never
  produces the "frozen in the middle of an intersection for no live
  reason" symptom a fixed-zone version of this rule once did.

  Separately: the HIGHER-priority party runs the exact same kind of
  check, but ONLY against candidates it outranks, and with a much
  SHORTER horizon (OBSTACLE_HORIZON) -- "is a collision with a lower-
  priority vehicle already essentially unavoidable at our current
  speeds," not "might one develop in the next few seconds." This is not
  a right-of-way decision (no tie-break, no negotiation) -- it fires only
  when a lower-priority vehicle is ALREADY, physically about to be in the
  way regardless of anything either side does about it, the same category
  of thing find_front_vehicle already does for an ordinary same-lane
  leader, extended here to a conflicting lane's own vehicles. It's what
  makes trusting the lower-priority party to get out of the way actually
  safe, without needing any priority/yield negotiation (which is what
  would reopen the deadlock question below) to get that safety.

WHY THIS IS DEADLOCK-PROOF BY CONSTRUCTION, not just tuned to avoid it in
testing (unlike RegulatedRoad's own respect_priorities()+enforce_road_
rules(), see this project's own scene_background.VisibleRegulatedRoad for
the documented case where that DID deadlock): rank here is a FIXED
property of which lane a vehicle is on, computed once from static
geometry (lane_priority_rank below), never from live vehicle state (no
distance comparison, no "who's been waiting longer," nothing that could
ever flip based on what the OTHER vehicle happens to be doing). Two
lanes' own ranks are therefore always in the same order, every single
tick, for as long as either vehicle occupies them -- a strict, static
total order over the (small, fixed) set of rank values a given conflict
pair can ever take, and a strict total order cannot contain a cycle
(A > B and B > A can never both hold). A lower-priority vehicle can only
ever be braking for a SPECIFIC higher-priority vehicle that is, by
definition, not itself braking for THIS one back -- there is no pair of
mutually-braking vehicles this rule can ever produce, and therefore
nothing for a tie-break/release/timeout mechanism to need to resolve at
all. (An ordinary same-lane traffic jam -- ordinary IDM car-following
behind a real, physically-there leader -- is a separate, expected kind of
"stopped" this rule doesn't touch and can't remove; see this project's
own find_front_vehicle for that.)
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layout"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layout", "layouts"))

import numpy as np

import scene_background as sb  # noqa: E402 -- read-only import, nothing here edits it
from highway_env.road.regulation import RegulatedRoad  # noqa: E402 -- reusing its own
# is_conflict_possible(v1, v2, horizon, step) verbatim, not reimplementing it -- see
# module docstring. A staticmethod, so this needs no actual RegulatedRoad instance.

# ---------------------------------------------------------------------------
# Rank values. Only ever compared WITHIN one conflict pair, and
# lane_conflict_table only ever pairs lanes belonging to the same local
# primitive (a junction's own arms, one roundabout's own ring+approaches,
# one merge's own ramp+highway) -- so these never need to be meaningfully
# comparable ACROSS different primitives, only within one.
PAST_CONFLICT = -1  # already exiting/through -- always the highest priority there is
MERGING = 0          # entering a roundabout, or an on-ramp joining a highway
THROUGH = 1          # roundabout's own circulating ring traffic, or a highway's own ordinary lanes
# Junction compass ranks, keyed by build_scene.py's own corner numbering
# (0=South/down, 1=West/left, 2=North/up, 3=East/right -- see
# _layout_utils.corner_for_heading's own docstring). User's own stated
# order, clockwise from the right (east): right > down > left > up.
_JUNCTION_COMPASS_RANK = {3: 0, 0: 1, 1: 2, 2: 3}  # corner -> rank, lower = higher priority


def _corner_rank(corner: int) -> int:
    return _JUNCTION_COMPASS_RANK[corner]


# One compiled (regex, handler) pair per node-naming convention build_
# scene.py's own add_four_way/add_three_way/round_about/merge use (see
# their own source for these exact patterns -- not guessed). Tried in
# order; first match wins. Every pattern requires the keyword immediately
# after an underscore, matching this codebase's own "prefix always ends in
# _" convention throughout (fw_, tw1_, ra_, mg_, ...) -- safe against a
# keyword substring appearing inside a DIFFERENT keyword (verified
# directly: no two of these keywords share a "_keyword" substring).
_RE_JUNCTION_APPROACH = re.compile(r"_o(\d+)_(\d+)$")     # o{corner}_{i} -- approach, about to enter
_RE_JUNCTION_ENTERED = re.compile(r"_ir(\d+)_(\d+)$")     # ir{corner}_{i} -- just entered, mid-turn
_RE_JUNCTION_EXIT = re.compile(r"_il(\d+)_(\d+)$")        # il{corner}_{i} -- about to exit via corner
_RE_RA_MERGE_IN_1 = re.compile(r"_farin(\d+)_(\d+)$")     # farin{k}_{i} -- entering the ring
_RE_RA_MERGE_IN_2 = re.compile(r"_bendin(\d+)_(\d+)$")    # bendin{k}_{i} -- entering the ring
_RE_RA_EXIT = re.compile(r"_exit(\d+)_(\d+)$")            # exit{k}_{i} -- circulating OR about to leave (see TO check)
_RE_RA_ENTRY = re.compile(r"_entry(\d+)_(\d+)$")          # entry{k}_{i} -- circulating (only ever -> exit{(k+1)%4})
_RE_RA_BENDOUT = re.compile(r"_bendout(\d+)_(\d+)$")      # bendout{k}_{i} -- already leaving the ring
_RE_MG_RAMP_JK = re.compile(r"_[jk]$")                    # j or k -- merge's own ramp taper start/mid
_RE_MG_RAMP_B = re.compile(r"_b_ramp$")                   # b_ramp -- ramp, still merging in
_RE_MG_HIGHWAY = re.compile(r"_[abcd]_(\d+)$")            # a_i/b_i/c_i/d_i -- ordinary forward highway lane
_RE_MG_REVERSE = re.compile(r"_r[ad]_(\d+)$")             # ra_i/rd_i -- ordinary reverse highway lane


def lane_priority_rank(road, lane_index):
    """This vehicle's own fixed right-of-way rank on `lane_index` -- lower
    number always means higher priority, see module docstring for the
    full rule and why a FIXED (never live-state-dependent) rank is what
    makes this deadlock-proof. Falls back to THROUGH (a safe, ordinary,
    "must check against anyone ranked 0" default) for any lane this
    module doesn't recognize the naming convention of -- never crashes on
    an unrecognized lane, just treats it as ordinary traffic.

    THE THIRTEENTH FIX for a real, measured crash, only surfaced at real
    background-traffic density (count=100, not this module's own lighter
    18-vehicle stress tests): the junction compass rank was originally
    keyed on CORNER alone, so two vehicles approaching from the SAME arm
    but different LANE numbers (e.g. one turning right from lane 0, one
    going straight from lane 1) got the exact same rank -- a tie, which
    neither the yield check nor the obstacle check ever looks at (both
    require a strict outrank). Confirmed directly: two same-arm, adjacent-
    lane vehicles, spawned normally on ordinary approach lanes (not a
    spawn-placement case), drove into each other inside the junction
    because nothing was ever comparing them at all. Adding a small
    (0.01 * lane number) offset breaks the tie with a lane number that is
    itself a fixed, static property of the lane (not live vehicle state),
    so this is still a strict total order, still deadlock-proof by the
    same construction argument -- it only decides who checks whom for a
    pair the geometry (lane_conflict_table) may or may not actually flag
    as conflicting; it doesn't invent a new conflict where none exists.
    """
    f, t, _ = lane_index

    m = _RE_JUNCTION_EXIT.search(f)
    if m:
        # THE SEVENTEENTH FIX for a real, measured crash -- the SAME
        # same-arm tie the THIRTEENTH fix already found and fixed for
        # approach/entered lanes, just missed here: two exit lanes from
        # the same corner, different lane numbers, both returned the bare
        # PAST_CONFLICT constant with no lane-number tiebreak, so they
        # never checked each other -- confirmed directly, this exact
        # untied pair produced a repeated pile-up at the same spot
        # throughout a real 100-vehicle run (4 of 9 crashes in one trace
        # were this same pair, since the resulting wreckage never clears
        # and keeps drawing in more traffic). Still always the highest
        # priority tier overall (PAST_CONFLICT - 0.01*lane_number stays
        # below MERGING=0 for any realistic lane count), just no longer
        # tied against a DIFFERENT lane at the same corner.
        return PAST_CONFLICT - 0.01 * int(m.group(2))
    m = _RE_JUNCTION_APPROACH.search(f)
    if m:
        return _corner_rank(int(m.group(1))) + 0.01 * int(m.group(2))
    m = _RE_JUNCTION_ENTERED.search(f)
    if m:
        return _corner_rank(int(m.group(1))) + 0.01 * int(m.group(2))

    if _RE_RA_MERGE_IN_1.search(f) or _RE_RA_MERGE_IN_2.search(f):
        return MERGING
    if _RE_RA_BENDOUT.search(f):
        return PAST_CONFLICT
    m = _RE_RA_EXIT.search(f)
    if m:
        k = m.group(1)
        # exit{k} -> bendout{k} (same k): already leaving, past the conflict.
        # exit{k} -> entry{k} (same k): still circulating on the ring.
        return PAST_CONFLICT if re.search(rf"_bendout{k}_(\d+)$", t) else THROUGH
    if _RE_RA_ENTRY.search(f):
        return THROUGH

    if _RE_MG_RAMP_JK.search(f) or _RE_MG_RAMP_B.search(f):
        return MERGING
    if _RE_MG_HIGHWAY.search(f) or _RE_MG_REVERSE.search(f):
        return THROUGH

    return THROUGH


def _already_committed(lane_index):
    """True once a vehicle has actually entered a junction's own interior
    (mid-turn, ir{corner}) rather than still approaching it (o{corner}).

    Used ONLY to SHRINK apply_priority_driving's own yield-check horizon
    from YIELD_HORIZON down to OBSTACLE_HORIZON (see the NINETEENTH fix,
    apply_priority_driving's own docstring) -- NOT to skip the check
    outright (an earlier version did exactly that, see the TENTH fix
    below for why it existed, and the NINETEENTH fix for the real crash
    that outright skipping caused), NOT lane_priority_rank's own compass
    ranking above, which deliberately stays the SAME for o and ir, and
    NOT the obstacle-safety-net check, which needs a vehicle already
    inside to still count as a real hazard to anyone else's own safety
    net. THE TENTH FIX for a real, measured
    bug -- a permanent two-vehicle standoff (avg speed 0.0, confirmed
    directly by inspecting the final state at a 3-way junction): a lower-
    priority vehicle that had already turned into the junction interior
    (o -> ir) before a higher-priority candidate ever became detectable
    kept being told to brake by the yield check regardless, because
    _conflict_possible only asks "would our predicted paths cross soon,"
    which says nothing about whether one side already committed first --
    once both vehicles were fully stopped, each one's own predicted
    position (constant-speed-zero) never moved, so BOTH the higher-
    priority vehicle's own obstacle check (seeing the lower-priority one
    as a permanent, unmoving hazard) and the lower-priority vehicle's own
    yield check (seeing the higher-priority one as a permanent, unmoving
    threat) stayed true forever -- a genuine two-mechanism cycle, not
    resolved by the NINTH fix's own rank-direction restriction alone,
    since this cycle runs through two DIFFERENT mechanisms (one vehicle's
    obstacle check, the other's yield check), not the same one pointed
    both ways. Real drivers don't stop mid-intersection just because
    someone with the right of way is still approaching once they've
    already committed to crossing -- they finish it; this is that same
    principle expressed directly, since removing the old fixed-zone
    geometry (see this module's own git history) also removed the
    mechanism that used to express it.

    THE NINETEENTH FIX refines this further: "committed" doesn't mean
    "permanently exempt," it means "no longer watched for a merely-
    approaching threat." Confirmed directly, twice, in a real 100-vehicle
    run: a vehicle already on `ir` got stuck there for several real
    seconds (ordinary same-lane congestion, unrelated to priority) with
    the yield check fully off, and a genuinely new higher-priority
    vehicle arrived during that window with nothing on this side watching
    for it -- crashed despite the higher-priority side's own obstacle
    check also trying and failing to react in time alone. A vehicle can
    sit on `ir` far longer than it takes to actually cross it, so "already
    committed" cannot mean "nothing can possibly still go wrong here."
    """
    return bool(_RE_JUNCTION_ENTERED.search(lane_index[0]))


def is_unsafe_spawn_lane(lane_index):
    """True for a lane a background vehicle should never be SPAWNED onto --
    reuses the SAME node-naming regexes lane_priority_rank already trusts
    (_RE_JUNCTION_ENTERED, _RE_RA_MERGE_IN_1/2, _RE_MG_RAMP_JK/B), doesn't
    guess new ones. Purely advisory -- this module never calls scene_
    background.add_background_traffic itself, see spawn_safe_lane_indexes
    below for how a caller is meant to use this.

    THE FOURTEENTH FIX for a real, measured crash CATEGORY -- not a
    priority-rank bug at all, a gap in WHERE add_background_traffic (an
    existing, validated scene_background.py function, deliberately left
    untouched -- this is pure ADDED code, see spawn_safe_lane_indexes'
    own docstring) is willing to place a vehicle: it picks a uniformly
    random position along a uniformly random lane from whatever candidate
    list it's given, with no notion that some lanes are ALREADY inside an
    active conflict (a junction's own interior, mid-turn) or ARE the
    short final approach that dumps directly into one (a roundabout's own
    merge-in ramp, a highway on-ramp's own taper). Confirmed directly,
    repeatedly, across many distinct crash traces this session: a vehicle
    spawned at s~0 of exactly this kind of lane gives whoever it turns
    out to conflict with zero real-world lead time to react, regardless
    of how sound the driving logic afterward is -- no yield/obstacle
    check, however well designed, can react to a threat that was already
    there before the simulation's first tick.
    """
    f, _t, _i = lane_index
    return bool(
        _RE_JUNCTION_ENTERED.search(f)
        or _RE_RA_MERGE_IN_1.search(f)
        or _RE_RA_MERGE_IN_2.search(f)
        or _RE_MG_RAMP_JK.search(f)
        or _RE_MG_RAMP_B.search(f)
    )


def spawn_safe_lane_indexes(lane_indexes):
    """Filters `lane_indexes` down to ones is_unsafe_spawn_lane doesn't
    flag -- meant to be passed straight to scene_background.add_
    background_traffic's OWN existing `lane_indexes` parameter (see its
    own docstring: "Pass a restricted list... to bias density toward
    where a route-following vehicle actually drives" -- this is that same
    documented mechanism, used for a different, additional reason, not a
    new one invented here). add_background_traffic itself is NOT modified
    by this -- it already accepts a candidate-lane list; this only
    narrows the list a CALLER (see highway_env/test/watch.py) chooses to
    hand it. Falls back to the full, unfiltered list if filtering would
    leave nothing to spawn on at all, so this can never make add_
    background_traffic error out on an empty candidate list.
    """
    safe = [idx for idx in lane_indexes if not is_unsafe_spawn_lane(idx)]
    return safe if safe else list(lane_indexes)


SPAWN_SAFE_DISTANCE = 25.0  # m -- see maintain_vehicle_count's own docstring
# (the EIGHTEENTH fix) for why this replaces add_background_traffic's own
# default (10.0), not add_background_traffic itself. Comfortably more than
# the ~12.25m pure kinematic stopping distance from this codebase's own
# top background speed (14 m/s) at BRAKE_DECEL -- 10.0 alone is enough to
# stop two spawns from literally overlapping, not enough real room for a
# freshly-spawned, already-moving vehicle to react to one that happens to
# already be stopped nearby on a lane that runs close to (without being
# the exact same lane_index as) the one it's given.


def maintain_vehicle_count(road, target_count, lane_indexes, rng, safe_distance=SPAWN_SAFE_DISTANCE):
    """Tops `road.vehicles` back up to `target_count` by calling scene_
    background.add_background_traffic AGAIN, for however many are
    missing -- meant to be called once per tick, right after scene_
    background.advance_vehicles (which is what actually removes a vehicle,
    only once it's reached a genuine dead end / completed its own route --
    see its own docstring). Neither of those two functions is modified by
    this; add_background_traffic already reads road.vehicles' own current
    positions to keep new spawns a safe distance from everyone already on
    the road (see its own `placed = [v.position for v in road.vehicles]`),
    so calling it again on an already-populated road works exactly the
    same way it does for the very first, initial spawn -- this just passes
    a LARGER value for its own existing `safe_distance` parameter than its
    own default.

    THE SIXTEENTH FIX / requirement: a real stress test of "N vehicles, no
    crash, no freeze" needs N vehicles ACTIVE for the WHOLE run, not just
    at t=0 -- confirmed directly, a 100-vehicle spawn decayed to under 50
    within 40 simulated seconds as vehicles simply finished their own
    routes and left, which quietly tests a far sparser, easier scenario
    than the one actually being asked for.

    THE EIGHTEENTH FIX for a real, measured crash -- not the same category
    as is_unsafe_spawn_lane's own lane-TYPE exclusions: confirmed directly,
    a vehicle spawned (via replenishment) onto an ordinary, otherwise-safe
    exit lane, only ~7m in, on a DIFFERENT lane_index that happens to run
    close and parallel to a lane where another vehicle was already
    stopped -- add_background_traffic's own default safe_distance (10.0)
    only guards against spawning literally on top of another vehicle's
    CURRENT position, not against spawning an already-MOVING vehicle too
    close to react to one it's converging toward. Passing a larger
    safe_distance through the SAME existing parameter (not a new one)
    covers this without needing to enumerate every lane pair that happens
    to run close together.

    `rng` is a caller-owned numpy Generator (e.g. np.random.default_rng
    (seed), created ONCE by the caller, NOT re-seeded on every call) --
    add_background_traffic itself takes a single `seed` int, not a
    generator, so calling it with the SAME seed every tick would draw the
    exact same lane/position every time; drawing a fresh seed from a
    generator that keeps advancing keeps the whole run's replenishment
    sequence reproducible end to end from one initial --seed, without
    ever repeating the same spawn.
    """
    missing = target_count - len(road.vehicles)
    if missing > 0:
        fresh_seed = int(rng.integers(0, 2**31 - 1))
        sb.add_background_traffic(road, count=missing, seed=fresh_seed, lane_indexes=lane_indexes,
                                   safe_distance=safe_distance)


def advance_vehicles_keep_crashed(road, lane_indexes):
    """Wraps scene_background.advance_vehicles (unmodified, called
    straight through) so a CRASHED vehicle is never among the ones it
    removes -- advance_vehicles' own removal criterion is off_road() with
    no real continuation (see its own docstring: "genuinely off real
    road... despawn"), which has nothing to do with crash status and can
    coincidentally be true for a vehicle that crashed right near the very
    end of its own lane. A crash or a freeze must stay VISIBLE, not be
    quietly removed -- that would hide the very failure this whole module
    exists to eliminate, not fix it (see is_unsafe_spawn_lane and this
    module's own ongoing fixes for actually fixing causes; this function
    is a visibility guarantee, not a substitute for that).
    """
    crashed = [v for v in road.vehicles if v.crashed]
    road.vehicles = [v for v in road.vehicles if not v.crashed]
    sb.advance_vehicles(road, lane_indexes)
    road.vehicles.extend(crashed)


# ---------------------------------------------------------------------------
BRAKE_DECEL = 8.0  # m/s^2, the hard override applied once a conflict fires

YIELD_HORIZON = 3.0   # seconds -- how far ahead a LOWER-priority vehicle looks
# when checking an outranking candidate. Matches RegulatedRoad.is_conflict_
# possible's own default (see module docstring): plenty of real stopping
# room at this codebase's own top background speed (14 m/s, add_background_
# traffic's own speed_range) against BRAKE_DECEL (stops in 1.75s).

OBSTACLE_HORIZON = 2.5  # seconds -- how far ahead the HIGHER-priority
# vehicle's own last-resort safety net looks. Deliberately shorter than
# YIELD_HORIZON: this is not a right-of-way decision, it should only ever
# fire once a lower-priority vehicle is genuinely, imminently in the way --
# using the SAME long horizon as the yield check would make the higher-
# priority party proactively slow down for a merely-nearby lower-priority
# vehicle, which is exactly the two-sided negotiation this rule's asymmetry
# (see module docstring) exists to avoid.
#
# THE ELEVENTH FIX for a real, measured crash: an earlier version set this
# to 1.5s, justified (wrongly) as "comfortably >= the ~1.1s BRAKE_DECEL
# needs to stop from this codebase's own top speed" -- that 1.1s was this
# specific crash's own vehicle's SPAWN speed (8.8 m/s), not the actual top
# background-traffic speed (14 m/s, add_background_traffic's own speed_
# range), whose real stopping time is 14/8 = 1.75s -- LONGER than the 1.5s
# horizon that was supposed to give it enough warning to stop. Confirmed
# directly: a rank-0 vehicle detected a lower-priority vehicle converging
# on it, braked at full BRAKE_DECEL exactly as designed, and still hadn't
# stopped clear of the shared corridor by the time the lower-priority
# vehicle (having ALREADY committed to crossing, per the TENTH fix, so no
# longer checking anything itself) arrived and hit it. 2.5s leaves about
# 0.75s of real margin beyond the 1.75s worst-case stopping time, instead
# of a horizon that was numerically incapable of giving enough warning at
# this codebase's own top speed in the first place.

TRUST_SPEED_THRESHOLD = 0.5   # m/s -- "stopped," for TRUST_TICKS_THRESHOLD's
# own purposes. Matches scene_background.VisibleRegulatedRoad's own
# FROZEN_SPEED_THRESHOLD, the same category of "close enough to zero that
# creeping/measurement noise shouldn't reset the counter" judgment call.

TRUST_TICKS_THRESHOLD = 15  # ~1s at this codebase's own 1/15s tick rate --
# how long a candidate must have been CONTINUOUSLY yielding-and-stopped to
# ME specifically before the TWELFTH fix's own trust applies. THE TWENTIETH
# FIX for a real, measured crash: an earlier version trusted a deferring
# candidate INSTANTLY, the very first tick its own is_priority_yielding
# flipped True -- confirmed directly, a rank-1 vehicle started yielding to
# a rank -1 (PAST_CONFLICT) vehicle and reached a full stop within about a
# second, but its STOPPED position was still genuinely inside the PAST_
# CONFLICT vehicle's own physical corridor (two roundabout exit lanes that
# run close and parallel rather than crossing at one sharp point, see
# is_unsafe_spawn_lane's own "converging lane" cases elsewhere in this
# module for the same recurring geometry) -- the PAST_CONFLICT vehicle
# trusted it from that very first tick and drove straight through, never
# running the obstacle check that would otherwise have correctly found a
# real conflict (confirmed directly by evaluating it anyway, unused, at
# every tick of the actual crash trace). Requiring the candidate to have
# ALREADY been settled, not just newly deferring, gives the checking
# vehicle's OWN obstacle check (which fires every tick regardless, up to
# OBSTACLE_HORIZON's own real lead time) a genuine chance to catch a fast-
# approaching conflict BEFORE trust ever kicks in, while still resolving
# the TWELFTH fix's own original permanent-standoff case once both
# vehicles have genuinely settled -- typically within about a second,
# not never.


def _conflict_possible(v1, v2, horizon):
    """Thin wrapper around highway_env's own RegulatedRoad.is_conflict_
    possible -- see module docstring for why this is reused verbatim
    rather than reimplemented (it's the same real-rectangle, real-heading
    forward-trajectory check RegulatedRoad's own enforce_road_rules()
    already relies on elsewhere in this codebase). Requires both vehicles
    to have a real `.route` set (see apply_priority_driving, which sets
    one via scene_background.build_lookahead_route before calling this)
    -- predict_trajectory_constant_speed falls back to a single bare lane
    otherwise, which runs out of road within this codebase's own short
    (10-25m) lane fragments well before a 1.5-3s horizon at any real
    speed.
    """
    return RegulatedRoad.is_conflict_possible(v1, v2, horizon=horizon, step=0.25)


def apply_priority_driving(road, lane_indexes, dt, radius=110.0):
    """Recompute every vehicle's acceleration using find_front_vehicle()
    (ordinary same-lane car-following, unchanged from scene_background's
    own version -- imported directly) plus THIS module's own deterministic
    priority rule instead of scene_background.crossing_conflict_brake/
    RegulatedRoad. Call after road.act() and before road.step(dt), same
    convention as scene_background.apply_better_car_following.

    A lower-priority vehicle brakes hard as soon as some outranking
    candidate within `radius` has a real, currently-plausible geometric
    conflict with it within YIELD_HORIZON seconds (_conflict_possible,
    see its own docstring for why this is a thin wrapper around highway_
    env's own RegulatedRoad.is_conflict_possible rather than a bespoke
    distance/lane-topology heuristic -- an earlier version of this
    function used exactly such a heuristic and required THREE successive,
    real-crash-driven fixes -- entry/exit intervals, a graph-wide Dijkstra
    distance field, then a radius-bounded reachability check that still
    produced a network-wide gridlock under real traffic density -- before
    being replaced outright). The higher-priority vehicle itself never
    runs this specific check -- see module docstring for why that's the
    rule, and why it's still deadlock-proof -- it only runs the much
    shorter-horizon safety net below.

    `radius` bounds which candidates are even considered (Euclidean
    distance, via scene_background.nearby_vehicles) -- separate from, and
    coarser than, YIELD_HORIZON/OBSTACLE_HORIZON's own time-based
    filtering; 110m is generous enough that nothing within either horizon
    at this codebase's own top background speed is ever excluded by it.

    Sets `vehicle.is_priority_yielding` and `vehicle.yield_to` (or clears
    both), and separately `vehicle.is_obstacle_braking`, on every vehicle
    every tick, purely for this test's own watch.py debug overlay to read
    -- no driving decision anywhere reads them back.
    """
    per_vehicle = {}
    for v in road.vehicles:
        if v.crashed or not hasattr(v, "acceleration"):
            continue
        # A real, multi-segment route -- needed for _conflict_possible's
        # own forward trajectory projection to walk further than a single
        # short lane fragment (see _conflict_possible's own docstring).
        v.route = sb.build_lookahead_route(road, v.lane_index)
        candidates = sb.nearby_vehicles(road, v, radius)
        front = sb.find_front_vehicle(road, v, lane_indexes, candidates)
        per_vehicle[id(v)] = (v, candidates, front)

    for v, candidates, front in per_vehicle.values():
        if front is not None:
            v.action["acceleration"] = min(
                v.action.get("acceleration", 0.0),
                v.acceleration(ego_vehicle=v, front_vehicle=front, rear_vehicle=None),
            )

        my_rank = lane_priority_rank(road, v.lane_index)
        v.priority_rank = my_rank  # purely for priority_debug_label to read

        # THE NINETEENTH FIX for a real, measured crash: an earlier version
        # of this loop skipped it ENTIRELY once _already_committed (TENTH
        # fix) -- correct for "don't stop mid-crossing just because
        # someone with the right of way is still on approach," but that
        # reasoning silently assumed the vehicle stays MOVING once
        # committed. Confirmed directly, twice, in a real 100-vehicle run:
        # a vehicle already on an `ir` lane got stuck there for several
        # real seconds (ordinary congestion, unrelated to priority), with
        # NOTHING watching for a genuinely new higher-priority vehicle
        # that arrived during that window, and got hit. Rather than skip
        # the check outright once committed, shrink it to OBSTACLE_HORIZON
        # instead of YIELD_HORIZON -- still won't stop for someone merely
        # APPROACHING within a generous 3s window (preserving "finish the
        # crossing"), but will still brake for one that's already,
        # imminently unavoidable, exactly the same standard the higher-
        # priority side already holds itself to against a lower-priority
        # vehicle (see OBSTACLE_HORIZON's own comment).
        yielding_to = None
        horizon = OBSTACLE_HORIZON if _already_committed(v.lane_index) else YIELD_HORIZON
        for other in candidates:
            if other.crashed:
                continue
            if lane_priority_rank(road, other.lane_index) >= my_rank:
                continue  # other does not outrank me -- I never check it, I just go
            if _conflict_possible(v, other, horizon):
                yielding_to = other
                break

        if yielding_to is not None:
            v.action["acceleration"] = min(v.action.get("acceleration", 0.0), -BRAKE_DECEL)
            v.is_priority_yielding = True
            v.yield_to = yielding_to
            v.color = (200, 0, 200)  # magenta -- matches this project's own YIELDING_COLOR convention
        else:
            v.is_priority_yielding = False
            v.yield_to = None

        # THE TWENTIETH FIX's own bookkeeping (see TRUST_TICKS_THRESHOLD's
        # own comment): consecutive ticks v has been BOTH yielding to
        # yielding_to specifically AND settled near zero speed -- resets
        # the instant either stops holding (still moving, no longer
        # yielding, or yielding to someone new), so a fresh yield never
        # inherits an old target's own settled count.
        if v.is_priority_yielding and v.speed < TRUST_SPEED_THRESHOLD:
            if getattr(v, "_yield_settle_target", None) is v.yield_to:
                v._yield_settled_ticks = getattr(v, "_yield_settled_ticks", 0) + 1
            else:
                v._yield_settle_target = v.yield_to
                v._yield_settled_ticks = 1
        else:
            v._yield_settle_target = None
            v._yield_settled_ticks = 0

        # Last-resort safety net: is a LOWER-priority vehicle already,
        # imminently in my way regardless of anything either of us does
        # about it? See OBSTACLE_HORIZON's own comment for why this uses a
        # much shorter horizon than the yield check above -- this is not a
        # right-of-way decision (no tie-break, restricted to candidates I
        # outrank so it can never cycle against the yield check above).
        #
        # THE TWELFTH FIX for a real, measured bug -- a permanent two-
        # vehicle standoff at a roundabout's own merge point (confirmed
        # directly: a rank-0 merging vehicle stopped, its own obstacle
        # check finding a rank-1 circulating vehicle 4.8m away within
        # OBSTACLE_HORIZON, while that SAME rank-1 vehicle was ALREADY,
        # actively is_priority_yielding specifically to THIS rank-0
        # vehicle -- both true, forever, the exact cross-mechanism cycle
        # the NINTH fix targeted, just at a roundabout merge instead of a
        # junction, which the TENTH fix's own exemption doesn't cover):
        # skip a candidate that is CURRENTLY, ACTIVELY yielding
        # specifically to ME. If the yield mechanism is kinematically
        # sound (YIELD_HORIZON gives real stopping room, see its own
        # comment), a vehicle already yielding to me has, by definition,
        # stopped (or is stopping) BEFORE physically reaching where our
        # paths cross -- it is not yet a real obstacle, so there is
        # nothing here to distrust. This is narrower than "skip anyone
        # yielding to anyone" -- a candidate yielding to a THIRD vehicle
        # gets no such pass, since nothing then guarantees it has stopped
        # clear of MY OWN path.
        is_obstacle_braking = False
        for other in candidates:
            if other.crashed:
                continue
            if lane_priority_rank(road, other.lane_index) <= my_rank:
                continue  # only ever a safety net against a LOWER-priority vehicle
            if (getattr(other, "is_priority_yielding", False) and getattr(other, "yield_to", None) is v
                    and getattr(other, "_yield_settled_ticks", 0) >= TRUST_TICKS_THRESHOLD):
                continue  # settled, correctly deferring to ME specifically -- trust it, see the TWELFTH/TWENTIETH fixes
            if _conflict_possible(v, other, OBSTACLE_HORIZON):
                is_obstacle_braking = True
                break

        if is_obstacle_braking:
            v.action["acceleration"] = min(v.action.get("acceleration", 0.0), -BRAKE_DECEL)
            v.is_obstacle_braking = True
            v.color = (255, 140, 0)  # orange -- distinct from magenta priority-yield
        else:
            v.is_obstacle_braking = False
            if not v.is_priority_yielding and hasattr(v, "color"):
                delattr(v, "color")

        v.action["acceleration"] = max(v.action["acceleration"], -v.speed / dt)


def priority_debug_label(vehicle):
    """Short debug string for watch.py's --debug-priority: this vehicle's
    own rank, and (while yielding) who it's yielding to and that vehicle's
    own rank -- lets you confirm directly that the higher-priority party
    (lower rank number) is never the one shown as yielding, for any pair,
    ever, since that would mean the rule itself has a bug, not just an
    unlucky scenario. Also flags OBS (the short-horizon safety net against
    a lower-priority vehicle, see apply_priority_driving's own second
    loop) separately from YIELD (the long-horizon, rank-based check) --
    different mechanisms, worth telling apart on screen."""
    label = f"P{vehicle.priority_rank}" if hasattr(vehicle, "priority_rank") else ""
    if getattr(vehicle, "is_priority_yielding", False):
        target = getattr(vehicle, "yield_to", None)
        target_rank = getattr(target, "priority_rank", "?") if target is not None else "?"
        label = f"{label} YIELD->P{target_rank}"
    if getattr(vehicle, "is_obstacle_braking", False):
        label = f"{label} OBS"
    return label or None
