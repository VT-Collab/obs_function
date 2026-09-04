"""A route-following FOV-limited human whose driving is gated by explicit,
discrete subtask selection (interface.py) instead of continuously
optimizing IDM on its own every tick, the way LimitedVisionHuman does.

THIS CAR IS GLUED TO ITS FIXED PATH. Every legal action IS the path's own
next segment, exactly: "go" only ever means "keep driving the current
ordinary stretch", and a maneuver button only ever means "execute THIS
path's own turn/lane-change/merge exactly as built" -- there is no free
steering, no alternate route, nothing to deviate onto. See advance() for
the two separate mechanisms that make this a hard guarantee rather than a
description: a per-tick end-of-route stop and a catastrophic-deviation
safety net (position-based), and current_span()'s own boundary logic
(subtask-based) for which action is legal when.

STATE:
  self.selected -- a single sticky string: "wait", "go", or one of
    subtasks.DISPLAY_NAME's own kinds ("turn", "lane_change", "merge_in",
    "merge_out"). "go" is the automatic default -- it drives an ordinary
    forward stretch on its own, with no button of its own; interface.py's
    own WAIT button is a toggle between "wait" and "go" (there is no
    separate resume button). Starts at "go": the car drives itself the
    moment a forward stretch is in front of it, no click needed -- WAIT is
    for the user to explicitly hold it back, not the default state.
  self._committed_span -- the maneuver span (if any) currently being
    executed. Once a maneuver is authorized, it is committed: it runs to
    completion regardless of anything selected afterwards (including
    "wait") and regardless of any hazard that becomes visible partway
    through -- gating only ever decides whether to ENTER a maneuver, never
    whether to abandon one already under way. Cleared once progress
    genuinely passes the committed span's own end.

NO AUTOMATIC COLLISION AVOIDANCE DURING A MANEUVER -- BUT ORDINARY DRIVING
STILL FOLLOWS NORMALLY. See _drive()'s own `ordinary` flag: a plain
"forward" stretch is not itself an FOV-gated decision -- it's just driving,
same as every other vehicle on the road -- so it still runs normal IDM
front-vehicle following plus crossing_conflict_brake (an earlier version
removed this UNCONDITIONALLY, which made the car rear-end ordinary same-
lane traffic during plain automatic driving, nothing to do with any
subtask choice at all). A maneuver, once authorized/committed, is
different: an earlier version ran that same avoidance logic underneath the
gating too, which meant clicking a maneuver the human genuinely couldn't
see a hazard for was RISK-FREE (IDM would just brake for whatever was
actually there anyway), defeating the entire point of an FOV study -- a
wrong call made on limited information has to have a real consequence. So
a committed maneuver's own motion is purely mechanical (accelerate/cruise
toward target_speed along the path's own lane geometry) with no
front_vehicle lookup and no crossing_conflict_brake call at all -- if that
mechanical motion collides with something, highway_env's own ordinary
physics marks it crashed, same as any other vehicle. The ONLY thing that
ever stops this car during ordinary driving is the user explicitly
selecting "wait" (and even that cannot interrupt an already-committed
maneuver -- see self._committed_span above).

PER TICK (see advance()):
  1. Reached the end of the fixed path -> stop here for good,
     unconditionally (nothing legal left to do).
  2. Caught by the catastrophic-deviation safety net (should never
     actually fire; see MAX_LATERAL_TOLERANCE) -> hard reset to the last
     confirmed-on-path snapshot.
  3. Already committed to a maneuver -> keep driving it, no re-checks.
  4. Otherwise: "wait" always freezes; a "forward" span always allows
     driving; any other span requires self.selected to exactly match its
     own kind AND (only at THIS entry instant) pass gated()'s FOV check,
     or it freezes in place until it does.
  5. Not authorized -> FREEZE, not reverse: zero speed/accel and leave
     position exactly where it is. An earlier version snapped position
     BACKWARD to the last authorized point instead, which is what
     produced a visible shake/button-flicker right at every boundary: the
     instant a creep-forward tick got "un-crossed" by the snap-back, the
     very next tick saw "forward" again, re-allowed the creep, and the
     whole cycle repeated every other frame. Freezing never moves the
     vehicle at all once stopped, so nothing to observe can oscillate --
     the tick that first touched the boundary is the only one that ever
     puts it fractionally past it, and it then simply stays there.
"""
import numpy as np

import scene1_background as sb
from limit_vision_human import LimitedVisionHuman, _route_progress, _unstick_if_frozen


class ApproximateLimitVisionHuman(LimitedVisionHuman):
    MAX_LATERAL_TOLERANCE = 10.0  # meters -- see advance()'s own safety-net comment

    def __init__(self, *args, subtasks=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.subtasks = tuple(subtasks)  # subtasks.segment_route(...) output for this vehicle's own route
        self.selected = "go"  # automatic default -- see module docstring
        self.active_span = None  # HUD/debug: the span last checked in advance()
        self._committed_span = None
        # Last (position, heading, lane bookkeeping) confirmed to be
        # genuinely ON the path -- restored only by the catastrophic safety
        # net (see advance()), never by the ordinary not-yet-authorized
        # case (that one just freezes -- see module docstring).
        self._safe_snapshot = None
        # _unstick_if_frozen (limit_vision_human.py) reads/writes this
        # directly with no getattr fallback -- it assumes a LimitedVisionHuman
        # spawned via add_human_vehicle, which happens to always start moving
        # (speed > 0.5) and so never hits its speed<0.5 read-branch on the
        # very first call. THIS class is deliberately spawned at speed=0.0
        # (see interface.py's own spawn comment: never move before the user
        # clicks anything), so its first-ever _drive(ordinary=True) call
        # WOULD hit that branch with no attribute set yet -- confirmed as a
        # real AttributeError, not theoretical, before this line was added.
        self._stalled_ticks = 0

    def select(self, choice):
        """choice: "wait", "go", or a subtasks.py kind string. Always sets
        self.selected -- the "committed maneuver can't be interrupted"
        guarantee (see module docstring) lives in advance()'s own point 3,
        which ignores self.selected entirely while self._committed_span is
        set, not here. So calling this during a commitment changes nothing
        about that maneuver's own outcome, only what's authorized once it
        completes. interface.py never calls this for a disabled button in
        the first place (available_choices() returns {} while committed),
        so this distinction is invisible from the UI; it only matters for
        a caller that sets self.selected directly."""
        self.selected = choice

    def current_span(self):
        if not self.subtasks:
            return None
        progress, _ = _route_progress(self.route_points, self.position)
        for span in self.subtasks:
            if span[1] <= progress < span[2]:
                return span
        return self.subtasks[-1]

    def _lane_conflict(self, lane_indexes, visible, danger_dist=32.0, lateral_tol=6.0):
        """True iff some visible vehicle sits within danger_dist along ANY
        of the maneuver's own upcoming lanes, laterally within
        lateral_tol -- e.g. cross-traffic already committed to a turn's
        own arc, a car occupying a lane-change's target lane, or ring
        traffic near a roundabout's own entry/exit point. Pure geometry
        (positions projected onto the maneuver's own lane objects), same
        category of check as scene1_background.find_front_vehicle's own
        lane-projection, just scoped to whatever's actually visible.

        Widened from an original (22.0, 3.0): even with 3 robots plus 35
        background vehicles densely packed onto this route, the tighter
        window essentially NEVER registered a single vehicle at the exact
        moment the human reached a maneuver (confirmed directly: zero
        informative posterior ticks across an entire 6000-step run at
        every candidate FOV). A maneuver's own arc is short (a junction
        turn or roundabout access spans maybe 15-30m) and lateral_tol=3.0
        is tighter than a single lane's own half-width -- a real car
        merely one lane over from the maneuver's own path, which a real
        driver would still reasonably factor into "is this safe", was
        being excluded outright. This still isn't "anything visible
        anywhere" (see subtasks-fov-gating's own design intent: the check
        stays scoped to the maneuver's own upcoming lane geometry, not a
        blanket radius), just wide enough that realistic nearby traffic
        actually has a chance to register."""
        for lane_idx in lane_indexes:
            lane = self.road.network.get_lane(lane_idx)
            for other in visible:
                if other is self:
                    continue
                s, lat = lane.local_coordinates(other.position)
                if 0.0 <= s <= danger_dist and abs(lat) < lateral_tol:
                    return True
        return False

    def gated(self, span, visible):
        """True iff `span`'s own maneuver is currently illegal to ENTER,
        given only what this human can see right now. Only ever consulted
        at the moment of deciding whether to authorize entry into a
        maneuver -- never again once committed (see module docstring)."""
        kind, _, _, lane_indexes = span
        if kind == "forward":
            return False
        return self._lane_conflict(lane_indexes, visible)

    def _snapshot(self):
        """Record the current (position, heading, lane bookkeeping) as
        genuinely on the path -- called every authorized tick, purely as
        the anchor for the catastrophic-deviation safety net below (NOT
        used by the ordinary not-yet-authorized case, which freezes
        instead of restoring -- see module docstring)."""
        self._safe_snapshot = (np.array(self.position, dtype=float), self.heading,
                                self.lane_index, self.lane, self.target_lane_index)

    def _restore_snapshot(self):
        """Last-resort hard reset: zero speed/acceleration and snap back to
        the last on-path snapshot. Reserved for the catastrophic-deviation
        safety net -- see advance()'s own comment on why the ordinary
        not-yet-authorized case uses a plain freeze instead."""
        if self._safe_snapshot is not None:
            pos, heading, lane_index, lane, target_lane_index = self._safe_snapshot
            self.position = np.array(pos, dtype=float)
            self.heading = heading
            self.lane_index, self.lane, self.target_lane_index = lane_index, lane, target_lane_index
        self.speed = 0.0
        self.action["acceleration"] = 0.0

    def _freeze(self):
        self.speed = 0.0
        self.action["acceleration"] = 0.0

    def advance(self, road, lane_indexes, dt, radius=35.0):
        """Call once per tick, in place of apply_human_aware_car_following's
        own per-human accel block, for THIS vehicle specifically (every
        other piece -- sb.apply_better_car_following for background
        traffic, route_aware_continuation, the stall unstick -- still runs
        exactly as it does for a plain LimitedVisionHuman)."""
        if not self.subtasks:
            self._freeze()
            return

        progress, dist = _route_progress(self.route_points, self.position)

        # 1. END OF THE FIXED PATH: stop here for good, unconditionally.
        # current_span() would otherwise fall back to self.subtasks[-1]
        # forever once progress runs past every span, reading as "still in
        # an ordinary forward span" and letting plain IDM free-flow
        # accelerate the car straight off the end of the road indefinitely
        # (confirmed: still climbing at -328m and beyond, never stopping).
        if progress >= self.subtasks[-1][2]:
            self.active_span = self.subtasks[-1]
            self._freeze()
            return

        # 2. CATASTROPHIC-DEVIATION SAFETY NET. Should never actually fire
        # in normal operation -- MAX_LATERAL_TOLERANCE is generous enough
        # that a genuine turn/lane-change/merge, which can briefly sit a
        # couple meters off the route polyline's own centerline, never
        # trips it. It exists purely as a last-resort catch for any other
        # bug that might someday put this car somewhere it has no business
        # being (this project has already observed one runaway reach
        # thousands of meters off-path) -- unlike the ordinary
        # not-yet-authorized case below, THIS one does restore position,
        # because by definition something has already gone wrong enough
        # that "freeze in place" would leave the car stuck off-path.
        if dist > self.MAX_LATERAL_TOLERANCE:
            self._restore_snapshot()
            return

        candidates = sb.nearby_vehicles(road, self, radius)
        visible = self.visible_candidates(candidates)
        span = self.current_span()
        self.active_span = span

        # 3. ALREADY COMMITTED: ride the current maneuver out to completion,
        # no re-checks -- gating only ever gates ENTRY (see module
        # docstring: a maneuver started must finish, whatever else gets
        # clicked or newly appears in view). Cleared once progress
        # genuinely passes its own end.
        if self._committed_span is not None:
            if progress >= self._committed_span[2]:
                self._committed_span = None
                self.selected = "go"
            else:
                self._drive(road, lane_indexes, visible, dt, radius, ordinary=False)
                return

        # 4. NOT YET COMMITTED: "wait" always freezes; "forward" never
        # needs authorization; any other span needs an exact, ungated
        # match right now to be newly entered/committed.
        if self.selected == "wait":
            self._freeze()
            return
        if span[0] != "forward":
            if self.selected != span[0] or self.gated(span, visible):
                self._freeze()
                return
            self._committed_span = span  # newly entered -- locks in for point 3 above from next tick on

        # 5. AUTHORIZED THIS TICK. "forward" is ordinary driving, not a
        # subtask decision point at all -- it keeps normal same-lane
        # car-following (front_vehicle + crossing_conflict_brake), exactly
        # like LimitedVisionHuman/every other vehicle in this codebase, so
        # this car doesn't rear-end whatever's directly ahead of it just
        # because nothing was clicked. A just-committed MANEUVER (falls
        # through here too, on its very first tick) is NOT ordinary -- see
        # _drive()'s own docstring on why that one has no avoidance at all.
        self._drive(road, lane_indexes, visible, dt, radius, ordinary=(span[0] == "forward"))

    def _drive(self, road, lane_indexes, visible, dt, radius, ordinary):
        """ordinary=True: normal IDM car-following against whatever's
        actually ahead in this car's own lane, plus crossing_conflict_brake
        -- for plain forward driving, which is not itself an FOV-gated
        decision and should behave like any other vehicle on the road.
        ordinary=False: purely mechanical acceleration toward target_speed,
        no front_vehicle lookup and no crossing_conflict_brake at all --
        for an authorized/committed maneuver, where the whole point is
        that a wrong FOV-based judgment call about it has to be able to
        produce a real collision, not be quietly caught by a safety net
        underneath it (see module docstring)."""
        self._snapshot()
        if ordinary:
            front = sb.find_front_vehicle(road, self, lane_indexes, visible)
            accel = self.acceleration(ego_vehicle=self, front_vehicle=front, rear_vehicle=None)
            heads = {id(v) for v in visible
                      if sb.find_front_vehicle(road, v, lane_indexes, sb.nearby_vehicles(road, v, radius)) is None}
            if front is None:
                heads.add(id(self))
            conflict = sb.crossing_conflict_brake(road, self, visible, heads=heads)
            if conflict is not None:
                accel = min(accel, conflict)
            # apply_human_aware_car_following's own per-human loop always
            # follows this same crossing_conflict_brake call with
            # _unstick_if_frozen -- missing here (this class never goes
            # through that function) left a real gap: crossing_conflict_brake
            # can correctly decide to yield indefinitely to a background
            # vehicle it has no way to know is itself stuck, and nothing was
            # ever clearing that from THIS car's own side. Confirmed as the
            # actual cause of a real stall, not theoretical: a wide FOV (180,
            # 360 degrees) sees far more nearby traffic as "visible" than a
            # narrow one, so crossing_conflict_brake has far more candidates
            # to react to -- measured directly, wide-FOV runs got stuck at
            # 8% route progress during plain ordinary forward driving (before
            # ever reaching a single FOV-gated subtask span), while narrower
            # FOV on the identical scene/seed reached 100%.
            _unstick_if_frozen(road, self, front, visible, dt)
        else:
            accel = self.acceleration(ego_vehicle=self, front_vehicle=None, rear_vehicle=None)
        self.action["acceleration"] = max(accel, -self.speed / dt)


def available_choices(human):
    """The subtask kinds LEGAL given the human's own fixed path right now --
    the human never deviates from HUMAN_ROUTE, so this is entirely
    determined by whatever subtasks.segment_route() says the current span
    requires, nothing else: "go" only when that span is "forward" (there
    is no maneuver to authorize, so a lane-change/turn/merge_in/merge_out
    click would be a no-op every bit as illegal as "go" is during an
    actual maneuver -- see the module docstring's "press go, never
    anything else, halts at the first required maneuver" behavior, which
    is exactly this same rule from the OTHER direction), the ONE specific
    maneuver kind the path itself requires otherwise (e.g. the path needs
    a merge right now -> only "merge_in" is legal, "turn" is not, even
    though "turn" is a real kind elsewhere on the route), and "wait"
    always -- holding is always a legal choice, regardless of the path.
    Returns {} (nothing legal, buttons should all show disabled) while a
    maneuver is already committed -- see ApproximateLimitVisionHuman's own
    module docstring on why a committed maneuver can't be interrupted.

    This is the PATH-legality check only. A maneuver kind returned here
    can still be further gated by ApproximateLimitVisionHuman.gated() --
    what this car's own FOV can currently see -- see interface.py's own
    _is_enabled() for how the two combine.
    """
    if human._committed_span is not None:
        return set()
    span = human.current_span()
    if span is None or span[0] == "forward":
        return {"wait", "go"}
    return {"wait", span[0]}
