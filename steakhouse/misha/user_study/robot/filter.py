"""Field-of-view filter laid over a nominal policy -- almost identical to
robot/filter/core/my_fov_filter.py (untouched, read-only reference), so the
robot can search for a plan that keeps it in the STUDY human's actual sight,
not the ladder-argmax human my_fov_filter.py was built against.

Pick the first action that keeps the robot in the human's cone, on a
budget: Q(a) = min over plans whose FIRST action was 'a'. After taking
action 'a' the robot is free to continue however is best. See
my_fov_filter.py's own docstring for the full account of _score/_constraint
/_forecast/action() -- none of that changed here.

THE ONE SUBSTANTIVE CHANGE. my_fov_filter.py's _forecast() rolls a shadow
human forward tick by tick to see whether the robot would be seen:
`human = self.post.shadows[fov]`, one deterministic LimitedVisionHuman per
cone. This file's `post` is a user_study/robot/fov_posterior.py
SubtaskFOVPosterior, which reweights every currently-legal SUBTASK for every
cone every tick rather than keeping one shadow at all -- see that file's own
docstring for the design and why. _forecast() below rolls forward
`self.post.best_shadow(fov)` instead -- a rollout-capable ApproximateHuman
Model, seeded on demand from that fov's current highest-weight subtask and
its observer's real belief, built for exactly this purpose (a planner needs
something it can `.clone()` and step forward several ticks; the posterior's
own per-tick reweighting never needs one). Everything else reads
identically, because everything else -- baseline ranking, remaining(),
true_pi(), the score/budget/incumbency machinery -- has nothing to do with
which class the human shadow is; it only ever calls `.action(state)` and
`.clone()` on it, and ApproximateHumanModel implements both the same way
LimitedVisionHuman does.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))              # .../user_study/robot
USER_STUDY = os.path.dirname(HERE)                              # .../user_study
MISHA = os.path.dirname(USER_STUDY)                              # .../misha
sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, MISHA)

from overcooked_ai_py.mdp.actions import Action                # noqa: E402
from common import geometry as geo                             # noqa: E402
from common.tasks import TIER_NAME                             # noqa: E402
from common.views import TruthView                             # noqa: E402
from robot.nominal_policy.baselines import true_pi              # noqa: E402


ALL_ACTIONS = [a for a in Action.ALL_ACTIONS]
ALL_BUT_INTERACT = [a for a in Action.MOTION_ACTIONS]


#---- This is for play/watch only for debugging fov convergences
#----------------------------------------------------------------------
def norm_entropy(p):
    vals = [q for q in p.values() if q > 0]
    if len(vals) < 2:
        return 0.0
    h = -sum(q * math.log(q) for q in vals)
    return h / math.log(len(p))


class FOVFilter:
    """
    Pick the first action that keeps the robot in human's cone, on a budget

    Q(a) = min over plans whose FIRST action was 'a'. After taking action 'a'
    the robot is free to continue however is best

    `posterior` must be a user_study/robot/fov_posterior.py SubtaskFOVPosterior
    (or anything exposing the same `.p` / `.best_shadow(fov)` surface) -- NOT
    robot/filter/core/fov_posterior.py's FOVPosterior, which has no
    `best_shadow` to call at all.
    """
    def __init__(self, mdp,
                 baseline, #baseline type, solo/greedy/bayes
                 posterior, #SubtaskFOVPosterior
                 certainty=0.9,
                 agent_index=0,
                 m =3, #number of DISTINCT JOBS from baseline (tier, verb) --
                       #every legal cell for each is kept, see _top_jobs
                 depth = 12,
                 weight_progress=1,   #_score: cost per tick of remaining distance
                 weight_seen=1,       #_score: reward per tick actually seen
                 seen_bonus=20,        #_score: extra reward for being seen at all
                 budget_mult=100,     #_constraint: how much longer than the
                                      #baseline's closest target a plan may take
                 stick=0.5,           #_forecast: incumbency bonus for the
                                      #committed cell, in score units
                 ):


        self.mdp, self.baseline, self.post = mdp, baseline, posterior
        self.agent_index, self.other_index = agent_index, 1 - agent_index
        self.certainty = certainty
        self.m = m
        self.depth = depth

        #---- every _score/_constraint/_forecast knob lives here, together,
        #     instead of scattered as local constants -- one place to find
        #     and tune all of them. Same as my_fov_filter.py.
        #----------------------------------------------------------------------
        self.weight_progress = weight_progress
        self.weight_seen = weight_seen
        self.seen_bonus = seen_bonus
        self.budget_mult = budget_mult
        self.stick = stick

        #---- the target cell _forecast committed to last tick, if any --
        #     see the incumbency bonus in _forecast's candidate loop. Cleared
        #     on the certainty gate and on a _constraint override in action().
        #----------------------------------------------------------------------
        self.committed = None
        #---- where the robot stood right before its last move toward
        #     self.committed -- lets the incumbency bonus tell "a step
        #     forward" from "a step back to where I just was", so two
        #     adjacent, equally-good positions can't swap the bonus back
        #     and forth every tick. Cleared whenever self.committed is.
        #----------------------------------------------------------------------
        self.committed_pos = None

    #---- self.m distinct JOBS (tier, verb), not a flat top-m slice of
    #     individual (tier, verb, cell) entries. A flat slice can silently
    #     split one job's own cells across ticks -- a cell demoted within
    #     its tier by the baseline's own within-tier ordering (contention,
    #     say) falls out of a flat top-m slice even though its job is
    #     still very much in play, so the search never gets the chance to
    #     weigh it against that job's other cells at all. This keeps every
    #     legal cell for whichever self.m jobs rank best, so a within-job
    #     reorder can shuffle which cell looks most attractive but can
    #     never make one disappear from consideration outright.
    #----------------------------------------------------------------------
    def _top_jobs(self, state):
        """
        Every (tier, verb, cell) entry from the baseline's full ranking
        whose job is among the self.m best-ranked distinct (tier, verb)
        jobs.
        """
        full = self.baseline.rank_subtasks(state)
        jobs = []
        for tier, verb, _ in full:
            if (tier, verb) not in jobs:
                jobs.append((tier, verb))
        top = set(jobs[:self.m])
        return [sub for sub in full if (sub[0], sub[1]) in top]


    def _map_cone(self):
        """
        (fov, p) for the MAP cone
        """
        p = self.post.p
        if not p:
            return None, 0.0
        f = max(p, key=p.get)
        return f, p[f]


    def walkable(self, state):
        """
        A set of all walkable cells
        """
        return set(TruthView(self.mdp, state).walkable)



    def _mask(self, ranked, pos, orient, walk):
        """
        Return the mask for top-m possible jobs
        Every legal target of the top-m jobs.
        Then make a action mask for it

        """
        #all target cells
        cells = {cell for _, _, cell in ranked}
        #the facing tile
        faced = (pos[0] + orient[0], pos[1] + orient[1])
        if faced in cells:
            base = ALL_ACTIONS
        else:
            base = ALL_BUT_INTERACT

        #cannot walk into a unmovable spot
        legal = []
        for a in base:
            if a == Action.STAY or a == Action.INTERACT:
                legal.append(a)
            else:
                target = (pos[0] + a[0], pos[1] + a[1])
                if target in walk or target in cells:
                    legal.append(a)
        return legal


    def _seen(self, robot_pos, human_pos, human_orient, fov):
        """
        Whether or not at this particular tick robot is seen by human or not
        """
        terrain = self.mdp.terrain_mtx
        return robot_pos in geo.visible_cells(terrain, human_pos, human_orient, fov)

    def _score(self, remaining, seen_count):
        #8 steps away, this detour will take
        #how far + how much in fov
        #how much for arrived, how much cost for never seen even once (but reme,ber this is every tick)
        #how much for each subsequent seen
        #what about repeats
        bonus = self.seen_bonus if seen_count > 0 else 0
        return (self.weight_progress * remaining
                - self.weight_seen * seen_count
                - bonus)

    def human_robot_position(self, state, fov):
        """robot pos/orient, human pos/orient, and whether the human sees the robot."""
        r_pos = tuple(state.players[self.agent_index].position)
        r_orient = tuple(state.players[self.agent_index].orientation)
        h_pos = tuple(state.players[self.other_index].position)
        h_orient = tuple(state.players[self.other_index].orientation)
        seen = 1 if self._seen(r_pos, h_pos, h_orient, fov) else 0
        return r_pos, r_orient, h_pos, h_orient, seen

    def _joint(self, r_act, h_act):
        """
        Joint robot and human action, each in the right index
        """
        j = [None, None]
        j[self.agent_index], j[self.other_index] = r_act, h_act
        return tuple(j)

    def _constraint(self, remaining, floor):
        """
        True if `remaining` is no more than self.budget_mult times
        `floor` -- the one budget check the whole file is built on.
        """
        return floor is None or remaining <= floor * self.budget_mult


    def _forecast(self, state, ranked, pos, orient, walk, fov, base_act):
        """
        Try every masked first action; after that, walk a LIVE rollout toward
        every candidate cell (live human reaction)

        Return (best first action, info) -- info is the HUD/debug dict
        watch.py and play.py's _footer/_draw_cands read (cands, plan, top3,
        ...), empty if nothing was ever scored.

        Q(a) = min over every (action, cell) plan whose first action was a.

        Forcast the up to depth what the human will do if robot does this
        and sum up cost per action
        Only first action is enumerated, then assume bfs per plan.

        1. arrival distance and fov bonus
        2. how much did counter and robot belief of human changed
        """

        #---- CHANGED FROM my_fov_filter.py: `self.post` is a
        #     SubtaskFOVPosterior (see that file), which reweights subtasks
        #     directly rather than keeping a shadow at all -- best_shadow()
        #     builds a rollout-capable one on demand from the fov's current
        #     highest-weight subtask. See this file's module docstring.
        #----------------------------------------------------------------------
        human = self.post.best_shadow(fov)

        #calculate action mask
        action_mask = self._mask(ranked, pos, orient, walk)

        #candidate target cells
        candidates = {cell for _, _, cell in ranked}

        #which (tier, verb) job owns each candidate cell, and the distinct
        #jobs in ranked order -- lets the HUD group cells by sub-task (the
        #"top3"/"weighs" line) instead of just listing cells.
        owner = {}
        jobs = []
        for tier, verb, cell in ranked:
            owner.setdefault(cell, (tier, verb))
            if (tier, verb) not in jobs:
                jobs.append((tier, verb))

        #how far each candidate was from done BEFORE this tick's action --
        #ar-independent, computed once so the progress check below (both for
        #the tick's own first action and for every rollout tick after it)
        #has something to compare against: a tick only earns its "seen"
        #credit if remaining actually went down, whatever action caused that.
        before = {}
        for cell in candidates:
            tier, verb = owner.get(cell, (None, None))
            before[cell] = self.baseline.remaining(state, (tier, verb, cell))

        #best action and score for it
        best_action, best_cell, best_score, best_t = None, None, float("inf"), None
        #incumbency-adjusted comparison, tracked separately from best_score
        #so info/cands still show the real, unadjusted score -- only the
        #SELECTION is biased toward the committed cell.
        best_adj = float("inf")
        scored = []            # every (ar, cell) plan actually scored -> `cands`
        qa = {}                 # ar -> best score seen for that first action -> `gain`.
                                 # diagnosis/HUD only -- nothing else reads qa.
        #every (ar, cell, remaining, score) plan actually scored. The winner
        #used to be picked inline as each plan was scored, but the budget
        #check further down needs each cell's CHEAPEST remaining first --
        #which isn't known until every action has been tried -- so picking a
        #winner has to wait for a second pass over this list instead.
        combos = []

        #loop over every possible robot action at this timestep
        for ar in action_mask:

            #is terminal?
            if self.mdp.is_terminal(state):
                break

            #clone human so this branch can't corrupt the shared shadow
            human1 = human.clone()
            ah, _ = human1.action(state)
            state1, _, _, _ = self.mdp.get_state_transition(state, self._joint(ar, ah))
            pos1, orient1, _, _, seen1 = self.human_robot_position(state1, fov)

            for cell in candidates:
                #asks the baseline for the whole walk-plus-work number for
                #this (tier, verb, cell) -- same call, right or wrong verb --
                #so a chop/wash mid-press correctly costs less than a STAY
                #next to it, without this file knowing what "chop" or "wash"
                #mean.
                tier, verb = owner.get(cell, (None, None))
                subtask = (tier, verb, cell)
                remaining = self.baseline.remaining(state1, subtask)
                if remaining is None:
                    continue

                #only bank this tick's own sighting if `ar` actually made
                #progress on THIS subtask -- a STAY or a step that goes
                #nowhere earns nothing, whatever action caused it. See
                #`before`, computed above.
                b = before.get(cell)
                seen = seen1 if (b is not None and remaining < b) else 0
                prev_remaining = remaining

                #state and human after first action
                cur_state, cur_human = state1, human1.clone()
                rp, ro = pos1, orient1

                #roll out until depth or remaning for candidate
                ticks = 0
                for _ in range(min(remaining, self.depth-1)):
                    #is terminal? (check the rolled-forward state, not the tick's start state)
                    if self.mdp.is_terminal(cur_state):
                        break

                    #robot action (basically BFS towards cell). step_towards returns
                    #(None, False) if the cell went unreachable -- fall back to STAY
                    #rather than feed None into get_state_transition
                    mv, arrived = geo.step_towards(walk, rp, ro, cell)
                    if arrived:
                        break

                    #new human action, state
                    ah, _ = cur_human.action(cur_state)
                    cur_state, _, _, _ = self.mdp.get_state_transition(
                        cur_state, self._joint(mv or Action.STAY, ah))
                    rp, ro, _, _, saw = self.human_robot_position(cur_state, fov)
                    #same progress check, per rollout tick -- a tick that
                    #doesn't push remaining below every prior tick in THIS
                    #rollout doesn't earn its sighting either.
                    cur_remaining = self.baseline.remaining(cur_state, subtask)
                    if cur_remaining is not None and cur_remaining < prev_remaining:
                        seen += saw
                        prev_remaining = cur_remaining
                    ticks += 1

                #this is the fov score function called
                score = self._score(remaining, seen)

                #record this plan for the HUD's per-counter cost table
                #(`cands`). `av`/`tl` split so av+tl == score, the invariant
                #the HUD's "C = av + tl" readout assumes -- computed post-hoc
                #so it holds for whatever _score does.
                verb = owner.get(cell, (None, "?"))[1]
                window = 1 + ticks     # ticks actually examined for visibility
                av = float(remaining)
                tl = score - av
                scored.append((verb, cell, remaining, round(score, 1),
                               seen, window, round(av, 1), round(tl, 1)))
                qa[ar] = min(qa.get(ar, float("inf")), score)
                #pos1 threaded through too -- the selection pass below needs
                #it to tell a step forward from a step back to
                #self.committed_pos.
                combos.append((ar, cell, remaining, score, pos1))

        #each cell's own floor -- the cheapest remaining ANY action reached
        #for it this tick. This is what the budget check below measures
        #every other action for that cell against, so a plan can never win
        #by taking longer than it had to for the same destination, no
        #matter how much extra "seen" credit it racks up along the way.
        floor = {}
        for ar, cell, remaining, score, pos1 in combos:
            if cell not in floor or remaining < floor[cell]:
                floor[cell] = remaining

        #the winner used to be picked inline, above, as each plan was
        #scored. Now it's picked here, in a second pass over `combos`, so
        #every plan for a cell is on the table before any of them can be
        #ruled out by that cell's own floor.
        for ar, cell, remaining, score, pos1 in combos:
            #skip a plan that costs more than self.budget_mult times the
            #cheapest plan available for the SAME cell -- the search may
            #still trade distance for visibility across DIFFERENT
            #cells/jobs below, just not dawdle on the way to whichever one
            #it's already picked.
            if not self._constraint(remaining, floor[cell]):
                continue

            #the committed cell gets a bonus for THIS comparison only -- a
            #challenger has to beat it by more than self.stick to take
            #over, so a tie/near-tie against per-tick position noise
            #doesn't cause a switch.
            #The bonus is withheld from a plan that would walk straight
            #back to self.committed_pos -- the position the robot stood at
            #right before its last move toward this same cell. Without
            #this, two adjacent, equally-good positions each win the SAME
            #bonus in turn (mine at this spot, theirs at that spot) and the
            #robot ping-pongs between them forever; withholding it from the
            #reversing move breaks the tie toward whichever option doesn't
            #undo the last step.
            adj = score
            if cell == self.committed and pos1 != self.committed_pos:
                adj = score - self.stick
            if adj < best_adj:
                best_adj = adj
                best_action, best_cell, best_score, best_t = ar, cell, score, remaining

        #commit to whatever won, so next tick's comparison above can give
        #it the incumbency bonus. Also remember where we stood AT THE START
        #of this tick, so next tick's comparison can tell a step forward
        #from a step back to here.
        self.committed_pos = pos if best_cell == self.committed else None
        self.committed = best_cell

        #--------------More diagnosis text logic-------------
        info = {}
        if best_action is not None:
            v = owner.get(best_cell)
            info["subtask"] = (TIER_NAME[v[0]], v[1], best_cell) if v is not None else None
            info["plan"] = (v[1], best_cell) if v is not None else None
            info["plan_t"] = best_t
            info["cands"] = sorted(scored, key=lambda x: x[3])
            info["cands_kind"] = "fov"
            info["top3"] = [
                ("%s/%s" % (TIER_NAME[j[0]], j[1]),
                 min((row[3] for row in scored if owner.get(row[1]) == j),
                     default=float("inf")),
                 owner.get(best_cell) == j)
                for j in jobs
            ]
            info["top3_kind"] = "t"
            info["deviated"] = best_action != base_act
            base_q = qa.get(base_act)
            if base_q is not None:
                info["gain"] = base_q - best_score
            info["n_mask"] = len(candidates)
            info["n_jobs"] = len(jobs)
        #------------------------------------------------------
        return (best_action if best_action is not None else Action.STAY), info


    def action(self, state):
        """
        Once per tick.
        """
        #actual action and subtask that the baseline chose
        base_act, base_info = self.baseline.action(state)

        #get the highest fov guess
        fov, pmap = self._map_cone()

        #always-on HUD keys, set before any return so watch.py's
        #"H(theta)"/belief-bar lines and the exec line's headline always
        #have something to read, even on a gated tick.
        info = {
            "gated": False,
            "deviated": False,
            "base_action": Action.ACTION_TO_INDEX[base_act],
            "base_subtask": base_info.get("subtask"),
            "subtask": base_info.get("subtask"),
            "theta_entropy": norm_entropy(self.post.p),
            "fov_post": dict(self.post.p),
        }

        #certainty about fov
        if fov is None or pmap < self.certainty:
            info["gated"] = True
            #a stale commitment shouldn't survive a stretch where the
            #posterior wasn't trusted enough to search at all.
            self.committed = None
            self.committed_pos = None
            return base_act, info


        #ranked subtask, which is NOT the actual subtask baseline chose (due to stickiness)
        ranked = self._top_jobs(state)
        if not ranked:
            return base_act, info



        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)

        #So this takes whatever set self.walkable(state) returns and unions in the robot's own current cell
        walk = self.walkable(state) | {pos}

        #{(tier, verb, cell): p} over the candidates a baseline proposed.
        pi, pi_src = true_pi(self.baseline, state, ranked, pos, orient, walk)

        act, forecast_info = self._forecast(state, ranked, pos, orient, walk, fov, base_act)
        info.update(forecast_info)

        #same safety cap as my_fov_filter.py (don't let the search's chosen
        #job cost more than self.budget_mult times what the baseline's own
        #target would), so action() has to gather the two numbers itself:
        #the search's own plan_t, and how long the baseline's OWN target
        #would take from here.
        base_subtask = base_info.get("subtask")
        plan_t = forecast_info.get("plan_t")
        closest = (self.baseline.remaining(state, base_subtask)
                   if base_subtask is not None else None)

        if (base_subtask is None or plan_t is None or closest is None
                or not self._constraint(plan_t, closest)):
            info["constrained"] = True
            #a rejected target shouldn't keep winning ties next tick just
            #because it's still "committed".
            self.committed = None
            self.committed_pos = None
            return base_act, info

        return act, info
