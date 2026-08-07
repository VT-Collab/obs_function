"""QMDP over the filter's (theta, tau) posterior, with a d-step rollout.

    Take the baseline's top-k candidates. Roll each one forward d steps -- robot
    on the baseline, human on its own policy -- scoring every tick with
    cost_function.tick_score. Average under the posterior. Pick the best.

        Q(s, a) = SUM_{theta,tau} b(theta,tau) . rollout_score(s, a | theta, tau)

===========================================================================
THE CANDIDATE SET IS A LOW-LEVEL ACTION DISTRIBUTION
===========================================================================
Everything here operates on p_base, a distribution over the SIX PRIMITIVES, and
the module emits a primitive. Same contract as the filter package had with the
trained network -- only the source of p_base differs.

The hand-written planner is deterministic GIVEN a subtask, so its randomness
lives one level up, in `subtask_distribution`. base_action_probs marginalises
that down onto the action space:

    p_base(a) = SUM_tau P(tau | s) . 1[execute(tau) = a]

Every unit of mass on an action is mass the baseline really would have played,
so this is a genuine low-level distribution and not a hand-made one.

Two properties fall out:

    k = 1 IS THE BASELINE, bit for bit, because the top-1 of p_base is what the
        planner plays. A free correctness check: run at k=1 and the two arms must
        be identical.
    A DEVIATION IS A ONE-STEP DEVIATION. The module changes only the primitive
        emitted this tick; the planner advances its own commitment on its own
        logic, untouched. Next tick it simply re-routes from wherever it landed.
        That is what makes this a one-step policy improvement over p_base rather
        than a replacement for it.

WATCH n_opinion. When the planner is mid-errand or its hands are full, p_base
collapses to a delta, only one action is live, and the module defers. That is the
commitment discipline showing through -- but if the baseline is almost never
uncertain, the module almost never speaks, and n_opinion is where you see it.

===========================================================================
WHY QMDP
===========================================================================
The uncertainty is over a HIDDEN STATE -- the partner's cone and current subtask
-- not over the physics, which is deterministic. QMDP is the standard
approximation for that shape: value each action inside every hypothesis as if the
hidden state were about to become known, then average under the belief.

Known limitation, stated because it is the standard critique: QMDP will never act
purely to LEARN theta, since it assumes theta becomes known after one step. That
is acceptable here -- our information term is about the HUMAN's knowledge, not
the robot's -- but this planner cannot deliberately probe.

===========================================================================
WHY d STEPS AND NOT ONE
===========================================================================
At d=1 only INTERACT can change a station, so belief-level terms fire on a few
percent of ticks and pure movement gets no signal at all. The old campaign
measured exactly that: eight belief terms at 481 win / 482 loss, i.e. chance.
With a rollout, walking toward a station your partner is working shows up as an
actual clash a few ticks later. Depth REPLACES terms rather than adding them,
which is why the cost function next door is two counters instead of nine.

===========================================================================
CAUSALITY -- THE ORDER THAT MAKES THIS NOT A CHEAT
===========================================================================
Robot and human move SIMULTANEOUSLY, so at tick t the robot may condition on
s_0..s_t and h_0..h_{t-1}, and NOT on h_t:

    sync_shadows(s_t)       shadows perceive s_t. Free -- looking is not evidence
                            about the cone, and it is idempotent with the
                            observe() inside filter.update().
    predict(s_t)            PREDICTIVE posterior, conditioned through t-1 only.
    robot picks a_t         <- everything above, nothing below
    h_t = human.action()    the human moves
    filter.update(s_t,h_t)  NOW the evidence is admissible

Reading h_t before choosing a_t would be peeking at a simultaneous move.
"""

import copy
import math
import random

import _paths  # noqa: F401   MUST be first

from overcooked_ai_py.mdp.overcooked_mdp import Action, EVENT_TYPES     # noqa: E402
from fov.human.agent.limited_vision_human import SAMPLING_SUBTASKS      # noqa: E402

from cost_function import tick_score, wrong_beliefs                     # noqa: E402

N_ACTIONS = 6
ROBOT_INDEX = 0
HUMAN_INDEX = 1
#posterior mass on one cone above which we stop deep-rolling every hypothesis
CERTAIN = 0.7
N_CERTAIN = 2

#NEVER override the baseline when it wants to INTERACT.
#The cost function is pure, so it has no term for "work got done" and any
#positive credit outbids an INTERACT that scores 0 -- measured on the old setup,
#21% of overrides threw one away. INTERACT is the one primitive that CHANGES THE
#WORLD; the other five only reposition. So the module may re-rank among ways of
#MOVING and may never talk the robot out of acting. That is a structural
#distinction in the action space, NOT a reward, so the cost function stays
#task-agnostic. ON by default here, unlike the filter package.
PROTECT_INTERACT = True

#THE EFFORT GATE. Act on the cost function only where there is EVIDENCE the
#choice matters. The SCORE ranks candidates (theory: what is better); EFFORT
#decides whether to act on it (evidence: does it matter here). effort(a) counts
#the re-plans the partner is driven into during the rollout. It never enters the
#objective; it only gates it. If every candidate produces the same effort, the
#module has no evidence it is choosing between anything, and defers.
#
#Safe Policy Improvement with Baseline Bootstrapping (Laroche et al., ICML 2019):
#allow change only given sufficient evidence it is for the better.
#TAU_EFFORT = 0.0 disables the gate and recovers the ungated module exactly.
TAU_EFFORT = 1.0

#=========================================================================
#BETA: how loudly the partner terms speak relative to the baseline's own
#preference. The objective is
#
#    Q(a) = log p_base(a) + BETA . (credit - penalty)
#
#WHY THE FIRST TERM HAD TO EXIST, measured rather than argued. With partner
#terms alone the module was worse than the baseline at EVERY setting tried --
#+9.7 / +8.7 / +15.7 ticks at W_CREDIT=1, and WORSE still at W_CREDIT=0
#(+11.7 / +21.0 / +28.3), i.e. the deconfliction penalty on its own was the most
#harmful configuration of all.
#
#The reason is structural and it is worth stating plainly: a purely
#task-agnostic score can only ever tell the robot what NOT to do. Both terms are
#satisfied by withdrawing -- back off the line your partner is on, go somewhere
#you can be seen -- and under capacity pressure withdrawing is the expensive
#mistake. When two agents are both needed to finish at all, UNUSED effort costs
#more than DUPLICATED effort, so "avoid overlap" optimises the wrong thing.
#
#log p_base is the fix that costs no domain knowledge. It is the baseline's own
#opinion about what is worth doing -- a quantity the module already consumes by
#construction -- so the cost function still reads no sparse reward, no shaped
#reward, no recipe, no order count and no PRIORITY. The module simply has to pay
#for a deviation in the baseline's own currency, and the partner terms have to be
#worth the price. That is KL-regularised policy improvement, and it is what makes
#"defer unless you have a reason" a gradient instead of a threshold.
#
#BETA = 0 recovers the pure baseline; large BETA recovers the partner-only
#module that was measured to hurt.
BETA = 1.0


#=========================================================================
#MAX_DETOUR: how far off a shortest path the module may route.
#
#This is what un-mutes the module. Measured on setup_island: a situation the
#module's mechanisms could act on exists on 83-99% of ticks, but the module could
#SPEAK on only 22-29% of them -- because with its hands full there is exactly one
#sensible errand, the subtask marginal is a delta, and top-k is degenerate.
#
#But "one errand" constrains WHAT the robot is doing, not HOW IT GETS THERE. On
#these layouts the island gives two ways round to almost everything, and which
#way you go decides whether your partner sees you carrying the meat. That is the
#entire content of Hsu et al.'s scenarios 1 and 3 -- walk alongside the human,
#take the staircase path -- and it is available on every tick of every errand,
#not just at decision points.
#
#So the candidate set gains every move that stays within MAX_DETOUR steps of a
#shortest route to the CURRENT goal. The robot never abandons its errand and
#never does different work; it only chooses among ways of walking there. That
#keeps the module task-agnostic -- it is choosing a path, not a task -- while
#giving it something to say on nearly every tick.
#=========================================================================
MAX_DETOUR = 2
#how much of p_base's mass the route variants may hold in total
ROUTE_SHARE = 0.6


def _goal_cells(agent, subtask):
    """Where `subtask` is trying to get to, in the agent's OWN station map."""
    try:
        return list(agent.stations.get(agent.planner.target_kind(subtask), []))
    except Exception:
        return []


def _dist_from(agent, goals):
    """BFS distance to the nearest goal, over floor the agent KNOWS is floor.

    Uses known_terrain, not the true map: routing may never use a tile the agent
    has not seen. Goals are stations, so we search from the cells ADJACENT to
    them -- you stand beside a station to use it.
    """
    from collections import deque
    walk = {c for c, t in agent.known_terrain.items() if t == ' '}
    dist, q = {}, deque()
    for g in goals:
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = (g[0] + d[0], g[1] + d[1])
            if c in walk and c not in dist:
                dist[c] = 0
                q.append(c)
    while q:
        c = q.popleft()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if n in walk and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return dist


def route_variants(agent, state, subtask, robot_index=0):
    """{action_index: detour_cost} for every move that still gets there.

    detour 0 = on a shortest path. Anything above MAX_DETOUR is dropped, so the
    module can dawdle a little to be seen but can never wander off.
    """
    goals = _goal_cells(agent, subtask)
    if not goals:
        return {}
    dist = _dist_from(agent, goals)
    p = state.players[robot_index]
    here = dist.get(tuple(p.position))
    if here is None:
        return {}
    out = {}
    for d in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        nxt = (p.position[0] + d[0], p.position[1] + d[1])
        nd = dist.get(nxt)
        if nd is None:
            continue
        #cost of going this way instead of straight at it. A move that keeps the
        #distance the same is a 1-step detour, not a free one.
        detour = (nd + 1) - here
        if 0 <= detour <= MAX_DETOUR:
            out[Action.ACTION_TO_INDEX[d]] = detour
    return out


def counterfactual_step(mdp, state, joint_action):
    """T(state, joint_action) without touching the real episode."""
    #resolve_interacts wants somewhere to jot down statistics. We never read it.
    events = {e: [False] * mdp.num_players for e in EVENT_TYPES}
    #THE PHOTOCOPY, and the single most important line here. The three resolve_*
    #calls do NOT return a new state -- they EDIT the one you hand them, in
    #place. Pass the live state and your daydream becomes the actual episode.
    nxt = state.deepcopy()
    #Interact BEFORE move: you act on what you are facing at the START of the
    #tick, not after you walk. Environment effects LAST, once the tick's business
    #is settled. rollout=True keeps imaginary objects out of the mdp's shared id
    #registry -- otherwise it grows without bound and the ids the REAL trajectory
    #hands out shift underneath it.
    mdp.resolve_interacts(nxt, joint_action, events, rollout=True)
    mdp.resolve_movement(nxt, joint_action)
    mdp.step_environment_effects(nxt)
    return nxt


def sync_shadows(filt, state):
    """Let every shadow perceive `state` at the filter's current clock.

    filter.update() does this too, but it cannot run until the human has acted
    and the robot must decide first. Idempotent: same state, same clock, so every
    belief is rewritten with the value and timestamp it already had.

    The parameter is `filt`, not `filter`, deliberately: `filter` is a python
    built-in, so `filter.tick` dies with a confusing AttributeError that reads
    like the filter object is broken rather than like a typo.
    """
    t = filt.tick
    for sh in filt.shadows.values():
        sh.t = t
        sh.observe(state)


def clone_agent(agent, seed):
    """A throwaway copy that can actually be RUN -- action(), counters, RNG.

    The filter drives its shadows through the PURE methods (execute /
    subtask_distribution). A rollout needs the full action(), which writes
    _current, _sampled, t and the counters AND draws from _rng. Doing that to a
    live agent would desynchronise it from the real trajectory and correlate its
    dice with the real human's.
    """
    sc = copy.copy(agent)                    #new object, attributes still shared
    sc.beliefs = dict(agent.beliefs)
    #stations is a dict OF LISTS, so it needs both levels -- copying just the
    #dict would leave the lists shared and the rollout would "discover" stations
    #for the real agent.
    sc.stations = {k: list(v) for k, v in agent.stations.items()}
    sc.seen_cells = set(agent.seen_cells)
    sc.known_terrain = dict(agent.known_terrain)
    sc.subtask_log = []
    sc._rng = random.Random(seed)
    #Zeroed so a caller can read them afterwards as "what happened during THIS
    #rollout" rather than "since the episode began".
    for c in ("n_checks", "n_wasted_commits", "n_delivered", "n_abandoned",
              "n_explore"):
        setattr(sc, c, 0)
    return sc


def predict(filt, state, human_index=HUMAN_INDEX):
    """One predictive step of the filter's own transition rule, consuming no
    evidence. -> list of {fov, shadow, subtask, sampled, prob} summing to 1.

    THIS IS THE TRANSITION HALF ONLY. A Bayes filter has two halves; this is the
    first and it stops. The UPDATE half reweights by how likely the action you
    just WATCHED was -- and this tick's human action does not exist yet, because
    the robot decides first. Using it would be peeking at a simultaneous move.

    We do not need a predicted action either: in the rollout the human simply
    ACTS, so their moves come out of the simulation rather than a distribution.

    filt.b is one flat dict whose KEY is nested:  (fov, (current, sampled))
        fov      which cone this hypothesis assumes. FIXED all episode.
        current  the errand they are on RIGHT NOW. Wiped the tick one finishes.
        sampled  the last thing they freely CHOSE. Survives a whole
                 pickup -> carry -> drop excursion, so it outlives many currents.
    """
    held = state.players[human_index].held_object
    held_name = held.name if held else None
    hyps = []

    for fov, shadow in filt.shadows.items():
        #pull out just this cone's entries, dropping the fov from the key
        mass = {tau: p for (f, tau), p in filt.b.items() if f == fov and p > 0}
        if not mass:
            continue                       #this cone is already ruled out
        try:
            #is their next move FORCED by their hands? You cannot freely choose
            #what to do with a full pair of hands.
            forced = shadow._forced_held(held_name) is not None
            for tau, p_tau in mass.items():
                #untouched = "was their standing line left alone?" True in the
                #two cases where the sampler is never consulted: the errand lock
                #held, or their hands are full.
                next_subtasks, untouched = filt._transition(shadow, state, tau,
                                                            forced)
                for subtask, p_z in next_subtasks.items():
                    if p_z <= 0.0:
                        continue
                    sampled = (tau[1] if untouched
                               else (subtask if subtask in SAMPLING_SUBTASKS
                                     else None))
                    hyps.append({"fov": fov, "shadow": shadow,
                                 "subtask": subtask, "sampled": sampled,
                                 "prob": p_tau * p_z})
        except Exception:
            #a shadow that raises contributes nothing this tick; the filter takes
            #the same view. Never fatal.
            continue

    total = sum(h["prob"] for h in hyps)
    if total <= 0.0:
        return []                          #caller falls back to the baseline
    for h in hyps:
        h["prob"] /= total
    return hyps


class QMDPModule:
    """baseline's top-k subtasks -> one primitive, chosen by a d-step rollout."""

    def __init__(self, mdp, planner, k=3, depth=5, alpha=0.5,
                 human_index=HUMAN_INDEX, robot_index=ROBOT_INDEX):
        self.mdp = mdp
        self.planner = planner             #the hand-written baseline (Planner)
        self.k = int(k)
        self.depth = int(depth)
        self.alpha = float(alpha)
        self.hi = human_index
        self.ri = robot_index
        #per-tick debug. n_opinion is built from this: if the module never has an
        #opinion, any difference in the results is noise.
        self.last = {}

    # ---- the baseline's LOW-LEVEL action distribution ---------------------
    def base_action_probs(self, state):
        """p_base over the six primitives, induced by the baseline's own
        uncertainty about which errand to run next:

            p_base(a) = SUM_tau P(tau | s) . 1[execute(tau) = a]

        The planner is deterministic GIVEN a subtask, so its randomness lives
        entirely at the subtask level -- `subtask_distribution` is the categorical
        it samples its next errand from. Pushing that through the planner's own
        pure `execute()` marginalises it down onto the action space, which is a
        genuine low-level distribution rather than a hand-made one: every unit of
        mass on an action is mass the baseline really would have played.

        Both halves are PURE -- subtask_distribution and execute write nothing
        and consume no RNG -- so this is safe to call speculatively.

        Two subtasks that route through the same cell this tick MERGE here, and
        that is correct: at the action level they are the same decision, and the
        module should not be asked to choose between them.

        ===================================================================
        committed=None, AND WHY THE COMMITMENT LOCK IS DROPPED *HERE ONLY*
        ===================================================================
        Passing the planner's live commitment produces a DELTA -- measured on
        setup_island, one live action on 385 of 387 ticks. A committed
        deterministic planner has no action-level uncertainty at all, so top-k is
        degenerate and the module is a structural no-op. That is not a property
        of the cost function; there is simply nothing to re-rank.

        With committed=None the question becomes "which errands would you
        consider if you re-decided right now", which on the same episode gives 2
        or more live actions on 114 of 387 ticks. Those are real alternatives:
        every one is a subtask the baseline itself scores as available and
        worth doing, routed by the baseline's own planner.

        This does NOT let the module break the baseline's commitment. The
        planner keeps its own lock and advances its own bookkeeping untouched;
        dropping the lock only widens the SET THE MODULE MAY CHOOSE FROM this
        tick. And a deviation stays a one-step deviation -- next tick the planner
        re-routes toward its standing errand from wherever it landed.

        The guards below use `play_action_probs` instead, which does respect the
        lock, so "would the baseline have interacted" is still asked of what the
        baseline is really about to do.
        """
        p = self._marginalise(state, committed=None)
        if p is None:
            return None
        #ROUTE VARIANTS. The subtask marginal answers "what should I do"; this
        #answers "which way should I walk while doing it", which is available
        #even when the errand is forced. Weighted by exp(-detour) so a shortest
        #path outranks a scenic one, and folded in at ROUTE_SHARE of the mass so
        #it can never outvote the baseline's own opinion about the task.
        a = self.planner.agent
        sub = a._current or a._sampled
        if sub:
            var = route_variants(a, state, sub, self.ri)
            if len(var) > 1:
                w = {i: math.exp(-float(d)) for i, d in var.items()}
                z = sum(w.values())
                for i, wi in w.items():
                    p[i] += ROUTE_SHARE * wi / z
                tot = sum(p)
                p = [x / tot for x in p]
        return p

    def play_action_probs(self, state):
        """p over the primitives the baseline is actually about to PLAY.

        Same marginalisation, but respecting the commitment lock, so this is a
        delta whenever the planner is mid-errand. Used for the two guards --
        never for the candidate set.
        """
        a = self.planner.agent
        return self._marginalise(state, committed=a._sampled)

    def _marginalise(self, state, committed):
        a = self.planner.agent
        p = [0.0] * N_ACTIONS
        try:
            dist = a.subtask_distribution(state, committed=committed)
        except Exception:
            return None
        for sub, pr in dist.items():
            if pr <= 0.0:
                continue
            try:
                act, _arrived, _explored = a.execute(state, sub)
            except Exception:
                continue
            p[Action.ACTION_TO_INDEX[act]] += pr
        z = sum(p)
        if z <= 0.0:
            return None
        return [x / z for x in p]

    # ---- one rollout ------------------------------------------------------
    def rollout(self, state, action, hyp, seed, t0):
        """Total score of taking `action` now under ONE hypothesis. Higher is
        better.

            robot   `action` on the first step, then the baseline's own choice
            human   its own full policy, on a throwaway copy of the shadow

        A single straight-line trajectory, not a tree: inside a hypothesis
        everything is determined, so there is one future to walk.
        """
        shadow = clone_agent(hyp["shadow"], seed)
        #start the copy on the standing line this hypothesis believes they are
        #on, and let their own decide() re-derive the errand from it rather than
        #us asserting it.
        shadow._sampled = hyp["sampled"]

        #the robot inside the rollout: a throwaway copy of the baseline, so
        #imagining the future cannot move the real planner's commitment
        rbt = clone_agent(self.planner.agent, seed + 7919)

        wrong = wrong_beliefs(shadow, state)
        s, a, total = state, action, 0.0

        try:
            for step in range(self.depth):
                #action() bumps n_explore when execute() falls back to wandering
                n_explore_before = shadow.n_explore
                #THE HUMAN DECIDES ON THE SAME STATE THE ROBOT DID -- simultaneous
                #move. action() calls observe(s) first, so the clone perceives
                #this state through its own cone before choosing.
                #_info["subtask"] is the ground-truth label. NEVER read it.
                h, _info = shadow.action(s)
                s = counterfactual_step(self.mdp, s, (a, h))
                explored = shadow.n_explore > n_explore_before

                #SCORE AFTER THE MOVE, AND MIND THE TIMING: the shadow's beliefs
                #are still the ones it formed observing the PREVIOUS state,
                #because it does not look again until the top of the next tick.
                #Truth is the state we just produced. So this asks exactly "did
                #what we just did make them wrong?" rather than "have they
                #noticed yet", which they have not had the chance to.
                sc, wrong = tick_score(self.mdp, shadow.planner, shadow, s,
                                       wrong, explored, self.ri)
                total += sc

                if self.mdp.is_terminal(s):
                    break
                #FROM HERE ON THE ROBOT IS JUST THE BASELINE AGAIN. That is what
                #makes this a one-step policy improvement: deviate once, then be
                #yourself.
                a, _ = rbt.action(s)
        except Exception:
            #a rollout that dies halfway hands back what it scored so far.
            #Returning 0.0 would be worse than useless: 0 is the "nothing
            #happened" score, so a crash would read as a clean candidate and
            #could win the argmax.
            pass

        #EFFORT: how much re-planning this candidate drove the partner into.
        #Measured, not asserted -- the human's own counters off a simulated
        #rollout. Used ONLY by the gate; never added to `total`.
        effort = shadow.n_abandoned + shadow.n_wasted_commits
        return total, effort

    def _to_roll(self, reps, mass):
        """Which cones are worth a full d-step rollout: all of them while the
        posterior is flat, the leaders once it is not.

        Take `mass` as an ARGUMENT -- do not re-derive it by summing reps' prob.
        Each cone has exactly one representative by now, and that representative
        carries its own (fov, tau) share, not the cone's. A cone owning 0.75
        spread over five tau has a representative at ~0.15, so a re-derived total
        would read "still unsure" and deep-roll everything forever. The pruning
        would silently never engage and the only symptom would be slowness.
        """
        if not mass:
            return reps
        if max(mass.values()) < CERTAIN:
            return reps
        keep = sorted(mass, key=mass.get, reverse=True)[:N_CERTAIN]
        return [h for h in reps if h["fov"] in keep]

    # ---- the decision -----------------------------------------------------
    def choose(self, state, p_base, filt, seed=0):
        """-> action INDEX. Called BEFORE the human has moved this tick.

        Returns None to DEFER, and an action index only when it is genuinely
        overriding. That distinction has to live here rather than in the caller,
        because argmax(p_base) is NOT always what the baseline plays: at a free
        choice the planner SAMPLES from its subtask distribution while p_base
        reports the whole categorical. Returning the mode there would silently
        replace sampling with greedy play, and the k=1 arms would diverge for a
        reason that has nothing to do with the module. Deferring by returning
        None lets the caller emit the baseline's own sampled action, so
        "the module said nothing" means bit-identical, always.

        The baseline's commitment bookkeeping is advanced separately by the
        caller, on the baseline's own logic -- so a deviation here is exactly a
        ONE-STEP deviation: play something else this tick, then go back to being
        yourself.
        """
        if p_base is None:
            return None
        #the six actions, the baseline's favourite first
        order = sorted(range(N_ACTIONS), key=lambda i: -p_base[i])
        #only actions the baseline gives real mass are candidates: re-rank among
        #things it would have played, never invent one out of a partner-shaped
        #preference
        live = [i for i in order if p_base[i] > 0.0]
        cand = live[:max(1, self.k)]
        base_idx = order[0]
        self.last = {"cand": cand, "q": None, "n_hyp": 0, "base": base_idx,
                     "n_live": len(live)}

        #one live action = the baseline had no choice to make (mid-errand, or
        #hands full). Defer. This is also the k=1 control arm, exactly.
        if len(cand) == 1:
            return None

        sync_shadows(filt, state)
        hyps = predict(filt, state, self.hi)
        if not hyps:
            return None                    #no idea who they are -> no opinion

        t0 = filt.tick
        #ONE ROLLOUT PER CONE, so inside each cone keep only its likeliest guess
        per_fov = {}
        for h in hyps:
            cur = per_fov.get(h["fov"])
            if cur is None or h["prob"] > cur["prob"]:
                per_fov[h["fov"]] = h
        #...but how much we BELIEVE a cone is all its guesses added together.
        #Weighting by the survivor would quietly discount the cone you believe in
        #most, purely for having spread its mass over several subtasks.
        mass = {}
        for h in hyps:
            mass[h["fov"]] = mass.get(h["fov"], 0.0) + h["prob"]

        rolled = self._to_roll(list(per_fov.values()), mass)
        z = sum(mass[h["fov"]] for h in rolled) or 1.0
        self.last["n_hyp"] = len(rolled)

        q, eff = {}, {}
        for idx in cand:
            a = Action.INDEX_TO_ACTION[int(idx)]
            acc = e_acc = 0.0
            for h in rolled:
                #SAME seed for every candidate, so the human rolls the same dice
                #in each branch and the only thing that differs is the robot
                w = mass[h["fov"]] / z
                sc, ef = self.rollout(state, a, h, seed, t0)
                acc += w * sc
                e_acc += w * ef
            #log p_base is the task-value term; the partner score is scaled by
            #BETA. A candidate the baseline dislikes starts in a hole it has to
            #climb out of, which is what stops the module buying information
            #with work.
            lp = math.log(max(p_base[int(idx)], 1e-9))
            q[int(idx)] = lp + BETA * acc
            eff[int(idx)] = e_acc
        self.last["q"] = q
        self.last["effort"] = eff

        #THE GATE. If every candidate drives the partner into the same amount of
        #re-planning, there is no measured difference to act on -- defer rather
        #than rank noise. This is what makes "never worse than the baseline"
        #structural instead of something to tune for.
        spread = max(eff.values()) - min(eff.values())
        self.last["spread"] = spread
        if spread < TAU_EFFORT:
            return None

        #best score wins; ties fall back to whatever the baseline liked more, so
        #on the many ticks where nothing scores we ARE the baseline
        best = max(cand, key=lambda i: (q[int(i)], p_base[int(i)]))

        #never let the module talk the robot out of acting on the world.
        #Asked of what the baseline is ACTUALLY about to play (lock respected),
        #not of the widened candidate set -- otherwise a spread p_base could hide
        #an imminent INTERACT behind a movement that happens to hold more mass.
        p_play = self.play_action_probs(state)
        if PROTECT_INTERACT and p_play is not None:
            play_idx = max(range(N_ACTIONS), key=lambda i: p_play[i])
            if Action.INDEX_TO_ACTION[int(play_idx)] == Action.INTERACT:
                return None
        #never take something the baseline really did not want
        if p_base[int(best)] < self.alpha * p_base[int(base_idx)]:
            return None

        if int(best) == int(base_idx):
            return None            #agrees with the baseline: say nothing
        self.last["override"] = True
        return int(best)
