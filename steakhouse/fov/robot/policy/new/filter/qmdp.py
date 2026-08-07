"""
QMDP over the filter's (fov, subtask) posterior, with a d-step rollout.

    Take the baseline's top-3 actions. Roll each one forward d steps -- robot on
    the baseline, human on its own policy -- scoring every tick with
    cost_function.tick_score. Average under the posterior. Pick the best.

===========================================================================
WHY TOP-K
===========================================================================
The baseline knows the TASK; the cost function knows the PARTNER and nothing
about the task. Restricting the choice to the K actions the baseline already
likes is what stops the module inventing a bad action out of a partner-shaped
preference: it RE-RANKS among sensible options instead of overruling competence
it does not have.

It also makes the control trivial: K = 1 IS the baseline, bit for bit, because
the top-1 of p_base is what the baseline would have played. Free correctness
check -- run at K=1 and the two arms must be identical.

===========================================================================
WHY QMDP
===========================================================================
The uncertainty is over a HIDDEN STATE -- the partner's cone theta and current
subtask tau -- not over the physics, which is deterministic. QMDP is the standard
approximation for that shape: value each action inside every hypothesis as if
the hidden state were about to become known, then average under the belief.

    Q(s, a)  =  SUM_{theta,tau}  b(theta,tau) . rollout_score(s, a | theta, tau)

b is inference.py's joint posterior, read verbatim.

Known limitation, stated because it is the standard critique: QMDP will never
take an action purely to LEARN theta, since it assumes theta becomes known after
one step. Fine here -- our information term is about the HUMAN's knowledge, not
the robot's -- but this planner cannot deliberately probe.


QMDP's approximation: pretend the hidden state becomes known right after this step. So you never branch. You evaluate each action inside each hypothesis as if you knew which one was true, then average under the belief:
        Q(s, a) = Σ_h  b(h) · Q_h(s, a)
                    ↑        ↑
                how much   how good is a
                I believe   IF h is true

The averaging is over who my partner is, not over what anyone might do.
PIPELINE OF ONE TICK:
tick t:  p_base = [.50 .20 .15 .10 .03 .02]
                     ↓  top-3
              candidates  a1, a2, a3
                     ↓
   hypotheses after the per-cone collapse:   θ=90 (mass .7)   θ=30 (mass .3)
                     ↓
              6 rollouts, one per (candidate, hypothesis) pair

   Q(a1) = .7·score(a1,θ90) + .3·score(a1,θ30)
   Q(a2) = ...
   Q(a3) = ...
                     ↓  argmax


And one rollout is a single straight-line trajectory, not a tree:
step 0:  robot plays a1        human = shadow_θ90 decides for itself
         → next state, score this tick
step 1:  robot plays argmax p_base(state)     human decides again
         → next state, score
   .. d = 5 times, sum the tick scores




===========================================================================
WHY d STEPS AND NOT ONE
===========================================================================
At d=1 only INTERACT can change a station, so belief-level terms fire on a few
percent of ticks and pure movement gets no signal at all. The old package
measured exactly that (RESULTS.md 6d: eight belief terms at 481 win / 482 loss,
i.e. chance) and had to bolt on a hand-written `approach` term to give walking a
gradient.

With a rollout, walking toward a station your partner is working shows up as an
actual clash a few ticks later. Depth REPLACES terms rather than adding them --
which is why the cost function next door is two counters instead of nine.

    ROBOT in the rollout: the baseline's own argmax. That makes this a one-step
        policy improvement over p_base -- "what if I deviate once, then go back
        to being myself". Hsu et al. (ICRA 2025, arXiv 2505.14805) do the same
        with their planner recursively; p_base is our equivalent.
    HUMAN in the rollout: its own full policy, on a throwaway copy of the shadow.
        Not a myopic stand-in -- the real thing, cone and all.

===========================================================================
CAUSALITY -- THE ORDER THAT MAKES THIS NOT A CHEAT
===========================================================================
Robot and human move SIMULTANEOUSLY, so at tick t the robot may condition on
s_0..s_t and h_0..h_{t-1}, and NOT on h_t:

    sync_shadows(s_t)       shadows perceive s_t. Free -- looking is not
                            evidence about the cone, and it is idempotent with
                            the observe() inside filter.update().
    predict(s_t)            PREDICTIVE posterior, conditioned through t-1 only.
    robot picks a_t         <- everything above, nothing below
    h_t = human.action()    the human moves
    filter.update(s_t,h_t)  NOW the evidence is admissible

Reading h_t before choosing a_t would be peeking at a simultaneous move.
"""

# =========================================================================
# WHAT TO WRITE, ONE STEP PER LINE OF CODE.
# Indentation of each comment is the indentation your line needs.
# =========================================================================

# ---- STEP 0: IMPORTS AND CONSTANTS --------------------------------------
import copy      #makes shallow/deep copies of objects
import random    #the standard random-number generator (the human uses it)
import baseline  # noqa: F401     MUST be first: runs the sys.path shim
import numpy as np                #arrays and argsort. `as np` renames it.
import torch                      #the tensor library the policy runs on
from overcooked_ai_py.mdp.overcooked_mdp import Action, EVENT_TYPES
from fov.human.agent.limited_vision_human import SAMPLING_SUBTASKS

from utils.features import build_full_state
from SP.self_play import action_probs as sp_action_probs

import cost_function
#WHICH COST FUNCTION IS IN USE. Swap this module reference to change what the
#rollout is scored by -- everything else (rollout, gate, QMDP average, harness)
#is untouched. play_episode.py --cost preference sets it to
#preference_cost_function. Both modules expose the same two entry points.
COST = cost_function

#then the module constants, ALL_CAPS by convention (= set once, never changed):
N_ACTIONS = 6
ROBOT_INDEX = 0
HUMAN_INDEX = 1
CERTAIN = 0.7      #posterior mass on one cone above which we stop
                    #deep-rolling every hypothesis
N_CERTAIN = 2      #how many cones to deep-roll once we ARE certain
#NEVER override the baseline when it wants to INTERACT.
#The score has no term for "work got done", so any positive credit outbids an
#INTERACT that scores 0 -- measured, 21% of overrides threw one away. INTERACT is
#the one primitive that CHANGES THE WORLD; the other five only reposition. So we
#let the module re-rank among ways of MOVING, and never let it talk the robot out
#of acting. That is a structural distinction in the action space, not a reward.
PROTECT_INTERACT = False   #CALIBRATION KNOB, default OFF = original behaviour

#=========================================================================
#THE EFFORT GATE.  Act on the cost function only where there is EVIDENCE the
#choice matters.
#
#The score (cost_function) says WHAT a partner-aware robot should prefer -- two
#normative claims, information gain and plan compatibility. It does not say
#whether those claims bear on the outcome at THIS state. Measured: credit fires
#on 76% of ticks, so the module was expressing an opinion almost always, most of
#them about nothing.
#
#So we separate the two questions:
#    the SCORE ranks the candidates          (theory: what is better)
#    the EFFORT decides whether to act on it (evidence: does it matter here)
#
#effort(a) = re-plans the partner is driven into during the rollout -- errands
#they abandon, plus trips they complete and find pointless. It never enters the
#objective; it only gates it. If every candidate produces the same effort, the
#module has no evidence it is choosing between anything, and defers.
#
#This is Safe Policy Improvement with Baseline Bootstrapping (Laroche et al.,
#ICML 2019): "allow change only when you have sufficient evidence that it is for
#the better", bootstrapping to the baseline wherever evidence is thin. It is
#also Dragan & Srinivasa's arbitration (IJRR 2013): assistance should scale with
#confidence rather than being applied uniformly.
#
#TAU_EFFORT = 0.0 disables the gate and recovers the ungated module exactly.
#=========================================================================
TAU_EFFORT = 1.0


# =========================================================================
# 1. PHYSICS, SPECULATIVELY
# =========================================================================
# ---- STEP 1: counterfactual_step(mdp, state, joint_action) --------------

def counterfactual_step(mdp, state, joint_action):
    """T(state, joint_action) without touching the real episode.

    A WHAT-IF MACHINE. "If these two chefs pressed these two buttons right now,
    what would the kitchen look like a tick later?" It runs the SAME physics the
    real game runs -- just on a photocopy, so nothing you imagine leaks into the
    episode actually being played.

    You call it 30-90 times per real tick. The real game calls its own version
    once. That ratio is the whole reason the precautions below exist.
    """
    #A SCRATCH PAD THE MDP INSISTS ON.
    #resolve_interacts wants somewhere to jot down "a delivery happened", "a
    #pickup happened" -- statistics for the real game's bookkeeping. We never
    #read it; we just have to hand it one or the call fails.
    #`[False] * mdp.num_players` is one flag per chef: [False, False].
    events = {e: [False] * mdp.num_players for e in EVENT_TYPES}

    #THE PHOTOCOPY, and the single most important line here.
    #The three mdp.resolve_* calls below do NOT return a new state -- they EDIT
    #the one you hand them, in place. Pass the live state and your daydream
    #becomes the actual episode: pots you imagined filling are really full,
    #chefs you imagined moving have really moved. Copy first, always.
    nxt = state.deepcopy()

    #THE THREE STEPS OF A TICK, AND THE ORDER IS NOT ARBITRARY.
    #
    #  1. anyone who pressed INTERACT does their thing -- pick up, put down,
    #     chop, plate -- acting on the cell they are FACING RIGHT NOW.
    #  2. everyone moves and turns.
    #  3. the kitchen's own clocks tick: pots cook, boards chop, sinks wash.
    #
    #Interact BEFORE move, because you act on what you are facing at the start
    #of the tick, not on what you will be facing after you walk. Swap them and a
    #chef would reach through the wall they are about to turn toward.
    #Environment effects LAST, because the timers advance once the tick's human
    #and robot business is settled.
    #
    #`rollout=True` IS THE ONE THAT MATTERS.
    #The mdp keeps a registry of every object it has ever created, so it can
    #hand out unique ids. Normally, creating an object writes into that shared
    #registry. Two consequences if a daydream writes there:
    #  - the registry grows without bound (you daydream ~50x per tick, 400
    #    ticks an episode), and
    #  - the ids the REAL trajectory hands out shift, because your imaginary
    #    objects consumed numbers first.
    #rollout=True says "this is scrap paper, not the ledger": take the counter
    #off the state itself and do not write home. Nothing else about the physics
    #changes -- same calls, same order, same results.
    mdp.resolve_interacts(nxt, joint_action, events, rollout=True)
    mdp.resolve_movement(nxt, joint_action)
    mdp.step_environment_effects(nxt)

    #hand back the photocopy. `state` is untouched, which is what lets the
    #caller run this once per candidate action from the SAME starting state.
    return nxt

# =========================================================================
# 2. THE PARTNER
# =========================================================================
# ---- STEP 2: sync_shadows(filt, state) ----------------------------------
#docstring: "let every shadow perceive `state` at the filter's current clock."
#filter.update() does this too, but it cannot run until the human has acted and
#the robot must decide first. Idempotent: same state, same clock, so every
#belief is rewritten with the value and timestamp it already had.

def sync_shadows(filt, state):
    """let every shadow perceive `state` at the filter's current clock."""
    #THE PARAMETER IS `filt`, NOT `filter`, AND THAT IS DELIBERATE.
    #`filter` is a python BUILT-IN (the function that filters a list). Writing
    #`filter.tick` does not raise "undefined" -- it happily reaches for `.tick`
    #on the built-in and dies with a confusing
    #    AttributeError: type object 'filter' has no attribute 'tick'
    #which reads like the filter object is broken rather than like a typo.
    #Never shadow a built-in; that is why every signature here says `filt`.
    t = filt.tick
    for sh in filt.shadows.values():
        sh.t = t
        sh.observe(state)


# ---- STEP 3: predict ----------------------------------------------------
def predict(filt, state, human_index=HUMAN_INDEX):
    """One predictive step of inference.py's own transition rule, consuming no
    evidence. Returns a list of {fov, shadow, subtask, sampled, prob}, summing
    to 1 -- one entry per "who might my partner be, and what might they be
    doing".

    Reuse the filter's machinery rather than restating it, so the planner and
    the filter can never disagree about what the partner is doing.

    =======================================================================
    THIS IS THE TRANSITION HALF ONLY.  NO P(action | subtask) ANYWHERE.
    =======================================================================
    A Bayes filter has two halves. This function is the first one and stops:

        PREDICT   push the belief forward through the transition rule,
                  P(tau_t | tau_{t-1}, s)          <-- THIS FUNCTION
        UPDATE    reweight each hypothesis by how likely the action you just
                  WATCHED was under it,  P(action | tau, theta)   <-- NOT HERE

    inference.py's update() does both: it calls the same `_transition` we call,
    and THEN compares `shadow.execute(...)` against the action the human really
    emitted. That comparison is the emission / likelihood half, and it is the
    single place evidence ever enters the belief.

    We skip it on purpose, for two separate reasons:

      1. THERE IS NO EVIDENCE TO CONSUME YET. The robot decides BEFORE the human
         moves, so this tick's human action does not exist. Using it would be
         peeking at a simultaneous move -- the exact thing the causality note in
         the module header forbids.

      2. WE DO NOT NEED A PREDICTED ACTION EITHER. In the d-step rollout the
         human simply ACTS -- rollout() calls shadow.action() -- so their moves
         come out of the simulation instead of out of a distribution over
         actions. That is why there is no `execute()` call in here, and why
         `_explore_action` (the most expensive call in a tick) never runs.
         An earlier depth-1 draft did need it; the rollout made it dead weight.

    So what comes out of here is P(theta, tau) at time t given evidence through
    t-1, and nothing whatsoever about which button anyone will press.

    (`subtask` in each returned dict is a LABEL for reading logs. The two fields
    anything downstream actually uses are `sampled`, which seeds the rollout's
    shadow, and `fov`/`prob`, which weight the QMDP average.)

    =======================================================================
    THE SHAPE OF filt.b -- A PAIR WHOSE SECOND HALF IS ITSELF A PAIR
    =======================================================================
    This is the thing that makes the loop below look cryptic. The belief is one
    flat dict, but its KEY is nested:

        filt.b = {
            (30, (None,        None))            : 0.11,
            (30, ("drop_meat", "pickup_meat"))   : 0.04,
            (90, ("drop_meat", "pickup_meat"))   : 0.40,
            (90, ("check_pot", "pickup_onion"))  : 0.45,
        }
             ^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
            fov            tau
                   ^current^     ^sampled^

    So: key = (fov, tau)   and   tau = (current, sampled).  a, b -- but b is
    itself (b1, b2).

        fov      which cone this hypothesis assumes.  FIXED all episode.
        current  the errand they are on RIGHT NOW. Rewritten constantly, wiped
                 the tick an errand finishes.
        sampled  the last thing they freely CHOSE -- the standing line. Survives
                 a whole pickup -> carry -> drop excursion, so it outlives many
                 values of `current`.

    Two memories, because `current` is wiped the moment a step completes and so
    cannot answer "what was I working on?". Only `sampled` can, and the next
    decision needs it.
    """
    #what the human is carrying, as a plain string or None. `_forced_held` wants
    #the NAME, not the object.
    held = state.players[human_index].held_object
    held_name = held.name if held else None

    hyps = []

    for fov, shadow in filt.shadows.items():
        #PULL OUT JUST THIS CONE'S ENTRIES, dropping the fov from the key.
        #
        #    {tau: p for (f, tau), p in filt.b.items() if f == fov and p > 0}
        #     ^^^^^^  ^^^^^^^^^^^^^^^                    ^^^^^^^^^^^^^^^^^^
        #     what to  unpack BOTH levels of              keep only this cone,
        #     build    the (key, value) pair              and only live entries
        #
        #`.items()` hands you (key, value) = ((fov, tau), prob). Writing
        #`(f, tau), p` on the left unpacks the outer pair AND splits the key in
        #one go. The long way is two lines:
        #        for key, p in filt.b.items():
        #            f, tau = key
        #Same thing. `tau` stays packed as (current, sampled) -- you only take it
        #apart when you need a half, which is once, at `tau[1]` below.
        mass = {tau: p for (f, tau), p in filt.b.items() if f == fov and p > 0}

        if not mass:
            continue                #this cone is already ruled out

        try:
            #IS THEIR NEXT MOVE FORCED BY THEIR HANDS?
            #`_forced_held` returns a subtask when they are holding something
            #(you cannot freely choose what to do with a full pair of hands) and
            #None when empty-handed. `is not None` turns that into a yes/no.
            #_transition needs it to answer the `untouched` question below.
            forced = shadow._forced_held(held_name) is not None

            for tau, p_tau in mass.items():
                #ONE CALL, TWO ANSWERS. `a, b = f()` splits a returned pair.
                #
                #  next_subtasks  {subtask: probability} -- what they might be
                #                 doing this tick. Usually a single certain
                #                 answer; only at a free choice does it spread.
                #  untouched      "was their standing line left alone?"
                #
                #untouched is True in the two cases where the sampler is never
                #consulted: the errand lock held (mid-errand, still useful), or
                #their hands are full (the item forces the move). It is False
                #only when hands are empty AND the lock broke -- the one moment
                #they actually make a free choice, which overwrites `sampled`.
                next_subtasks, untouched = filt._transition(shadow, state, tau,
                                                            forced)

                for subtask, p_z in next_subtasks.items():
                    if p_z <= 0.0:
                        continue        #impossible, do not carry it forward

                    #WHERE THEIR STANDING LINE LANDS NEXT TICK. Same bookkeeping
                    #inference.update() does when it builds next_tau, and the
                    #reason we need `untouched` at all:
                    #  untouched -> keep the old standing line, tau[1]
                    #  otherwise -> the sampler ran, so it becomes the new
                    #               choice... unless the "choice" was a look or
                    #               a wait, which are not choosable, and then it
                    #               is cleared to None.
                    #tau[1] is `sampled`; tau[0] would be `current`. You could
                    #also write `current, sampled = tau` up top if the index
                    #reads as a riddle later.
                    sampled = (tau[1] if untouched
                               else (subtask if subtask in SAMPLING_SUBTASKS
                                     else None))

                    #`prob` is the JOINT weight: how much I believed this
                    #(cone, tau) x how likely this subtask follows from it.
                    hyps.append({"fov": fov, "shadow": shadow,
                                 "subtask": subtask, "sampled": sampled,
                                 "prob": p_tau * p_z})
        except Exception:
            #a shadow that raises contributes nothing this tick; the filter
            #takes the same view (it discards the tick). Never fatal.
            continue

    #NORMALISE so the list sums to 1. Two things were multiplied together and
    #some tau were dropped, so the raw weights do not add up on their own.
    total = sum(h["prob"] for h in hyps)
    #`sum(... for ...)` is a GENERATOR expression -- a list comprehension without
    #the brackets. sum() takes one directly. (np.mean does NOT -- that is the bug
    #you hit in baseline.py.)
    if total <= 0.0:
        return []                   #nothing survives -> caller falls back to p_base
    for h in hyps:
        h["prob"] /= total
        #dicts are MUTABLE and `h` is the same object that sits in the list, so
        #editing it here edits the list. No reassignment needed.
    return hyps



# ---- STEP 4: clone_shadow(shadow, seed) ---------------------------------
#docstring: "a throwaway shadow that can actually be RUN -- action(), counters,
#RNG and all."
#WHY THIS EXISTS. The filter :drives its shadows through execute() and
#subtask_distribution(), which are PURE -- no writes, no RNG. A rollout needs the
#full action(), which writes _current, _sampled, t, subtask_log, the counters,
#AND DRAWS FROM _rng. Running that on a live shadow would desynchronise the
#filter from the real trajectory and correlate its draws with the real human's.
def clone_shadow(shadow, seed):
    """shadow copy, with its own RNG and counters, so the rollout can run it without touching the the real shadow"""
    sc = copy.copy(shadow) #shallow copy: new object, but attributes still point at the same things
    #beliefs and stations
    sc.beliefs = dict(shadow.beliefs)
    sc.stations = {k: list(v) for k, v in shadow.stations.items()}
#      dict(d) and list(l) each make a copy one level deep. stations is a dict OF
#      LISTS, so it needs both -- copying just the dict would leave the lists
#      shared and the rollout would "discover" stations for the real shadow.
    sc.seen_cells = set(shadow.seen_cells)
    sc.known_terrain = dict(shadow.known_terrain)
    sc.subtask_log = []
    sc._rng = random.Random(seed)
#      its own generator, seeded by us, so the rollout never consumes draws from
#      the real human's stream.
    for c in ("n_checks", "n_wasted_commits", "n_delivered", "n_abandoned",
                "n_explore"):
        #sets all of above to 0
        #Zeroing makes them mean "during this rollout" instead, which is what you'd want to read off afterwards — "did this candidate cause a wasted trip in the next 5 ticks?" as a diagnostic.
        setattr(sc, c, 0)
#      setattr(obj, "name", v) is `obj.name = v` with the name as a string, so
#      one loop zeroes all five. Zeroing them means a caller can read them
#      afterwards as "what happened during the rollout".
    return sc


# =========================================================================
# 3. THE MODULE
# =========================================================================                         anything at all" later.
class QMDPModule:
    """baseline's top-K -> one action, chosen by a d-step partner rollout."""

    def __init__(self, mdp, actor, k=3, depth=5, alpha=0.5,
                 horizon=400, human_index=HUMAN_INDEX, robot_index=ROBOT_INDEX):
        #NO cost object. cost_function is plain functions now, imported at the
        #top and called directly -- there is nothing to hold.
        self.mdp = mdp
        self.actor = actor
        self.k = int(k)
        self.depth = int(depth)
        self.alpha = float(alpha)
        self.horizon = int(horizon)
        self.hi = human_index
        self.ri = robot_index
        #per-tick debug: the candidates, their scores, how many hypotheses were
        #rolled. Costs nothing, and it is the only way to answer "is the module
        #doing anything at all" once this is running.
        self.last = {}


# ---- STEP 6: the robot's step inside the rollout ------------------------
    def _robot_action(self, state, rnn, t):
        """One step of the BASELINE inside the rollout, with a FORKED gru.

        Returns (action, new_rnn). The hidden state goes IN and comes BACK OUT;
        `self.actor.rnn` is never touched.

        THE HAZARD THIS EXISTS FOR: Actor.probs() writes self.rnn. Call that in
        here and imagining the future corrupts the memory of the policy actually
        playing the episode -- silently, with no error, for the rest of the run.
        The only symptom would be slightly worse numbers.
        """
        #PASS THE REAL TICK, never a hardcoded 0. Channel [21] of the
        #observation is "time left in the episode, normalised" and [22] is
        #"orders remaining". With t=0 you tell the baseline it has the full 400
        #ticks left on EVERY step of EVERY rollout, at minute six of the
        #episode. It still returns an action, so nothing looks broken -- you are
        #just rolling out a policy that thinks it has all the time in the world.
        obs = build_full_state(self.mdp, state, agent_index=self.ri,
                               t=t, horizon=self.horizon)

        #numpy -> torch, and onto whatever device the actor lives on.
        #  .astype(np.float32)  the net wants float32; MPS refuses float64
        #  [None, ...]          adds the batch axis: (23,W,H) -> (1,23,W,H)
        #  .to(device)          no-op on cpu, required if the actor is on gpu
        obs = torch.from_numpy(obs.astype(np.float32)[None, ...])
        obs = obs.to(self.actor.device)

        #ALWAYS ONES. NEVER ZEROS. This is the one that will bite you.
        #masks=0 WIPES the GRU (utils/rnn.py:61-63) and is correct in exactly
        #one place: the true first tick of a real episode, in Actor.probs. A
        #rollout is not that -- you forked the hidden state from the middle of a
        #live episode SPECIFICALLY to keep its memory, so wiping it on step one
        #throws away the whole reason you forked. Nothing crashes and nothing
        #warns; the rollout just silently models a memoryless policy that is not
        #the one playing the episode.
        masks = torch.ones(1, 1, device=self.actor.device)

        #CATCH THE RETURNED rnn. That is the entire point of this function --
        #the GRU advances on every forward pass, so the state that comes back is
        #what the NEXT rollout step must be given. Drop it and each step of the
        #rollout starts from the same stale memory.
        p, rnn = sp_action_probs(self.actor.policy, obs, rnn, masks)

        #back to a plain (6,) numpy array. detach() drops the autograd history
        #(we are not training), cpu() moves it off the gpu, reshape(-1) flattens
        #(1,6) to (6,).
        p = p.detach().cpu().numpy().reshape(-1)

        #ARGMAX, not a sample. The rollout should be the baseline's INTENDED
        #continuation -- "what would I do if I stopped deviating" -- and a
        #sample would inject noise that differs between candidates, so you would
        #be comparing dice rolls as much as actions.
        return Action.INDEX_TO_ACTION[int(np.argmax(p))], rnn



# ---- STEP 7: one rollout ------------------------------------------------
    def rollout(self, state, action, hyp, seed, t0):
        """Total score of taking `action` now, under ONE (fov, subtask)
        hypothesis, measured by cost_function. Higher is better.

            robot   `action` on the first step, then the baseline's argmax
            human   its own full policy, on a throwaway copy of the shadow

        `t0` is the REAL episode tick this starts from, threaded down from
        choose() so the imagined observations carry a truthful clock.

        A single straight-line trajectory. No branching -- that is the point of
        QMDP: inside a hypothesis everything is determined, so there is one
        future to walk, not a tree to search.
        """
        #A THROWAWAY HUMAN. The filter drives its shadows through the PURE
        #methods (execute / subtask_distribution). A rollout needs the full
        #action(), which writes _current, _sampled, t, the counters, AND draws
        #from _rng. Doing that to a live shadow would desynchronise the filter
        #from the real trajectory and correlate its dice with the real human's.
        shadow = clone_shadow(hyp["shadow"], seed)

        #start the copy on the standing line this hypothesis believes they are
        #on. We set `_sampled` and leave `_current` as None on purpose: their own
        #decide() then re-derives the errand from the standing line using their
        #own rules, rather than us asserting it. If the first tick ever turns out
        #to matter, setting `_current = hyp["subtask"]` too is the more literal
        #reading of the hypothesis.
        shadow._sampled = hyp["sampled"]

        #FORK THE GRU. .clone() gives an independent tensor; self.actor.rnn is
        #never written. See _robot_action for what goes wrong otherwise.
        rnn = self.actor.rnn.clone()

        #the BEFORE set for the first tick's credit, taken on the state we are
        #standing in, before anybody has moved.
        wrong = COST.wrong_beliefs(shadow, state)

        s = state
        a = action
        total = 0.0

        try:
            for step in range(self.depth):
                #remember the counter so we can tell whether they explored on
                #THIS tick specifically -- action() bumps it when execute()
                #falls back to wandering.
                n_explore_before = shadow.n_explore

                #THE HUMAN DECIDES ON THE SAME STATE THE ROBOT DID. Simultaneous
                #move: `a` was chosen for `s`, and so is `h`. action() also calls
                #observe(s) first, so the clone perceives this state through its
                #own cone before choosing.
                #_info["subtask"] is the ground-truth label. NEVER read it -- the
                #leading underscore is the reminder.
                h, _info = shadow.action(s)

                #resolve the joint move on a photocopy
                s = counterfactual_step(self.mdp, s, (a, h))

                explored = shadow.n_explore > n_explore_before

                #SCORE AFTER THE MOVE, AND MIND THE TIMING: the shadow's beliefs
                #are still the ones it formed observing the PREVIOUS state,
                #because it does not look again until the top of the next tick.
                #Truth is the state we just produced. So comparing them asks
                #exactly the right question -- "did what we just did make them
                #wrong?" -- rather than "have they noticed yet", which they have
                #not had the chance to.
                sc, wrong = COST.tick_score(self.mdp, shadow.planner, shadow, s,
                                       wrong, explored, self.ri)
                total += sc

                #orders ran out inside the daydream; nothing after this matters
                if self.mdp.is_terminal(s):
                    break

                #FROM HERE ON THE ROBOT IS JUST THE BASELINE AGAIN. That is what
                #makes this a one-step policy improvement: deviate once, then be
                #yourself. t0 + step + 1 because `s` is now the state after that
                #many joint moves.
                a, rnn = self._robot_action(s, rnn, t0 + step + 1)
        except Exception:
            #a rollout that dies halfway hands back what it scored so far rather
            #than killing the tick. Returning 0.0 instead would be worse than
            #useless: 0 is the "nothing happened" score, so a crash would read as
            #a clean candidate and could win the argmax.
            pass

        #DIAGNOSTICS, NEVER SCORED. These are why clone_shadow zeroes the
        #counters -- read here they mean "during this rollout", not "since the
        #episode began". They explain WHY a condition won; no decision reads them.
        self.last["wasted"] = self.last.get("wasted", 0) + shadow.n_wasted_commits
        self.last["abandoned"] = self.last.get("abandoned", 0) + shadow.n_abandoned

        #EFFORT: how much re-planning this candidate drove the partner into.
        #Measured, not asserted -- these are the human's own counters, read off a
        #rollout we simulated. Used ONLY by the gate in choose(); never added to
        #`total`, so it cannot overwrite the theory.
        effort = shadow.n_abandoned + shadow.n_wasted_commits
        return total, effort



# ---- STEP 8: which hypotheses get the deep roll -------------------------
    def _to_roll(self, reps, mass):
        """Which cones are worth a full d-step rollout.

            reps   one representative hypothesis per cone (built in choose)
            mass   {fov: total posterior mass of that cone}

        Returns a subset of `reps`. All of them while the posterior is flat; the
        leaders once it is not.

        WHY NOT ALWAYS JUST THE LEADER. While the belief is spread out the robot
        genuinely does not know who it is playing with, and rolling only the top
        cone would commit it to a hypothesis it has no evidence for -- confidently
        deferring to a 90-degree partner who is actually blind. Once one cone owns
        most of the mass the others cannot move the argmax anyway, so rolling them
        is compute spent on a foregone conclusion.

        So the rule is "spend compute exactly while it can change the answer",
        and CERTAIN is where that stops being true.
        """
        #TAKE `mass` AS AN ARGUMENT. Do NOT re-derive it by summing reps' "prob".
        #By the time this is called each cone has exactly ONE representative, and
        #that representative's prob is its own (fov, tau) share -- not the cone's.
        #A cone owning 0.75 spread over five tau has a representative at ~0.15,
        #so a re-derived total would read 0.15, conclude "still unsure", and
        #deep-roll all six cones forever. The pruning would silently never engage
        #and the only symptom would be unexplained slowness.
        if not mass:
            return reps

        #max() over .values() is the biggest single cone's mass. Below the
        #threshold, everybody rolls.
        if max(mass.values()) < CERTAIN:
            return reps

        #SORT THE CONES BY THEIR MASS, BIGGEST FIRST, AND KEEP THE TOP FEW.
        #    sorted(mass, ...)      walks a dict's KEYS (the fovs)
        #    key=mass.get           ...but orders them by the VALUE each maps to
        #    reverse=True           descending, so biggest first
        #    [:N_CERTAIN]           a SLICE: the first N items. Safe when the
        #                           list is shorter -- it just gives you all of
        #                           them, no error.
        #`key=` takes a FUNCTION that sorted() calls on each item to get the
        #thing to compare. mass.get is that function, unapplied -- note there are
        #no parentheses after it. Writing mass.get() would call it right here
        #with no argument and crash.
        keep = sorted(mass, key=mass.get, reverse=True)[:N_CERTAIN]

        #keep the representatives whose cone survived. `h["fov"] in keep` is a
        #membership test against the short list of winning cones.
        return [h for h in reps if h["fov"] in keep]



# ---- STEP 9: the decision ----------------------------------------------
    def choose(self, state, p_base, filt, seed=0):
        """-> action index. Called BEFORE the human has moved this tick."""
        #the baseline's six actions, its favourite first
        order = list(np.argsort(-np.asarray(p_base)))
        #the only ones we will even look at
        cand = order[:max(1, self.k)]
        self.last = {"cand": cand, "q": None, "n_hyp": 0}

        #k=1 means "just do what the baseline wanted" -- that IS the control arm
        if len(cand) == 1:
            return int(cand[0])

        #let the shadows look at the world, then ask who the partner might be
        sync_shadows(filt, state)
        hyps = predict(filt, state, self.hi)
        #no idea who they are -> no opinion -> the baseline's own pick
        if not hyps:
            return int(cand[0])

        #what time it really is, so the daydreams do not think the episode just started
        t0 = filt.tick

        #ONE ROLLOUT PER CONE, so inside each cone keep only its likeliest guess.
        per_fov = {}
        for h in hyps:
            #cur = the best guess kept for this cone SO FAR (None the first time we meet it)
            cur = per_fov.get(h["fov"])
            if cur is None or h["prob"] > cur["prob"]:
                per_fov[h["fov"]] = h

        #...but how much we BELIEVE a cone is all of its guesses added together,
        #not just the one we kept. Weighting by the survivor would quietly
        #discount the cone you believe in most, purely for having spread its
        #mass over several subtasks.
        mass = {}
        for h in hyps:
            mass[h["fov"]] = mass.get(h["fov"], 0.0) + h["prob"]

        #which cones are worth simulating: all of them while unsure, the leaders once sure
        rolled = self._to_roll(list(per_fov.values()), mass)
        #the survivors no longer add up to 1, so rescale over what is left
        z = sum(mass[h["fov"]] for h in rolled) or 1.0
        #how many cones we actually simulated -- the number that tells you
        #whether pruning is doing anything
        self.last["n_hyp"] = len(rolled)

        #score each candidate: simulate it against every cone, average by belief
        q, eff = {}, {}
        for idx in cand:
            a = Action.INDEX_TO_ACTION[int(idx)]
            acc = e_acc = 0.0
            for h in rolled:
                #SAME seed for every candidate, so the human rolls the same dice
                #in each branch and the only thing that differs is the robot
                w = mass[h["fov"]] / z
                sc, ef = self.rollout(state, a, h, seed, t0)
                acc += w * sc          #the THEORY: info gain + plan compatibility
                e_acc += w * ef        #the EVIDENCE: re-planning we forced on them
            q[int(idx)] = acc
            eff[int(idx)] = e_acc
        self.last["q"] = q
        self.last["effort"] = eff

        #THE GATE. If every candidate drives the partner into the same amount of
        #re-planning, there is no measured difference to act on -- so defer to the
        #baseline rather than rank noise. This is what makes "never worse than the
        #baseline" structural instead of something to tune for.
        #TAU_EFFORT = 0 disables it and recovers the ungated module exactly.
        spread = max(eff.values()) - min(eff.values())
        self.last["spread"] = spread
        if spread < TAU_EFFORT:
            return int(order[0])

        #best score wins. Ties fall back to whatever the baseline liked more --
        #so on the many ticks where nothing scores, we ARE the baseline.
        best = max(cand, key=lambda i: (q[int(i)], p_base[int(i)]))

        #never take something the baseline really did not want. We may re-rank
        #among options it finds comparable; we may not trade away its task
        #competence to protect a partner's plan. alpha=0 turns this off.
        if PROTECT_INTERACT and Action.INDEX_TO_ACTION[int(order[0])] == Action.INTERACT:
            best = order[0]

        if p_base[int(best)] < self.alpha * p_base[int(order[0])]:
            best = order[0]

        return int(best)
