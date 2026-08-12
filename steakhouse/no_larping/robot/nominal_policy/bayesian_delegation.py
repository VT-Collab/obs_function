"""B3  BayesianDelegationRobot -- inverse planning over sub-task ALLOCATIONS.

After Wu, Wang, Evans, Tenenbaum, Parkes and Kleiman-Weiner (2021), "Too Many
Cooks: Bayesian Inference for Coordinating Multi-Agent Collaboration"
(arXiv:2003.11778, AAMAS/JAAMAS). The mechanism, kept intact:

  1. ENUMERATE SUB-TASK ALLOCATIONS. An allocation says who is doing which
     sub-task: here a pair (tau_robot, tau_human). Both DIVIDE-AND-CONQUER
     allocations (the two agents on different sub-tasks) and JOINT allocations
     (both on the SAME sub-task) are in the set, because the paper's claim is
     that one piece of machinery chooses between them rather than a rule
     deciding in advance which regime we are in.
  2. HOLD A BELIEF P(allocation | history) -- "Bayesian delegation".
  3. UPDATE IT BY INVERSE PLANNING. For every allocation a level-0 planner says
     what the partner WOULD do next if that allocation were true; the observed
     partner action is scored against it with a Boltzmann likelihood; multiply
     into the prior and renormalise.
  4. ACT BY MARGINALISING. The robot takes the sub-task with the highest
     marginal posterior, which is the paper's selection rule and which, because
     the prior is a team-VALUE prior that penalises two cooks on one station,
     comes out COMPLEMENTARY to whatever the partner is probably doing -- and
     comes out JOINT when the sub-task genuinely rewards two pairs of hands.

THETA-BLIND, and that is the entire point of it
-----------------------------------------------
The partner model in this file assumes the human has FULL OBSERVABILITY. It
infers WHAT the human is doing; it has no model whatsoever of what the human can
SEE. There is no cone here, no visible_cells call, no assumed_fov, no
FOVPosterior, no reasoning about line of sight. Every sub-task the human might
be pursuing is enumerated from a TruthView, i.e. from the state the ROBOT
observes, exactly as if the human observed it too. That mis-specification is
deliberate: it is the strongest intent-inference control we can build without
crossing into observability, so whatever an FOV-aware policy beats it by is
attributable to theta and to nothing else.

Nor is there any collision handling, here or anywhere else in the package. The
partner is not an obstacle, nothing routes around them and nothing gives way:
every layout is two rooms joined only by pass-through counters, so the two agents
share no floor and were measured adjacent on 0 of 4800 ticks. What they can share
is a station embedded in the divide, worked from opposite sides -- and this robot
handles that through the value term rather than through a rule, since an
allocation that puts both of us on one station is paid once.

Relationship to the other baselines
-----------------------------------
    greedy    no partner model at all
    solo      no partner model at all beyond contention over shared stations
    handoff   solo plus blind staging
    bayes     partner as an AGENT WITH INTENT, inferred from their actions

Domain adaptation, and where it departs from the paper
------------------------------------------------------
  * SUB-TASKS are the entries of legal_subtasks(TruthView, held) -- the same
    (tier, verb, cell) ladder the human and the other robots run, so the
    comparison is like for like. The paper's sub-tasks come from a recipe graph;
    ours come from the ladder, which is the same thing written differently.
  * LEVEL-0 PLANNER: shortest path to the target cell, turn, INTERACT. Cheap,
    and it is exactly the plan the human model actually executes, which is what
    makes inverse planning identifiable here at all. It is partner-free (it does
    not route around anybody), which is what "level 0" means.
  * THE LIKELIHOOD IS SOFT, not a match test. Read FOVPosterior in
    robot/filter/fov_posterior.py: it faces the identical problem -- one predicted
    action per hypothesis, compared against one observed action -- and its
    lesson is that a hard match lets a single modelling error zero a hypothesis
    forever, after which the posterior can never recover. Same two-part fix
    here: a Boltzmann softmax over the level-0 planner's action values (beta is
    how rational we assume the partner is), then a uniform floor mixed in
    (alpha is how much we trust our own model of them, not a property of them).
  * VALUE, NOT STRICT TIER ORDER. The other baselines are lexicographic:
    tier first, distance only as a tie-break. This one cannot be, because
    Bayesian delegation compares ALLOCATIONS and needs them commensurable. So a
    sub-task is worth TIER_GAIN**(9 - tier) discounted by GAMMA**(steps to
    finish it). With the defaults one rung of the ladder is worth about 21
    steps of walking, so tier dominance holds over any distance these kitchens
    contain -- but it is a soft dominance, and on a big enough map this robot
    will take the ready garnish under its nose over the finished dish across the
    room. That is a deliberate difference from solo/handoff/greedy, not a bug.
  * THE PRIOR is proportional to that team value. The paper uses a prior
    inversely proportional to the COST of an allocation, which is the special
    case where every sub-task is worth the same. Ours are not -- delivering and
    stashing sit eight rungs apart -- so cost alone would put as much prior mass
    on putting a dish down as on serving it.
  * BELIEF DRIFT. The set of legal sub-tasks changes every tick as the world
    moves, so the belief is re-projected onto the current allocation set each
    tick and mixed `drift` of the way back toward the prior. That is the same
    guarantee as the uniform floor in the likelihood, one level up: no
    allocation can be permanently dead, and an allocation that only becomes
    possible now enters at its prior weight instead of at zero.

Interface: identical to the other baselines -- action(state) -> (act, info)
with info["subtask"], plus rank_subtasks(state), set_agent_index, set_mdp and
reset. It additionally exposes update(state, human_action), the same call
FOVPosterior takes and driven from the same place in play.py / watch.py.
"""
import math
import os
import sys
from collections import deque

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from overcooked_ai_py.mdp.actions import Action              # noqa: E402
from common import geometry as geo                            # noqa: E402
from common.tasks import legal_subtasks, TIER_NAME, T_EXPLORE  # noqa: E402
from common.views import TruthView                            # noqa: E402

N_ACTIONS = len(Action.ALL_ACTIONS)          # 6: four steps, STAY, INTERACT

# What a rung of the ladder is worth, and how fast that decays with the walk.
# TIER_GAIN**1 == GAMMA**-21, so one tier beats twenty-one tiles of detour: on
# kitchens this size the ladder still dominates, but it dominates by arithmetic
# instead of by construction. See the header.
TIER_GAIN = 3.0
GAMMA = 0.95

# Sub-tasks that are advanced by REPEATED INTERACT rather than by one. The env
# resolves each player's interact separately in the same tick
# (SteakHouseGridworld.resolve_interacts), so two cooks facing the same board
# from two different tiles really do chop twice as fast. These are the sub-tasks
# for which a JOINT allocation is worth more than a split one, and having any at
# all is what stops the joint half of the hypothesis space being decorative.
PARALLEL_VERBS = {"chop", "wash"}

# Enumerating every legal sub-task for both agents squares into the allocation
# set, so keep each agent's list to its best few by ladder rank. The tail is
# made of far-away stashes the ladder would never choose anyway.
MAX_SUBTASKS = 10

IDLE = None          # "nothing I can name" -- see _candidates()


def _cost_field(walk, target):
    """Steps from every walkable cell to a cell you can INTERACT with `target`.

    One backwards BFS per target replaces one A* per (agent, action, target),
    which is what makes enumerating a hundred allocations a tick affordable.
    """
    starts = geo.adjacent_standing_cells(walk, target)
    dist = {c: 0 for c in starts}
    q = deque(starts)
    while q:
        c = q.popleft()
        d = dist[c] + 1
        for dx, dy in geo.DIRECTIONS:
            n = (c[0] + dx, c[1] + dy)
            if n in walk and n not in dist:
                dist[n] = d
                q.append(n)
    return dist


def _reachable_from(walk, pos):
    seen = {pos}
    q = deque([pos])
    while q:
        c = q.popleft()
        for dx, dy in geo.DIRECTIONS:
            n = (c[0] + dx, c[1] + dy)
            if n in walk and n not in seen:
                seen.add(n)
                q.append(n)
    return seen


class _Snapshot:
    """Everything about one tick that both the belief update and the choice need.

    Built once and reused, because action(state) and update(state, a_H) are
    called back to back on the SAME state by both harnesses.
    """

    def __init__(self, bot, state):
        self.bot = bot
        self.view = TruthView(bot.mdp, state)
        self.walk = set(self.view.walkable)
        me, you = state.players[bot.agent_index], state.players[bot.other_index]
        self.me = (tuple(me.position), tuple(me.orientation),
                   me.held_object.name if me.held_object else None)
        self.you = (tuple(you.position), tuple(you.orientation),
                    you.held_object.name if you.held_object else None)
        # Both agents' own tiles are forced walkable so neither cost field can be
        # handed a start node outside its own graph. Note this ADDS rather than
        # removes: the partner is never an obstacle here either.
        self.walk |= {self.me[0], self.you[0]}
        self._fields = {}
        self.mine = self._candidates(self.me)
        self.yours = self._candidates(self.you)
        self.steps = {}                       # (who, sub) -> steps to finish
        for s in self.mine:
            self.steps[(0, s)] = self._steps(self.me, s)
        for s in self.yours:
            self.steps[(1, s)] = self._steps(self.you, s)
        self.allocs = [(r, h) for r in self.mine for h in self.yours]
        self.prior = self._prior()

    # -- geometry -----------------------------------------------------------
    def field(self, cell):
        f = self._fields.get(cell)
        if f is None:
            f = self._fields[cell] = _cost_field(self.walk, cell)
        return f

    def _steps(self, who, sub):
        """Ticks the level-0 planner needs to finish `sub` from `who`.

        walk there (d), turn to face it (1), press INTERACT (1). Standing beside
        it already facing it costs the one INTERACT and nothing else.
        """
        if sub is IDLE:
            return 0
        pos, orient, _ = who
        return self.steps_from(pos, orient, sub[2])

    def steps_from(self, pos, orient, cell):
        d = self.field(cell).get(pos)
        if d is None:
            return None
        if d == 0:
            return 1 if (pos[0] + orient[0], pos[1] + orient[1]) == cell else 2
        return d + 2

    # -- the sub-task menu --------------------------------------------------
    def _candidates(self, who):
        """This agent's legal, reachable sub-tasks, best few by ladder rank.

        FROM THE TRUTH VIEW FOR BOTH AGENTS. That is the theta-blind assumption
        stated as code: the robot enumerates the partner's options as though the
        partner saw the kitchen exactly as it does.
        """
        pos, orient, held = who
        reach = _reachable_from(self.walk, pos)

        def ok(cell):
            return any(n in reach
                       for n in geo.adjacent_standing_cells(self.walk, cell))

        subs = legal_subtasks(self.view, held, ok)

        # A STASH IS ONLY A HANDOFF IF THEY CAN GET TO IT. Putting a steak_dish
        # on our own bench is not staging it for anybody -- the split layouts
        # give the robot private worktop the human can never walk to, and
        # free_counters() over a TruthView returns all of it. So when WE are the
        # one stashing, the targets are narrowed to counters with a tile of the
        # PARTNER'S floor component beside them, still ordered by OUR distance:
        # "nearest counter to me that they can actually reach".
        #
        # Reachability, not visibility. Which counters they can WALK to is a fact
        # about the floor plan and this policy observes s, so it is fair game.
        # WHICH of those reachable counters they happen to be LOOKING at is the
        # thing it does not and must not know -- that is the FOV-aware policy's
        # job, and leaving it undone here is what keeps this an honest control.
        #
        # Never to zero: if none of them is reachable by the partner we stash
        # anywhere rather than stand there holding it. An urgent job elsewhere is
        # allowed to win over a tidy handoff.
        if who is self.me:
            theirs = _reachable_from(self.walk, self.you[0])
            shared = [s for s in subs if s[1] != "stash"
                      or any(n in theirs for n in
                             geo.adjacent_standing_cells(self.walk, s[2]))]
            if any(s[1] == "stash" for s in shared):
                subs = shared

        scored = []
        for sub in subs:
            n = self.steps_from(pos, orient, sub[2])
            if n is not None:
                scored.append((sub[0], n, sub[2], sub[1]))
        scored.sort()
        out = [(t, verb, cell) for t, _, cell, verb in scored[:MAX_SUBTASKS]]
        # An agent with nothing on the menu is EXPLORING, or doing something the
        # ladder cannot name from ground truth -- which for a limited-vision
        # partner happens often. Rather than leave the allocation set empty we
        # carry one IDLE hypothesis: worth nothing, and with a flat action
        # likelihood, so it soaks up exactly the evidence no real sub-task
        # explains instead of forcing that evidence onto a wrong one.
        return out or [IDLE]

    def parallel(self, sub):
        """Does a second pair of hands actually help with this sub-task?"""
        return (sub is not IDLE and sub[1] in PARALLEL_VERBS
                and len(geo.adjacent_standing_cells(self.walk, sub[2])) >= 2)

    # -- team value ---------------------------------------------------------
    def _worth(self, who_idx, sub):
        if sub is IDLE:
            return 0.0
        n = self.steps[(who_idx, sub)]
        if n is None:
            return 0.0
        return (TIER_GAIN ** (T_EXPLORE - sub[0])) * (GAMMA ** n)

    def value(self, alloc):
        """Expected team value of an allocation. Positive; used as the prior."""
        sr, sh = alloc
        vr, vh = self._worth(0, sr), self._worth(1, sh)
        if sr is IDLE or sh is IDLE or sr[2] != sh[2]:
            return vr + vh                    # divide and conquer: both are paid
        if sr == sh and self.parallel(sr):
            # JOINT WORK, and the only case where doubling up beats splitting:
            # a station that advances once per INTERACT advances twice when two
            # cooks stand at it, so the team is paid MORE than either alone.
            n = max(self.steps[(0, sr)] or 0, self.steps[(1, sh)] or 0)
            return self.bot.parallel_bonus * (TIER_GAIN ** (T_EXPLORE - sr[0])) \
                * (GAMMA ** n)
        # one station, one pair of hands' worth of progress: the team is paid
        # once, by whichever of us gets there first.
        return max(vr, vh)

    def _prior(self):
        v = {a: self.value(a) for a in self.allocs}
        z = sum(v.values())
        if z <= 0:
            n = float(len(self.allocs)) or 1.0
            return {a: 1.0 / n for a in self.allocs}
        return {a: x / z for a, x in v.items()}


class BayesianDelegationRobot:
    """Bayesian delegation over (tau_robot, tau_human). See the module docstring."""

    name = "bayes"

    def __init__(self, mdp, agent_index=0, seed=0, beta=2.0, alpha=0.9,
                 drift=0.1, parallel_bonus=1.3, inverse_planning=True):
        self.mdp = mdp
        self.seed = seed
        self.agent_index = agent_index
        self.other_index = 1 - agent_index
        self.beta = beta                  # partner rationality (Boltzmann temp)
        self.alpha = alpha                # trust in our own partner model
        self.drift = drift                # belief mixed back toward the prior
        self.parallel_bonus = parallel_bonus
        # The control for the control: with inverse planning off, the belief is
        # the prior every tick and the robot is a pure value-allocation planner.
        # Any concentration the posterior shows above this line is the evidence
        # doing work rather than the prior.
        self.inverse_planning = inverse_planning
        self.reset()

    def reset(self):
        import random
        self._rng = random.Random(self.seed * 2 + 3)
        self.t = 0
        self.belief = {}                  # allocation -> probability
        self.last_subtask = None
        self.last_action = None
        self._snap_for = None
        self._snap = None
        self.log = []

    # -- the HUD contract ---------------------------------------------------
    @property
    def p(self):
        """Deliberately EMPTY.

        play.py and watch.py read `.p` off whatever object is driving the
        per-tick update and print it as a posterior over CONE WIDTHS. This robot
        has no such posterior -- it is theta-blind by construction -- so the
        readout is empty and both HUDs fall through to saying exactly that. The
        belief this robot does hold is over sub-task allocations and is reported
        through info["partner_subtask"] and partner_posterior().
        """
        return {}

    # -- snapshot -----------------------------------------------------------
    def _snapshot(self, state):
        if state is not self._snap_for:
            self._snap_for, self._snap = state, _Snapshot(self, state)
        return self._snap

    # -- level-0 partner model ---------------------------------------------
    def _act_dist(self, snap, who, sub):
        """P(action | this agent is pursuing `sub`), Boltzmann over a shortest-path
        planner's action values.

        Q(a) = -(ticks still needed after taking a). Every action costs one tick,
        so the value of an action is entirely how much closer to the INTERACT it
        leaves you: a step toward the target is worth one more than a step away,
        turning into furniture aims you without moving you, and the INTERACT that
        completes the sub-task is worth the whole remainder.

        Then a uniform floor. Both halves matter and they are doing different
        jobs: beta says the partner is only approximately rational, alpha says
        our MODEL of the partner is only approximately right -- and it is alpha
        that keeps a hypothesis alive through the tick where we simply had them
        wrong, which is the failure FOVPosterior's docstring is about.
        """
        floor = (1.0 - self.alpha) / N_ACTIONS
        if sub is IDLE:
            return {a: 1.0 / N_ACTIONS for a in Action.ALL_ACTIONS}
        pos, orient, _ = who
        cur = snap.steps_from(pos, orient, sub[2])
        if cur is None:
            return {a: 1.0 / N_ACTIONS for a in Action.ALL_ACTIONS}

        q = {}
        for a in Action.ALL_ACTIONS:
            if a == Action.INTERACT:
                r = 0.0 if cur == 1 else cur   # cur == 1 means "beside it, facing it"
            elif a == Action.STAY:
                r = cur
            else:
                nxt = (pos[0] + a[0], pos[1] + a[1])
                if nxt not in snap.walk:
                    nxt = pos                  # into furniture: turns you, no step
                r = snap.steps_from(nxt, a, sub[2])
                r = cur + 4.0 if r is None else r
            q[a] = -float(r)

        top = max(q.values())
        w = {a: math.exp(self.beta * (v - top)) for a, v in q.items()}
        z = sum(w.values())
        return {a: self.alpha * (x / z) + floor for a, x in w.items()}

    # -- belief -------------------------------------------------------------
    def _carried(self, snap):
        """Last tick's belief re-projected onto this tick's allocation set."""
        prior = snap.prior
        keep = {a: self.belief.get(a, 0.0) for a in prior}
        z = sum(keep.values())
        if z <= 0:
            return dict(prior)
        r = self.drift
        return {a: (1.0 - r) * keep[a] / z + r * prior[a] for a in prior}

    def update(self, state, human_action):
        """Fold one observed partner action into the belief. INVERSE PLANNING.

        Same call signature and same place in the loop as FOVPosterior.update,
        so this robot drops straight into play.py / watch.py: the harness acts,
        then hands us the action the human took on the state we both saw.

        The observed action of BOTH agents is scored, which is what the paper
        does and is not redundant: our own action is the evidence that keeps the
        belief committed to the sub-task we are already walking towards, so
        commitment falls out of the same inference instead of needing a rule.
        """
        snap = self._snapshot(state)
        b = self._carried(snap)
        if self.inverse_planning:
            lik_h = {s: self._act_dist(snap, snap.you, s).get(human_action, 0.0)
                     for s in snap.yours}
            lik_r = {s: 1.0 for s in snap.mine}
            if self.last_action is not None:
                lik_r = {s: self._act_dist(snap, snap.me, s).get(self.last_action, 0.0)
                         for s in snap.mine}
            new = {a: b[a] * lik_r[a[0]] * lik_h[a[1]] for a in b}
            z = sum(new.values())
            b = dict(snap.prior) if z <= 0 else {a: v / z for a, v in new.items()}
        self.belief = b
        return self.partner_posterior()

    # -- readouts -----------------------------------------------------------
    def _marginal(self, i):
        out = {}
        for alloc, prob in self.belief.items():
            out[alloc[i]] = out.get(alloc[i], 0.0) + prob
        return out

    def partner_posterior(self):
        """P(tau_human | history). The thing inverse planning is supposed to buy."""
        return self._marginal(1)

    def partner_map(self):
        """(named_subtask, probability) or (None, 0.0)."""
        m = self.partner_posterior()
        if not m:
            return None, 0.0
        sub = max(m, key=lambda s: (m[s], s is not IDLE))
        name = None if sub is IDLE else (TIER_NAME[sub[0]], sub[1], sub[2])
        return name, m[sub]

    def partner_entropy(self):
        """Shannon entropy of P(tau_human), in bits. Falls as the belief sharpens."""
        m = self.partner_posterior()
        return -sum(p * math.log(p, 2) for p in m.values() if p > 0)

    # -- choice -------------------------------------------------------------
    def rank_subtasks(self, state):
        """[(tier, verb, cell)] best first, by MARGINAL POSTERIOR over my sub-task.

        This is the paper's selection rule. It reads as complementary-seeking
        because the prior is a team-value prior: once the belief says the human
        is probably on a station, every allocation that puts me on the same
        station is paying for one pair of hands' worth of progress, so its value
        -- and with it my marginal for that sub-task -- collapses. When the
        station is one that rewards two cooks, the same arithmetic runs the other
        way and doubling up wins.

        Does not mutate the belief; update() does that.
        """
        snap = self._snapshot(state)
        b = self._carried(snap)
        marg = {}
        for alloc, prob in b.items():
            marg[alloc[0]] = marg.get(alloc[0], 0.0) + prob
        out = []
        for sub, prob in marg.items():
            if sub is IDLE:
                continue
            out.append((-prob, sub[0], snap.steps[(0, sub)] or 0, sub[2], sub[1]))
        out.sort()
        return [(t, verb, cell) for _, t, _, cell, verb in out]

    def action(self, state):
        snap = self._snapshot(state)
        ranked = self.rank_subtasks(state)
        self.t += 1
        if not ranked:
            self.last_action = Action.STAY
            return Action.STAY, {"subtask": None, "partner_subtask": None}
        tier, verb, cell = ranked[0]
        self.last_subtask = (TIER_NAME[tier], verb, cell)

        pos, orient, _ = snap.me
        walk = snap.walk
        # Nothing follows the step. Where the other baselines have a contention
        # term, this robot has the value term: once the belief says the human is
        # probably on a station, every allocation putting me on the same one is
        # paid for a single pair of hands, so its marginal collapses and I go
        # elsewhere -- before we ever want the same board, not after.
        move, arrived = geo.step_towards(walk, pos, orient, cell)
        act = Action.INTERACT if arrived else (move or Action.STAY)

        self.last_action = act
        self.log.append(self.last_subtask)
        name, prob = self.partner_map()
        return act, {"subtask": self.last_subtask,
                     "partner_subtask": name,
                     "partner_p": prob,
                     "partner_entropy": self.partner_entropy(),
                     "n_alloc": len(self.belief)}

    # -- plumbing -----------------------------------------------------------
    def set_agent_index(self, i):
        self.agent_index = i
        self.other_index = 1 - i
        self.belief = {}          # the allocation set was keyed to the old seats
        self._snap_for = self._snap = None

    def set_mdp(self, mdp):
        self.mdp = mdp
        self.belief = {}
        self._snap_for = self._snap = None
