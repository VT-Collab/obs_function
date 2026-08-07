"""
Environment wrapper for SELF-PLAY.

Both chefs are driven by the SAME network, so this wrapper is deliberately
symmetric: it hands back one observation per agent, built from that agent's
own point of view, and one SHARED team reward.

===========================================================================
WHY TWO OBSERVATIONS PER ENV
===========================================================================
There are 2 agents in one kitchen. Each needs its own row through the
network, because features.build_full_state is EGO-CENTRIC -- agent_index
decides who lands in channels [0,1] ("me") and who lands in [2,3] ("them").

    obs for agent 0 : channels [0,1] = agent 0,  [2,3] = agent 1
    obs for agent 1 : channels [0,1] = agent 1,  [2,3] = agent 0

Same weights, different input, different behaviour. That is the only thing
stopping self-play from making both chefs do the identical dance.

===========================================================================
SHAPES
===========================================================================
one env      reset()/step() -> obs (2, 23, W, H)      one row per agent
vec of E     reset()/step() -> obs (E, 2, 23, W, H)

the runner then flattens (E, 2, ...) -> (N, ...) with N = E*2, and N is the
"kitchens" axis every layer in cnn/rnn/act already expects. The flatten is
thread-major:  [env0-agent0, env0-agent1, env1-agent0, env1-agent1, ...]
-- keep that order consistent everywhere or the agents get scrambled.

===========================================================================
REWARD
===========================================================================
SHARED. Both agents receive the same number, because a steakhouse order is
a team outcome -- there is no meaningful way to say which chef "earned" a
delivery when one of them chopped and the other plated.

Two components come back separately so the trainer can anneal them:

  sparse   the real objective: +delivery_reward when a dish is served.
           this is what you actually care about.
  shaped   hand-written breadcrumbs (meat in pan, dish pickup, ...) from
           BASE_REW_SHAPING_PARAMS. Sparse reward alone is far too rare to
           learn from early on, so you mix shaped in at the start and anneal
           it to zero, leaving only the true objective.

===========================================================================
MULTI-LAYOUT + PADDING
===========================================================================
One policy cannot span layouts of different grid sizes on its own: the CNN's
Linear(32*W*H, 64) weight shape is fixed by W*H. Tier 1 of the layout library
runs from steak_gc00 at (5,5) to steak_mid_2 at (15,11), so nothing would
transfer.

The fix is the same one ZSC-Eval uses (layout_generator.py: mdp_gen_fn_from_
dict / embed_grid): pick ONE canvas as big as the biggest layout, and drop
each smaller kitchen inside it.

    pad_shape = (max(widths), max(heights))  over the layouts being trained

Two details that matter:

  WHAT THE PADDING LOOKS LIKE.  Padded cells are written as 'X' terrain
  (counter) -- channel 18 in features.py -- NOT left as all-zeros. All-zeros
  across the 8 terrain channels means "floor" in this encoding, which would
  tell the agent the dead space outside the kitchen is walkable. Marking it
  counter says "solid, unreachable", which is what it actually is.

  RANDOM OFFSET.  Where the kitchen sits inside the canvas is resampled at
  every reset, so the policy cannot memorize absolute coordinates and has to
  read the terrain instead. Fixed WITHIN an episode -- the kitchen must not
  teleport mid-episode. Same trick as ZSC-Eval's embed_grid.

The two global scalar planes ([21] time-left, [22] orders-left) are filled
uniformly across the whole canvas, since they describe the episode and not
any particular cell.

COST: padding is not free. steak_gc00 is 25 real cells inside a 15x11 = 165
cell canvas, so ~85% of its conv work is dead space. That is the price of one
generalist policy instead of 25 specialists. Train on a narrower size band if
it bites -- e.g. the contention layouts alone pad to only (13,9).

===========================================================================
AUTO-RESET
===========================================================================
When an episode ends the vec wrapper resets that env immediately and hands
back the FIRST observation of the new episode. The done flag it returns
belongs to the step that just finished -- the runner turns it into
masks = 1 - done, which is exactly what wipes the GRU memory in rnn.py so
the new episode does not inherit the old one's memory.
"""

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import (
    SteakHouseGridworld, Action, BASE_REW_SHAPING_PARAMS)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from utils.features import build_full_state, N_LAYERS, TERRAIN, station_locs

#channel index of the 'X' (counter) terrain plane. features.py lays out
#[0-3] agents, [4-10] objects, [11-18] terrain in TERRAIN order, [19-22] the rest.
_COUNTER_CH = 11 + TERRAIN.index("X")
_TIME_CH = 21
_ORDERS_CH = 22


#===========================================================================
# DENSE SHAPING, ADDED ON TOP OF THE MDP  (mdp code is NOT touched)
#===========================================================================
# The steakhouse fork only fires TWO shaped rewards, both at the pan:
#     PLACEMENT_IN_POT_REW   meat goes in
#     COOKING_STEAK_REW      each cook tick
# Everything after that -- chop the onion, wash the plate, plate the steak,
# add the garnish, pick up the dish -- gives nothing. PLATE_PICKUP_REWARD is
# commented out in the fork and chop/wash/plate were never wired at all.
#
# So the agent gets dense signal for the first 2 steps of an 8-step chain and
# has to find the other 6 from delivery reward alone. That is the difference
# between this and ZSC-Eval, whose soup task is 4 steps and fully shaped.
#
# Rather than edit overcooked_mdp.py, we re-derive the missing events by
# DIFFING the state before and after each step, and add our own reward.
# Magnitudes mirror ZSC-Eval's BASE_REW_SHAPING_PARAMS (pickups 1-3,
# completions 3-5).
#
# ANTI-FARMING: every event type is capped per episode. Without a cap the
# agent finds the pick-up / put-down loop and farms it forever instead of
# cooking -- the classic shaping exploit. n_orders * 3 is generous enough to
# never bind on honest play.

#picked up off a dispenser/counter -- small, it is only the first move
_PICKUP_REW = {"meat": 1.0, "onion": 1.0, "plate": 1.0}
#a NEW object appearing at its station = a real step forward
_PLACE_REW = {"garnish": 3.0, "washed_plate": 3.0}
#each tick of chopping / washing, mirroring COOKING_STEAK_REW for the pan
_PROGRESS_REW = {"garnish": 1.0, "washed_plate": 1.0}
#carrying away a FINISHED intermediate, or assembling the dish
_COLLECT_REW = {"steak_dish": 3.0, "washed_plate": 3.0, "garnish": 3.0, "dish": 5.0}


class _DenseShaper:
    """Diffs consecutive states and pays out the events the mdp forgot."""

    def __init__(self, mdp, cap_per_event):
        self.mdp = mdp
        self.cap = cap_per_event
        self.counts = {}
        self._station_locs = set()
        for group in station_locs(mdp).values():
            self._station_locs.update(tuple(l) for l in group)

    def reset(self):
        self.counts = {}

    def _pay(self, key, amount):
        n = self.counts.get(key, 0)
        if n >= self.cap:
            return 0.0
        self.counts[key] = n + 1
        return amount

    @staticmethod
    def snapshot(state):
        held = tuple(p.held_object.name if p.held_object else None
                     for p in state.players)
        objs = {loc: (o.name, o.state) for loc, o in state.objects.items()}
        return held, objs

    def reward(self, prev, cur):
        (p_held, p_objs), (c_held, c_objs) = prev, cur
        r = 0.0

        #--- picked something up, or walked off with a finished item
        for i, (was, now) in enumerate(zip(p_held, c_held)):
            if was == now or now is None:
                continue
            if now in _COLLECT_REW:
                r += self._pay(f"collect_{now}", _COLLECT_REW[now])
            elif now in _PICKUP_REW:
                r += self._pay(f"pickup_{now}", _PICKUP_REW[now])

        #--- station events. counters are ignored on purpose: dropping an
        #    onion on a counter is not progress, only putting it on a board is
        for loc, (name, st) in c_objs.items():
            if loc not in self._station_locs:
                continue
            was = p_objs.get(loc)
            if was is None:
                if name in _PLACE_REW:
                    r += self._pay(f"place_{name}", _PLACE_REW[name])
            else:
                w_name, w_st = was
                #chop / wash tick. these states are bare ints; a steak's state
                #is a 3-tuple and the mdp already pays for its cook ticks.
                if (w_name == name and name in _PROGRESS_REW
                        and isinstance(st, (int, float))
                        and isinstance(w_st, (int, float)) and st > w_st):
                    r += self._pay(f"progress_{name}", _PROGRESS_REW[name])
        return r


def resolve_layouts(spec):
    """'a,b,c' or ['a','b'] -> list of layout names. Accepts a bare string."""
    if isinstance(spec, str):
        return [s.strip() for s in spec.split(",") if s.strip()]
    return list(spec)


class SteakSelfPlayEnv:
    """One kitchen (sampled from a pool), two chefs, one shared reward."""

    def __init__(self, layouts="steak_side_2", n_orders=4, horizon=260,
                 seed=None, pad_shape=None, dense_shaping=True):
        self.layout_names = resolve_layouts(layouts)
        self.horizon = horizon
        self.n_orders = n_orders

        #build every mdp ONCE up front -- from_layout_name parses a file, and
        #doing it per reset would show up in the rollout fps
        #start_order_list MUST be a real list. the .layout file declares it as
        #a string, and features.py asserts against that -- see the note there.
        self.mdps = [
            SteakHouseGridworld.from_layout_name(
                name, start_order_list=["steak"] * n_orders,
                rew_shaping_params=dict(BASE_REW_SHAPING_PARAMS))
            for name in self.layout_names
        ]

        #WHY rew_shaping_params IS PASSED EXPLICITLY:
        #every .layout file in the library declares  "rew_shaping_params": None
        #and overcooked_mdp.py:672 maps None -> NO_REW_SHAPING_PARAMS, which is
        #all ZEROS. Leave it alone and shaped reward is identically 0, so with
        #no deliveries yet the TOTAL reward is 0 for the whole rollout.
        #
        #That does not merely slow learning down, it actively destroys the run:
        #   rewards 0 -> returns 0 -> critic learns V=0 -> vloss 0.000
        #   -> advantages are float noise off V, ~1e-9
        #   -> the buffer normalizes them by their own std
        #   -> 1e-9 / 1e-9 = unit-scale garbage, at full gradient strength
        #Observed exactly that: vloss 0.000, clip_frac 0.45, entropy falling
        #1.79 -> 1.11 in 30 episodes. The policy was collapsing onto noise.
        assert any(v != 0 for v in self.mdps[0].reward_shaping_params.values()), (
            "reward shaping is all zeros -- the run will train on pure noise")

        #one canvas big enough for the biggest kitchen in the pool
        if pad_shape is None:
            pad_shape = (max(m.shape[0] for m in self.mdps),
                         max(m.shape[1] for m in self.mdps))
        self.pad_w, self.pad_h = pad_shape

        self.obs_shape = (N_LAYERS, self.pad_w, self.pad_h)
        self.n_actions = Action.NUM_ACTIONS          # 6
        self.num_agents = 2

        #one shaper per mdp, built once. cap generously: n_orders*3 payouts
        #per event type per episode is far above honest play but blocks the
        #pick-up/put-down farming loop.
        self.dense_shaping = dense_shaping
        self._shapers = ([_DenseShaper(m, n_orders * 3) for m in self.mdps]
                         if dense_shaping else None)
        self._shaper = None
        self._prev_snap = None

        self._rng = np.random.RandomState(seed)
        self.mdp = None
        self.env = None
        self.layout = None
        self._ox = self._oy = 0

    def _pad(self, obs):
        """(23, w, h) -> (23, pad_w, pad_h), kitchen dropped in at (ox, oy)."""
        if obs.shape[1:] == (self.pad_w, self.pad_h):
            return obs

        out = np.zeros((N_LAYERS, self.pad_w, self.pad_h), dtype=np.float32)
        #dead space reads as solid counter, NOT as walkable floor
        out[_COUNTER_CH] = 1.0

        w, h = obs.shape[1], obs.shape[2]
        out[:, self._ox:self._ox + w, self._oy:self._oy + h] = obs

        #global scalars describe the episode, not a cell -- keep them uniform
        out[_TIME_CH] = obs[_TIME_CH].flat[0]
        out[_ORDERS_CH] = obs[_ORDERS_CH].flat[0]
        return out

    def _obs(self):
        #one observation per agent, each from that agent's own perspective
        return np.stack([
            self._pad(build_full_state(self.mdp, self.env.state, agent_index=i,
                                       t=self.env.t, horizon=self.horizon))
            for i in range(self.num_agents)
        ]).astype(np.float32)                        # (2, 23, pad_w, pad_h)

    def reset(self):
        #resample which kitchen this episode is played in
        idx = self._rng.randint(len(self.mdps))
        self.mdp = self.mdps[idx]
        self.layout = self.layout_names[idx]

        #and resample WHERE it sits on the canvas, so the policy cannot
        #memorize absolute coordinates. fixed for the whole episode.
        w, h = self.mdp.shape
        self._ox = self._rng.randint(self.pad_w - w + 1)
        self._oy = self._rng.randint(self.pad_h - h + 1)

        #+10 so is_done() is driven by our own horizon check, not the env's
        self.env = OvercookedEnv.from_mdp(self.mdp, info_level=0,
                                          horizon=self.horizon + 10)

        if self.dense_shaping:
            self._shaper = self._shapers[idx]
            self._shaper.reset()
            self._prev_snap = _DenseShaper.snapshot(self.env.state)

        return self._obs()

    def step(self, actions):
        """
        actions: array-like of 2 ints in [0, 6).
        returns obs, sparse, shaped, done, truncated

        TRUNCATED is the "proper time limits" flag (ZSC-Eval sets
        --use_proper_time_limits in train_sp.sh). It marks an episode that was
        CUT OFF by the horizon rather than genuinely finishing.

        Why it matters here more than anywhere: the only true terminal is
        running the order list down, and early in training that essentially
        never happens -- so EVERY episode boundary is a truncation. Treating
        those as real terminals tells GAE the world ends at t=horizon with
        value 0, which systematically depresses the value of every state near
        the end of every episode. The critic then learns that the last ~50
        ticks are worthless, and the policy stops trying there.
        """
        joint = tuple(Action.INDEX_TO_ACTION[int(a)] for a in actions)
        _, sparse_reward, done, info = self.env.step(joint)

        #both agents get the team total -- deliveries are not attributable
        shaped_reward = float(sum(info.get("shaped_r_by_agent", (0.0, 0.0))))

        #plus everything the mdp forgot to reward, re-derived by diffing state
        if self.dense_shaping:
            snap = _DenseShaper.snapshot(self.env.state)
            shaped_reward += self._shaper.reward(self._prev_snap, snap)
            self._prev_snap = snap

        hit_horizon = self.env.t >= self.horizon
        terminal = bool(done) and not hit_horizon      # order list ran out
        done = bool(done or hit_horizon)
        truncated = done and not terminal

        return self._obs(), float(sparse_reward), shaped_reward, done, truncated


class VecSteakEnv:
    """
    E copies of SteakSelfPlayEnv stepped in lockstep, with auto-reset.

    Sequential on purpose: this is the simple, debuggable version. It is also
    the throughput bottleneck -- if you need more, this is the thing to swap
    for a SubprocVecEnv, NOT the network. See the GPU note in rMAPPOPolicy.
    """

    def __init__(self, layouts="steak_side_2", n_threads=8, n_orders=4,
                 horizon=260, seed=0, dense_shaping=True):
        self.layout_names = resolve_layouts(layouts)

        #every env must agree on the canvas, so compute it ONCE here and pass
        #it down. letting each env compute its own max would work today but
        #would silently diverge the moment the pools differ.
        probe = [SteakHouseGridworld.from_layout_name(
                     n, start_order_list=["steak"] * n_orders)
                 for n in self.layout_names]
        pad_shape = (max(m.shape[0] for m in probe),
                     max(m.shape[1] for m in probe))
        self.pad_shape = pad_shape
        self.layout_sizes = {n: m.shape for n, m in zip(self.layout_names, probe)}

        self.envs = [SteakSelfPlayEnv(self.layout_names, n_orders, horizon,
                                      seed + i, pad_shape, dense_shaping)
                     for i in range(n_threads)]
        self.n_threads = n_threads
        self.num_agents = self.envs[0].num_agents
        self.obs_shape = self.envs[0].obs_shape
        self.n_actions = self.envs[0].n_actions

        #per-episode bookkeeping, so the runner can log real episode returns
        self._ep_sparse = np.zeros(n_threads, dtype=np.float32)
        self._ep_len = np.zeros(n_threads, dtype=np.int32)

    def reset(self):
        self._ep_sparse[:] = 0.0
        self._ep_len[:] = 0
        return np.stack([e.reset() for e in self.envs])      # (E, 2, 23, W, H)

    def step(self, actions):
        """
        actions: (E, 2) ints
        returns obs (E,2,23,W,H), sparse (E,), shaped (E,), dones (E,), infos
        """
        obs, sparse, shaped, dones, truncs, infos = [], [], [], [], [], []

        for i, env in enumerate(self.envs):
            o, sp, sh, d, tr = env.step(actions[i])

            self._ep_sparse[i] += sp
            self._ep_len[i] += 1

            if d:
                #record what the finished episode achieved BEFORE wiping it.
                #layout is logged too so per-layout returns can be broken out
                #-- with a mixed pool the aggregate average hides which
                #kitchens the policy is actually failing.
                infos.append({"episode_sparse": float(self._ep_sparse[i]),
                              "episode_length": int(self._ep_len[i]),
                              "layout": env.layout})
                self._ep_sparse[i] = 0.0
                self._ep_len[i] = 0
                #auto-reset: o becomes the FIRST obs of the NEW episode.
                #the done flag still refers to the step that just ended, and
                #the runner turns it into masks=0 so the GRU memory gets wiped.
                o = env.reset()
            else:
                infos.append({})

            obs.append(o); sparse.append(sp); shaped.append(sh)
            dones.append(d); truncs.append(tr)

        return (np.stack(obs),
                np.array(sparse, dtype=np.float32),
                np.array(shaped, dtype=np.float32),
                np.array(dones, dtype=bool),
                np.array(truncs, dtype=bool),
                infos)
