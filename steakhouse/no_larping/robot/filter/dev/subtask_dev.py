"""How often does the layer pursue a job its baseline did not pick?

NOT "a non-argmax subtask", which is what this was first written to measure. Every
baseline in the registry is now a `-stoch` rung, so the baseline's own choice is a
STICKY DRAW from pi rather than an argmax of anything -- `argmax pi` is a job the
baseline may well not be doing. The question that survives is the one in the title:
the layer's committed job against the job the baseline actually realised.

    python -m robot.filter.dev.subtask_dev --layouts divide --fovs 60 --seeds 0
    python -m robot.filter.dev.subtask_dev --all --out dev.jsonl
    python -m robot.filter.dev.subtask_dev --load OUT/*.jsonl --only qmdp

`deviated_frac` in evaluate.py answers a DIFFERENT question and has been mistaken
for this one. It counts ticks where the emitted ACTION differs from the
baseline's action, which conflates three things the design keeps apart:

    same job, different target cell     job matches, cell does not
    same job, same cell, later tick     both match, t_end differs
    a different job altogether          job does not match

Only the third is "pursuing a non-argmax subtask", and it is invisible in
`deviated` both ways round. A different job can produce the SAME action -- two
jobs on opposite sides of a corridor share their first ten steps -- and the same
job can produce a different action. So the two counts are not nested and neither
bounds the other.

WHAT IT COMPARES AGAINST. `base_subtask`: the wrapped baseline's own realised pick
for this tick, taken from the very `baseline.action(state)` call the filter already
makes. NOT `ranked[0]` -- a `-stoch` rung draws its job from pi and then holds it,
and even a deterministic ladder sticks within a tier (baselines.py:157), so the head
of the ranking is not what either would do. `top_subtask` records `ranked[0]`
alongside it, and the gap between the two is now expected to be LARGE rather than
small: it is the stochastic policy declining its own argmax, which is the whole
point of the `-stoch` rungs. Both numbers are printed; the realised pick is the one
the comparison is about.

WHAT COUNTS AS PURSUING. Attribution and follow-through are reported separately
because they are not the same claim and the weaker one is easy to overread:

  ATTRIBUTION (per tick)   the (v, g) the winning Q came from. This is what the
                           filter commits to and what the HUD shows, but on a
                           tick where the action is unchanged the divergence is
                           inert -- the robot moves exactly as the baseline
                           would and only the label differs.
  PURSUIT (per run)        a maximal stretch of consecutive ticks holding one
                           (v, g). Classified by the tick the choice was MADE.
                           This is the unit that costs something: a run of k
                           ticks is k ticks of walking committed to that job.
  FOLLOW-THROUGH           did the run end on an INTERACT? A run that ends
                           because the commitment was dropped mid-walk bought
                           nothing and paid k ticks for it -- the abandonment tax
                           RESULTS.md section 5 measures, restricted here to the
                           runs that were off-argmax in the first place.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))        # .../robot/filter/dev
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../no_larping
sys.path.insert(0, os.environ.get("STEAK_ROOT", os.path.dirname(ROOT)))
sys.path.insert(0, ROOT)

import overcooked_ai_py                                            # noqa: E402
overcooked_ai_py.LAYOUTS_DIR = os.path.join(ROOT, "layout", "layouts")

from overcooked_ai_py.mdp.actions import Action                    # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv      # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld  # noqa: E402
from human.limited_vision_human import (LimitedVisionHuman,        # noqa: E402
                                        FORGET_HORIZON)
from common.tasks import TIER_NAME                                 # noqa: E402
from robot.filter.harness.evaluate import (build_robot, layout_sha,   # noqa: E402
                                           FOVS, LAYOUTS, _expand)

# name -> ladder index, smaller = more urgent (tasks.py:34). Needed because the
# subtask tuples carry the tier NAME and the DIRECTION of a tier change is what
# matters: dropping a rung means the layer preferred a less urgent job on
# estimated completion time, which is the trade C exists to make.
#
# DO NOT ASSUME THE DIRECTION. Under the old deterministic ladders it was forced:
# they sort tier-first and stick only within a tier, so the realised pick always
# sat on the most urgent legal tier and every tier change was necessarily
# downwards -- a tautology, not a finding. That no longer holds. Every registered
# baseline is now a -stoch rung, and StochasticSubtask holds its job across ticks
# unless it finishes, goes illegal, or a strictly MORE urgent tier appears, so its
# realised pick can sit below the top of the ladder and a change can go either
# way. Both directions are counted here for that reason.
TIER_IDX = {v: k for k, v in TIER_NAME.items()}


def _job(sub):
    """(tier_name, verb) -- the JOB, dropping the target cell.

    The same unit QMDPFilter._jobs aggregates on, and for the same reason:
    the cell is a property of the PLAN, not of the job, so a cell change is a
    different question from a job change and must not be counted as one.
    """
    return None if sub is None else tuple(sub[:2])


class Tally:
    """Per-episode counters. One instance per episode, fed one tick at a time."""

    def __init__(self):
        self.n = 0                  # ticks where the search produced a plan
        self.n_nosearch = 0         # ticks it fell straight through to baseline
        # attribution, per tick
        self.same = 0
        self.off_cell = 0           # same job, different target cell
        self.off_job = 0            # different job -- the number asked for
        self.off_tier = 0           # off_job, and a different tier
        self.off_verb = 0           # off_job, same tier, different verb
        self.off_less_urgent = 0    # off_tier, and OUR tier is the higher number
        self.off_more_urgent = 0    # off_tier the other way -- see TIER_IDX
        self.tier_gap = 0           # sum |our tier - theirs| over off_tier ticks
        self.off_top = 0            # vs top_subtask (bare argmax) instead
        self.pair_idx_nz = 0        # winning Q came from a non-baseline pair
        self.base_unscored = 0      # baseline's own pair not even in `pairs`
        # THE STRICT NUMBER. off_job says the committed job is not the one the
        # baseline picked; it does NOT say the layer chose that. `pairs[0]` is the
        # top job by pi mass, which is not always the baseline's realised pick, so
        # an off_job tick can be the candidate list simply starting elsewhere.
        # off_job_q requires BOTH: the job differs AND the winning Q came from a
        # pair other than pairs[0], i.e. the search moved it. enum_gap counts the
        # other case so the two never have to be inferred from each other.
        self.off_job_q = 0
        self.enum_gap = 0           # pairs[0]'s job != baseline's realised job
        # cross-tab with the ACTION
        self.dev = 0                # action != baseline action
        self.off_job_dev = 0        # off-job AND action changed
        self.off_job_inert = 0      # off-job, action identical anyway
        self.same_job_dev = 0       # same job, action changed (route/wait)
        # runs
        self.runs = 0
        self.off_runs = 0
        self.off_run_ticks = 0
        self.off_runs_done = 0      # ended on an INTERACT
        self.off_runs_q = 0         # off_job AND search-driven, per above
        self.off_run_q_ticks = 0
        self.off_runs_q_done = 0
        self._cur = None            # [Q, off_at_start, off_q_at_start, ticks]

    def tick(self, info, act):
        q, b = info.get("subtask"), info.get("base_subtask")
        top = info.get("top_subtask")
        if "plan" not in info:
            # No plan this tick: no candidates, no cones, or empty C. The filter
            # returned the baseline's action verbatim, so there is nothing to
            # attribute and counting it as agreement would inflate `same`.
            self.n_nosearch += 1
            self._close(False)
            return
        self.n += 1
        if info.get("deviated"):
            self.dev += 1

        off = _job(q) != _job(b)
        if q is not None and top is not None and _job(q) != _job(top):
            self.off_top += 1
        if off:
            self.off_job += 1
            # off_job_q used to require "and the search moved it", separating a real
            # re-rank from the pi-ordered candidate list merely starting elsewhere.
            # One union mask has no such list, so the two counts are now identical
            # by construction and `enum_gap` is structurally 0. Kept equal rather
            # than deleted so a pre-consolidation grid and a current one still print
            # in the same shape.
            self.off_job_q += 1
            if q is not None and b is not None and q[0] != b[0]:
                self.off_tier += 1
                dq, db = TIER_IDX.get(q[0], 0), TIER_IDX.get(b[0], 0)
                self.tier_gap += abs(dq - db)
                if dq > db:
                    self.off_less_urgent += 1
                else:
                    self.off_more_urgent += 1
            else:
                self.off_verb += 1
            if info.get("deviated"):
                self.off_job_dev += 1
            else:
                self.off_job_inert += 1
        elif q is not None and b is not None and q[2] != b[2]:
            self.off_cell += 1
            if info.get("deviated"):
                self.same_job_dev += 1
        else:
            self.same += 1
            if info.get("deviated"):
                self.same_job_dev += 1

        # A run is keyed on the whole (v, g) the filter committed to, so a cell
        # switch inside one job ends the run -- it is a new approach to a new
        # counter and its ticks are not the first one's.
        # With one mask there is no "did the search move it" to separate out:
        # every off-job tick IS the search choosing, because the only other
        # candidate source the old filter had -- a pi-ordered pair list -- is
        # gone. off_q is kept equal to off so the run bookkeeping below and the
        # saved JSONL schema stay readable against older grids.
        off_q = off
        if self._cur is None or self._cur[0] != q:
            self._close(False)
            self._cur = [q, off, off_q, 0]
            self.runs += 1
            if off:
                self.off_runs += 1
            if off_q:
                self.off_runs_q += 1
        self._cur[3] += 1
        if self._cur[1]:
            self.off_run_ticks += 1
        if self._cur[2]:
            self.off_run_q_ticks += 1
        # The first INTERACT is what certifies the job (DESIGN.md 3.2), and the
        # filter releases the commitment on exactly that tick, so an INTERACT
        # both ends the run and marks it completed.
        if act == Action.INTERACT:
            self._close(True)

    def _close(self, done):
        if self._cur is None:
            return
        _q, off, off_q, _k = self._cur
        if off and done:
            self.off_runs_done += 1
        if off_q and done:
            self.off_runs_q_done += 1
        self._cur = None

    def result(self):
        self._close(False)
        n = max(self.n, 1)
        r = self.off_runs or 1
        return {
            "scored_ticks": self.n, "nosearch_ticks": self.n_nosearch,
            "same": self.same, "off_cell": self.off_cell,
            "off_job": self.off_job, "off_job_frac": self.off_job / n,
            "off_job_q": self.off_job_q, "off_job_q_frac": self.off_job_q / n,
            "enum_gap": self.enum_gap,
            "off_tier": self.off_tier, "off_verb": self.off_verb,
            "off_less_urgent": self.off_less_urgent,
            "off_more_urgent": self.off_more_urgent,
            "tier_gap": self.tier_gap,
            "off_top_job": self.off_top, "off_top_frac": self.off_top / n,
            "pair_idx_nz": self.pair_idx_nz, "base_unscored": self.base_unscored,
            "deviated": self.dev, "deviated_frac": self.dev / n,
            "off_job_dev": self.off_job_dev, "off_job_inert": self.off_job_inert,
            "same_job_dev": self.same_job_dev,
            "runs": self.runs, "off_runs": self.off_runs,
            "off_run_ticks": self.off_run_ticks,
            "off_runs_done": self.off_runs_done,
            "off_follow_through": self.off_runs_done / r,
            "off_runs_q": self.off_runs_q,
            "off_run_q_ticks": self.off_run_q_ticks,
            "off_runs_q_done": self.off_runs_q_done,
        }


def run_episode(layout, fov, seed, method, horizon=400, forget=FORGET_HORIZON,
                top_k=3, depth=40, human_index=1, trace=None):
    """One episode, with the tick order copied from evaluate.run_episode.

    The order is load-bearing and is not re-argued here: robot decides, human
    decides on the same un-mutated state, THEN the posterior folds in the human's
    action. See evaluate.py's docstring. Deliveries and ticks are returned too so
    a deviation rate can be read next to the outcome it produced instead of on
    its own.
    """
    mdp = SteakHouseGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env.reset()
    human_idx, robot_idx = human_index, 1 - human_index
    human = LimitedVisionHuman(mdp, fov, agent_index=human_idx,
                               forget_horizon=forget, seed=seed)
    robot, post = build_robot(method, mdp, robot_idx, human_idx, seed, top_k, depth)

    orders_total = len(env.state.order_list or [])
    tally = Tally()
    done = False
    while not done:
        state = env.state
        r_act, r_info = robot.action(state)
        h_act, _ = human.action(state)
        if post is not None:
            post.update(state, h_act)
        tally.tick(r_info, r_act)
        if trace is not None and "plan" in r_info:
            trace.append({
                "layout": layout, "fov": fov, "seed": seed, "t": env.t,
                "q": str(r_info.get("subtask")),
                "base": str(r_info.get("base_subtask")),
                "top": str(r_info.get("top_subtask")),
                "act": str(r_act), "dev": bool(r_info.get("deviated")),
                "gain": r_info.get("gain"),
            })
        joint = [None, None]
        joint[robot_idx], joint[human_idx] = r_act, h_act
        nxt, _, done, _ = env.step(tuple(joint))
        done = done or mdp.is_terminal(nxt)

    # layout_sha for the same reason evaluate.py stamps it: four of the six
    # .layout files were edited mid-experiment, and two grids with different
    # fingerprints are measuring different kitchens however tempting the diff
    # looks. analysis/layout_facts.py's docstring has the incident.
    return {"layout": layout, "fov": fov, "seed": seed, "method": method,
            "layout_sha": layout_sha(layout),
            "delivered": orders_total - len(env.state.order_list or []),
            "ticks": env.t, **tally.result()}


def summarise(rows):
    """Pooled over seeds within a cell, then over cells. Printed, not returned.

    Ticks are the denominator throughout and they are pooled by SUM rather than
    by averaging per-episode fractions: an episode that ended at tick 200
    contributes half the ticks of one that ran to the horizon and should carry
    half the weight, which averaging fractions would not do.
    """
    keys = ["scored_ticks", "nosearch_ticks", "same", "off_cell", "off_job", "off_job_q",
            "enum_gap", "off_tier", "off_verb", "off_less_urgent",
            "off_more_urgent", "tier_gap", "off_top_job", "pair_idx_nz",
            "deviated", "off_job_dev", "off_job_inert", "same_job_dev", "runs",
            "off_runs", "off_run_ticks", "off_runs_done", "base_unscored",
            "off_runs_q", "off_run_q_ticks", "off_runs_q_done"]
    by = collections.defaultdict(lambda: collections.Counter())
    eps = collections.Counter()
    for r in rows:
        k = (r["method"], r["layout"], r["fov"])
        for key in keys:
            by[k][key] += r.get(key, 0)
        by[k]["delivered"] += r.get("delivered", 0)
        eps[k] += 1

    hdr = ("method       layout        fov dish ticks | off-job%  strict%  "
           "act-chg%| runs  off  len  follow%| dev%  cell%")
    print(hdr)
    print("-" * len(hdr))
    tot = collections.Counter()
    for k in sorted(by):
        c = by[k]
        n = max(c["scored_ticks"], 1)
        r = max(c["off_runs_q"], 1)
        print("%-12s %-13s %4d %4.1f %5d | %6.1f%% %6.1f%% %7.1f%%| "
              "%4d %4d %4.1f %6.1f%%| %4.1f%% %5.1f%%"
              % (k[0], k[1], k[2], c["delivered"] / max(eps[k], 1),
                 c["scored_ticks"],
                 100.0 * c["off_job"] / n, 100.0 * c["off_job_q"] / n,
                 100.0 * c["off_job_dev"] / max(c["off_job"], 1),
                 c["runs"], c["off_runs_q"], c["off_run_q_ticks"] / r,
                 100.0 * c["off_runs_q_done"] / r,
                 100.0 * c["deviated"] / n, 100.0 * c["off_cell"] / n))
        for key in keys:
            tot[key] += c[key]

    n = max(tot["scored_ticks"], 1)
    r = max(tot["off_runs"], 1)
    rq = max(tot["off_runs_q"], 1)
    pct = lambda x, d=n: 100.0 * x / max(d, 1)
    print("-" * len(hdr))
    print("POOLED  %d scored ticks (+%d with no plan) over %d episodes"
          % (tot["scored_ticks"], tot.get("nosearch_ticks", 0), sum(eps.values())))
    print()
    print("ATTRIBUTION, per tick -- which job the winning Q came from")
    print("  the baseline's own job      %6d  %5.1f%%"
          % (tot["same"] + tot["off_cell"], pct(tot["same"] + tot["off_cell"])))
    print("    same job, different CELL  %6d  %5.1f%%"
          % (tot["off_cell"], pct(tot["off_cell"])))
    print("  a DIFFERENT job            %6d  %5.1f%%"
          % (tot["off_job"], pct(tot["off_job"])))
    print("    search moved it (STRICT)  %6d  %5.1f%%"
          % (tot["off_job_q"], pct(tot["off_job_q"])))
    print("    candidate list started   %6d  %5.1f%%   <- not a choice; pairs[0]"
          % (tot["off_job"] - tot["off_job_q"], pct(tot["off_job"] - tot["off_job_q"])))
    print("      elsewhere                                 disagreed with the baseline"
          " on %.1f%% of ticks" % pct(tot["enum_gap"]))
    print("    a different TIER          %6d  %5.1f%%  (mean %.1f rungs apart)"
          % (tot["off_tier"], pct(tot["off_tier"]),
             tot["tier_gap"] / max(tot["off_tier"], 1)))
    print("      LESS urgent than the baseline's  %6d  %5.1f%% of off-tier"
          % (tot["off_less_urgent"], pct(tot["off_less_urgent"], tot["off_tier"])))
    print("      MORE urgent                     %6d  %5.1f%% of off-tier"
          % (tot["off_more_urgent"], pct(tot["off_more_urgent"], tot["off_tier"])))
    print("    same tier, different VERB %6d  %5.1f%%"
          % (tot["off_verb"], pct(tot["off_verb"])))
    print("  vs bare argmax ranked[0]   %6d  %5.1f%%"
          % (tot["off_top_job"], pct(tot["off_top_job"])))
    print("  baseline pair never scored %6d  %5.1f%%"
          % (tot["base_unscored"], pct(tot["base_unscored"])))
    print()
    print("DID IT CHANGE THE ROBOT'S MOVE?")
    print("  off-job AND action changed  %6d  %5.1f%% of off-job ticks"
          % (tot["off_job_dev"], pct(tot["off_job_dev"], tot["off_job"])))
    print("  off-job, action identical   %6d  %5.1f%% of off-job ticks (inert)"
          % (tot["off_job_inert"], pct(tot["off_job_inert"], tot["off_job"])))
    print("  ACTION deviations, all      %6d  %5.1f%%  (same-job %d)"
          % (tot["deviated"], pct(tot["deviated"]), tot["same_job_dev"]))
    print()
    print("PURSUIT, per commitment run -- the unit that costs walking")
    print("  runs                        %6d" % tot["runs"])
    print("  off-job runs                %6d  %5.1f%% of runs"
          % (tot["off_runs"], pct(tot["off_runs"], tot["runs"])))
    print("    mean ticks held           %6.1f    total %d ticks"
          % (tot["off_run_ticks"] / r, tot["off_run_ticks"]))
    print("    ended on an INTERACT      %6d  %5.1f%%  (rest dropped mid-walk)"
          % (tot["off_runs_done"], pct(tot["off_runs_done"], r)))
    print("  STRICT (search-moved) runs  %6d  %5.1f%% of runs"
          % (tot["off_runs_q"], pct(tot["off_runs_q"], tot["runs"])))
    print("    mean ticks held           %6.1f    total %d ticks"
          % (tot["off_run_q_ticks"] / rq, tot["off_run_q_ticks"]))
    print("    ended on an INTERACT      %6d  %5.1f%%  (rest dropped mid-walk)"
          % (tot["off_runs_q_done"], pct(tot["off_runs_q_done"], rq)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layouts", default="divide")
    p.add_argument("--fovs", default="60")
    p.add_argument("--seeds", default="0")
    p.add_argument("--methods", default="qmdp")
    p.add_argument("--all", action="store_true")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--forget", type=int, default=FORGET_HORIZON)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--depth", type=int, default=40)
    p.add_argument("--out", default=None, help="per-episode JSONL")
    p.add_argument("--trace-out", default=None, help="per-tick JSONL")
    p.add_argument("--load", nargs="*", default=None,
                   help="summarise existing JSONL instead of running anything")
    p.add_argument("--only", default=None,
                   help="with --load: pool one method at a time. The pooled block "
                        "is meaningless across methods -- qmdp-base is a parity "
                        "control whose whole point is not to deviate, so leaving "
                        "it in the same pool drags every rate down.")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    if a.load:
        rows = []
        for path in a.load:
            with open(path) as f:
                rows += [json.loads(l) for l in f if l.strip()]
        if a.only:
            keep = set(a.only.split(","))
            rows = [r for r in rows if r["method"] in keep]
        # Refuse to pool across kitchens. Rows written before layout_sha existed
        # report "unknown" and are counted as one group, which is the honest
        # answer rather than assuming they match.
        shas = collections.defaultdict(set)
        for r in rows:
            shas[r["layout"]].add(r.get("layout_sha", "unknown"))
        mixed = {k: v for k, v in shas.items() if len(v) > 1}
        if mixed:
            sys.stderr.write(
                "REFUSING TO POOL: these layouts appear with more than one "
                "layout_sha, so the rows describe different kitchens:\n")
            for k, v in sorted(mixed.items()):
                sys.stderr.write("  %-13s %s\n" % (k, ", ".join(sorted(v))))
            raise SystemExit(2)
        summarise(rows)
        return

    if a.all:
        layouts, fovs, seeds = LAYOUTS, FOVS, list(range(10))
    else:
        layouts = _expand(a.layouts, LAYOUTS)
        fovs = _expand(a.fovs, FOVS)
        seeds = _expand(a.seeds, [0])
    methods = [m for m in a.methods.split(",") if m]

    rows, trace = [], ([] if a.trace_out else None)
    sink = open(a.out, "a") if a.out else None
    try:
        for layout in layouts:
            for fov in fovs:
                for seed in seeds:
                    for m in methods:
                        r = run_episode(layout, fov, seed, m, a.horizon,
                                        a.forget, a.top_k, a.depth, trace=trace)
                        rows.append(r)
                        if sink:
                            sink.write(json.dumps(r) + "\n")
                            sink.flush()
                        if not a.quiet:
                            sys.stderr.write(
                                "%-13s fov%-4d s%-2d %-12s -> %d dishes/%dt  "
                                "off-job %d/%d (%.1f%%, strict %.1f%%)  "
                                "dev %.1f%%\n"
                                % (layout, fov, seed, m, r["delivered"],
                                   r["ticks"], r["off_job"], r["scored_ticks"],
                                   100 * r["off_job_frac"],
                                   100 * r["off_job_q_frac"],
                                   100 * r["deviated_frac"]))
    finally:
        if sink:
            sink.close()
    if trace is not None:
        with open(a.trace_out, "w") as f:
            for row in trace:
                f.write(json.dumps(row) + "\n")
    print()
    summarise(rows)


if __name__ == "__main__":
    main()
