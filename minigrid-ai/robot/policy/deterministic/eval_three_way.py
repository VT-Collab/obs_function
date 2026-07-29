"""
eval_three_way.py — Multi-condition assistance comparison.

Conditions compared
--------------------
  no_assist      : no robot intervention
  static_60      : assumes human FOV=60  (under-estimates — over-assists wide-FOV)
  static_120     : assumes human FOV=120 (middle ground)
  static_180     : assumes human FOV=180 (over-estimates — under-assists narrow-FOV)
  dynamic        : Bayesian MAP estimate of true FOV; LocationAssist
  gateway_120    : static assumed FOV=120; GatewayAssist (reveals entry door, not key)
  gateway_dyn    : dynamic Bayesian FOV;  GatewayAssist

Communication modes
--------------------
  LocationAssist  (static/dynamic) : reveals exact (x,y) of key → door → goal
  GatewayAssist   (gateway_*)      : reveals ENTRY DOOR to room containing key
                                     (coarser guidance; human still explores inside)

Metric
------
  adjusted_steps = steps + (n_assists * 1)
    Each robot communication costs 1 "time unit" (signalling overhead).
  success         True if agent reached goal before step limit
  n_assists       robot interventions per episode

Per-episode timing:
    robot.step(state, kb)           ← may inject info; costs 0 env steps but
                                      1 adjusted-step per intervention
    subtask = human.select_subtask(state)
    action  = planner.next_action(...)
    if dynamic: robot.observe(state, action)
    env.step(action)

Usage
-----
  # full evaluation (50 seeds × 3 fovs × 7 conditions)
  python robot/policy/determinstic/eval_three_way.py
  python robot/policy/determinstic/eval_three_way.py --seeds 50 --patience 5
  python robot/policy/determinstic/eval_three_way.py --seeds 10 --verbose

  # watch rendered episodes (6 assist conditions, seed=0, fov=120 by default)
  python robot/policy/determinstic/eval_three_way.py --render
  python robot/policy/determinstic/eval_three_way.py --render --render-seed 5 --render-fov 60
  python robot/policy/determinstic/eval_three_way.py --render --delay 0.05   # faster

"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[3]))

import gymnasium as gym

from human.agents.bayes_agent import BayesHumanAgent
from human.planning.bayes_planner import BayesianPlanner
from robot.policy.deterministic.no_assist       import NoAssist
from robot.policy.deterministic.static_assist   import StaticAssist
from robot.policy.deterministic.dynamic_assist  import DynamicAssist
from robot.policy.deterministic.gateway_assist  import GatewayAssist
from robot.estimation.bayesian_posterior.bayes_fov import BayesFOVInference, CANDIDATE_FOVS
try:
    # end_to_end/ was removed in the module rebuild; keep the CLI's neural_comm
    # condition optional so run_episode / evaluate stay importable without it.
    from robot.policy.neural.end_to_end.neural_assist import NeuralAssist  # MISHA NEW CHANGE
except ModuleNotFoundError:
    NeuralAssist = None

NEURAL_CHECKPOINT = "robot/policy/neural/baseline/checkpoints/model.pt"  # MISHA NEW CHANGE

ENV_ID = "MiniGrid-LockedRoom-v0"
RANDOM_WALLS = False  # MISHA NEW CHANGE — toggle random 2-cell wall segments in rooms
MAX_STEPS = None  # MISHA NEW CHANGE — None = env default (10 * size = 190)


# ── Dynamic gateway: Bayesian FOV + gateway communication ────────────────────

class DynamicGatewayAssist(GatewayAssist):
    """GatewayAssist with Bayesian FOV inference instead of a fixed assumption."""

    def __init__(self, patience: int = 5, candidate_fovs=None):
        super().__init__(assumed_fov=120, patience=patience)
        self.inf = BayesFOVInference(candidate_fovs=candidate_fovs or CANDIDATE_FOVS)

    def reset(self, state) -> None:
        super().reset(state)
        self.inf.reset(state)

    def observe(self, state, action: int) -> None:
        self.inf.update(state, action)

    def step(self, state, human_kb: dict):
        """
        MISHA NEW CHANGE (fix) — was judging _next_needed from human_kb
        (cheating: reading the real human's KB instead of the shadow's
        belief) and calling a _shadow_knows() method that doesn't exist
        anywhere in the codebase, so this condition has never actually run.
        Now mirrors DynamicAssist.step(): judge purely from skb.
        """
        map_fov = self.inf.map_fov()
        shadow  = self.inf.hypothesis_agents[map_fov]
        shadow.select_subtask(state)
        skb = shadow.knowledge_base

        target = self._next_needed(state, skb)
        if target is None:
            self.timer = 0
            return None
        self.timer += 1
        if self.timer >= self._effective_patience():  # MISHA NEW CHANGE
            self._reveal(human_kb, target, state)
            self.n_assists += 1
            self.timer = 0
            return target
        return None

    def _reveal(self, human_kb: dict, target, state) -> None:
        """
        MISHA NEW CHANGE — mirror into all 3 hypothesis agents' KBs instead
        of self.shadow, which this class (like DynamicAssist) never keeps in
        sync. Reuses GatewayAssist's entry_door write for Phase 1 reveals,
        and the shared _write_reveal helper for everything else.
        """
        kind, color, loc = target
        kbs = [human_kb] + [a.knowledge_base for a in self.inf.hypothesis_agents.values()]

        if kind == "entry_door":
            obj = state.grid.get(*loc)
            if obj is None:
                return
            for kb in kbs:
                kb.setdefault("grid_cache", {})[loc] = obj
                kb.setdefault("explored_cells", set()).add(loc)
                kb.setdefault("seen_doors", {})[color] = loc
            return

        for kb in kbs:
            self._write_reveal(kb, target, state)

    def _get_assumed_fov(self) -> int:
        return self.inf.map_fov()


# ── Single-episode runner ─────────────────────────────────────────────────────

def run_episode(seed: int, true_fov: int, robot, verbose: bool = False,
                 reveal_log: List[dict] = None) -> dict:
    """
    Run one episode.  Returns dict with seed, true_fov, steps, success,
    n_assists, and adjusted_steps (= steps + n_assists).

    MISHA NEW CHANGE — if reveal_log is given, every (kind, color, loc)
    reveal gets appended to it as a flat dict, so the raw coordinate
    injections can be inspected/audited after the run instead of just
    trusting the episode-level aggregates.
    """
    env = gym.make(ENV_ID, random_walls=RANDOM_WALLS, max_steps=MAX_STEPS)  # MISHA NEW CHANGE
    env.reset(seed=seed)
    state = env.unwrapped

    human   = BayesHumanAgent(fov=true_fov)
    planner = BayesianPlanner()
    human.init_knowledge_base(state)
    robot.reset(state)

    done = False
    t    = 0
    terminated = False

    while not done:
        state = env.unwrapped

        reveal = robot.step(state, human.knowledge_base)
        if verbose and reveal is not None:
            print(f"  t={t:3d} REVEAL {reveal}")
        if reveal_log is not None and reveal is not None:  # MISHA NEW CHANGE
            kind, color, loc = reveal
            x, y = loc
            reveal_log.append(dict(seed=seed, true_fov=true_fov, step=t,
                                    kind=kind, color=color, x=x, y=y))

        subtask = human.select_subtask(state)
        action  = planner.next_action(subtask, state, human.knowledge_base)
        if action is None:
            action = 2  # FWD fallback

        if hasattr(robot, "observe"):
            robot.observe(state, action)

        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        t += 1

    env.close()
    adj = t + (robot.n_assists * 1)  # each intervention costs 1 adjusted step
    return {
        "seed":            seed,
        "true_fov":        true_fov,
        "steps":           t,
        "adjusted_steps":  adj,
        "success":         terminated,
        "n_assists":       robot.n_assists,
    }


# ── Rendered episode ──────────────────────────────────────────────────────────

def run_episode_render(seed: int, true_fov: int, robot, label: str,
                       step_delay: float = 0.15) -> dict:
    """
    Run one episode with a pygame window and a live text overlay showing
    the condition, current subtask, and every robot injection.
    """
    import pygame

    env = gym.make(ENV_ID, random_walls=RANDOM_WALLS, max_steps=MAX_STEPS, render_mode="human")  # MISHA NEW CHANGE
    env.reset(seed=seed)
    state = env.unwrapped

    human   = BayesHumanAgent(fov=true_fov)
    planner = BayesianPlanner()
    human.init_knowledge_base(state)
    robot.reset(state)

    pygame.font.init()
    font = pygame.font.SysFont("monospace", 15, bold=True)

    done       = False
    t          = 0
    terminated = False
    last_reveal      = None   # most recent reveal tuple (persists between steps)
    last_reveal_step = -1     # step it happened on

    while not done:
        state = env.unwrapped

        reveal = robot.step(state, human.knowledge_base)
        if reveal is not None:
            last_reveal      = reveal
            last_reveal_step = t

        subtask = human.select_subtask(state)
        action  = planner.next_action(subtask, state, human.knowledge_base)
        if action is None:
            action = 2

        if hasattr(robot, "observe"):
            robot.observe(state, action)

        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        t   += 1

        # ── overlay ───────────────────────────────────────────────────────────
        surf = pygame.display.get_surface()
        if surf:
            if last_reveal is None:
                reveal_str   = "—"
                reveal_color = (140, 140, 140)
            else:
                kind, color, loc = last_reveal
                age = t - last_reveal_step
                reveal_str   = f"INJECT  {kind}  {color or ''}  @ {loc}  (t-{age})"
                reveal_color = (255, 255, 80) if age == 0 else (200, 200, 120)

            #overlay message board 
            rows = [
                (f" {label} ", (160, 200, 255)),
                (f" seed={seed}  fov={true_fov}°  step={t} ", (210, 210, 210)),
                (f" subtask : {subtask} ", (140, 255, 160)),
                (f" robot   : {reveal_str} ", reveal_color),
                (f" assists : {robot.n_assists} ", (210, 160, 255)),
            ]

            surfs   = [font.render(text, True, color) for text, color in rows]
            box_w   = max(s.get_width() for s in surfs) + 12
            line_h  = surfs[0].get_height() + 3
            box_h   = line_h * len(rows) + 8

            bg = pygame.Surface((box_w, box_h))
            bg.set_alpha(210)
            bg.fill((10, 10, 10))
            surf.blit(bg, (4, 4))

            y = 8
            for s in surfs:
                surf.blit(s, (10, y))
                y += line_h

            pygame.display.flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    done = True

        time.sleep(step_delay)

    env.close()
    adj = t + robot.n_assists
    return dict(seed=seed, true_fov=true_fov, steps=t,
                adjusted_steps=adj, success=terminated, n_assists=robot.n_assists)


# ── Render demo (assist conditions only, one seed + fov) ─────────────────────

def render_demo(conditions, seed: int = 0, fov: int = 120,
                step_delay: float = 0.15) -> None:
    """
    Show one rendered episode per assist condition.
    Skips no_assist — nothing ever fires there, nothing to watch.
    """
    for label, robot in conditions:
        if label == "no_assist":
            continue
        print(f"\n▶  {label:<14}  seed={seed}  fov={fov}")
        r  = run_episode_render(seed, fov, robot, label, step_delay=step_delay)
        ok = "✓" if r["success"] else "✗"
        print(f"   {ok}  steps={r['steps']}(+{r['n_assists']})  adj={r['adjusted_steps']}")


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate(robot, seeds: List[int], fovs: List[int],
             verbose: bool = False, reveal_log: List[dict] = None) -> List[dict]:
    results = []
    for fov in fovs:
        for seed in seeds:
            r = run_episode(seed, fov, robot, verbose=verbose, reveal_log=reveal_log)  # MISHA NEW CHANGE
            results.append(r)
            if verbose:
                ok = "✓" if r["success"] else "✗"
                print(f"  seed={seed:3d} fov={fov:3d} {ok} "
                      f"steps={r['steps']:4d}(+{r['n_assists']}) "
                      f"adj={r['adjusted_steps']:4d}")
    return results


def summarise(results: List[dict], label: str) -> dict:
    n = len(results)
    successes = [r for r in results if r["success"]]
    sr   = len(successes) / n if n else 0.0
    avg_steps = sum(r["steps"]          for r in results) / n if n else 0.0
    avg_adj   = sum(r["adjusted_steps"] for r in results) / n if n else 0.0
    avg_ass   = sum(r["n_assists"]      for r in results) / n if n else 0.0
    avg_adj_s = (sum(r["adjusted_steps"] for r in successes) / len(successes)
                 if successes else float("nan"))
    return dict(condition=label, n=n, success_rate=sr,
                avg_steps=avg_steps, avg_adj=avg_adj,
                avg_adj_succ=avg_adj_s, avg_assists=avg_ass)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    type=int, default=50)
    parser.add_argument("--fovs",     type=int, nargs="+", default=[60, 120, 180])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out",      type=str, default=None)
    parser.add_argument("--summary-out", type=str, default=None, dest="summary_out",
                        help="write the per-condition summary table (success rate, avg steps) to this CSV")  # MISHA NEW CHANGE
    parser.add_argument("--random-walls", action="store_true", dest="random_walls",
                        help="enable random 2-cell wall segments in rooms + hallway")  # MISHA NEW CHANGE
    parser.add_argument("--max-steps", type=int, default=None, dest="max_steps",
                        help="override env max_steps (default: env's own 10*size=190)")  # MISHA NEW CHANGE
    parser.add_argument("--reveal-log-out", type=str, default=None, dest="reveal_log_out",
                        help="write every individual (kind, color, loc) reveal event to this CSV")  # MISHA NEW CHANGE
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--render",      action="store_true",
                        help="watch one episode per assist condition (skips no_assist)")
    parser.add_argument("--render-seed", type=int, default=0, dest="render_seed",
                        help="seed to use when rendering (default 0)")
    parser.add_argument("--render-fov",  type=int, default=120, dest="render_fov",
                        help="human FOV to use when rendering (default 120)")
    parser.add_argument("--delay",       type=float, default=0.15,
                        help="seconds per step when rendering (default 0.15)")
    args = parser.parse_args()

    global RANDOM_WALLS, MAX_STEPS  # MISHA NEW CHANGE
    RANDOM_WALLS = args.random_walls
    MAX_STEPS = args.max_steps

    seeds = list(range(args.seeds))
    fovs  = args.fovs
    K     = args.patience

    conditions = [
        ("no_assist",    NoAssist()),
        
        # Wrong-FOV-assumption baselines — shows how model mismatch hurts/helps
        ("static_60",    StaticAssist(assumed_fov=60,  patience=K)),
        ("static_120",   StaticAssist(assumed_fov=120, patience=K)),
        ("static_180",   StaticAssist(assumed_fov=180, patience=K)),
        
        # Bayesian adaptive — exact location communication
        ("dynamic",      DynamicAssist(patience=K)),
        
        # Gateway communication — region guidance (entry-door reveal for Phase 1)
        ("gateway_120",  GatewayAssist(assumed_fov=120, patience=K)),
        ("gateway_dyn",  DynamicGatewayAssist(patience=K)),
    ]

    # MISHA NEW CHANGE — learned reveal-decision policy, only added once trained
    # (train.py's checkpoint) so this script still runs for everyone else.
    if NeuralAssist is not None and os.path.exists(NEURAL_CHECKPOINT):
        conditions.append(("neural_comm", NeuralAssist(checkpoint=NEURAL_CHECKPOINT)))

    if args.render:
        render_demo(conditions, seed=args.render_seed, fov=args.render_fov,
                    step_delay=args.delay)
        return

    all_results = []
    summaries   = []
    reveal_log  = [] if args.reveal_log_out else None  # MISHA NEW CHANGE

    for label, robot in conditions:
        log_start = len(reveal_log) if reveal_log is not None else 0  # MISHA NEW CHANGE
        results = evaluate(robot, seeds, fovs, verbose=args.verbose, reveal_log=reveal_log)
        for r in results:
            r["condition"] = label
        all_results.extend(results)
        s = summarise(results, label)
        summaries.append(s)
        if reveal_log is not None:  # MISHA NEW CHANGE — backfill condition on this batch's entries
            for rv in reveal_log[log_start:]:
                rv["condition"] = label

    # ── Per-FOV breakdown ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Per-FOV breakdown  (adj_steps = steps + n_assists)")
    print(f"{'='*70}")
    print(f"{'Condition':<14} {'FOV':>5} {'Succ%':>7} {'Adj Steps':>11} {'Raw Steps':>11} {'Assists':>8}")
    print("-" * 60)
    for label, _ in conditions:
        for fov in fovs:
            sub = [r for r in all_results
                   if r["condition"] == label and r["true_fov"] == fov]
            if not sub:
                continue
            sr   = sum(r["success"]         for r in sub) / len(sub)
            avga = sum(r["adjusted_steps"]  for r in sub) / len(sub)
            avgs = sum(r["steps"]           for r in sub) / len(sub)
            avam = sum(r["n_assists"]       for r in sub) / len(sub)
            print(f"{label:<14} {fov:>5} {sr:>6.1%} {avga:>11.1f} {avgs:>11.1f} {avam:>8.2f}")
        print()

    # ── Summary comparison ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Summary (all FOVs combined)")
    print(f"{'='*70}")
    print(f"{'Condition':<14} {'Succ%':>7} {'AdjSteps':>10} {'RawSteps':>10} {'Assists':>8}")
    print("-" * 55)
    for s in summaries:
        print(f"{s['condition']:<14} {s['success_rate']:>6.1%} "
              f"{s['avg_adj']:>10.1f} {s['avg_steps']:>10.1f} {s['avg_assists']:>8.2f}")

    print(f"\nNote: adjusted_steps = raw_steps + n_assists (each robot signal costs 1 time unit)")

    # ── Optional CSV ──────────────────────────────────────────────────────────
    if args.out:
        fields = ["condition", "seed", "true_fov",
                  "steps", "adjusted_steps", "success", "n_assists"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows({k: r[k] for k in fields} for r in all_results)
        print(f"\nWrote {len(all_results)} rows → {args.out}")

    # ── Optional summary CSV (avg steps, success rate per condition) ─────────
    # MISHA NEW CHANGE
    if args.summary_out:
        fields = ["condition", "n", "success_rate",
                  "avg_steps", "avg_adj", "avg_adj_succ", "avg_assists"]
        with open(args.summary_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows({k: s[k] for k in fields} for s in summaries)
        print(f"Wrote {len(summaries)} summary rows → {args.summary_out}")

    # ── Optional reveal-event log (every individual coordinate injection) ────
    # MISHA NEW CHANGE
    if args.reveal_log_out:
        fields = ["condition", "seed", "true_fov", "step", "kind", "color", "x", "y"]
        with open(args.reveal_log_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows({k: rv[k] for k in fields} for rv in reveal_log)
        print(f"Wrote {len(reveal_log)} reveal-event rows → {args.reveal_log_out}")


if __name__ == "__main__":
    main()
