"""
MISHA NEW CHANGE - final verification of two claims that the aggregate accuracy
numbers do NOT by themselves establish:

  1. The divergence is genuinely DIFFERENT SUBTASK BEHAVIOUR, not the same plan
     executed at different speeds. Printed side by side so it can be eyeballed.
  2. The inference is really reading that behaviour, and is not trivially or
     tautologically correct.

For (2) three things are checked that a high accuracy alone would hide:
  a. Per-hypothesis action-agreement with the real human. The TRUE hypothesis
     should agree far more than the others. If every hypothesis agreed equally,
     accuracy would be luck.
  b. A MISSPECIFIED control: run the true human at an FOV that is NOT in the
     candidate set. A filter that is actually reading evidence should be
     uncertain or wrong here. If it still reports high confidence, it is not
     reading the human at all.
  c. Shadow-vs-human divergence rate, to confirm evidence actually arrives.
"""
import random, sys
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from overcooked_ai_py.agents.agent import GreedySteakHumanModel
from fov.human.agent.limited_vision_human import LimitedVisionSteakHuman
from fov.human.planning.steak_planner import SteakMotionPlanner
from fov.robot.inference.bayes_fov import BayesFOVInference

N = 260
def runs(q):
    o = []
    for x in q:
        if not o or o[-1] != x: o.append(x)
    return o

def go(layout, triple, true_fov, seed, verbose=False):
    mdp = SteakHouseGridworld.from_layout_name(layout, start_order_list=['steak']*4)
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)
    np.random.seed(seed); random.seed(seed)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=N+10)
    human = LimitedVisionSteakHuman(mdp, true_fov, SteakMotionPlanner(mdp, mlp), agent_index=1)
    rb = GreedySteakHumanModel(mlp); rb.set_agent_index(0)
    inf = BayesFOVInference(mdp, mlp, triple, human_agent_index=1)
    agree = {f: 0 for f in triple}
    ests = []
    for _ in range(N):
        s = env.state
        try: a_r, _ = rb.action(s)
        except Exception: a_r = Action.STAY
        a_h, _ = human.action(s)
        # MISHA NEW CHANGE - read agreement from the filter's own record.
        # The previous version called sh.action(s) here AND inf.update() called
        # it again, stepping every shadow TWICE per timestep. That desynchronised
        # each shadow's beliefs and exploration from the real trajectory, so the
        # measured agreement was of a corrupted agent - which is why it appeared
        # to favour the wrong hypothesis.
        inf.update(s, a_h)
        ests.append(inf.map_fov())
        _, _, d, _ = env.step((a_r, a_h))
        if d: break
    n = len(ests)
    return dict(n=n, ests=ests, subtasks=human.subtask_log,
                agree={f: inf.n_agree[f]/max(1,n) for f in triple},
                post=inf.posterior(), div=inf.n_divergent_steps/max(1,inf.n_steps),
                delivered=human.n_delivered)

LAY, TRI = "steak_side_2", (30, 90, 180)
print(f"===== CLAIM 1: is it really DIFFERENT SUBTASK BEHAVIOUR? ({LAY}, seed 0) =====\n")
seqs = {}
for f in TRI:
    r = go(LAY, TRI, f, 0)
    seqs[f] = runs(r["subtasks"])
    print(f"fov={f:>3}  delivered={r['delivered']}  steps={r['n']}")
    print(f"   {seqs[f]}\n")
allsets = {f: set(s) for f, s in seqs.items()}
for i in range(len(TRI)):
    for j in range(i+1, len(TRI)):
        a, b = TRI[i], TRI[j]
        only_a = allsets[a] - allsets[b]; only_b = allsets[b] - allsets[a]
        print(f"fov{a} vs fov{b}: subtasks ONLY in {a}: {sorted(only_a) or '-'} | ONLY in {b}: {sorted(only_b) or '-'}")

print(f"\n===== CLAIM 2a: does the TRUE hypothesis agree more? =====\n")
for f in TRI:
    r = go(LAY, TRI, f, 0)
    line = "  ".join(f"fov{h}={r['agree'][h]:.2f}" for h in TRI)
    best = max(r['agree'], key=r['agree'].get)
    print(f"true={f:>3} | agreement: {line} | argmax={best} {'OK' if best==f else 'MISMATCH'}"
          f" | shadow-div={r['div']:.2f}")

print(f"\n===== CLAIM 2b: MISSPECIFIED control (true FOV not among candidates) =====\n")
for hidden in (45, 150):
    r = go(LAY, TRI, hidden, 0)
    p = r["post"]; mx = max(p, key=p.get)
    print(f"true fov={hidden} (NOT in {TRI}): posterior={ {k: round(v,3) for k,v in p.items()} } "
          f"-> picks {mx}, P={p[mx]:.3f}")
print("\nA filter reading real evidence should NOT be confidently right here -")
print("there is no correct answer available. High confidence would mean it is")
print("keying on something other than the human's behaviour.")
