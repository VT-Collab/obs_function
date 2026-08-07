"""
BASELINE ARM + the shared plumbing the whole package runs on.

Run the frozen self-play specialist for one layout against a LimitedVisionSteakHuman
at a given FOV, for N episodes. No filter, no module, no cost function. Print the
performance and write it to a file.

This is also the file play_episode.py imports: the sys.path shim, the checkpoint
lookup, the kitchen, the human factory and the actor wrapper all live here so the
module arm and the baseline arm cannot drift apart.

    python baseline.py --layout steak_gc00 --fov 90 --episodes 10

===========================================================================
THINGS THAT SILENTLY BREAK IF DONE WRONG  (all documented in SP/CARC_RUNS.md)
===========================================================================
1. start_order_list must be a real LIST (["steak"] * n_orders). The .layout files
   declare it as a string, len() then counts characters and NO delivery ever fires.
2. obs_shape comes from the layout, never hardcoded. CNNBase ends in
   Linear(32*W*H, 64), so a checkpoint only loads into its own grid.
3. rew_shaping_params=BASE_REW_SHAPING_PARAMS. Nothing here reads reward, but it
   is part of the state transition, so training and eval must match.
4. The layouts live in fov/layouts_final/layouts/ and from_layout_name only reads
   overcooked_ai_py/data/layouts/ -> stage them first.
5. Checkpoints: <root>/<layout>_seed1/sp_<layout>.pt. NEVER glob - every layout
   dir also holds smoke/sp_<layout>.pt, same name, same size, junk weights.
"""
import argparse
import json
import os
import random
import sys

import glob #file finder
import shutil #file copier
import types #SimpleNamespace for R_MAPPOPolicy args

import numpy as np
import torch


# =========================================================================
# 0. PATHS.  Import shim first: official_baselines is written to be run with
#    ITSELF on sys.path (utils.features / algorithm.rMAPPOPolicy are top level).
# =========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
#  .../fov/robot/policy/new/filter  ->  5 up is steakhouse/
STEAKHOUSE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
BASELINES = os.path.join(STEAKHOUSE, "fov", "robot", "policy", "new",
                         "official_baselines")
for _p in (STEAKHOUSE, BASELINES, os.path.dirname(STEAKHOUSE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from overcooked_ai_py.mdp.overcooked_mdp import (                   # noqa: E402
    SteakHouseGridworld, Action, BASE_REW_SHAPING_PARAMS)
from overcooked_ai_py.mdp.actions import Direction                  # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv       # noqa: E402

from utils.features import build_full_state, N_LAYERS               # noqa: E402
from algorithm.rMAPPOPolicy import R_MAPPOPolicy                    # noqa: E402
from SP.self_play import action_probs as sp_action_probs            # noqa: E402

from fov.human.agent.limited_vision_human import LimitedVisionSteakHuman  # noqa: E402
from fov.human.planning.steak_planner import SteakMotionPlanner     # noqa: E402


ROBOT_INDEX = 0
HUMAN_INDEX = 1
N_ACTIONS = 6
HORIZON = 400
N_ORDERS = 4
#DNF = "did not finish": the clock ran out before all the orders went out.
#
#The metric is completion_time = the tick the LAST order was served. A team that
#never finished has no such tick, so it needs a stand-in number -- you cannot
#leave a hole in a column you are about to average.
#    completion = delivery_ticks[-1] if finished else horizon + DNF_PENALTY
#i.e. 400 + 100 = 500. Real completion times can never exceed the horizon, so
#500 is guaranteed to rank below every genuine finish.
#
#Why not the obvious alternatives:
#  drop the episode  -- deletes exactly the failures, and "finishes more often"
#                       may be how the module wins. You would hide your result.
#  infinity          -- one DNF and the mean is infinity.
#  just the horizon  -- a team that squeaked in at 399 and a team that served
#                       nothing would score 399 vs 400. Not remotely the same.
#
#Both arms use this same constant, so a DNF is charged identically on each side
#and the paired difference stays clean. CAVEAT: 100 is arbitrary, and its size
#controls how hard DNFs drag the mean -- which is why deliveries and
#finished_rate are always reported next to completion_time, never instead of it.
#(Same convention as old/module/QMDP/RESULTS.md, so the numbers are comparable.)
DNF_PENALTY = 100

#where to look for sp_<layout>.pt, in order. CARC first, then anything local.
CKPT_DIRS = [
    os.environ.get("STEAK_SP_CKPT_ROOT",
                   "/scratch1/%s/steakhouse_sp/specialist"
                   % os.environ.get("USER", "mishafu")),
    os.path.join(HERE, "ckpt"),
    os.path.join(STEAKHOUSE, "fov", "robot", "policy", "old", "module",
                 "QMDP", "ckpt"),
]


def resolve_device(spec="auto"):
    """"auto" | "cpu" | "cuda" | "mps" -> a torch.device.

    "auto" prefers cuda, then mps (the Apple Silicon backend), then cpu.
    Anything else is passed straight to torch.device, so "cuda:1" works too.

    ===========================================================================
    READ THIS BEFORE REACHING FOR --device cuda
    ===========================================================================
    A GPU will very likely make THIS workload SLOWER, and it is worth knowing
    why rather than being surprised:

      * batch size is 1. One kitchen, one chef, one forward pass per tick. GPUs
        win by doing thousands of rows at once; at N=1 the launch overhead of
        each kernel (tens of microseconds) dwarfs the arithmetic.
      * the network is tiny. Three convs over a 5x5 .. 15x10 grid, then
        Linear(32*W*H, 64) and a GRU with hidden 64. That is microseconds of
        real work either way.
      * every tick pays TWO transfers across the bus: obs host->device in
        probs(), and the 6 probabilities device->host in .cpu().numpy(). Both
        synchronise, so you also lose any pipelining.
      * the actual bottleneck is not the network at all. It is the pure-python
        env loop -- mdp.get_state_transition, the human's BFS, and (in
        play_episode.py) six shadow humans doing cone tests. None of that is a
        tensor op, so none of it moves to the GPU.

    CARC_RUNS.md section 12 measured the same thing for TRAINING, where the
    batch is 50x bigger: "GPU is not worth it." On CARC compute nodes
    torch.cuda.is_available() is False anyway, so "auto" resolves to cpu there
    and this whole function is a no-op.

    It is here so the option exists and so nothing in the file hardcodes cpu.
    Benchmark before you believe it helps.
    """
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    #getattr(...) because torch.backends.mps does not exist on older versions,
    #and an AttributeError here would be a silly way to fail on a linux box.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stage_layouts():
    """cp -n fov/layouts_final/layouts/*.layout -> overcooked_ai_py/data/layouts/

    Copy any .layout that is missing. NEVER overwrite an existing one.
    Needed b/c overcooked_mdp.py:2131 calls read_layout_dict(layout_name), defined
    in overcooked_ai_py/__init__.py:21-22, which only ever looks inside
    LAYOUTS_DIR = overcooked_ai_py/data/layouts/. A layout that is not sitting in
    THAT folder does not exist as far as from_layout_name is concerned.

    Returns the list of filenames actually copied. Empty list = everything was
    already there, which is the normal case after the first run.

    Each comment below is ONE line of code for you to write, in order. The
    indentation of the comment is the indentation your line needs -- python
    uses indenting instead of { } to decide what sits inside what, so it is
    syntax, not style. 4 spaces per level, never tabs.
    """

    #make a name `src` holding the folder the REAL layouts live in:
    #    <STEAKHOUSE>/fov/layouts_final/layouts
    #`name = value` makes a name and points it at a value -- you never declare
    #a type, python works it out. Build the path with os.path.join(...), giving
    #each folder as its OWN argument: os.path.join(STEAKHOUSE, "fov", ...).
    #join glues them with the right separator for this OS. Never write
    #a + "/" + b by hand: it breaks on windows, and gives you "folder//file"
    #whenever `a` already ends in a slash.
    src = os.path.join(STEAKHOUSE, "fov", "layouts_final", "layouts")
    

    #make `dst` the same way, the destination:
    #    <STEAKHOUSE>/overcooked_ai_py/data/layouts
    #this is the ONLY folder from_layout_name will ever look in.
    dst = os.path.join(STEAKHOUSE, "overcooked_ai_py", "data", "layouts")
    

    #the guard: `if not os.path.isdir(src):`
    #os.path.isdir(p) asks the actual disk "is there a FOLDER at p?" and hands
    #back True or False. (os.path.exists = folder OR file; os.path.isfile =
    #file only.) `if not <test>:` runs the indented block when the test is
    #False. The ":" opens the block, exactly like it did for `def`.
    if not os.path.isdir(src):
        #raise FileNotFoundError, with a message that includes `src`.
        #this line is INDENTED one more level because it lives inside the if.
        #`raise` stops the program right here and shouts. Use it when carrying
        #on would be a lie: no layout library means the checkout is broken, and
        #quietly doing nothing would look exactly like success.
        #build the message as "no layout library at %s" % src -- the %s is a
        #blank that the value after the % drops into (this codebase's style;
        #f"...{src}" is the newer way and equally fine).
        raise FileNotFoundError("no layout library at %s" % src)

    #create `dst` if it is missing: os.makedirs(dst, exist_ok=True)
    #exist_ok=True means "already there is fine, do not complain" -- leave it
    #out and the SECOND run crashes. Passing it BY NAME like that is a keyword
    #argument, worth it here because os.makedirs(dst, True) reads like nothing.
    if not os.path.isdir(dst):
        os.makedirs(dst, exist_ok=True)
    
    #make an empty list called `copied`, written as []
    #a list is an ordered box you can add to. This is what you hand back at the
    #end, so the caller can log "staged 3 layouts" if it wants to.
    copied = []
    
    #the loop header: `for path in sorted(glob.glob(os.path.join(src, "*.layout"))):`
    #read it inside out. os.path.join(src, "*.layout") builds a PATTERN, where
    #"*" means "any run of characters" -- so it matches every file in src whose
    #name ends in .layout.
    # glob.glob(pattern) turns that into a list of full
    #paths. 
    # sorted(...) puts that list in alphabetical order, because glob's own
    #order depends on the filesystem and two machines can disagree; sorting
    #makes the run identical everywhere, for free. `for path in things:` then
    #repeats the indented block once per item, with `path` holding that item.
    for path in sorted(glob.glob(os.path.join(src, "*.layout"))):
        #`name = os.path.basename(path)` -- just the filename, folders dropped.
        #"/a/b/steak_gc00.layout" -> "steak_gc00.layout".
        #(os.path.dirname is the opposite: keep the folders, drop the file.)
        name = os.path.basename(path)
        #`target = os.path.join(dst, name)` -- where this file would land.
        target = os.path.join(dst, name)
        #the skip test: `if os.path.exists(target):`
        if os.path.exists(target):
            continue
            #`continue`
            #skips the rest of THIS turn of the loop and jumps to the next file.
            #(`break` would leave the loop entirely -- not what you want.)
            #THIS IS THE -n IN `cp -n`: never overwrite, so a file some other
            #run depends on can never be clobbered by this one.

        #`shutil.copy2(path, target)` -- do the copy.
        #copy2 brings the timestamps along; plain shutil.copy drops them. Keep
        #them: in this repo modification time is how you tell a real file from
        #a stale one (see the smoke-checkpoint trap in the header).
        shutil.copy2(path, target)
        
        #`copied.append(name)` -- stick the name on the end of the list.
        #still inside the loop, so it runs once per file you actually copied.
        copied.append(name)
        
    #back out at function level (one indent, not two): `return copied`
    #return hands the value back to whoever called this and finishes. A
    #function with no return hands back None. Prefer RETURNING data over
    #PRINTING inside a helper -- a helper that prints is useless inside a loop
    #over 11 layouts, a helper that returns is useful everywhere.
    return copied 
    #ONE JUDGMENT CALL, and find_checkpoint wants the same split: a missing
    #SOURCE folder raises, but a destination that is already full just returns
    #an empty list. Those look similar and are not. No layout library = broken
    #checkout, stop the run. Nothing to copy = the normal second-run case.


def find_checkpoint(layout, seed=1):
    """Absolute path of the specialist for `layout`, or raise.

    Only ONE checkpoint per layout exists (seed 1, the LAST save, not a
    best-of). "best" here means the real one rather than the decoy:

        <dir>/<layout>_seed<seed>/sp_<layout>.pt     <- CARC layout
        <dir>/sp_<layout>.pt                         <- flat local ckpt dir

    Returns the absolute path of the first candidate that exists. Raises if
    none do -- see the note at the bottom for why raising beats returning None.

    Same deal as stage_layouts: each comment below is one line of code, and the
    indentation of the comment is the indentation your line needs.
    """
    #NEW PYTHON HERE: function ARGUMENTS. `layout` is required -- the caller
    #must supply it. `seed=1` has a DEFAULT, so both find_checkpoint("steak_gc00")
    #and find_checkpoint("steak_gc00", 2) are legal calls. Arguments with
    #defaults must come after the ones without. Give a default when one value is
    #almost always right: CARC_RUNS.md says seed 1 is the ONLY seed that exists,
    #so seed is a knob nobody normally touches.
    

    #make an empty list `tried`, same [] as before.
    #it collects every path you looked at, in order, so the error at the bottom
    #can say WHERE it searched. "checkpoint not found" wastes ten minutes; the
    #same error listing the four folders it checked is fixed in ten seconds.
    tried = []
    
    #the outer loop: `for d in CKPT_DIRS:`
    #CKPT_DIRS is the constant defined above (ALL_CAPS = set once, never
    #reassigned). It is already written in priority order: CARC scratch first,
    #then the local ckpt/ folder, then the old QMDP ckpt/ folder. Looping over a
    #list beats writing three if-branches -- adding a fourth location later is
    #one line in the constant and zero lines in here.
    for d in CKPT_DIRS: 
        #build the CARC-shaped candidate:
        #    <d>/<layout>_seed<seed>/sp_<layout>.pt
        #that needs string formatting in two of the three pieces:
        #    "%s_seed%d" % (layout, seed)  ->  "steak_gc00_seed1"
        #    "sp_%s.pt" % layout           ->  "sp_steak_gc00.pt"
        #%s drops in a string, %d drops in a whole number. When there is MORE
        #than one blank to fill, the values go in parentheses after the %, in
        #order -- that is a TUPLE. One blank needs no parentheses.
        #then wrap the three pieces in os.path.join(d, ..., ...) as before.
        #call it something like `carc`.
        carc = os.path.join(d, "%s_seed%d" % (layout, seed), "sp_%s.pt" %layout)
        

        #build the flat-local candidate: `flat` = <d>/sp_<layout>.pt
        #the laptop ckpt folders have no per-seed subfolder, just the file.
        flat = os.path.join(d, "sp_%s.pt" % layout)

        #the inner loop: `for cand in (carc, flat):`
        #(carc, flat) is a TUPLE -- like a list, but written with () and not
        #changeable after you make it. Right choice for a fixed pair you walk
        #once. Order matters: CARC shape first, so a real cluster checkpoint
        #beats a laptop copy of the same layout.
        for cand in (carc, flat):
            #the decoy guard: `if "smoke" in cand:`
            #`in` on a string is a substring test -- "smoke" in "/a/smoke/b" is
            #True. This is the CARC_RUNS.md trap: EVERY layout folder also holds
            #smoke/sp_<layout>.pt -- same filename, same file size, 800-step junk
            #weights. Size will not save you, only the path will.
            #You build these paths yourself so they should never contain it; the
            #guard is for a STEAK_SP_CKPT_ROOT env var pointed somewhere bad.
            if "smoke" in cand:
                #`continue` -- skip this candidate, go to the next one.
                continue
            
            #record it: `tried.append(cand)`
            #do this BEFORE the existence test, so the error message lists every
            #place you looked, not just the ones that happened to be there.
            tried.append(cand)
            
            #make sure this is a file and not just a folder the hit test: `if os.path.isfile(cand):`
            #isfile, not exists -- you want "a FILE is sitting there". A folder
            #with that name would pass exists() and then fail to load.
            if os.path.isfile(cand):
                #`return cand`
                #return inside a loop leaves the WHOLE function immediately --
                #both loops, not just the inner one. So the first hit wins and
                #the priority order of CKPT_DIRS is what decides. No break needed.
                return cand
            
    raise FileNotFoundError(
        "no self-play checkpoint for %s (looked in %s)"
        % (layout, "\n  ".join(tried)))
    #fell out of both loops = nothing found. Raise FileNotFoundError with the
    #layout name AND every path in `tried`.
    #to put the list into the message, use "\n  ".join(tried).
    #  ** THIS IS A DIFFERENT join FROM os.path.join **, and the name collision
    #  is unfortunate. str.join glues the items of a list together, using the
    #  string you call it on as the glue: "-".join(["a", "b"]) -> "a-b".
    #  So "\n  ".join(tried) puts each path on its own line, indented two
    #  spaces. os.path.join builds ONE path out of folder names; str.join
    #  builds ONE string out of a list. Unrelated functions, same word.
    
    #WHY RAISE INSTEAD OF RETURNING None: the caller feeds this straight to
    #torch.load(). Hand back None and the crash happens one call later, with a
    #message about NoneType that names neither the layout nor the folders you
    #searched. Same judgment call as stage_layouts -- a broken setup should stop
    #the run loudly, at the line that knows what went wrong.

    #OPTIONAL, once the file is loading in Actor: after torch.load, check
    #ck["episode"]. The real checkpoints say 499 (475 for steak_mid_2); a smoke
    #decoy says 3. That is the only content-level way to tell them apart.
    


# =========================================================================
# 1. THE KITCHEN.  Chef 0 = the policy under test, chef 1 = the FOV human.
# =========================================================================
def disable_collisions(mdp):
    """Let the two chefs walk through each other. MUTATES `mdp` in place.

    ===========================================================================
    WHAT THE MDP DOES BY DEFAULT, AND WHERE
    ===========================================================================
    overcooked_mdp.py:1115

        def _handle_collisions(self, old_positions, new_positions):
            if self.is_transition_collision(old_positions, new_positions):
                return old_positions
            return new_positions

    A "transition collision" is either chefs landing on the SAME tile, or the
    two of them SWAPPING tiles (walking through each other). Note what happens
    then: BOTH revert to where they were. Not just the one who caused it -- one
    chef walking into a stationary partner costs the innocent partner a tick as
    well. So collisions are a real, two-sided cost in the dynamics.

    SteakHouseGridworld inherits straight from OvercookedGridworld and overrides
    NONE of resolve_movement / compute_new_positions_and_orientations /
    _handle_collisions, so this one function is the only place to patch.

    ===========================================================================
    THIS IS THE DEFAULT FOR THE WHOLE PACKAGE.  Kitchen.__init__ calls it every
    time, unconditionally -- there is no flag and nothing to remember to pass.
    Bumping into your partner is simply free here, in every arm and every run.
    ===========================================================================

    ===========================================================================
    WHY OFF
    ===========================================================================
    A partner-aware module can win in two completely different ways:

        (a) it knows what the partner can SEE, and stops blindsiding them
        (b) it physically stops standing in their way

    (a) is the thesis. (b) is motion planning, and it would happen with a
    partner model that knew nothing about vision at all. With collisions ON,
    every result mixes the two and you cannot say which one you measured. The
    old QMDP package hit exactly this: it DELETED its collision cost terms
    because they "dominated everything else" (cost.py header) -- but the
    ENVIRONMENT still had collisions, so channel (b) was still live.

    Turn collisions off and channel (b) is gone by construction. A win in this
    mode is informational, and that is a much stronger claim.

    ===========================================================================
    THE ONE THING TO SAY OUT LOUD IN THE WRITE-UP
    ===========================================================================
    The baselines were TRAINED with collisions ON. Evaluating with them off
    moves the transition dynamics slightly off the distribution the policy
    learned, so the trained arm scores a little differently than it would under
    training physics -- and you cannot predict the sign in advance.

    That is NOT a problem for any comparison in this package, because every arm
    runs under the same physics, so the differences stay clean. It IS something
    a reader must be told once, plainly: "all evaluation is run with inter-agent
    collisions disabled, so that measured coordination is informational rather
    than physical." State it, do not bury it.

    ===========================================================================
    THE COMMENTS, ONE LINE OF CODE EACH
    ===========================================================================
    """
    #the patch itself is one line:
    #    mdp._handle_collisions = lambda old, new: new
    #
    #NEW PYTHON: `lambda args: expression` is a function with no name, written
    #inline. `lambda old, new: new` is the same as
    #    def f(old, new): return new
    #Use one only when the body is a single short expression, as here.
    #
    #WHY ASSIGNING TO THE INSTANCE WORKS. mdp._handle_collisions(...) looks in
    #the INSTANCE first and only then in the class, so your lambda wins. And
    #because it is an instance attribute rather than a class attribute, python
    #does NOT bind `self` to it -- the call passes exactly (old, new), which is
    #why the lambda takes two arguments and not three. Patch the CLASS instead
    #and you would need `def f(self, old, new)`. This trips everyone once.
    #
    #Do it per-mdp (per Kitchen), NOT on the class, so a single process can run
    #the collisions-on and collisions-off conditions without them contaminating
    #each other.

    #WHAT ELSE NOTICES, so nothing surprises you later:
    #  * get_valid_joint_player_positions (mdp:1153,1176) still refuses
    #    overlapping positions, but it is only used to build START states, so it
    #    does not matter here.
    #  * the two chefs can now occupy one tile. build_full_state writes their
    #    positions to SEPARATE channels, so the observation is still well formed.
    #  * LimitedVisionSteakHuman.avoid_robot detours around the teammate's cell.
    #    It defaults to False and should stay False here -- avoiding a collision
    #    that can no longer happen is pure wasted walking.
    #  * resolve_interacts is untouched: two chefs interacting with the same
    #    station in one tick already resolves in player order, collisions or not.
    mdp._handle_collisions = lambda old, new: new

    #a flag you can read back later. The leading underscore says "ours, not the
    #library's", so nobody mistakes it for part of the mdp's own API.
    mdp._collisions_enabled = False

    #return the SAME object it was handed -- mutated, not replaced -- so callers
    #may write either `disable_collisions(mdp)` or `mdp = disable_collisions(mdp)`.
    return mdp


#NEW PYTHON: a CLASS. Everything so far has been a plain function. A class is a
#box that holds DATA plus the functions that work on that data. `class Kitchen:`
#opens it; every function indented inside is a METHOD of it.
#
#`self` is the first argument of every method and is the box itself. You never
#pass it -- python passes it for you. `k = Kitchen("steak_gc00")` calls __init__
#with self = the new box, and `k.reset()` calls reset with self = k.
#Anything you write as `self.NAME = value` sticks to the box and every other
#method can read it later. That is the whole point: `self.mdp` set in __init__
#is still there when step() runs 400 ticks later.
class Kitchen:
    """One episode's worth of environment. Deliberately thin: it owns the mdp
    and the clock only. The tick loop lives in the callers, because the robot
    has to interleave perceive / decide / observe-human / step in a specific
    order and an env that hides the human inside step() cannot express it."""

    def __init__(self, layout, n_orders=N_ORDERS, horizon=HORIZON):
        """Build the mdp for `layout`. Does NOT start an episode -- reset() does.

        __init__ is the special method python runs when you write Kitchen(...).
        It returns nothing; its whole job is to attach things to `self`.
        Note the defaults come from the module constants, so the numbers live in
        ONE place and a sweep can override them per call.
        """
        #stash the three arguments on self, so later methods can read them:
        #self.layout = layout, self.n_orders = n_orders, self.horizon = horizon.
        #`layout` alone is just a local name that dies when __init__ ends;
        #`self.layout` survives for the life of the object.
        self.layout = layout
        self.n_orders = n_orders
        self.horizon = horizon
        
        #build the mdp:
        #    SteakHouseGridworld.from_layout_name(layout,
        #        start_order_list=["steak"] * n_orders,
        #        rew_shaping_params=dict(BASE_REW_SHAPING_PARAMS))
        #store it as self.mdp.
        #  * ["steak"] * 4 makes ["steak","steak","steak","steak"] -- multiplying
        #    a list repeats it. This is TRAP 1 in the header: the .layout file
        #    declares start_order_list as a STRING, len() then counts characters,
        #    and no delivery ever fires. Passing a real list overrides it.
        #  * dict(X) makes a COPY of dict X. Pass the copy, not X itself, so the
        #    mdp cannot mutate the shared module-level constant under you.
        #  * keyword arguments (name=value) let you skip everything in between
        #    and make the call readable.
        self.mdp = SteakHouseGridworld.from_layout_name(
            layout, start_order_list=["steak"] * n_orders,
            rew_shaping_params=dict(BASE_REW_SHAPING_PARAMS))
        
        #assert the trap is actually dead:
        #    assert not isinstance(self.mdp.start_order_list, str)
        #`assert <test>` raises AssertionError when the test is False. Use it for
        #things that must be true if the code is correct -- a self-check, not
        #input validation. This one costs nothing and catches the single most
        #expensive silent bug in this repo.
        assert not isinstance(self.mdp.start_order_list, str)
        
        #work out the observation shape and store it as self.obs_shape:
        #    (N_LAYERS, self.mdp.shape[0], self.mdp.shape[1])
        #i.e. (23, W, H). This is TRAP 2: CNNBase ends in Linear(32*W*H, 64), so
        #a checkpoint only fits the grid it trained on. Ask the mdp, never hardcode.
        #The ( ) with commas makes a TUPLE; [0] and [1] index into mdp.shape.
        self.obs_shape = (N_LAYERS, self.mdp.shape[0], self.mdp.shape[1])
        #set self.env = None -- no episode yet. Declaring the attribute here,
        #even as None, tells a reader it exists and that reset() fills it in.

        # ---- COLLISIONS: OFF, ALWAYS ---------------------------------------
        #no parameter, no flag: bumping into your partner is free everywhere in
        #this package. Every Kitchen gets it, so nobody can forget to pass it
        #and no two runs can disagree.
        #
        #It lives HERE and nowhere else. play_episode.py builds its own Kitchen,
        #so putting it in main() would leave the module arm running under
        #different physics from this one -- the exact silent mismatch that makes
        #a paired comparison meaningless. One line, one place, both files.
        disable_collisions(self.mdp)

    def reset(self):
        """Start a fresh episode. Returns the starting state."""
        #build the env and store it on self:
        #    OvercookedEnv.from_mdp(self.mdp, info_level=0,
        #                           horizon=self.horizon + 10)
        #  * from_mdp is a "make me one of these" helper on the class, so you
        #    call it on the CLASS (OvercookedEnv.from_mdp), not on an instance.
        #  * info_level=0 keeps it quiet.
        #  * +10 ON PURPOSE: you want YOUR horizon check to end the episode, so
        #    the mdp's own terminal (the order list running out) stays a clean
        #    "they finished early" signal instead of colliding with the timeout.
        #    Same thing SteakSelfPlayEnv does -- match it or the numbers differ.

        self.env = OvercookedEnv.from_mdp(self.mdp, info_level=0,
                                          horizon=self.horizon + 10)

        #hand back the starting state: return self.state
        #(self.state is the property defined just below -- no parentheses.)
        return self.state

    @property
    def state(self):
        """The current state object."""
        #NEW PYTHON: @property. That line above the def is a DECORATOR -- it
        #changes how the function is used. With it, callers write `k.state`
        #with NO parentheses, as if it were a stored value, and python quietly
        #runs this method. Use it for cheap read-only lookups that feel like
        #attributes. `k.state()` would now be an ERROR (you would be calling the
        #state object), which is the usual way this bites.

        #one line: return self.env.state
        return self.env.state

    @property
    def t(self):
        """Ticks elapsed in this episode."""
        #one line: return self.env.t
        #the env owns the clock; do not keep a second counter here or the two
        #will disagree the first time something is off by one.
        return self.env.t
    
    def robot_obs(self):
        """(1, 23, W, H) float32 -- the robot's ego-centric view, unpadded,
        exactly what the specialist trained on."""
        #call the feature builder:
        #    build_full_state(self.mdp, self.state, agent_index=ROBOT_INDEX,
        #                     t=self.t, horizon=self.horizon)
        #ROBOT_INDEX is 0 -- the constant, not a bare 0, so the day the indices
        #swap you edit one line at the top instead of hunting for zeros.
        #features.py is EGO-CENTRIC: agent_index decides who lands in the "me"
        #channels and who lands in "them". Pass the wrong one and the network
        #plays as the human.
        obs = build_full_state(self.mdp, self.state, agent_index=ROBOT_INDEX,
                         t = self.t, horizon=self.horizon)
        #CATCH THE VALUE. A call is an EXPRESSION -- it produces an array. Call
        #it on a line by itself and python computes it and throws it away.
        #So either name it first:  obs = build_full_state(...)
        #then                      return obs.astype(np.float32)[None, ...]
        #or do the whole thing as one returned expression:
        #    return build_full_state(...).astype(np.float32)[None, ...]

        #the two fix-ups, in order:
        #    .astype(np.float32)      the net wants float32, not float64
        #    [None, ...]              adds a new axis at the FRONT
        #  (23, W, H) -> (1, 23, W, H). That leading 1 is the BATCH dimension:
        #  torch layers always expect "how many at once" first, and here it is
        #  one. `...` means "all the remaining axes, unchanged", and putting
        #  None in an index slot INSERTS an axis there.
        #
        #WATCH OUT: .astype() does NOT edit the array in place -- it returns a
        #NEW one. `obs.astype(np.float32)` on its own line changes nothing.
        #Most numpy and string methods work this way (return a copy);
        #list.append is the opposite (edits in place, returns None).
        return obs.astype(np.float32)[None, ...]
        
    def step(self, robot_action, human_action):
        """Apply the joint action. Returns (sparse_reward, done).

        sparse_reward is METRIC ONLY -- no decision anywhere in this package
        reads it.
        """
        #make the joint action: a TUPLE (robot_action, human_action).
        #ORDER IS THE CONTRACT: index 0 is the robot, index 1 is the human,
        #matching ROBOT_INDEX / HUMAN_INDEX. Swap them and everything still
        #runs, silently, with the two chefs' bodies exchanged.
        joint = (robot_action, human_action)
        
        #step the env and unpack what comes back:
        #    _, sparse, done, _info = self.env.step(joint)
        #env.step hands back FOUR things (next_state, sparse, done, info).
        #Writing four names on the left splits the tuple in one line -- that is
        #TUPLE UNPACKING, and the count must match exactly.
        #`_` is the convention for "I must name this but will not use it". It is
        #a normal variable; the underscore is a message to humans. `_info` is
        #the same idea while keeping the word, so a reader knows what was there.
        _, sparse, done, _info = self.env.step(joint)
        
        #decide whether the episode is over:
        #    hit_horizon = self.env.t >= self.horizon
        #a comparison evaluates to True/False, so this is a plain boolean.
        #then done = done or hit_horizon -- either the mdp finished the orders,
        #or we ran out of time.
        hit_horizon = self.env.t >= self.horizon
        done = done or hit_horizon

        #return TWO values: return float(sparse), bool(done)
        #python returns them as a tuple, so the caller writes
        #    sparse, done = kitchen.step(a, h)
        #float()/bool() force the types, so a numpy scalar cannot sneak into
        #your JSON output later and blow up json.dumps.
        return float(sparse), bool(done)

#the three reference arms. Order is worst-to-best on purpose: that is how the
#comparison table reads, and `none` is the floor everything else is measured
#against.
ARMS = ("none", "random", "policy")


def face_away_from_stations(kitchen):
    """Point the robot at a cell that is not a pot / board / sink. Returns the
    direction chosen.

    Only used by the `none` arm, and it is what makes that arm mean "no robot"
    rather than "a statue". limited_vision_human._robot_faced_kind (line 824)
    reports the station kind the SEEN teammate is FACING, and
    _available_advancing then BLOCKS every task in STATION_TASKS[kind] -- from
    pose alone, no held object needed. A robot frozen facing the pot would
    suppress the human's pickup_meat for all 400 ticks just by existing.

    Facing a wall, counter or floor makes that lookup return None, and the
    human's own code is left completely untouched -- we change OUR agent, not
    theirs, so the arms stay comparable.

    It sticks for the whole episode: overcooked_mdp.py:1134 is
        new_orientation = orientation if action == Action.STAY else action
    so STAY preserves facing, and INTERACT is not a motion action either.
    """
    rp = kitchen.state.players[ROBOT_INDEX]
    stations = (list(kitchen.mdp.get_pot_locations())
                + list(kitchen.mdp.get_chopping_board_locations())
                + list(kitchen.mdp.get_sink_locations()))
    for d in Direction.ALL_DIRECTIONS:
        faced = (rp.position[0] + d[0], rp.position[1] + d[1])
        if faced not in stations:
            rp.update_pos_and_or(rp.position, d)
            return d
    #boxed in by stations on all four sides. Do NOT continue quietly: this
    #layout's human-solo number would not be comparable to the others, and you
    #would never find out from the numbers alone.
    raise RuntimeError(
        "%s: robot start tile %s faces a station in every direction, so the "
        "'none' arm cannot be made neutral here"
        % (kitchen.layout, rp.position))


def make_human(mdp, fov, seed, temperature=0.5):
    """A fresh LimitedVisionSteakHuman with a reproducible sampler."""
    #seed the GLOBAL random stream FIRST: random.seed(seed)
    #this line must come BEFORE you build the human, and that ordering is the
    #whole point of the function. LimitedVisionSteakHuman.reset() seeds its own
    #private RNG by drawing from `random`, so seeding `random` here is what
    #makes an episode replayable -- and what makes the module-on and module-off
    #arms start from the SAME draw sequence, diverging only because the world
    #diverged. Get the order wrong and your "paired" comparison is not paired.
    random.seed(seed)
    
    #build the planner: SteakMotionPlanner(mdp, None)
    #None is CORRECT, not a placeholder. The second argument is a
    #MediumLevelPlanner, and the human deliberately does not use one -- it
    #routes with its own BFS over floor it has actually SEEN. Hand it a real
    #planner and it silently gains a map of the whole kitchen, which destroys
    #the entire experiment. The planner object is kept only to look up which
    #station kind a subtask targets.
    planner = SteakMotionPlanner(mdp, None)
    
    #build and return the human in one line:
    #    return LimitedVisionSteakHuman(mdp, fov, planner,
    #                                   agent_index=HUMAN_INDEX,
    #                                   temperature=temperature)
    #`fov` is the TRUE cone. It is set here, on the human, and passed to nothing
    #on the robot's side -- that is the no-cheating boundary in one line.
    return LimitedVisionSteakHuman(mdp, fov, planner, agent_index=HUMAN_INDEX,
                                      temperature=temperature)

# =========================================================================
# 2. THE FROZEN BASELINE.  The only place the trained network is touched.
# =========================================================================
class Actor:
    """obs -> 6 probabilities, carrying the GRU memory forward.

    sp_action_probs is the same forward pass policy.act() uses, stopped one step
    before the sample, so these ARE the numbers it would have sampled from. The
    recurrent state is the easy thing to get wrong: it must be threaded through
    the whole episode or this is not the policy that was trained.
    """

    def __init__(self, layout, obs_shape, ckpt_path=None, device=None):
        """Load the frozen checkpoint and rebuild the network around it."""
        
        #resolve the checkpoint: if ckpt_path is None, call find_checkpoint(layout).
        #    if ckpt_path is None:
        #        ckpt_path = find_checkpoint(layout)
        #`is None` not `== None` -- `is` asks "the same object?", which is the
        #right question for None and cannot be fooled by a weird __eq__.
        #An argument defaulting to None that gets filled in like this is the
        #standard way to say "optional, and here is what you get if you skip it".
        if ckpt_path is None:
            ckpt_path = find_checkpoint(layout)
            
        #resolve the device the same way. `device` may arrive as None (pick one
        #for me), as a string ("cuda", "mps", "cpu"), or as a torch.device.
        #torch.device(<a torch.device>) is a no-op, so one line covers all three.
        #NOTE `is None`, not `or`: torch.device("cpu") is a perfectly good value
        #and `device or ...` would be asking whether it is TRUTHY, which is not
        #the question. Use `or` for defaults only when no legitimate value is
        #falsy -- here it happens to be safe, but the habit is not.
        device = resolve_device() if device is None else torch.device(device)
        
        #load the file: torch.load(ckpt_path, map_location="cpu",
        #                          weights_only=False)
        #  * map_location="cpu" loads a GPU-trained file on a CPU box. Without
        #    it you get a CUDA error on a machine with no GPU.
        #  * weights_only=False because the file holds a dict with "args" in it,
        #    not just tensors. Newer torch defaults this to True and refuses.
        #the result is a plain dict: {"actor", "critic", "episode", "args"}.
        #STAY ON "cpu" HERE even when running on a GPU. R_MAPPOPolicy below
        #already builds its modules on `device`, and load_state_dict copies each
        #saved tensor INTO the existing parameter, converting device as it goes.
        #So the weights land on the GPU either way, and map_location="cpu" is the
        #one spelling that works on cuda, mps and cpu without a special case.
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        #rebuild the hyperparameters the run actually used:
        #    saved = ck.get("args", {}) or {}
        #    args = types.SimpleNamespace(
        #        hidden_size=int(saved.get("hidden_size", 64)),
        #        recurrent_N=int(saved.get("recurrent_N", 1)),
        #        lr=5e-4, critic_lr=5e-4, opti_eps=1e-5)
        #  * d.get(k, default) reads a dict key WITHOUT crashing when it is
        #    missing -- d[k] would raise KeyError. The default is the fallback.
        #  * SimpleNamespace turns keywords into an object with dots, so
        #    args.hidden_size works. R_MAPPOPolicy expects dots, not a dict.
        #  * lr / critic_lr / opti_eps are only used to build optimizers you will
        #    never step. They must exist or the constructor fails.
        #READ the numbers from the checkpoint rather than assuming: ck["args"]
        #holds every hyperparameter that run used, so nothing is guessed.
        saved = ck.get("args", {}) or {}
        args = types.SimpleNamespace(
            hidden_size=int(saved.get("hidden_size", 64)),
            recurrent_N=int(saved.get("recurrent_N", 1)),
            lr=5e-4, critic_lr=5e-4, opti_eps=1e-5
        )
        #build the network:
        #    policy = R_MAPPOPolicy(args, obs_shape, obs_shape, act_dim=6,
        #                           device=device)
        #obs_shape twice: the actor's obs and the critic's shared obs are the
        #same thing here. act_dim=6 is the six primitives.
        #pass device= or the net is built on CPU while probs() moves obs to
        #self.device -- fine today (both cpu), a crash the day one is not.
        policy = R_MAPPOPolicy(args, obs_shape, obs_shape, act_dim=6,
                               device=device)
        #load the weights into it:
        #    policy.actor.load_state_dict(ck["actor"])
        #    if "critic" in ck: policy.critic.load_state_dict(ck["critic"])
        #`"critic" in ck` is a dict key test (on a dict, `in` checks KEYS).
        #The critic is not used for acting -- guard it so a checkpoint saved
        #without one still loads.
        policy.actor.load_state_dict(ck["actor"])
        if "critic" in ck:
            policy.critic.load_state_dict(ck["critic"])
            
        #switch both to eval mode: policy.actor.eval(), policy.critic.eval()
        #eval() turns off dropout/batchnorm training behaviour. FROZEN means
        #frozen: nothing here ever calls .train(), .backward() or an optimizer.
        policy.actor.eval()
        policy.critic.eval()
        
        #stash what the other methods need on self: self.policy, self.device,
        #self.hidden = int(args.hidden_size), self.layers = int(args.recurrent_N)
        self.hidden = int(args.hidden_size)
        self.layers = int(args.recurrent_N)
        self.policy = policy
        self.device = device
        #finish by calling self.reset() -- one place builds the GRU state, and
        #__init__ just uses it. Calling your own method from __init__ is normal.
        self.reset()
        
        
    def reset(self):
        """New episode: zero the GRU and arm the first-tick mask."""
        #    self.rnn = torch.zeros(1, self.layers, self.hidden, device=self.device)
        #shape (1, recurrent_N, hidden_size): one row because you run one chef
        #at a time here, unlike training where both chefs were rows in a batch.
        self.rnn = torch.zeros(1, self.layers, self.hidden, device=self.device)
        
        #    self._first = True
        #a flag saying "the next probs() call is tick 0 of a new episode".
        #The leading underscore is a convention meaning PRIVATE -- python does
        #not enforce it, it tells a reader "internal, do not poke from outside".
        self._first = True

    def probs(self, obs_np):
        """(1, 23, W, H) -> np.ndarray (6,), sums to 1."""
        #convert numpy -> torch: torch.from_numpy(obs_np).to(self.device)
        #from_numpy SHARES memory with the numpy array rather than copying, so
        #do not mutate obs_np afterwards and expect the tensor to be unchanged.
        #(.to(device) DOES copy when the device differs, so on a GPU the sharing
        #stops there -- this is one of the two bus transfers per tick.)
        #.float() is cheap insurance: robot_obs already casts to float32, but MPS
        #refuses float64 outright and a stray float32 cast costs nothing.
        obs = torch.from_numpy(obs_np).float().to(self.device)
        
        #build the mask: zeros on the first tick, ones after.
        #    masks = torch.zeros(1, 1) if self._first else torch.ones(1, 1)
        #NEW PYTHON: the conditional expression `A if test else B` -- one line,
        #evaluates to A or B. Different from a normal `if` statement: this one
        #IS a value, so it can sit on the right of an `=`.
        #mask 0 WIPES the GRU memory. That is how self_play.py starts an
        #episode, so you must match it or tick 0 carries the last episode's
        #memory and the policy is not the one that was trained.
        masks = (torch.zeros(1, 1, device=self.device) if self._first
                 else torch.ones(1, 1, device=self.device))
        
        #then set self._first = False -- every later tick is a continuation.
        self._first = False
        
        #call the hook and CATCH THE NEW RNN STATE:
        #    p, self.rnn = sp_action_probs(self.policy, obs, self.rnn, masks)
        #the assignment back into self.rnn is the load-bearing part. This is a
        #forward pass through the GRU, so it advances the memory; drop the
        #returned state and the policy is memoryless -- it will still run, still
        #look reasonable, and quietly be a different, worse agent.
        p, self.rnn = sp_action_probs(self.policy, obs, self.rnn, masks)
        
        #convert back to a plain (6,) numpy array:
        #    p.detach().cpu().numpy().reshape(-1).astype(np.float64)
        #  * detach() drops the autograd history (you are not training)
        #  * cpu() moves it off the GPU if it was on one
        #  * numpy() hands back an array sharing the same memory
        #  * reshape(-1) flattens (1,6) to (6,); -1 means "work this axis out"
        #these are chained left to right, each returning something the next
        #one is called on.
        out = p.detach().cpu().numpy().reshape(-1).astype(np.float64)
        
        #renormalise defensively before returning:
        #    s = out.sum()
        #    return out / s if s > 0 else np.full(6, 1.0 / 6.0)
        #float rounding can leave the sum a hair off 1.0, and np.random.choice
        #REJECTS a p that does not sum to 1. np.full(n, v) makes an array of n
        #copies of v -- here the uniform distribution, as a last-resort fallback.
        s = out.sum()
        return out / s if s > 0 else np.full(6, 1.0 / 6.0)

def choose(p, sample, rng):
    """Turn a distribution over the 6 primitives into an action index.

    WHICH RULE MATTERS. CARC_RUNS.md section 6: gc01 / gc05 / cram2 DEADLOCK
    under argmax and are perfect when sampled, gc07 is the other way round.
    Whatever is chosen, BOTH arms in play_episode.py must use the same rule or
    the comparison measures the rule and not the module.
    """
    #one line, using the conditional expression again:
    #    return int(rng.choice(N_ACTIONS, p=p)) if sample else int(np.argmax(p))
    #  * rng.choice(6, p=p) draws one integer 0..5 with probability p[i]. `rng`
    #    is a np.random.RandomState the CALLER owns -- taking it as an argument
    #    instead of calling np.random.choice directly is what makes a run
    #    reproducible, because the caller controls the seed.
    #  * np.argmax(p) is the index of the biggest number: the greedy choice.
    #  * int(...) because numpy hands back np.int64, which json.dumps refuses.
    #Taking `sample` as a flag rather than writing two functions means the two
    #arms cannot drift apart -- there is exactly one selection rule in the file.
    return int(rng.choice(N_ACTIONS, p=p)) if sample else int(np.argmax(p))

# =========================================================================
# 3. ONE BASELINE EPISODE
# =========================================================================
def play_baseline_episode(kitchen, actor, fov, seed, sample=True, rng=None,
                          temperature=0.5, robot="policy"):
    """Robot = the frozen policy alone. Human = LimitedVisionSteakHuman(fov).

    TICK ORDER (identical to the module arm, so the two are paired):
        s = kitchen.state
        a = choose(actor.probs(kitchen.robot_obs()))     robot decides on s
        h, info = human.action(s)                        human decides on s
                                                         info["subtask"] is the
                                                         ground-truth label --
                                                         NEVER read it
        kitchen.step(a, h)                               simultaneous resolve
    """
    #default the rng: rng = rng or np.random.RandomState(seed)
    #so a caller can pass one in for a whole campaign, or let each episode make
    #its own from the seed.
    rng = rng or np.random.RandomState(seed)

    # ---- REFERENCE ARMS: add a `robot="policy"` parameter to this function --
    #signature, last, with a default. Three values, and you want all three in
    #the paper:
    #
    #  "policy"  the trained SP checkpoint. The thing under test.
    #
    #  "random"  a UNIFORM robot: p = np.full(N_ACTIONS, 1.0 / N_ACTIONS).
    #            THE FLOOR, and do not skip it. old/module/QMDP/RESULTS.md 6b
    #            measured a uniform-random robot BEATING the trained SP policy
    #            paired with this human: 2.946 vs 2.814 deliveries, 206 vs 251
    #            ticks. If that reproduces on your baselines, then "we beat the
    #            baseline" means very little on its own and every table needs
    #            this row next to it.
    #
    #  "none"    NO ROBOT AT ALL -- what the human achieves alone. The other
    #            end of the scale, and the number that says whether the robot is
    #            worth having in the kitchen in the first place. See the section
    #            below: this arm takes TWO lines, not one, because a chef that
    #            merely stands still is not the same thing as a chef who is not
    #            there.
    #
    #the branch goes inside the tick loop below, replacing the one line that
    #currently computes idx. Shape of it:
    #    if robot == "policy":   p = actor.probs(kitchen.robot_obs())
    #    elif robot == "random": p = np.full(N_ACTIONS, 1.0 / N_ACTIONS)
    #    else:                   p = None      -> idx = Action.ACTION_TO_INDEX[Action.STAY]
    #then idx = choose(p, sample, rng) for the first two.
    #`elif` is "else if" -- one word in python, and the chain stops at the first
    #branch that matches.
    #
    #for "random" and "none" you can skip actor.probs() entirely: no network, no
    #GRU, and the episode runs much faster. actor.reset() below is then a no-op,
    #which is fine -- leave it in so the tick structure stays identical.

    # ---- MAKING "none" ACTUALLY MEAN "NO ROBOT" ---------------------------
    #The mdp has exactly two chefs, always. num_players comes from the layout's
    #two start positions, so you cannot delete one -- the closest you can get is
    #a chef who does nothing AND affects nothing. "Does nothing" is one line.
    #"Affects nothing" needs one more, and this is the part that is easy to
    #miss, because it fails silently and looks like a result.
    #
    #A robot standing still still reaches the human through three channels in
    #limited_vision_human.py. I checked all three:
    #
    #  _robot_redundant (line 631, 840) -- reads beliefs[ROBOT].value and looks
    #      it up in ROBOT_HELD_FETCH. A parked robot holds nothing, the belief
    #      is "none", and ROBOT_HELD_FETCH.get("none") is None.  HARMLESS.
    #
    #  _weights (line 783) -- same belief, same lookup, soft down-weight.
    #      Also nothing to match on empty hands.                  HARMLESS.
    #
    #  _robot_faced_kind (line 824) -- returns the station kind the SEEN
    #      teammate is FACING, and _available_advancing then BLOCKS every task
    #      in STATION_TASKS[kind]. Pose alone; no held object required.
    #      *** THIS ONE BITES. *** A robot frozen facing the pot suppresses the
    #      human's pickup_meat for all 400 ticks just by existing there, so the
    #      "human alone" number is really "human who refuses to touch the pot".
    #      And it is cone-gated, so it hits hard at fov 360 and not at all at
    #      fov 30 -- the artefact varies with the exact axis you are studying,
    #      which is the worst possible shape for a confound.
    #
    #TWO WAYS TO KILL THAT CHANNEL, and we are taking the second one:
    #
    #  (a) blind the human:  human._robot_faced_kind = lambda: None
    #      Effective, but it EDITS THE PARTNER. The human is now running
    #      different code in this arm than in the others, and "the human alone"
    #      stops meaning the same human. Rejected.
    #
    #  (b) POINT THE ROBOT AT NOTHING.  <-- what this file does.
    #      _robot_faced_kind only returns a kind when the faced cell is in
    #      shadow.stations[kind]. Face the robot at a wall, a counter or empty
    #      floor and it returns None for the whole episode, with the human's
    #      code completely untouched. The comparison stays fair because we
    #      changed OUR agent, not theirs.
    #
    #WHY IT STICKS FOR ALL 400 TICKS: overcooked_mdp.py:1134 is
    #    new_orientation = orientation if action == Action.STAY else action
    #so STAY preserves orientation exactly, and INTERACT is not a motion action
    #so it does not touch orientation either. A robot that only ever plays STAY
    #keeps whatever facing you give it. Set it ONCE, right after kitchen.reset().
    #
    #THE LINES, in the "none" arm only, after reset:
    #  * get the robot: rp = kitchen.state.players[ROBOT_INDEX]
    #  * collect the station cells you must not face. The mdp knows them:
    #        mdp.get_pot_locations() + mdp.get_chopping_board_locations()
    #        + mdp.get_sink_locations()
    #    a python list + list is CONCATENATION, so those three join into one.
    #  * walk Direction.ALL_DIRECTIONS (import it alongside Action) and keep the
    #    first d where (rp.position[0] + d[0], rp.position[1] + d[1]) is NOT in
    #    that list.
    #  * apply it: rp.update_pos_and_or(rp.position, d)   -- same position, new
    #    facing. (mdp:228. Setting rp.orientation directly also works, but the
    #    method is the API and will keep working if the class gains bookkeeping.)
    #  * if NO direction is free -- a robot boxed in by stations on all four
    #    sides -- do not silently continue. raise, or at minimum print a loud
    #    warning naming the layout, because that layout's "human alone" number
    #    is not comparable to the others and you need to know which one it was.
    #
    #Record which way it ended up facing in the row if you want to be thorough;
    #it costs one key and makes the arm reproducible.
    #
    #With collisions already off, the robot is now a body that blocks nothing,
    #signals nothing and does nothing. That is as close to "not there" as a
    #two-player mdp allows, and every deviation from it is written down above.
    #start the episode: kitchen.reset()
    kitchen.reset()
    #`none` arm only: turn the robot to face nothing, once, right after reset.
    #This is what makes the arm mean "no robot" instead of "a statue in the
    #kitchen" -- see face_away_from_stations.
    faced_away = face_away_from_stations(kitchen) if robot == "none" else None
    #build the human: human = make_human(kitchen.mdp, fov, seed,
    #                                    temperature=temperature)
    #AFTER reset, and once per episode -- a human carries beliefs, discovered
    #cells and a decay clock, so reusing one across episodes hands it a map it
    #earned somewhere else.
    human = make_human(kitchen.mdp, fov, seed, temperature=temperature)
    
    #wipe the network's memory: actor.reset()
    #easy to forget, and the failure is invisible: episode 2 starts with
    #episode 1's GRU state and the policy behaves slightly wrong all run.
    actor.reset()
    
    #set up the counters you will fill in:
    #    deliveries = 0                 how many orders went out
    #    delivery_ticks = []            the tick each one landed on
    #    action_hist = [0] * 6          how often each button was pressed
    #[0] * 6 makes [0,0,0,0,0,0]. Worth the six integers: a module whose "win"
    #is really "the robot learned to stand still" shows up here and nowhere else.
    deliveries = 0
    delivery_ticks = []
    action_hist = [0] * 6
    
    #the loop header:
    #    while kitchen.t < kitchen.horizon and not kitchen.mdp.is_terminal(kitchen.state):
    #`while <test>:` repeats as long as the test is True (a `for` walks a fixed
    #list; you do not know the length here). `and` is only True when both sides
    #are. Two exit routes: out of time, or the order list ran down.
    while kitchen.t < kitchen.horizon and not kitchen.mdp.is_terminal(kitchen.state):
        #grab the state ONCE at the top: s = kitchen.state
        #both agents must decide on the SAME state -- that is what makes the
        #move simultaneous. Re-reading kitchen.state after stepping would be
        #the classic off-by-one that turns this into a turn-based game.
        s = kitchen.state
        
        #robot decides an action:
        #    idx = choose(actor.probs(kitchen.robot_obs()), sample, rng)
        #    a = Action.INDEX_TO_ACTION[idx]
        #INDEX_TO_ACTION is a lookup table from 0..5 to the action object the
        #mdp wants. (Action.ACTION_TO_INDEX goes the other way.)
        if robot == "policy":
            idx = choose(actor.probs(kitchen.robot_obs()), sample, rng=rng)
        elif robot == "random":
            #always a uniform DRAW, regardless of `sample`. argmax of a uniform
            #distribution is degenerate -- np.argmax returns index 0 every time,
            #so a "random" robot under argmax would just walk NORTH for 400
            #ticks. That is not the floor anyone means.
            idx = int(rng.choice(N_ACTIONS))
        else:
            #`none`: stand still forever. STAY preserves orientation, so the
            #facing set after reset holds for the whole episode.
            idx = Action.ACTION_TO_INDEX[Action.STAY]
        a = Action.INDEX_TO_ACTION[idx]
        
        #count it: action_hist[idx] += 1
        #`+=` means "add to what is already there".
        action_hist[idx] += 1
        
        #human decides:
        #    h, info = human.action(s)
        #TWO return values again. `info` holds {"subtask": ...} -- the human's
        #TRUE internal label, and the single most tempting cheat in this
        #codebase. Read it for a metric and the whole no-cheating claim dies.
        #Name it `_info` if you want the reminder in the code itself.
        h, info = human.action(s)
        #step the world: sparse, done = kitchen.step(a, h)
        sparse, done = kitchen.step(a, h)
        
        #count deliveries, if any reward landed:
        #    if sparse > 0:
        #        n = int(round(sparse / float(kitchen.mdp.delivery_reward)))
        #        for _ in range(max(1, n)):
        #            deliveries += 1
        #            delivery_ticks.append(kitchen.t)
        #  * delivery_reward is 20, so sparse=40 means TWO orders in one tick.
        #    Dividing recovers the count instead of assuming one.
        #  * `for _ in range(n):` repeats n times when you do not care about the
        #    loop variable -- `_` again.
        #  * max(1, n) guards against a rounding result of 0 on a positive reward.
        #THIS IS THE ONLY PLACE REWARD IS READ, and it is a metric, not a
        #decision. Nothing upstream of the action choice ever sees it.
        if sparse >0:
            n = int(round(sparse / float(kitchen.mdp.delivery_reward)))
            for _ in range(max(1, n)):
                deliveries += 1
                delivery_ticks.append(kitchen.t)
        
        #stop early if the env says so:
        #    if done:
        #        break
        #`break` leaves the loop entirely (vs `continue`, which skips one turn).
        if done:
            break
    #after the loop, work out whether the team finished:
    #    finished = len(delivery_ticks) >= (kitchen.n_orders - 1)
    #n_orders - 1 because is_terminal fires at one order LEFT, so with 4 the
    #ceiling is 3 deliveries. len() is the number of items in a list.
    finished = len(delivery_ticks) >= (kitchen.n_orders - 1)
    
    #    completion = delivery_ticks[-1] if finished else kitchen.horizon + DNF_PENALTY
    #[-1] is the LAST item (negative indices count from the end). A team that
    #never finished is scored at horizon+100 so a DNF is worse than any real
    #time and the two arms stay comparable -- same convention the old RESULTS.md
    #used, so the numbers are directly comparable to it.
    #if they didn't finish, add 100 steps or DNF_PENALTY to the time
    completion = delivery_ticks[-1] if finished else kitchen.horizon + DNF_PENALTY
    
    #build the result dict and return it. A dict is {key: value} pairs, and this
    #one is the row that lands in the JSONL file, so keep the keys stable:
    #    "layout", "fov", "seed", "module" (False here), "deliveries", "steps",
    #    "completion_time", "finished", "t_delivery", "action_hist",
    #    "h_wasted", "h_explore", "h_checks", "h_abandoned", "h_delivered"
    #the h_* ones come off the human's own counters: human.n_wasted_commits,
    #n_explore, n_checks, n_abandoned, n_delivered. They explain WHY a condition
    #won; no score depends on them.
    #Force the types: int(...) around counts, bool(...) around finished. numpy
    #and python ints look identical until json.dumps refuses one of them.
    #For "t_delivery" use a LIST COMPREHENSION: [int(x) for x in delivery_ticks]
    #-- "build a new list by running int(x) on every x". Same as a for-loop with
    #.append, one line
    return dict(
        layout=kitchen.layout,
        fov=fov,
        seed=seed,
        module=False,
        robot=robot,
        faced_away=list(faced_away) if faced_away else None,
        #RECORD THE ARM IN THE ROW. Add one more key here:
        #    robot=robot,                    "policy" / "random" / "none"
        #without it a .jsonl is just numbers and you cannot tell a random-robot
        #run from a policy run six weeks later. Every knob that changes the
        #experiment belongs in the output, not only in the sbatch script that
        #launched it. (Collisions need no key -- they are off in every run.)
        deliveries=int(deliveries),
        steps=int(kitchen.t),
        completion_time=int(completion),
        finished=bool(finished),
        t_delivery=[int(x) for x in delivery_ticks],
        action_hist=action_hist,
        h_wasted=int(human.n_wasted_commits),
        h_explore=int(human.n_explore),
        h_checks=int(human.n_checks),
        h_abandoned=int(human.n_abandoned),
        h_delivered=int(human.n_delivered)
    )
    


# =========================================================================
# 4. HARNESS
# =========================================================================
def summarize(rows):
    """rows -> one dict of means. Shared with play_episode.py so both arms are
    reported on the same numbers."""
    #guard the empty case first: if not rows: return {}
    #an empty list is FALSY in python, so `if not rows:` reads as "if there are
    #no rows". Same trick works for "", 0, {} and None. Returning early keeps
    #the rest of the function unindented -- easier to read than wrapping
    #everything in an else.
    if not rows:
        return {}
    
    #compute each mean with a list comprehension inside float(np.mean(...)):
    #    float(np.mean([r["deliveries"] for r in rows]))
    #read the inner part as "the deliveries value out of every row r".
    #np.mean averages the list; float() converts np.float64 to a plain float so
    #this dict is JSON-safe.
    #for the rate, average the booleans:
    #    float(np.mean([r["finished"] for r in rows]))
    #True counts as 1 and False as 0, so the mean IS the fraction finished.

    #return a dict with at least: "n" (len(rows)), "deliveries",
    #"completion_time", "finished_rate". Keep the key names identical to the
    #row keys where possible -- one vocabulary for the whole package.
    #NOTE THE SQUARE BRACKETS. [x for x in ...] builds a LIST; (x for x in ...)
    #makes a lazy GENERATOR, and np.mean cannot average one -- it raises
    #    TypeError: unsupported operand type(s) for /: 'generator' and 'int'
    #A bare `np.mean(r["k"] for r in rows)` is the generator form, because the
    #function-call parentheses double as the generator's parentheses.
    return dict(
        n=len(rows),
        deliveries=float(np.mean([r["deliveries"] for r in rows])),
        completion_time=float(np.mean([r["completion_time"] for r in rows])),
        finished_rate=float(np.mean([r["finished"] for r in rows]))
    )

def print_table(summary, title=""):
    """One short block of human-readable output."""
    #print the title, then one line per number. Use % formatting with widths:
    #    print("%-18s %6.2f" % ("deliveries", summary["deliveries"]))
    #  * %-18s = a string, left-aligned, padded to 18 characters
    #  * %6.2f = a float, 2 decimal places, right-aligned in 6 characters
    #the padding is what makes columns line up when you print several rows.
    print(title)
    print("%-18s %6.2f" % ("deliveries", summary["deliveries"]))
    print("%-18s %6.2f" % ("completion_time", summary["completion_time"]))
    print("%-18s %6.2f" % ("finished_rate", summary["finished_rate"]))
    #pass flush=True to print() if you are running under sbatch. SLURM buffers
    #stdout, so without it a job that dies shows you nothing at all.
    print(flush=True)
    #this function PRINTS and returns nothing -- the opposite of summarize(),
    #on purpose. Keep the thing that computes and the thing that displays
    #separate: you can then log the numbers, or table them, or both.


def compare_arms(rows, title=""):
    """One table, one line per arm, worst-to-best. Returns {arm: summary}.

    The whole point of running the arms together is that they share the seeds,
    so every column is PAIRED: the same human, from the same draw sequence, met
    a different robot. Read it as a scale --

        none     what the human manages with no robot at all. The floor.
        random   what a body that does SOMETHING is worth.
        policy   what training bought on top of that.

    The comparison that matters is not `policy` vs zero, it is `policy` vs the
    other two. old/module/QMDP/RESULTS.md section 6b measured a uniform-random
    robot BEATING the trained policy paired with this human (2.946 vs 2.814
    deliveries). If that shows up here, it is the headline, not a footnote.
    """
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r.get("robot", "policy"), []).append(r)

    floor = summarize(by_arm.get("none", []))
    print("\n" + (title or "arm comparison"))
    print("%-8s %4s %11s %11s %9s %10s"
          % ("arm", "n", "deliveries", "completion", "finished", "vs none"))
    print("-" * 58)
    out = {}
    for arm in ARMS:                       # fixed order, so runs look alike
        if arm not in by_arm:
            continue
        s = summarize(by_arm[arm])
        out[arm] = s
        #delta against the human-solo floor. Negative = finished sooner.
        delta = ("%+10.1f" % (s["completion_time"] - floor["completion_time"])
                 if floor else "%10s" % "--")
        print("%-8s %4d %11.2f %11.2f %9.2f %s"
              % (arm, s["n"], s["deliveries"], s["completion_time"],
                 s["finished_rate"], delta))
    print(flush=True)
    return out


def save(rows, path):
    """Append one JSON object per episode to `path` (JSONL)."""
    #make sure the folder exists first:
    #    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    #abspath turns a bare "results.jsonl" into a full path, dirname takes the
    #folder off it, and `or "."` covers the case where that comes back empty
    #(meaning the current directory).
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    
    #open the file and write:
    #    with open(path, "a") as fh:
    #        for row in rows:
    #            fh.write(json.dumps(row) + "\n")
    #NEW PYTHON: `with`. It borrows something, and guarantees it is given back
    #when the block ends -- even if an exception is thrown mid-loop. For a file
    #that means it is always closed and flushed. Always open files this way.
    #  * "a" is APPEND (add to the end). "w" would TRUNCATE -- it silently wipes
    #    an existing file, which on a long CARC run is a very bad afternoon.
    #  * json.dumps(obj) turns a dict into a string. One object per line, with
    #    "\n" between, is JSONL: you can append forever, and read it back one
    #    line at a time without loading the whole file.
    #    (json.dump with no "s" writes to a file directly; json.loads is the
    #    reverse. Easy to mix up.)
    #for a long run, flush after each write, so a job killed at the wall clock
    #still leaves you the episodes it finished. It must be INSIDE the `with`:
    #once the block ends the file is closed, and flushing a closed file raises
    #ValueError: I/O operation on closed file.
    with open(path, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
            fh.flush()

def main(argv=None):
    """CLI entry point. Returns an exit code (0 = fine)."""
    #build the parser:
    #    p = argparse.ArgumentParser("baseline arm: SP policy vs the FOV human")
    #then one add_argument per flag:
    #  * the leading -- makes it optional; type= converts the text from the
    #    command line into the right python type.
    #  * action="store_true" is a FLAG: present means True, absent means False,
    #    and it takes no value. That is how --sample works.
    #  * dashes in a flag become underscores in the result: --n_orders is
    #    args.n_orders.
    p = argparse.ArgumentParser("baseline arm: SP policy vs the FOV human")
    p.add_argument("--layout", type=str, default="steak_gc00")
    p.add_argument("--fov", type=int, default=90)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--n_orders", type=int, default=N_ORDERS)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--out", type=str, default="baseline.jsonl")
    #"auto" picks cuda > mps > cpu. Read resolve_device's docstring before using
    #anything but cpu here: at batch size 1 with a tiny conv stack, the GPU is
    #very likely SLOWER, and on CARC compute nodes there is no cuda anyway.
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | cuda | mps  (see resolve_device)")
    #every arm runs on the SAME seeds in one invocation, so the comparison at
    #the end is paired by construction and you never have to line up two files.
    p.add_argument("--arms", type=str, default="none,random,policy",
                   help="comma separated subset of: none, random, policy")
    # ---- the reference arms ------------------------------------------------
    #    p.add_argument("--robot", type=str, default="policy",
    #                   choices=["policy", "random", "none"],
    #                   help="policy = the checkpoint; random = uniform floor; "
    #                        "none = no robot at all (human solo)")
    #choices=[...] makes argparse reject a typo AT PARSE TIME with a readable
    #message, instead of you finding out from a silently wrong arm three hours
    #into a sweep. Use it on every string flag with a fixed set of values.
    #
    #NO collisions flag. Collisions are off in every run, unconditionally, from
    #Kitchen.__init__ -- there is nothing to pass and nothing to forget.
    #
    #then thread it through:
    #    row = play_baseline_episode(..., robot=args.robot)
    #and put args.robot in the progress print, or every arm's output looks
    #identical scrolling past.
    #
    #the three runs worth having side by side, all else equal:
    #    python baseline.py --layout steak_gc00 --fov 90 --episodes 10 --sample
    #    python baseline.py ... --sample --robot random
    #    python baseline.py ... --sample --robot none
    #read them as a scale: `none` is what the human manages alone, `random` is
    #what a body that does SOMETHING is worth, `policy` is what training bought
    #on top of that. If policy does not clear random, the interesting result is
    #that fact, not anything downstream of it.
    #parse: args = p.parse_args(argv)
    #argv=None means "read the real command line". Taking it as an argument
    #anyway lets a test call main(["--layout", "steak_gc00"]) directly.
    args = p.parse_args(argv)
    
    #stage the layouts: stage_layouts()
    #before anything touches from_layout_name, or you get "layout not found".
    stage_layouts()
    
    #torch.set_num_threads(1)
    #the bottleneck is the python env loop, not the conv stack. Left alone,
    #torch grabs every core and they fight -- this is FASTER, not slower.
    #set_num_threads, NOT set_num_interop_threads. The first caps the threads
    #INSIDE one op (the conv stack) -- that is the one you want. The second
    #caps parallelism BETWEEN ops, does nothing useful here, and raises if
    #anything parallel has already run.
    #CPU ONLY: set_num_threads governs the CPU op threadpool, so it is pointless
    #once the work is on a GPU. Resolve the device first, then decide.
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    print("[device] %s" % device, flush=True)
    
    #build the kitchen and the actor ONCE, outside the episode loop:
    #    kitchen = Kitchen(args.layout, args.n_orders, args.horizon)
    #    actor = Actor(args.layout, kitchen.obs_shape)
    #loading the checkpoint is the slowest thing here and the weights never
    #change, so doing it per episode wastes most of the run.
    kitchen = Kitchen(args.layout, args.n_orders, args.horizon)
    actor = Actor(args.layout, kitchen.obs_shape, device=device)
    #make the rng: rng = np.random.RandomState(args.seed0)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise ValueError("unknown arm(s) %s; pick from %s" % (bad, list(ARMS)))

    rows = []
    #SEED-MAJOR, arms inside. A run killed halfway then still holds COMPLETE
    #sets of seeds rather than one finished arm and two empty ones, so a partial
    #file is still a valid paired comparison.
    for seed in range(args.seed0, args.seed0 + args.episodes):
        for arm in arms:
            #a FRESH rng per episode, seeded from the episode\'s own seed.
            #Sharing one stream across episodes would make seed 5 depend on how
            #many episodes ran before it -- and here it is worse than that: the
            #arms draw different numbers of times (`none` draws not at all), so
            #one shared stream would desynchronise them and quietly stop the
            #comparison being paired.
            row = play_baseline_episode(
                kitchen, actor, args.fov, seed, sample=args.sample,
                rng=np.random.RandomState(seed), temperature=args.temperature,
                robot=arm)
            rows.append(row)
            print("  %-7s %s fov=%d seed=%d: deliveries=%d steps=%d finished=%s"
                  % (arm, args.layout, args.fov, seed, row["deliveries"],
                     row["steps"], row["finished"]), flush=True)

    save(rows, args.out)
    compare_arms(rows, title="%s  fov=%d  %d seeds  (collisions off)"
                 % (args.layout, args.fov, args.episodes))
    return 0
    #0 means success to the shell; anything else means failure. That is what
    #`sys.exit(main())` at the bottom of the file passes on, and it is how a
    #sbatch script or a CI job knows whether the run worked.

if __name__ == "__main__":
    #this line runs ONLY when the file is executed directly
    #(`python baseline.py`). When play_episode.py does `import baseline`,
    #__name__ is "baseline" instead of "__main__" and this block is skipped --
    #which is exactly why importing this file does not start an experiment.
    sys.exit(main())
