"""Registry + factory for every robot policy, plain baselines and the
FOV-aware one alike -- mirrors steakhouse/misha's own robot/methods.py
(a registry table selected by name, `make_robot(kind, ...)`), scaled down
to this project's 3 policies instead of a full QMDP-planning suite.

nominal_policy/ and filter/ never need to know about each other's
existence beyond this one file: nominal_policy/baselines.py exposes plain
baselines, filter/core/fov_aware_baseline.py wraps one of them with the
FOV posterior/filter, and this module is the only place both are imported
together.
"""
from robot.nominal_policy.baselines import REGISTRY as _NOMINAL_REGISTRY
from robot.filter.core import fov_aware_baseline as _fov_aware_baseline_module
from robot.filter.core.fov_aware_baseline import FOVAwareBaseline
from robot.nominal_policy.vehicles import REGISTRY as _VEHICLE_REGISTRY

REGISTRY = dict(_NOMINAL_REGISTRY)
REGISTRY[FOVAwareBaseline.name] = FOVAwareBaseline


def make_baseline(name, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline {name!r}, choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def combo_names():
    """Every "{vehicle}_{policy}" name -- e.g. "idm_fov_aware",
    "defensive_cautious" -- the single-flag shorthand play.py/watch.py/
    interface.py's own --robot argument accepts for when the two axes
    (nominal_policy.vehicles.REGISTRY's underlying dynamics, this module's
    own REGISTRY's policy wrapper) don't need setting independently via
    --robot-vehicle/--robot-policy. Mirrors steakhouse/misha's own
    robot/methods.py single-named-baseline convention (--partner <name>),
    scaled to this domain's own two orthogonal axes instead of one flat
    method table."""
    return sorted(f"{v}_{p}" for v in _VEHICLE_REGISTRY for p in REGISTRY)


def resolve_combo(name):
    """"{vehicle}_{policy}" -> (vehicle, policy), or None if `name` isn't
    one of combo_names()'s own entries. Matched by POLICY suffix first
    (there are only 3: nominal/cautious/fov_aware, all fixed strings with
    no underscores of their own) since a vehicle name could in principle
    contain one."""
    for p in REGISTRY:
        suffix = f"_{p}"
        if name.endswith(suffix):
            v = name[: -len(suffix)]
            if v in _VEHICLE_REGISTRY:
                return v, p
    return None


def _first_sentence(doc, max_len=100):
    """The first sentence of a docstring's own first paragraph, whitespace-
    collapsed onto one line -- plain .splitlines()[0] cuts a wrapped
    docstring off mid-sentence at ~70 chars, which is exactly what a raw
    line-1 grab did here before this existed."""
    if not doc:
        return ""
    paragraph = doc.strip().split("\n\n", 1)[0]
    paragraph = " ".join(paragraph.split())
    sentence = paragraph.split(". ", 1)[0].rstrip(".") + "."
    return sentence if len(sentence) <= max_len else sentence[:max_len - 1].rstrip() + "…"


def describe_robots():
    """--list-robots output for play.py/watch.py/interface.py -- explained
    and grouped like steakhouse/misha/play.py's own header (baselines
    described one by one, "the same filter over each one"), not just the
    bare alphabetical combo_names() dump this used to print. The human's
    OWN field of view is a SEPARATE thing from anything below: it's always
    on (--fov sets its width; there's no flag that turns it off, because
    "the human has limited vision" is the whole premise of this project),
    never a robot-side choice. What --robot actually picks is the ROBOT's
    own policy -- does IT reason about your FOV -- crossed with which
    underlying vehicle dynamics it drives with; those are the two
    independent axes combo_names() enumerates the full product of.
    """
    lines = [
        "Your OWN field of view (the human's) is always on -- set its width",
        "with --fov; it is never something a --robot choice turns on or off.",
        "",
        "POLICY -- does the robot reason about your FOV at all:",
    ]
    for name, cls in _NOMINAL_REGISTRY.items():
        lines.append(f"  {name:<10} {_first_sentence(cls.__doc__)}")
    fov_doc = _first_sentence(_fov_aware_baseline_module.__doc__)
    lines.append(f"  {FOVAwareBaseline.name:<10} {fov_doc}  <- the headline option: try this one first")
    lines += [
        "",
        "VEHICLE -- underlying driving style, independent of policy:",
        *(f"  {v}" for v in sorted(_VEHICLE_REGISTRY)),
        "",
        '--robot NAME is a "{vehicle}_{policy}" shorthand for both at once, e.g.:',
        "  --robot idm_fov_aware       (the default)",
        "  --robot defensive_cautious",
        "",
        "Every combination:",
        *(f"  {n}" for n in combo_names()),
    ]
    return "\n".join(lines)
