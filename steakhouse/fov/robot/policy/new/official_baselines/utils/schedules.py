"""
Piecewise-linear coefficient schedules, ZSC-Eval style.

ZSC-Eval does not anneal entropy with a single straight line. train_sp.sh:

    entropy_coefs         "0.2  0.05  0.01"
    entropy_coef_horizons "0    5e6   1e7"

which is TWO segments -- a fast drop from 0.2 to 0.05 over the first half, then
a slow crawl from 0.05 to 0.01 over the second. E3T uses the same shape with the
knee at 6e6 instead of 5e6, and small_corridor (their hardest layout) pushes the
knee out to 8e6 to buy more exploration.

The validated seed-1 runs here used a SINGLE segment, 0.2 -> 0.01 over 1e7,
which at the midpoint sits at 0.105 versus ZSC's 0.05 -- about twice the
exploration pressure through the middle of training. Not obviously worse; just
different, and undocumented until now.

This module expresses both, so which one a run used is a recorded argument
rather than a property of whichever file it was launched from.

    piecewise(x, [0.2, 0.01],        [0, 1e7])        <- the old single line
    piecewise(x, [0.2, 0.05, 0.01],  [0, 5e6, 1e7])   <- ZSC-Eval

Outside the horizon range the value is clamped to the first / last coefficient,
so a run that overshoots num_env_steps holds the final value instead of
extrapolating to nonsense.
"""


def piecewise(x, values, knots):
    """Linear interpolation of `values` over `knots`, clamped at both ends.

    values and knots must be the same length and knots must be increasing.
    """
    if len(values) != len(knots):
        raise ValueError(f"{len(values)} values vs {len(knots)} horizons -- "
                         "they must come in pairs")
    if len(values) == 1:
        return float(values[0])

    if x <= knots[0]:
        return float(values[0])
    if x >= knots[-1]:
        return float(values[-1])

    for i in range(1, len(knots)):
        if x <= knots[i]:
            lo_x, hi_x = knots[i - 1], knots[i]
            lo_v, hi_v = values[i - 1], values[i]
            span = hi_x - lo_x
            if span <= 0:
                return float(hi_v)
            frac = (x - lo_x) / span
            return float(lo_v + (hi_v - lo_v) * frac)
    return float(values[-1])


def add_schedule_args(parser):
    """The two flags, with the OLD single-segment behaviour as the default.

    Defaulting to the old shape is deliberate: this wave changes valuenorm and
    checkpointing on purpose, and stacking a third silent change on top would
    make a regression impossible to attribute. Pass the ZSC values explicitly:

        --entropy_coefs 0.2 0.05 0.01 --entropy_coef_horizons 0 5e6 1e7
    """
    parser.add_argument("--entropy_coefs", type=float, nargs="+",
                        default=[0.2, 0.01],
                        help="entropy coefficient at each horizon knot")
    parser.add_argument("--entropy_coef_horizons", type=float, nargs="+",
                        default=[0.0, 1e7],
                        help="env-step positions of the knots above")
    return parser
