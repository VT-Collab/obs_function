"""What job does it swap FOR what, on the ticks it goes off-argmax?

Reads subtask_dev --trace-out JSONL. Counts (baseline job -> chosen job) pairs on
off-job ticks only, and the mean `gain` (baseline C minus chosen C, in ticks) the
layer thought it was buying, so a swap that is common can be told from one that
is expensive.
"""
import ast
import collections
import json
import sys


def job(s):
    if s in (None, "None", ""):
        return None
    t = ast.literal_eval(s)
    return "%s/%s" % (t[0], t[1])


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        continue
    swaps = collections.Counter()
    gain = collections.defaultdict(list)
    dev = collections.Counter()
    n = off = 0
    for r in rows:
        n += 1
        b, q = job(r["base"]), job(r["q"])
        if b == q:
            continue
        off += 1
        swaps[(b, q)] += 1
        gain[(b, q)].append(r.get("gain") or 0.0)
        if r["dev"]:
            dev[(b, q)] += 1
    print("=== %s   %d scored ticks, %d off-job (%.1f%%)"
          % (path.split("/")[-1], n, off, 100.0 * off / max(n, 1)))
    print("   %-28s -> %-28s  ticks  act-chg  mean gain" % ("baseline job", "chosen job"))
    for (b, q), c in swaps.most_common(12):
        g = sum(gain[(b, q)]) / len(gain[(b, q)])
        print("   %-28s -> %-28s %6d %7d %10.2f" % (b, q, c, dev[(b, q)], g))
    print()
