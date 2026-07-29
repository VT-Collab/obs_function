"""Aggregate CARC search results (each line: 'JSON {layout, results:[...]}').
Reports, per FAIR (150-iter) baseline: win rate, wins by fov-set, biggest
speedups, and the mean fair time-delta at narrow FOV. Reads a file of JSON lines."""
import sys, json

rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("JSON "):
        line = line[5:]
    if not line.startswith("{"):
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    if "results" not in d:
        continue
    for r in d["results"]:
        rows.append({"layout": d["layout"], **r})

if not rows:
    print("no rows parsed"); sys.exit()

wins = [r for r in rows if r.get("wins")]
by_fov = {}
for r in wins:
    by_fov[r["fov_set"]] = by_fov.get(r["fov_set"], 0) + 1
# best config by win count
by_cfg = {}
for r in wins:
    by_cfg[r["cfg"]] = by_cfg.get(r["cfg"], 0) + 1

wins.sort(key=lambda r: r["time_delta"])   # most negative (biggest speedup) first
narrow = [r for r in rows if r["fov_set"] in ("30", "60", "90")]
import statistics as st
narrow_delta = st.mean([r["time_delta"] for r in narrow]) if narrow else 0

print(f"=== CARC FAIR (150-iter baseline) all-layout aggregate ===")
print(f"layouts: {len(set(r['layout'] for r in rows))}  cells: {len(rows)}  WIN cells: {len(wins)} ({100*len(wins)//max(1,len(rows))}%)")
print(f"mean time-delta at narrow FOV (30/60/90): {narrow_delta:+.1f} steps  (negative = module faster)")
print(f"wins by fov-set: {dict(sorted(by_fov.items()))}")
print(f"wins by config: {dict(sorted(by_cfg.items(), key=lambda x:-x[1]))}")
print(f"\ntop 15 fair speedups (layout / fov / cfg : dt , d_del , base->mod):")
for r in wins[:15]:
    print(f"  {r['layout']:<16} {r['fov_set']:<10} {r['cfg']:<20} dt={r['time_delta']:>+7.1f} dd={r['del_delta']:>+.2f} ({r['b_time']:.0f}->{r['m_time']:.0f})")
