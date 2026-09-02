"""Analyze CDT memory sim results."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
log = ROOT / "experiments" / "emergence_sim" / "cdt_memory.log"

conditions = {}
with open(log) as f:
    for line in f:
        entry = json.loads(line)
        cond = entry.get("condition", "")
        if cond not in conditions:
            conditions[cond] = []
        if "step" in entry:
            conditions[cond].append(entry)

print("=== CDT MEMORY SIM RESULTS ===\n")
for cond, entries in conditions.items():
    print(f"--- {cond} ---")
    for e in entries:
        print(f"  step={e['step']:5d}  ce={e['ce']:.3f}  surp={e['surp']:.1f}  "
              f"persist={e['persist']:.3f}  hcm_n={e['hcm_n']}  "
              f"nu={e['nu']:.3f}  w={e['w']:.3f}  "
              f"action_prob={e['action_prob']:.5f}")
    if entries:
        first = entries[0]
        last = entries[-1]
        print(f"  CE change: {first['ce']:.3f} -> {last['ce']:.3f} ({last['ce']-first['ce']:+.3f})")
        print(f"  Persist change: {first['persist']:.3f} -> {last['persist']:.3f} ({last['persist']-first['persist']:+.3f})")
        print(f"  Writes: {last['hcm_writes']}  Reads: {last['hcm_reads']}")
        print()

# Read the full log for completion events
print("\n=== COMPLETION EVENTS ===")
with open(log) as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("event") == "complete":
            print(f"  {entry['condition']}: final_ce={entry['final_ce']:.3f} "
                  f"writes={entry['total_writes']} recalls={entry['total_recalls']} "
                  f"hits={entry['recall_hits']}")
