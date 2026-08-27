"""Analyze night4 no_hcm baseline trajectory."""
import json
import pathlib
import math

ROOT = pathlib.Path(__file__).resolve().parents[1]
log_path = ROOT / "experiments" / "p4_night4" / "train.log"

entries = []
for line in open(log_path, encoding="utf-8"):
    try:
        entries.append(json.loads(line))
    except:
        pass

print(f"Total entries: {len(entries)}")

# Val entries
vals = [e for e in entries if "val_ce_nats" in e]
print(f"\n=== VAL ===")
for v in vals:
    print(f"  step={v['step']:5d}  val_ce={v['val_ce_nats']:.4f}")

# CE trajectory
ces = [(e["step"], e["ce"]) for e in entries if "ce" in e and "val_ce" not in e]
print(f"\n=== CE TRAJECTORY ({len(ces)} points) ===")
if ces:
    print(f"  first={ces[0][1]:.4f}  last={ces[-1][1]:.4f}  min={min(c[1] for c in ces):.4f}")
    # Sample every ~10%
    step = max(1, len(ces) // 15)
    for i in range(0, len(ces), step):
        s, c = ces[i]
        print(f"  step={s:5d}  ce={c:.4f}")

# Surp trajectory
surps = [(e["step"], e["surp"]) for e in entries if "surp" in e and "val_ce" not in e]
print(f"\n=== SURP ===")
if surps:
    step = max(1, len(surps) // 15)
    for i in range(0, len(surps), step):
        s, p = surps[i]
        print(f"  step={s:5d}  surp={p:.4f}")

# Persist trajectory
pers = [(e["step"], e["persist"]) for e in entries if "persist" in e and "val_ce" not in e]
print(f"\n=== PERSIST ===")
if pers:
    step = max(1, len(pers) // 15)
    for i in range(0, len(pers), step):
        s, p = pers[i]
        print(f"  step={s:5d}  persist={p:.4f}")

# action_remember_prob
arps = [(e["step"], e["action_remember_prob"]) for e in entries if "action_remember_prob" in e]
print(f"\n=== ACTION REMEMBER PROB ===")
if arps:
    step = max(1, len(arps) // 15)
    for i in range(0, len(arps), step):
        s, a = arps[i]
        print(f"  step={s:5d}  action_prob={a:.6f}")

# Health at last val
if vals:
    last = vals[-1]
    h = last.get("health", {})
    print(f"\n=== HEALTH (step {last['step']}) ===")
    print(f"  rms={h.get('rms', '?')}")
    print(f"  tau_mean={h.get('tau_mean', '?')}")
    print(f"  tau_pinned_frac={h.get('tau_pinned_frac', '?')}")
    print(f"  chi_motion={h.get('chi_motion', {}).get('motion', '?')}")
    print(f"  gen_trigram_transient={h.get('gen', {}).get('gen_trigram_transient', '?')}")
    print(f"  gen_sites={h.get('gen', {}).get('gen_sites', '?')}")
    carrier = last.get("carrier", {})
    print(f"  carrier_tenure={carrier.get('carrier_tenure_frac', '?')}")
    print(f"  carrier_tau_mean={carrier.get('carrier_tau_mean', '?')}")
    excursion = h.get("excursion", {})
    for k, v in excursion.items():
        print(f"  excursion_{k}: norm={v.get('norm','?'):.3f} conf={v.get('conf','?'):.4f}")
    # divergence
    print(f"  excursion_t24/t48 ratio: {excursion.get('t24',{}).get('norm',0)/max(excursion.get('t48',{}).get('norm',1e-9),1e-9):.3f}")
    print(f"  excursion_t48/t96 ratio: {excursion.get('t48',{}).get('norm',0)/max(excursion.get('t96',{}).get('norm',1e-9),1e-9):.3f}")

# Beta analysis
print(f"\n=== BETA (divergence) ===")
divs = [(e["step"], e["div"]) for e in entries if "div" in e and "val_ce" not in e]
if divs:
    # Calculate rolling mean
    window = 200
    for i in range(0, len(divs), max(1, len(divs) // 10)):
        s, d = divs[i]
        # rolling mean around this point
        start = max(0, i - window)
        end = min(len(divs), i + window)
        mean_d = sum(x[1] for x in divs[start:end]) / (end - start)
        print(f"  step={s:5d}  div={d:.4f}  rolling_mean={mean_d:.4f}")
