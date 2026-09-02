import json

log = []
for line in open("C:/Users/Anand/Desktop/Projects/Zeus/experiments/p5_night5b/train.log", encoding="utf-8"):
    try: log.append(json.loads(line))
    except: pass

ces = [(e["step"], e["ce"]) for e in log if "ce" in e and "val_ce" not in e]
surps = [(e["step"], e["surp"]) for e in log if "surp" in e]
pers = [(e["step"], e["persist"]) for e in log if "persist" in e]
arps = [(e["step"], e["action_remember_prob"]) for e in log if "action_remember_prob" in e]
currs = [(e["step"], e.get("curriculum_prob", 1.0)) for e in log if "curriculum_prob" in e]
hcmr = [(e["step"], e.get("hcm_reads", 0), e.get("hcm_writes", 0)) for e in log if "hcm_reads" in e and "ce" in e]

# Phase analysis
warmup = [(s, c) for (s, c) in ces if s <= 500]
cooldown = [(s, c) for (s, c) in ces if 500 < s <= 1500]
post = [(s, c) for (s, c) in ces if s > 1500]

print("=== PHASE ANALYSIS ===")
if warmup:
    vals = [c for _, c in warmup]
    print(f"Warmup (0-500):    CE min={min(vals):.4f}  max={max(vals):.4f}  mean={sum(vals)/len(vals):.4f}  n={len(vals)}")
if cooldown:
    vals = [c for _, c in cooldown]
    print(f"Cooldown (500-1500): CE min={min(vals):.4f}  max={max(vals):.4f}  mean={sum(vals)/len(vals):.4f}  n={len(vals)}")
if post:
    vals = [c for _, c in post]
    print(f"Post-curriculum (>1500): CE min={min(vals):.4f}  max={max(vals):.4f}  mean={sum(vals)/len(vals):.4f}  n={len(vals)}")

print()
print("=== CE ===")
for s, c in ces:
    print(f"  step={s:>4}  ce={c:.4f}")

print()
print("=== PERSIST ===")
for s, p in pers:
    print(f"  step={s:>4}  persist={p:.4f}")

print()
print("=== ACTION PROB ===")
for s, a in arps:
    print(f"  step={s:>4}  action_prob={a:.6f}")

print()
print("=== CURRICULUM ===")
for s, c in currs:
    print(f"  step={s:>4}  curriculum={c:.3f}")

print()
print("=== HCM READS/WRITES ===")
for s, r, w in hcmr:
    print(f"  step={s:>4}  reads={r:>4}  writes={w:>4}")

# Val CE
vals = [e for e in log if "val_ce_nats" in e]
print()
print("=== VAL CE ===")
for v in vals:
    print(f"  step={v['step']:>4}  val_ce={v['val_ce_nats']:.4f}  floor_L1=7.10  floor_L2=4.47")
