import json

ROOT = "C:/Users/Anand/Desktop/Projects/Zeus"

def load_log(path):
    entries = []
    for line in open(path, encoding="utf-8"):
        try: entries.append(json.loads(line))
        except: pass
    return entries

n5 = load_log(f"{ROOT}/experiments/p5_night5/train.log")
n5b = load_log(f"{ROOT}/experiments/p5_night5b/train.log")

print("=== NIGHT5 (w=0.05) vs NIGHT5b (w=0.007) ===")
print()

n5_ces = {e["step"]: e for e in n5 if "ce" in e and "val_ce" not in e}
n5b_ces = {e["step"]: e for e in n5b if "ce" in e and "val_ce" not in e}

for step in sorted(set(n5_ces.keys()) | set(n5b_ces.keys()))[:12]:
    e5 = n5_ces.get(step, {})
    e5b = n5b_ces.get(step, {})
    ce5 = f'{e5.get("ce", 0):.2f}' if e5 else "-"
    ce5b = f'{e5b.get("ce", 0):.2f}' if e5b else "-"
    p5 = f'{e5.get("persist", 0):.3f}' if e5 and "persist" in e5 else "-"
    p5b = f'{e5b.get("persist", 0):.3f}' if e5b and "persist" in e5b else "-"
    a5 = f'{e5.get("action_remember_prob", 0):.5f}' if e5 and "action_remember_prob" in e5 else "-"
    a5b = f'{e5b.get("action_remember_prob", 0):.5f}' if e5b and "action_remember_prob" in e5b else "-"
    print(f"  step={step:>4}  CE: n5={ce5:>7} n5b={ce5b:>7}  |  persist: n5={p5:>7} n5b={p5b:>7}  |  aprob: n5={a5:>9} n5b={a5b:>9}")

# HCM comparison
n5_250 = [e for e in n5 if e.get("step") == 250 and "hcm" in e]
n5b_250 = [e for e in n5b if e.get("step") == 250 and "hcm" in e]
if n5_250 and n5b_250:
    h5 = n5_250[0]["hcm"]
    h5b = n5b_250[0]["hcm"]
    print()
    print("=== HCM at step 250 ===")
    print(f"  Night5:  writes={h5['total_writes']:>6}  action={h5['action_writes']:>6}  auto={h5['auto_writes']:>6}  recalls={h5['total_recalls']:>6}  str={h5['avg_strength']:.1f}")
    print(f"  Night5b: writes={h5b['total_writes']:>6}  action={h5b['action_writes']:>6}  auto={h5b['auto_writes']:>6}  recalls={h5b['total_recalls']:>6}  str={h5b['avg_strength']:.1f}")

# Val CE comparison
n5_vals = [e for e in n5 if "val_ce_nats" in e]
n5b_vals = [e for e in n5b if "val_ce_nats" in e]
print()
print("=== VAL CE ===")
for v in n5_vals:
    print(f"  Night5  step={v['step']:>4}  val_ce={v['val_ce_nats']:.4f}")
for v in n5b_vals:
    print(f"  Night5b step={v['step']:>4}  val_ce={v['val_ce_nats']:.4f}")
