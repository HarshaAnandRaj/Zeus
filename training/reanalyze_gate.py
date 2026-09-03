"""Apply the new strict free_run_gate to existing ceiling report replies
(re-analysis, no GPU). Writes reports/free_run_gate_reanalysis.json."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from training.free_run_gate import analyze, passes_gate, aggregate, GateResult

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "zeus_sandbox" / "universe" / "reports"
OUT = {}


def _median(xs):
    v = [48 if x is None else x for x in xs]
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n else None

for fname in ["decode_robust_ceiling.json", "decode_robust_ceiling_hcm_on.json"]:
    d = json.load(open(REPORTS / fname, encoding="utf-8"))
    label = "primary_hcm_off" if "hcm_on" not in fname else "secondary_hcm_on"
    byck = {}
    for ck, ckr in d["checkpoints"].items():
        res = {}
        for mode in sorted({r["mode"] for r in ckr["replies"]}):
            reps = [r for r in ckr["replies"] if r["mode"] == mode]
            grs = []
            for r in reps:
                m = analyze(r["raw"])
                reasons = passes_gate(m)
                grs.append(GateResult(m, reasons))
            agg = aggregate(grs)
            # also compare old word-gate vs new strict
            old_raw = {r["seed"]: r for r in ckr["replies"] if r["mode"] == mode}
            res[mode] = {
                "new_strict_pass": agg["pass_frac"],
                "n": agg["n"], "k": agg["k"],
                "ci": agg["ci"], "reasons": agg["reasons"],
                "med_onset": float(_median([g.metrics["rep_onset"] for g in grs])),
            }
        byck[ck] = res
    OUT[label] = byck

with open(REPORTS / "free_run_gate_reanalysis.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2)

for label, byck in OUT.items():
    print(f"===== {label} (new STRICT free_run_gate pass fraction / 80) =====")
    for ck, modes in byck.items():
        print(f"  {ck}:")
        for mode, r in modes.items():
            print(f"    {mode:>8}: {r['k']:>2}/80 ({r['new_strict_pass']:.3f})"
                  f"  ci={r['ci'][0]:.2f}-{r['ci'][1]:.2f}"
                  f"  med_onset={r['med_onset']:.1f}"
                  f"  reasons={r['reasons']}")
