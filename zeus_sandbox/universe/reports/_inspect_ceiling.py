import json, pathlib, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open('zeus_sandbox/universe/reports/decode_robust_ceiling.json', encoding='utf-8'))
ck = d['checkpoints']['stage1c']
by = {}
for r in ck['replies']:
    by.setdefault(r['mode'], []).append(r)
out = []
for mode, lst in by.items():
    out.append(f"===== {mode} =====")
    for r in lst:
        out.append(f"  [s{r['seed']} {r['prompt']!r}] leg={r['leg']:.2f} loop={r['loop']:.2f} onset={r['onset']}")
        out.append(f"    RAW: {r['raw'][:140]!r}")
pathlib.Path('zeus_sandbox/universe/reports/ceiling_inspect.txt').write_text(
    "\n".join(out), encoding='utf-8')
print("wrote ceiling_inspect.txt")
