import json
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[1]
log = root / sys.argv[1] / "train.log"
rows = []
for line in log.read_text(encoding="utf-8").splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    if "ce" in r:
        rows.append((r["step"], r["ce"]))

buckets = [(0, 3000), (3000, 6000), (6000, 9000), (9000, 12000), (12000, 16000)]
header = f"{'steps':>12} {'n':>4} {'mean':>6} {'min':>6} {'pct<6.8':>8} {'pct<7.1':>8}"
print(header)
for lo, hi in buckets:
    xs = [c for s, c in rows if lo <= s < hi]
    if not xs:
        continue
    f68 = sum(1 for c in xs if c < 6.8) / len(xs) * 100
    f71 = sum(1 for c in xs if c < 7.1) / len(xs) * 100
    print(f"{str(lo)+'-'+str(hi):>12} {len(xs):>4} {sum(xs)/len(xs):>6.2f} {min(xs):>6.2f} {f68:>7.0f}% {f71:>7.0f}%")
