import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core.hcm import HCM
import torch

h = HCM(768, max_patterns=512, n_clusters=32, min_age=0)
for i in range(100):
    s = torch.randn(768)
    h.write(s, 2.0)
snap = h.snapshot()
print(f"patterns={h.n_patterns} regions={snap['n_regions']}")
dist = snap["region_distribution"]
print(f"region distribution: min={min(dist.values())} max={max(dist.values())} unique={len(dist)}")
n = h.consolidate(torch.randn(768), 0.15)
print(f"pruned with threshold=0.15: {n}")
# State dict roundtrip
sd = h.state_dict()
h2 = HCM(768, max_patterns=512, n_clusters=32, min_age=0)
h2.load_state_dict(sd)
print(f"roundtrip: patterns={h2.n_patterns}")
print("OK")
