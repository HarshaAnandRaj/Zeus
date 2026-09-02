import json, sys, numpy as np, torch, torch.nn.functional as F
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core.model as m
from training.stage1b import readout_logits, rollout_loss

device = "cuda"
knobs = json.loads(open("runs/broca_pretrain/run_config.json").read())
cfg = m.ZeusConfig()
for k in ("readout_layers", "readout_ffn_mult", "readout_heads", "ctx_anchor", "cross_attn"):
    setattr(cfg, k, knobs.get(k, getattr(cfg, k)))
net = m.ZeusCore(cfg).to(device)
ro, emb = net.readout, net.embed
ro.load_state_dict(torch.load("runs/broca_pretrain/readout.pt", map_location=device, weights_only=True), strict=False)
emb.load_state_dict(torch.load("runs/broca_pretrain/emb.pt", map_location=device, weights_only=True), strict=False)

ids = np.load("corpus/data/val_ids.npy").astype(np.int64)[:40000]
R, G = 16, 44
B = 16
base = np.random.RandomState(0).randint(0, 100, size=B)
rows = np.stack([ids[b: b + R + G + 1] for b in base])
ids_all = torch.from_numpy(rows).to("cuda")

ro.train(); emb.train()
loss = rollout_loss(ro, emb, ids_all, None, R, G, cfg, "cuda", True)
loss.backward()
print("loss", loss.item())
print("max alloc MB", torch.cuda.max_memory_allocated() / 1e6)
print("reserved MB", torch.cuda.memory_reserved() / 1e6)
import re
s = str(torch.cuda.memory_summary()).split("\n")[:30]
for line in s:
    if "alloc" in line or "reserved" in line or "CUR" in line or "PEAK" in line:
        print(line)