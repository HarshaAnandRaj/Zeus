import json, sys, numpy as np, torch, torch.nn.functional as F
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core.model as m
import training.stage1b as s1
import torch.nn.functional as F_

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
R, G, B = 16, 44, 16
base = np.random.RandomState(0).randint(0, 100, size=B)
rows = np.stack([ids[b: b + R + G + 1] for b in base])
ids_all = torch.from_numpy(rows).to("cuda")
ro.train(); emb.train()
mb = lambda: round(torch.cuda.memory_allocated() / 1e6)
print("start MB", mb())
W = cfg.ctx_window
prefix = ids_all[:, :R]; pad_tok = prefix[:, 0]
layouts = torch.zeros((B, G, W), dtype=torch.long, device=device)
gen = []
for g in range(G):
    pad_cnt = W - R - g
    seq = torch.zeros((B, W), dtype=torch.long, device=device)
    if pad_cnt > 0:
        seq[:, :pad_cnt] = pad_tok.unsqueeze(1).expand(B, pad_cnt)
    seq[:, pad_cnt: pad_cnt + R] = prefix
    if g > 0:
        seq[:, pad_cnt + R: pad_cnt + R + g] = torch.stack(gen, dim=1)
    layouts[:, g] = seq
    logit_g = s1.readout_logits(ro, emb, seq, cfg, device, True, last_only=True)[:, 0].float()
    gen.append(logit_g.argmax(-1).detach())
    if g in (0, 10, 20, 30, 40):
        print("gen g", g, "MB", mb())
print("after gen loop MB", mb())
all_l = layouts.reshape(B * G, W)
tgt = ids_all[:, R: R + G].reshape(-1)
for st in range(0, all_l.shape[0], 256):
    lg = s1.readout_logits(ro, emb, all_l[st:st + 256], cfg, device, True, last_only=True)[:, 0].float()
    l = F_.cross_entropy(lg, tgt[st:st + 256])
    l.backward()
    print("chunk", st, "MB", mb())
print("final")