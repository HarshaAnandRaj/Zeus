import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch, torch.nn.functional as F
import core.model as m
from training.train import eval_health

cfg = m.ZeusConfig()
net = m.ZeusCore(cfg)
ckpt = sorted((ROOT / "runs" / "proof_lm_lm").glob("zeus_step*.pt"))[-1]
net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
net.eval()

print("readout coupling: s_scale=%.2f gate_gain=%.2f ctx_gain=%.2f"
      % (net.readout.s_scale, net.readout.gate_gain, net.readout.ctx_gain))

# 1) Is the transformer OUTPUT coupled to S? Run free-run health with transformer ON vs OFF.
h_on = eval_health(net, steps=200)
net.readout.ctx_gain = 0.0
h_off = eval_health(net, steps=200)
print("\n[transformer OUTPUT -> S] health identical with ctx_gain ON vs OFF?")
for k in ["state_regime", "nu_state", "d_w_state", "rho_exact"]:
    print(f"  {k}: ON={h_on[k]}  OFF={h_off[k]}")

# 2) Reverse gradient: how much does the LM loss pull on S / the dynamics weights?
net.readout.ctx_gain = 2.0
net.train()
tok = torch.randint(3, cfg.vocab, (1,)).item()
net.S.requires_grad_(True)
logits, _ = net.step(tok)
tgt = torch.randint(3, cfg.vocab, (1,)).item()
lm_loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tgt]))
lm_loss.backward()
s_grad = net.S.grad.norm().item() if net.S.grad is not None else 0.0
rec_grad = net.rec.weight.grad.norm().item() if net.rec.weight.grad is not None else 0.0
print("\n[LM loss -> self] reverse-coupling strength (after one step):")
print(f"  ||grad LM_loss / dS_state||      = {s_grad:.4f}   (vs ||S||~{net.S.norm().item():.1f})")
print(f"  ||grad LM_loss / drec.weight||    = {rec_grad:.4f}")
print("  (tiny => transformer only weakly observes S; it does NOT write into S)")
