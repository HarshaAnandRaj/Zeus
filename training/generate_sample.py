import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import core.model as m

cfg = m.ZeusConfig()
net = m.ZeusCore(cfg)
ckpts = sorted((ROOT / "runs" / "proof_lm_lm").glob("zeus_step*.pt"))
ckpt = ckpts[-1]
print(f"loading {ckpt.name}")
payload = torch.load(ckpt, map_location="cpu", weights_only=False)
net.load_state_dict(payload["model"])
net.eval()

prompts = [
    "The",
    "He said that",
    "In the beginning",
    "We believe that the",
]

for temp in (0.0, 0.8):
    print(f"\n===== temperature={temp} =====")
    for p in prompts:
        pid = net.encode(p)
        nxt = net.reply(pid, max_tokens=80, temperature=temp)
        text = net.decode(nxt)
        print(f"\nPROMPT: {p}")
        print(f"GEN   : {text}")
