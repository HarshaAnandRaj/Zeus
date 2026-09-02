import pathlib, sys, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.argv = ["zsession.py", "--mode", "interact"]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import zsession

m, step = zsession.load_safe(str(ROOT / "zeus_sandbox/universe/shadow/milestone.pt"))
h = m.hcm
psd = torch.load(str(ROOT / "zeus_sandbox/universe/sessions/sPONR01/hcm.pt"),
                 map_location="cpu", weights_only=True)
sd = psd.get("hcm") if isinstance(psd, dict) and "hcm" in psd else psd
h.load_state_dict(sd)
print(f"restored live voice step={step} | self_source={m.deploy_self_source} "
      f"| patterns={h.n_patterns} | cross_attn={getattr(m.cfg,'cross_attn',None)}")

for p in ["hello", "tell me about the past", "what happened"]:
    m.reset_state(0.5)
    ids, recalls = zsession.reply_ids(m, h, p, max_tokens=32, temperature=0.72)
    print(f"{p!r} recalls={recalls} => {m.decode(ids)[:100]}")