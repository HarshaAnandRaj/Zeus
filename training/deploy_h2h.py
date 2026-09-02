import pathlib, sys, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.argv = ["zsession.py", "--mode", "interact"]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import zsession

torch.manual_seed(11)
prompts = ["hello", "tell me about the past", "once upon a time",
           "what happened", "the river flowed quietly"]

def run(ckpt, self_source):
    m, _ = zsession.load_safe(str(ROOT / ckpt))
    m = m.to(zsession.DEVICE); m.eval()
    m.deploy_self_source = self_source
    h = m.hcm
    psd = torch.load(str(ROOT / "zeus_sandbox/universe/sessions/sPONR01/hcm.pt"),
                     map_location="cpu", weights_only=True)
    sd = psd.get("hcm") if isinstance(psd, dict) and "hcm" in psd else psd
    h.load_state_dict(sd)
    return m, h

print("===== BROCA voice (self-source) =====")
mb, hb = run("zeus_sandbox/universe/shadow/milestone.pt", True)
for p in prompts:
    mb.reset_state(0.5)
    ids, _ = zsession.reply_ids(mb, hb, p, max_tokens=32, temperature=0.72)
    print(f"{p!r}: {mb.decode(ids)[:110]}")

print("\n===== OLD voice (val 3.16) =====")
mo, ho = run("zeus_sandbox/universe/shadow/milestone.pt.pre-broca", False)
for p in prompts:
    mo.reset_state(0.5)
    ids, _ = zsession.reply_ids(mo, ho, p, max_tokens=32, temperature=0.72)
    print(f"{p!r}: {mo.decode(ids)[:110]}")