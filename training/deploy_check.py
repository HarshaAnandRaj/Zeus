import pathlib, sys, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.argv = ["zsession.py", "--mode", "interact"]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import zsession

torch.manual_seed(7)
model, ckpt_step = zsession.load_safe(str(ROOT / "zeus_sandbox/universe/shadow/milestone.pt"))
model = model.to(zsession.DEVICE)
model.eval()
model.deploy_self_source = True   # mirrors config voice_self_source
hcm = model.hcm
# restore pruned session HCM like zsession.main does
psd = torch.load(str(ROOT / "zeus_sandbox/universe/sessions/sPONR01/hcm.pt"),
                 map_location="cpu", weights_only=True)
sd = psd.get("hcm") if isinstance(psd, dict) and "hcm" in psd else psd
hcm.load_state_dict(sd)
print(f"deployed broca voice: {ckpt_step} step | cross_attn={model.cfg.cross_attn} "
      f"| self_source={model.deploy_self_source} | patterns={hcm.n_patterns}")

for p in ["hello", "what happened before", "tell me about the past",
          "i remember the old times", "once upon a time"]:
    model.reset_state(0.5)
    ids, recalls = zsession.reply_ids(model, hcm, p, max_tokens=40, temperature=0.72)
    mem = getattr(model, "last_recalled_ctx", None)
    mctx = None if mem is None else model.decode([int(t) for t in mem.tolist() if int(t) != 0])
    print(f"\nPROMPT {p!r} recalls={recalls}")
    if mctx:
        print(f"  memory: {mctx!r}")
    print("  REPLY:", model.decode(ids)[:160])