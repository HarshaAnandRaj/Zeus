import pathlib, sys, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.argv = ["zsession.py", "--mode", "interact"]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import zsession

torch.manual_seed(7)
m = zsession.load_safe(str(ROOT / "zeus_sandbox/universe/shadow/milestone.pt"))[0]
m = m.to(zsession.DEVICE)
m.eval()
hcm = m.hcm
print("patterns:", hcm.n_patterns)
for p in ["hello", "what happened", "tell me about the past"]:
    m.reset_state(noise=0.5)
    ids, recalls = zsession.reply_ids(m, hcm, p, max_tokens=36, temperature=0.72)
    mem = getattr(m, "last_recalled_ctx", None)
    mctx = None if mem is None else m.decode([int(t) for t in mem.tolist() if int(t) != 0])
    print(f"\nPROMPT {p!r} recalls={recalls} ctx={mctx!r}")
    print("REPLY:", m.decode(ids))