import pathlib, sys, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.argv = ["zsession.py", "--mode", "interact"]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import zsession

m, _ = zsession.load_safe(str(ROOT / "zeus_sandbox/universe/shadow/milestone.pt"))
m = m.to(zsession.DEVICE); m.eval(); m.deploy_self_source = True

for p in ["hello", "the river flowed quietly", "tell me about the past",
          "once upon a time", "the old librarian climbed the stairs"]:
    m.reset_state(0.5)
    ids, _ = zsession.reply_ids(m, None, p, max_tokens=36, temperature=0.72, recall=False)
    print(f"{p!r} => {m.decode(ids)[:110]}")