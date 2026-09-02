import pathlib, sys, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.argv = ["zsession.py", "--mode", "interact"]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import zsession
from core.hcm import text_is_clean

torch.manual_seed(7)
m = zsession.load_safe(str(ROOT / "zeus_sandbox/universe/shadow/milestone.pt"))[0]
m = m.to(zsession.DEVICE)
m.eval()
hcm = m.hcm
print("before patterns:", hcm.n_patterns, "context_len:", hcm.context_len)

# write a clean passage through the real commit path with the real stream
m.reset_state(noise=0.6)
passage = ("the fisherman rowed out at dawn and cast his net "
           "into the grey calm water before the sunrise")
ids = list(m.encode(passage))
for i in ids:
    m.step(i)
recent = ids
n0 = hcm.n_patterns
zsession.commit_memory(m, hcm, ids, recent_tokens=recent)
print("wrote pattern:", hcm.n_patterns > n0)
if hcm.n_patterns > n0:
    stored = hcm.context_tokens[hcm.n_patterns - 1]
    nz = [int(t) for t in stored.tolist() if int(t) != 0]
    print("stored ctx clean:", text_is_clean(m.decode(nz)),
          "| ctx:", m.decode(nz)[:90])

# junk-gate: write must be rejected for markup
junk_ids = list(m.encode("id=||33||\n|1994 rowspan=4 Kashiwa Reysol"))
n1 = hcm.n_patterns
zsession.commit_memory(m, hcm, junk_ids, recent_tokens=junk_ids)
print("junk write rejected:", hcm.n_patterns == n1)

# live replay still works
for p in ["hello"]:
    m.reset_state(noise=0.5)
    out, recalls = zsession.reply_ids(m, hcm, p, max_tokens=24, temperature=0.72)
    print(f"live reply ({recalls} recalls):", m.decode(out)[:80])