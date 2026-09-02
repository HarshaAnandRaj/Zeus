import sys, pathlib, shutil
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
from tokenizers import Tokenizer
from core.hcm import text_is_clean

MEM = pathlib.Path("zeus_sandbox/universe/sessions/sPONR01/hcm.pt")
tok = Tokenizer.from_file("corpus/data/tokenizer/bpe_8192.json")

sd = torch.load(MEM, map_location="cpu", weights_only=True)["hcm"]
n = sd["n_patterns"]
ctx = sd["context_tokens"][:n]

def decode_row(r):
    ids = [int(t) for t in r.tolist() if int(t) != 0]
    return tok.decode(ids) if ids else ""

keep = [i for i in range(n) if text_is_clean(decode_row(ctx[i]))
        and len([int(t) for t in ctx[i].tolist() if int(t) != 0]) >= 4]
print(f"patterns: {n} -> keep {len(keep)} (drop {n - len(keep)})")

for key in ("patterns", "target_embed", "region_id", "strengths", "birth_step",
            "usage", "target_token", "context_tokens", "utility"):
    assert key in sd, key
    if sd[key].shape[0] == n:
        sd[key] = sd[key][keep].clone()
    elif sd[key].shape[0] == sd["context_tokens"].shape[0]:
        pass  # region_patterns-class buffers keep full size
sd["n_patterns"] = len(keep)
sd["step_count"] = int(sd.get("step_count", 0))
sd["total_writes"] = int(sd.get("total_writes", 0))
sd["total_recalls"] = int(sd.get("total_recalls", 0))
sd["recall_hits"] = int(sd.get("recall_hits", 0))
sd["action_writes"] = int(sd.get("action_writes", 0))
sd["auto_writes"] = int(sd.get("auto_writes", 0))

backup = MEM.with_name("hcm.pruned_backup.pt")
shutil.copy2(MEM, backup)
print("backup saved:", backup)

torch.save({"hcm": sd}, MEM)
print("pruned HCM saved:", MEM)
for i in range(min(5, len(keep))):
    print("  keeps:", repr(decode_row(ctx[keep[i]])[:70]))