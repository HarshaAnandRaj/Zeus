import pathlib, sys, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.argv = ["zsession.py", "--mode", "interact"]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import zsession
from tokenizers import Tokenizer

tok = Tokenizer.from_file(str(ROOT / "corpus/data/tokenizer/bpe_8192.json"))

m, _ = zsession.load_safe(str(ROOT / "zeus_sandbox/universe/shadow/milestone.pt"))
m = m.to(zsession.DEVICE); m.eval(); m.deploy_self_source = True

full = ("the old librarian climbed the stairs of the west wing every evening carrying a "
        "lantern and a stack of dusty ledgers, and as the months went by he learned the ")

tids = [int(x) for x in tok.encode(full).ids]
print("prefix len:", len(tids))

m.ingest([20])          # seed recent so pad_window has a key
ids, _ = zsession.reply_ids(m, None, full, max_tokens=36, temperature=0.72, recall=False)
print("CONTINUATION:", m.decode(ids)[:160])