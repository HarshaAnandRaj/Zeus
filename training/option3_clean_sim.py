import pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from core.model import ZeusCore, ZeusConfig
from core.hcm import HCM


def load(ckpt, device):
    p = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = ZeusCore(ZeusConfig(**p["config"]), tokenizer_path=p.get("tokenizer")).to(device)
    m.load_state_dict(p["model"])
    m.eval()
    return m


def seed_from_state(hcm, model, phrase):
    """Store a memory whose pattern is a REAL free-run S-state, tagged with clean
    prose context_tokens (the remembered text). Recall fires by S similarity."""
    model.reset_state(noise=0.6)
    g = torch.Generator(device=model.S.device); g.manual_seed(3)
    for _ in range(48):
        model.step(None)
    pat = model.S.detach().clone()
    ids = list(model.encode(phrase))
    ctx = torch.zeros(hcm.context_len, dtype=torch.long, device=hcm.device)
    ctx[-len(ids):] = torch.tensor(ids[: hcm.context_len], device=hcm.device)
    i = hcm.n_patterns
    hcm.patterns[i] = pat
    hcm.target_embed[i] = model.embed(torch.tensor(ids[-1], device=hcm.device))
    hcm.target_token[i] = ids[-1]
    hcm.context_tokens[i] = ctx
    hcm.strengths[i] = 5.0
    hcm.birth_step[i] = 0
    hcm.usage[i] = 0
    hcm.region_id[i] = hcm._assign_region(pat)
    hcm.n_patterns += 1
    return pat


def reply(model, hcm, prompt, seed=11, max_tokens=40, temp=0.8):
    g = torch.Generator(device=model.S.device); g.manual_seed(seed)
    with torch.no_grad():
        model.pad_window(model.encode(prompt)[0])
        model.ingest(list(model.encode(prompt)))
        got = hcm.read(model.S.detach())
        memorized = None
        if got is not None and got[0] is not None:
            cids = [int(t) for t in got[4].tolist() if int(t) != 0]
            if cids:
                memorized = model.decode(cids)
                model.ingest(cids)  # Option 3: prepend remembered text
        logits = model.observe()
        out = []
        for _ in range(max_tokens):
            probs = torch.softmax(logits / temp, dim=-1)
            if out:
                u = torch.tensor(sorted(set(out)), device=probs.device)
                probs[u] = probs[u] / 1.15
            probs = probs / probs.sum()
            nxt = torch.multinomial(probs, 1, generator=g).item()
            out.append(nxt)
            logits, _ = model.step(nxt)
    return model.decode(out), memorized


def main():
    m = load("runs/broca_voice/milestone.pt", "cuda")
    m.deploy_self_source = True
    hcm = HCM(dim=m.cfg.dim, max_patterns=64, recall_threshold=0.3, device="cuda")
    mems = [
        "the old librarian climbed the stairs to the dusty stacks every evening",
        "a small boy sailed his paper boat down the rain filled gutter",
        "the blacksmith hammered iron until sparks flew into the night sky",
    ]
    for phr in mems:
        seed_from_state(hcm, m, phr)
    # stay on the last free-run state (cos to the 3rd stored pattern ~ 1.0)
    s0 = m.S.detach().clone()
    for p in ["hello", "tell me something", "what do you remember"]:
        m.S.copy_(s0)
        r_on, mem = reply(m, hcm, p)
        m.S.copy_(s0)
        r_off, _ = reply(m, hcm, p)
        print(f"\nPROMPT {p!r}")
        print(f"  recalled: {mem!r}")
        print(f"  [prepend] {r_on[:140]!r}")
        print(f"  [noPrpnd] {r_off[:140]!r}")


if __name__ == "__main__":
    main()