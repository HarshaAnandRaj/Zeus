import pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from core.model import ZeusCore, ZeusConfig
from core.hcm import HCM


def load_model(ckpt_path, device):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ZeusConfig(**payload["config"])
    m = ZeusCore(cfg, tokenizer_path=payload.get("tokenizer")).to(device)
    m.load_state_dict(payload["model"])
    m.eval()
    return m


def seed_memory(hcm, model, phrase):
    """Store phrase as a memory pattern: pattern = embed-mean (a state-like key),
    target_embed = last-token embed, context_tokens = phrase tokens."""
    ids = list(model.encode(phrase))
    with torch.no_grad():
        e = model.embed(torch.tensor(ids, device=hcm.device, dtype=torch.long))
        pat = e.mean(0).detach()          # state-like key
        tgt = e[-1].detach()              # what the voice should predict next
    ctx = torch.zeros(hcm.context_len, dtype=torch.long, device=hcm.device)
    ctx[-len(ids):] = torch.tensor(ids[: hcm.context_len], device=hcm.device)
    hcm.patterns[hcm.n_patterns] = pat
    hcm.target_embed[hcm.n_patterns] = tgt
    hcm.target_token[hcm.n_patterns] = ids[-1]
    hcm.context_tokens[hcm.n_patterns] = ctx
    hcm.strengths[hcm.n_patterns] = 5.0
    hcm.birth_step[hcm.n_patterns] = 0
    hcm.usage[hcm.n_patterns] = 0
    hcm.region_id[hcm.n_patterns] = hcm._assign_region(pat)
    hcm.n_patterns += 1
    return ids


def reply(model, hcm, prompt, prepend=True, seed=7, max_tokens=48):
    g = torch.Generator(device=model.S.device); g.manual_seed(seed)
    with torch.no_grad():
        tokens = list(model.encode(prompt))
        model.ingest(tokens)
        got = hcm.read(model.S.detach())
        ctx_txt = None
        if got is not None and got[0] is not None:
            cids = [int(t) for t in got[4].tolist() if int(t) != 0]
            if cids:
                ctx_txt = model.decode(cids)
                if prepend:
                    model.ingest(cids)
        logits = model.observe()
        out = []
        for _ in range(max_tokens):
            probs = torch.softmax(logits / 0.72, dim=-1)
            if out:
                uniq = torch.tensor(sorted(set(out)), device=probs.device)
                probs[uniq] = probs[uniq] / 1.15
            probs = probs / probs.sum()
            nxt = torch.multinomial(probs, 1, generator=g).item()
            out.append(nxt)
            logits, _ = model.step(nxt)
    return model.decode(out), ctx_txt


def main():
    model = load_model("zeus_sandbox/universe/shadow/milestone.pt", "cuda")
    hcm = HCM(dim=model.cfg.dim, max_patterns=64, recall_threshold=0.05, device="cuda")
    mems = [
        "the old librarian climbed the stairs to the dusty stacks every evening",
        "the blacksmith hammered iron until the sparks flew into the night sky",
        "a small boy sailed his paper boat down the rain filled gutter",
    ]
    print("seeding clean memories:")
    for phr in mems:
        seed_memory(hcm, model, phr)
        print(f"  {phr!r}")

    for p in ["hello", "what do you remember", "tell me", "the boy"]:
        model.reset_state(noise=0.6)
        r_on, ctx = reply(model, hcm, p, prepend=True)
        model.reset_state(noise=0.6)
        r_off, ctx2 = reply(model, hcm, p, prepend=False)
        print(f"\nPROMPT: {p!r}")
        if ctx is not None:
            print(f"  recalled: {ctx!r}")
        print(f"  [grounded  ] {r_on!r}")
        print(f"  [ungrounded] {r_off!r}")


if __name__ == "__main__":
    main()