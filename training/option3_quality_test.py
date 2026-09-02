import pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from core.model import ZeusCore, ZeusConfig
from core.hcm import HCM, text_is_clean


def load(ckpt, device):
    p = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = ZeusCore(ZeusConfig(**p["config"]), tokenizer_path=p.get("tokenizer")).to(device)
    m.load_state_dict(p["model"])
    m.eval()
    return m


def main():
    m = load("runs/broca_voice/milestone.pt", "cuda")
    m.deploy_self_source = True
    hcm = HCM(dim=m.cfg.dim, max_patterns=64, recall_threshold=0.3,
              context_len=30, device="cuda")

    # --- write a CLEAN passage via the new gated path (real token stream) ---
    passage = ("the old librarian climbed the stairs to the dusty stacks "
               "every evening carrying a lantern")
    pid = list(m.encode(passage))
    clean = text_is_clean(m.decode(pid), min_chars=hcm.write_min_chars)
    print(f"gate clean={clean} context_len={hcm.context_len}")

    m.reset_state(noise=0.6)
    for _ in range(48):
        m.step(None)
    pat = m.S.detach().clone()
    tgt = int(pid[-1])
    tgt_emb = m.embed(torch.tensor(tgt, device=m.S.device))
    wrote = hcm.write(pat, 2.5, from_action=True, target_token=tgt,
                      target_embed=tgt_emb, recent_tokens=pid,
                      quality_ok=text_is_clean(m.decode(pid)))
    print(f"wrote={wrote} n_patterns={hcm.n_patterns}")
    hcm.step_count = 100   # age the pattern so recall is permitted (min_age=10)

    # --- recall on the same/neighbor state ---
    with torch.no_grad():
        got = hcm.read(m.S.detach())
    if got is not None and got[0] is not None:
        ctx_ids = [int(t) for t in got[4].tolist() if int(t) != 0]
        print(f"recalled ctx ({len(ctx_ids)} tokens): {m.decode(ctx_ids)!r}")
    else:
        print("recall: none (S drifted)")

    # --- generate, Option 3: prepend remembered passage into token stream ---
    m.reset_state(noise=0.6)
    g = torch.Generator(device=m.S.device); g.manual_seed(11)
    with torch.no_grad():
        m.pad_window(pid[0])
        m.ingest(pid)                    # memory passage IS the recent stream
        logits = m.observe()
        out = []
        for _ in range(48):
            probs = torch.softmax(logits / 0.8, dim=-1)
            if out:
                u = torch.tensor(sorted(set(out)), device=probs.device)
                probs[u] = probs[u] / 1.15
            probs = probs / probs.sum()
            nxt = torch.multinomial(probs, 1, generator=g).item()
            out.append(nxt)
            logits, _ = m.step(nxt)
    print("\nCONTINUATION of the remembered passage:")
    print(" ", m.decode(out)[:220])


if __name__ == "__main__":
    main()