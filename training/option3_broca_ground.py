import pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from core.model import ZeusCore, ZeusConfig


def load(ckpt, device):
    p = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = ZeusCore(ZeusConfig(**p["config"]), tokenizer_path=p.get("tokenizer")).to(device)
    m.load_state_dict(p["model"])
    m.eval()
    return m


def reply(model, prompt, prepend_phrase=None, seed=11, max_tokens=50, temp=0.8):
    g = torch.Generator(device=model.S.device); g.manual_seed(seed)
    with torch.no_grad():
        pid = list(model.encode(prompt))
        model.pad_window(pid[0])
        model.ingest(pid)
        if prepend_phrase:
            model.ingest(list(model.encode(prepend_phrase)))
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
    return model.decode(out)


def main():
    m = load("runs/broca_voice/milestone.pt", "cuda")
    m.deploy_self_source = True
    tests = [
        ("hello", "the old librarian climbed the stairs to the dusty stacks every evening"),
        ("tell me a story", "a small boy sailed his paper boat down the rain filled gutter"),
        ("i was thinking", "the blacksmith hammered iron until sparks flew into the night sky"),
    ]
    for p, mem in tests:
        r_on = reply(m, p, prepend_phrase=mem)
        m.reset_state(noise=0.6)
        r_off = reply(m, p, prepend_phrase=None) if False else reply(m, p, None)
        m.reset_state(noise=0.6)
        print(f"\nPROMPT {p!r} | MEMORY {mem!r}")
        print(f"  [prepend] {r_on[:150]!r}")
        print(f"  [noPrpnd] {r_off[:150]!r}")


if __name__ == "__main__":
    main()