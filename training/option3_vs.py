import pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from core.model import ZeusCore, ZeusConfig
from core.hcm import HCM


def load(ckpt, device):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ZeusConfig(**payload["config"])
    m = ZeusCore(cfg, tokenizer_path=payload.get("tokenizer")).to(device)
    m.load_state_dict(payload["model"])
    m.eval()
    return m


def load_hcm(path, device, dim, threshold):
    hcm = HCM(dim=dim, max_patterns=512, recall_threshold=threshold, device=device)
    hcm.load_state_dict(torch.load(path, map_location=device, weights_only=True)["hcm"])
    return hcm


def reply(model, hcm, prompt, prepend=True, seed=7, max_tokens=48, temp=0.72):
    g = torch.Generator(device=model.S.device); g.manual_seed(seed)
    with torch.no_grad():
        model.ingest(list(model.encode(prompt)))
        recall_vec, ctx_txt = None, None
        got = hcm.read(model.S.detach())
        if got is not None and got[0] is not None:
            recall_vec = got[0].to(model.S.device)
            model.hcm_pending = recall_vec
            cids = [int(t) for t in got[4].tolist() if int(t) != 0]
            if cids:
                ctx_txt = model.decode(cids)
                if prepend:
                    model.ingest(cids)
        logits = model.observe()
        out = []
        for _ in range(max_tokens):
            probs = torch.softmax(logits / temp, dim=-1)
            if out:
                uniq = torch.tensor(sorted(set(out)), device=probs.device)
                probs[uniq] = probs[uniq] / 1.15
            probs = probs / probs.sum()
            nxt = torch.multinomial(probs, 1, generator=g).item()
            out.append(nxt)
            if recall_vec is not None:
                model.hcm_pending = recall_vec
            logits, _ = model.step(nxt)
    return model.decode(out), ctx_txt


def main():
    mem_path = "zeus_sandbox/universe/sessions/sPONR01/hcm.pt"
    prompts = ["hello", "what happened before", "tell me about the past",
               "the king returned", "i remember", "do you recall"]
    for label, ckpt, self_src in [
        ("CURRENT voice_v2", "zeus_sandbox/universe/shadow/milestone.pt", False),
        ("BROCA  self-source", "runs/broca_voice/milestone.pt", True),
    ]:
        m = load(ckpt, "cuda")
        m.deploy_self_source = self_src
        hcm = load_hcm(mem_path, "cuda", m.cfg.dim, threshold=0.12)
        print(f"\n===== {label} =====")
        for p in prompts:
            m.reset_state(noise=0.5)
            r_on, ctx = reply(m, hcm, p, prepend=True)
            m.reset_state(noise=0.5)
            r_off, _ = reply(m, hcm, p, prepend=False)
            print(f"\nPROMPT {p!r}")
            print(f"  recalled: {ctx!r}")
            print(f"  [prepend] {r_on[:110]!r}")
            print(f"  [noPrpnd] {r_off[:110]!r}")


if __name__ == "__main__":
    main()