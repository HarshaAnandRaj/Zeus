import json, pathlib, sys
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
SANDBOX = ROOT / "zeus_sandbox"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SANDBOX))
import shims
from core.model import ZeusCore, ZeusConfig
from core.hcm import HCM


def load_model(ckpt_path, device):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ZeusConfig(**payload["config"])
    m = ZeusCore(cfg, tokenizer_path=payload.get("tokenizer")).to(device)
    m.load_state_dict(payload["model"])
    m.eval()
    return m


def load_hcm(path, device, model):
    hcm = HCM(dim=model.cfg.dim, max_patterns=512, recall_threshold=0.12, device=device)
    hcm.load_state_dict(torch.load(path, map_location=device, weights_only=True)["hcm"])
    return hcm


def reply(model, hcm, prompt, max_tokens=48, temperature=0.72, prepend=True, seed=7):
    g = torch.Generator(device=model.S.device); g.manual_seed(seed)
    with torch.no_grad():
        out, recalls, ctx_txt = [], 0, None
        tokens = list(model.encode(prompt))
        model.ingest(tokens)
        recall_vec = None
        got = hcm.read(model.S.detach())
        if got is not None and got[0] is not None:
            recall_vec = got[0].to(model.S.device)
            recalled_ctx = got[4]
            model.hcm_pending = recall_vec
            recalls = 1
            ctx_ids = [int(t) for t in recalled_ctx.tolist() if int(t) != 0]
            if ctx_ids:
                ctx_txt = model.decode(ctx_ids)
                if prepend:
                    model.ingest(ctx_ids)
        if getattr(model.cfg, "ctx_anchor", False) and recall_vec is not None:
            model.anchor_vec = recall_vec
        logits = model.observe()
        for _ in range(max_tokens):
            probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
            if out:
                uniq = torch.tensor(sorted(set(out)), device=probs.device)
                probs[uniq] = probs[uniq] / 1.15
            probs = probs / probs.sum()
            nxt = torch.multinomial(probs, 1, generator=g).item()
            out.append(nxt)
            if recall_vec is not None:
                model.hcm_pending = recall_vec
            logits, _ = model.step(nxt)
        model.anchor_vec = None
    return model.decode(out), recalls, ctx_txt


def main():
    device = "cuda"
    ckpt = str(SANDBOX / "universe" / "shadow" / "milestone.pt")
    mem_path = SANDBOX / "universe" / "sessions" / "sPONR01" / "hcm.pt"

    model = load_model(ckpt, device)
    hcm = load_hcm(mem_path, device, model)
    print(f"model: {pathlib.Path(ckpt).name}, patterns: {hcm.n_patterns}")

    prompts = ["hello", "what happened before", "tell me about the past", "the king returned", "i remember"]
    for p in prompts:
        r_on, rc_on, ctx = reply(model, hcm, p, prepend=True)
        # reset to fresh walk for the ungrounded comparison
        model.reset_state(noise=0.6)
        r_off, rc_off, ctx2 = reply(model, hcm, p, prepend=False)
        model.reset_state(noise=0.6)
        print(f"\nPROMPT: {p!r}")
        print(f"  recalled-memory: {ctx!r}")
        print(f"  reply  [grounded]  : {r_on!r}")
        print(f"  reply  [ungrounded]: {r_off!r}")


if __name__ == "__main__":
    main()