import json, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from core.model import ZeusCore, ZeusConfig
from core.hcm import HCM


def load_broca(pdir, device):
    ro = torch.load(f"{pdir}/readout.pt", map_location="cpu", weights_only=True)
    em = torch.load(f"{pdir}/emb.pt", map_location="cpu", weights_only=True)
    rcfg = json.loads(open(f"{pdir}/run_config.json", encoding="utf-8").read())
    cfg = ZeusConfig()
    for k in ("readout_layers", "readout_ffn_mult", "readout_heads", "ctx_anchor", "cross_attn"):
        if k in rcfg:
            setattr(cfg, k, rcfg[k])
    m = ZeusCore(cfg).to(device)
    m.readout.load_state_dict(ro, strict=False)
    m.embed.load_state_dict(em, strict=False)
    m.eval()
    return m


def plain_cont_reply(model, prompt, prepend_phrase=None, max_tokens=48, temp=0.8, seed=9):
    """Use the Broca voice as a plain continuation model: feed the token window
    (E_hist embeds) as the cross-source (= stage-1 self-source mode), so it
    continues the token context fluently. Optionally prepend a memory phrase."""
    g = torch.Generator(device=model.S.device); g.manual_seed(seed)
    with torch.no_grad():
        model.ingest(list(model.encode(prompt)))
        ctx_ids = []
        if prepend_phrase:
            ctx_ids = list(model.encode(prepend_phrase))
            model.ingest(ctx_ids)
        # readout needs history=E_hist embeds as cross source
        model._use_ctx_as_source = True
        logits = model.observe()
        out = []
        for _ in range(max_tokens):
            probs = torch.softmax(logits / max(temp, 1e-4), dim=-1)
            if out:
                uniq = torch.tensor(sorted(set(out)), device=probs.device)
                probs[uniq] = probs[uniq] / 1.15
            probs = probs / probs.sum()
            nxt = torch.multinomial(probs, 1, generator=g).item()
            out.append(nxt)
            logits, _ = model.step(nxt)
    return model.decode(out)


def main():
    model = load_broca("runs/broca_pretrain", "cuda")
    # Patch: make observe()/step() use history=None -> the cross-attn sublayer
    # self-sources (kv=q), exactly the stage-1 training condition. The token
    # window (E_hist, with any prepended memory) is both query and source, so the
    # voice continues the token context fluently.
    def observe_plain():
        return model.readout(model.S, None, model.last_e, model.E_hist, None)
    model.observe = observe_plain
    _orig_step = model.step
    def step_plain(nxt, **kw):
        out, aux = _orig_step(nxt, **kw)
        logits = model.readout(model.S, None, model.last_e, model.E_hist, None)
        return logits, aux
    model.step = step_plain

    mems = [
        "the old librarian climbed the stairs to the dusty stacks every evening",
        "a small boy sailed his paper boat down the rain filled gutter",
        "the blacksmith hammered iron until sparks flew into the night sky",
    ]
    for p in ["hello there", "do you have a story to tell", "what comes next"]:
        model.reset_state(noise=0.4)
        plain = plain_cont_reply(model, p, prepend_phrase=None)
        model.reset_state(noise=0.4)
        with_mem = plain_cont_reply(model, p, prepend_phrase=mems[0])
        print(f"\nPROMPT: {p!r}")
        print(f"  [no memory    ] {plain!r}")
        print(f"  [memory(librar)] {with_mem!r}")


if __name__ == "__main__":
    main()