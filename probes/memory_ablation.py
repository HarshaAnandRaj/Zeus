"""Memory ablation probe: disable each tier, measure behavioral delta.
§5.7: 'Removing recall must measurably change behavior.'"""
import torch
import json
from common import load_model
from init_helper import seeded_reset


def ablation_test(model, label, hcm_enabled=True, slow_enabled=True, h_enabled=True):
    if not hcm_enabled:
        model.hcm = None
    if not slow_enabled:
        model.slow = torch.zeros_like(model.slow)
    if not h_enabled:
        model.H = torch.zeros_like(model.H)

    g = torch.Generator(device="cpu").manual_seed(4242)
    texts, traj = [], []
    for p in ("hello", "the little girl"):
        model.reset_state(noise=0.05, generator=g)
        ids = model.encode(p)
        for i in ids:
            model.step(i)
        logits = model.observe()
        out = []
        for _ in range(48):
            probs = torch.softmax(logits / 0.7, dim=-1)
            nxt = torch.multinomial(probs.cpu(), 1, generator=g).item()
            out.append(nxt)
            logits, _ = model.step(nxt)
            traj.append(model.S.detach().clone())
        texts.append(model.decode(out))
    text = " ".join(texts).lower()
    tris = [text[i:i + 3] for i in range(len(text) - 2)]
    transient = len(set(tris)) / max(len(tris), 1)
    traj_tensor = torch.stack(traj).cpu()
    rms = float((traj_tensor - traj_tensor.mean(0)).norm(dim=1).mean())
    com = float(traj_tensor.mean(0).norm())
    return {"label": label, "transient": round(transient, 4), "rms": round(rms, 4),
            "com": round(com, 4), "text_preview": text[:200]}


def main(model=None):
    model = model or load_model()
    original_hcm = model.hcm
    results = []

    results.append(ablation_test(model, "baseline", hcm_enabled=True))
    model.hcm = original_hcm

    results.append(ablation_test(model, "no_hcm", hcm_enabled=False))
    model.hcm = original_hcm

    results.append(ablation_test(model, "no_slow", slow_enabled=False))
    model.hcm = original_hcm

    results.append(ablation_test(model, "no_h", h_enabled=False))
    model.hcm = original_hcm

    baseline = results[0]
    for r in results[1:]:
        r["delta_transient"] = round(r["transient"] - baseline["transient"], 4)
        r["delta_rms"] = round(r["rms"] - baseline["rms"], 4)

    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
