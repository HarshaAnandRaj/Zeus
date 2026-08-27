"""Small-scale emergence simulation: observe what arises from properly-gated HCM.
Minimal model (d=64, l=1), short training (2000 steps), detailed monitoring."""
import json
import pathlib
import random
import sys
import time

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusCore, ZeusConfig
from core.hcm import HCM
from tokenizers import Tokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = ZeusConfig(dim=64, attn_heads=2, vocab=8192, window=16, slow_dim=16)
    model = ZeusCore(cfg).to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params, dim={cfg.dim}")

    hcm = HCM(cfg.dim, max_patterns=128, recall_threshold=0.3, top_k=4,
              write_surp_thresh=1.5, min_age=5, strength_decay=0.995, device=device)
    model.hcm = hcm
    print(f"HCM: {hcm.max_patterns} max, min_age={hcm.min_age}, threshold={hcm.recall_threshold}")

    import numpy as np
    ids_cache = ROOT / "corpus" / "data" / "train_ids.npy"
    if ids_cache.exists():
        train_ids = np.load(ids_cache).tolist()[:500_000]
    else:
        tokenizer = Tokenizer.from_file(str(ROOT / "corpus" / "data" / "tokenizer" / "bpe_8192.json"))
        train_text = (ROOT / "corpus" / "data" / "train.txt").read_text(encoding="utf-8")
        chunks = [train_text[i:i + 2_000_000] for i in range(0, len(train_text), 2_000_000)]
        train_ids = []
        for enc in tokenizer.encode_batch(chunks):
            train_ids.extend(enc.ids)
        train_ids = train_ids[:500_000]
    print(f"Corpus: {len(train_ids)} tokens")

    dyn = [p for n, p in model.named_parameters() if "readout" not in n and "hcm" not in n]
    lm = [p for n, p in model.named_parameters() if "readout" in n]
    opt = torch.optim.AdamW([
        {"params": dyn, "lr": 3e-4},
        {"params": lm, "lr": 1e-3}
    ])

    bptt = 33
    teacher_p = 0.9
    log_path = ROOT / "experiments" / "emergence_sim" / "sim.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(entry):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    log({"event": "start", "dim": cfg.dim, "hcm_max": hcm.max_patterns,
         "min_age": hcm.min_age, "device": device, "n_params": n_params})

    t0 = time.time()
    window_metrics = []

    for step in range(1, 2001):
        off = random.randint(0, len(train_ids) - bptt - 1)
        seg = train_ids[off:off + bptt]
        model.reset_state(noise=0.05)
        nxt_input = int(seg[0])
        T = len(seg) - 1
        loss_total = 0.0
        ce_sum, surp_sum = 0.0, 0.0
        hcm_writes, hcm_reads = 0, 0
        action_logits_sum = 0.0

        for t in range(T):
            pred_before = model.self_pred(model.S)
            logits, aux = model.step(nxt_input)
            target = torch.tensor(seg[t + 1], device=device)
            ce = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            surp = (model.S - pred_before.detach()).norm()
            loss_t = ce + 0.1 * (-surp) + 0.1 * aux["rent"] + aux["div"]
            loss_total = loss_total + loss_t / T
            ce_sum += ce.item()
            surp_sum += surp.item()
            action_logits_sum += float(torch.softmax(logits, dim=-1)[cfg.remember_id])
            pred_token = int(logits.argmax().item())
            if random.random() < teacher_p:
                nxt_input = int(seg[t + 1])
                if surp.item() > hcm.write_surp_thresh:
                    nxt_input = cfg.remember_id
            else:
                nxt_input = pred_token
            if nxt_input == cfg.remember_id:
                wrote = hcm.write(model.S.detach().clone(), surp.item(),
                                  from_action=(pred_token == cfg.remember_id))
                if wrote:
                    hcm_writes += 1
                retrieved, sim = hcm.read(model.S.detach())
                if retrieved is not None:
                    model.hcm_pending = retrieved
                    hcm_reads += 1

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        hcm.decay()

        action_prob = action_logits_sum / T
        entry = {"step": step, "ce": round(ce_sum / T, 4),
                 "surp": round(surp_sum / T, 4),
                 "persist": round(-aux["tau_mean_t"].item(), 4),
                 "hcm_writes": hcm_writes, "hcm_reads": hcm_reads,
                 "action_prob": round(action_prob, 6),
                 "hcm_n": hcm.n_patterns, "hcm_str": round(float(hcm.strengths[:hcm.n_patterns].mean()), 3) if hcm.n_patterns > 0 else 0}

        window_metrics.append(entry)

        if step % 50 == 0:
            avg_ce = sum(m["ce"] for m in window_metrics[-50:]) / 50
            avg_surp = sum(m["surp"] for m in window_metrics[-50:]) / 50
            avg_action = sum(m["action_prob"] for m in window_metrics[-50:]) / 50
            total_reads = sum(m["hcm_reads"] for m in window_metrics[-50:])
            total_writes = sum(m["hcm_writes"] for m in window_metrics[-50:])
            sps = step / (time.time() - t0)
            log({"step": step, "avg_ce": round(avg_ce, 4), "avg_surp": round(avg_surp, 4),
                 "avg_action_prob": round(avg_action, 6), "reads_50": total_reads,
                 "writes_50": total_writes, "hcm_n": hcm.n_patterns,
                 "hcm_action_writes": hcm.action_writes, "hcm_auto_writes": hcm.auto_writes,
                 "hcm_total_recalls": hcm.total_recalls,
                 "sps": round(sps, 3)})
            print(f"step={step:5d} ce={avg_ce:.3f} surp={avg_surp:.2f} "
                  f"action={avg_action:.6f} reads={total_reads} writes={total_writes} "
                  f"hcm_n={hcm.n_patterns} sps={sps:.3f}")

    log({"event": "complete", "final_ce": round(ce_sum / T, 4),
         "hcm_patterns": hcm.n_patterns, "action_writes": hcm.action_writes,
         "auto_writes": hcm.auto_writes, "total_recalls": hcm.total_recalls})
    print(f"\nComplete. Final CE: {ce_sum/T:.3f}")
    print(f"HCM: {hcm.n_patterns} patterns, {hcm.action_writes} action writes, {hcm.total_recalls} recalls")


if __name__ == "__main__":
    main()
