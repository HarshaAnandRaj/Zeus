"""Ablation sim: test separate action head and extended curriculum.
Usage:
  python sim_ablation.py action_head        -- separate action head for REMEMBER
  python sim_ablation.py extended_curriculum -- 4x longer warmup/cooldown
  python sim_ablation.py baseline           -- current approach for comparison
"""
import json
import pathlib
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusCore, ZeusConfig
from core.hcm import HCM
from tokenizers import Tokenizer


class ActionHead(nn.Module):
    """Separate head for REMEMBER action. Single logit output."""
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, s):
        return self.head(s).squeeze(-1)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    w_action = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Mode: {mode}, Device: {device}, w_action={w_action}")

    cfg = ZeusConfig(dim=64, attn_heads=2, vocab=8192, window=16, slow_dim=16)
    model = ZeusCore(cfg).to(device).train()
    n_params = sum(p.numel() for p in model.parameters())

    action_head = None
    if mode == "action_head":
        action_head = ActionHead(cfg.dim).to(device).train()
        n_params += sum(p.numel() for p in action_head.parameters())
        print(f"Action head: {sum(p.numel() for p in action_head.parameters())} params")

    print(f"Model: {n_params/1e6:.2f}M params, dim={cfg.dim}")

    hcm = HCM(cfg.dim, max_patterns=128, recall_threshold=0.3, top_k=4,
              write_surp_thresh=1.5, min_age=5, strength_decay=0.995, device=device)
    model.hcm = hcm
    print(f"HCM: {hcm.max_patterns} max, min_age={hcm.min_age}")

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

    params = [p for n, p in model.named_parameters() if "readout" not in n and "hcm" not in n]
    lm = [p for n, p in model.named_parameters() if "readout" in n]
    if action_head is not None:
        params = params + list(action_head.parameters())
    opt = torch.optim.AdamW([
        {"params": params, "lr": 3e-4},
        {"params": lm, "lr": 1e-3}
    ])

    bptt = 33
    teacher_p = 0.9

    warmup_steps = 2000 if mode == "extended_curriculum" else 500
    cooldown_steps = 4000 if mode == "extended_curriculum" else 1000
    total_steps = 4000 if mode == "extended_curriculum" else 2000

    log_path = ROOT / "experiments" / "emergence_sim" / f"sim_{mode}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(entry):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    log({"event": "start", "mode": mode, "dim": cfg.dim, "w_action": w_action,
         "warmup": warmup_steps, "cooldown": cooldown_steps, "device": device})

    t0 = time.time()
    window_metrics = []

    for step in range(1, total_steps + 1):
        if step <= warmup_steps:
            curriculum_prob = 1.0
        elif step <= warmup_steps + cooldown_steps:
            curriculum_prob = 1.0 - (step - warmup_steps) / cooldown_steps
        else:
            curriculum_prob = 0.0
        off = random.randint(0, len(train_ids) - bptt - 1)
        seg = train_ids[off:off + bptt]
        model.reset_state(noise=0.05)
        nxt_input = int(seg[0])
        T = len(seg) - 1
        loss_total = 0.0
        ce_sum, surp_sum = 0.0, 0.0
        hcm_writes, hcm_reads = 0, 0
        action_logits_sum = 0.0
        action_head_loss_sum = 0.0
        curriculum_injects = 0

        for t in range(T):
            pred_before = model.self_pred(model.S)
            logits, aux = model.step(nxt_input)
            target = torch.tensor(seg[t + 1], device=device)
            ce = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            surp = (model.S - pred_before.detach()).norm()
            loss_t = ce + 0.1 * (-surp) + 0.1 * aux["rent"] + aux["div"]

            surp_ratio_t = min(1.0, surp.item() / max(hcm.write_surp_thresh, 1.0))

            if action_head is not None:
                ah_logit = action_head(model.S)
                ah_prob = torch.sigmoid(ah_logit)
                action_loss = w_action * surp_ratio_t * (-torch.log(ah_prob + 1e-8))
                loss_t = loss_t + action_loss
                action_head_loss_sum += action_loss.item()
                action_logits_sum += ah_prob.item()
            else:
                log_prob_remember = torch.log_softmax(logits, dim=-1)[cfg.remember_id]
                loss_t = loss_t + w_action * surp_ratio_t * (-log_prob_remember)
                action_logits_sum += float(torch.softmax(logits, dim=-1)[cfg.remember_id])

            loss_total = loss_total + loss_t / T
            ce_sum += ce.item()
            surp_sum += surp.item()

            pred_token = int(logits.argmax().item())
            surp_ratio = min(1.0, surp.item() / max(hcm.write_surp_thresh, 1.0))
            effective_teacher_p = teacher_p * (1.0 - 0.5 * surp_ratio)
            if random.random() < effective_teacher_p:
                nxt_input = int(seg[t + 1])
                if random.random() < curriculum_prob:
                    nxt_input = cfg.remember_id
                    curriculum_injects += 1
                elif action_head is not None and ah_prob.item() > 0.5:
                    nxt_input = cfg.remember_id
                elif surp.item() > hcm.write_surp_thresh:
                    nxt_input = cfg.remember_id
            else:
                if action_head is not None and ah_prob.item() > 0.5:
                    nxt_input = cfg.remember_id
                else:
                    nxt_input = pred_token
            if nxt_input == cfg.remember_id:
                is_action = (pred_token == cfg.remember_id) or (action_head is not None and ah_prob.item() > 0.5 and random.random() >= curriculum_prob)
                wrote = hcm.write(model.S.detach().clone(), surp.item(),
                                  from_action=is_action)
                if wrote:
                    hcm_writes += 1
                retrieved, sim, _, _, _ = hcm.read(model.S.detach())
                if retrieved is not None:
                    model.hcm_pending = retrieved
                    hcm_reads += 1

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if action_head is not None:
            torch.nn.utils.clip_grad_norm_(action_head.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        hcm.decay()

        action_prob = action_logits_sum / T
        entry = {"step": step, "ce": round(ce_sum / T, 4),
                 "surp": round(surp_sum / T, 4),
                 "persist": round(-aux["tau_mean_t"].item(), 4),
                 "hcm_writes": hcm_writes, "hcm_reads": hcm_reads,
                 "action_prob": round(action_prob, 6),
                 "curriculum_prob": round(curriculum_prob, 4),
                 "curriculum_injects": curriculum_injects,
                 "hcm_n": hcm.n_patterns}

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

    log({"event": "complete", "mode": mode, "final_ce": round(ce_sum / T, 4),
         "hcm_patterns": hcm.n_patterns, "action_writes": hcm.action_writes,
         "auto_writes": hcm.auto_writes, "total_recalls": hcm.total_recalls})
    print(f"\nComplete [{mode}]. Final CE: {ce_sum/T:.3f}")
    print(f"HCM: {hcm.n_patterns} patterns, {hcm.action_writes} action writes, {hcm.total_recalls} recalls")


if __name__ == "__main__":
    main()
