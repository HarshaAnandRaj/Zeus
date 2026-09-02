"""CDT Memory Sim: test qualitative predictions of CDT on memory.

Tests:
1. Phase boundary: critical memory capacity where recall transitions from working to failing
2. Consolidation as dimension reduction: does pruning stale patterns improve recall?
3. Coarse-grained vs exact storage: which sustains recall longer?
4. Dynamic capacity vs fixed capacity: which outperforms?

Measures:
- ν (correlation dimension of memory manifold)
- w (drift rate from state trajectories)
- recall accuracy: how often recall improves prediction
- phase transition: at what capacity does recall start failing?
"""
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusCore, ZeusConfig
from core.hcm import HCM
from tokenizers import Tokenizer


class CoarseHCM:
    """CDT-consistent HCM: coarse-grained storage, region-based retrieval."""
    def __init__(self, dim, max_patterns=512, recall_threshold=0.3,
                 top_k=4, write_surp_thresh=1.5, strength_decay=0.995,
                 min_age=10, n_clusters=32, device="cpu"):
        self.dim = dim
        self.max_patterns = max_patterns
        self.recall_threshold = recall_threshold
        self.top_k = top_k
        self.write_surp_thresh = write_surp_thresh
        self.strength_decay = strength_decay
        self.min_age = min_age
        self.device = device
        self.n_clusters = n_clusters

        # Region centers (coarse-grained storage)
        self.centers = torch.zeros(n_clusters, dim, device=device)
        self.center_count = torch.zeros(n_clusters, device=device)
        self.center_sum = torch.zeros(n_clusters, dim, device=device)
        self.n_filled = 0

        # Pattern storage (associated with regions)
        self.patterns = torch.zeros(max_patterns, dim, device=device)
        self.region_id = torch.zeros(max_patterns, dtype=torch.long, device=device)
        self.strengths = torch.zeros(max_patterns, device=device)
        self.birth_step = torch.zeros(max_patterns, dtype=torch.long, device=device)
        self.usage = torch.zeros(max_patterns, device=device)
        self.n_patterns = 0
        self.step_count = 0
        self.total_writes = 0
        self.total_recalls = 0
        self.recall_hits = 0

    def _assign_region(self, pattern):
        """Assign pattern to nearest cluster center."""
        if self.n_filled == 0:
            self.centers[0] = pattern.clone()
            self.center_count[0] = 1
            self.center_sum[0] = pattern.clone()
            self.n_filled = 1
            return 0
        # Find nearest center
        sims = F.normalize(pattern.unsqueeze(0), dim=-1) @ F.normalize(self.centers[:self.n_filled], dim=-1).t()
        region = sims.argmax().item()
        # Update center (online k-means)
        self.center_count[region] += 1
        self.center_sum[region] += pattern
        self.centers[region] = self.center_sum[region] / self.center_count[region]
        return region

    def _cosine_sim(self, query, bank):
        if bank.shape[0] == 0:
            return torch.zeros(0, device=query.device)
        q = F.normalize(query.unsqueeze(0), dim=-1)
        b = F.normalize(bank[:self.n_patterns], dim=-1)
        return (q @ b.t()).squeeze(0)

    def write(self, pattern, surprisal, from_action=False):
        if surprisal < self.write_surp_thresh:
            return False
        region = self._assign_region(pattern)
        if self.n_patterns >= self.max_patterns:
            scores = self.strengths[:self.n_patterns] * (self.usage[:self.n_patterns] + 1)
            victim = scores.argmin()
            self.patterns[victim] = pattern.clone()
            self.region_id[victim] = region
            self.strengths[victim] = 1.0
            self.birth_step[victim] = self.step_count
            self.usage[victim] = 0
        else:
            self.patterns[self.n_patterns] = pattern.clone()
            self.region_id[self.n_patterns] = region
            self.strengths[self.n_patterns] = 1.0
            self.birth_step[self.n_patterns] = self.step_count
            self.usage[self.n_patterns] = 0
            self.n_patterns += 1
        self.total_writes += 1
        return True

    def read(self, query):
        if self.n_patterns == 0:
            return None, None
        sim = self._cosine_sim(query, self.patterns)
        age = self.step_count - self.birth_step[:self.n_patterns]
        fresh = (age >= self.min_age) & (self.strengths[:self.n_patterns] > 0.1)
        if not fresh.any():
            return None, None
        sim_masked = sim.clone()
        sim_masked[~fresh] = -1.0
        topk_sim, topk_idx = sim_masked.topk(min(self.top_k, self.n_patterns))
        mask = topk_sim >= self.recall_threshold
        if not mask.any():
            return None, None
        valid_idx = topk_idx[mask]
        valid_sim = topk_sim[mask]
        self.strengths[valid_idx] += 1.0
        self.usage[valid_idx] = self.step_count
        self.total_recalls += 1
        self.recall_hits += 1
        weights = F.softmax(valid_sim, dim=0)
        retrieved = (self.patterns[valid_idx] * weights.unsqueeze(-1)).sum(0)
        return retrieved, valid_sim.mean()

    def consolidate(self, current_state, drift_threshold=0.5):
        """Prune patterns far from current trajectory (dimension reduction)."""
        if self.n_patterns == 0:
            return 0
        sims = self._cosine_sim(current_state, self.patterns)
        stale = sims < drift_threshold
        n_pruned = stale.sum().item()
        if n_pruned > 0:
            # Compact: move non-stale patterns to front
            keep = ~stale
            self.patterns[:keep.sum()] = self.patterns[keep]
            self.region_id[:keep.sum()] = self.region_id[keep]
            self.strengths[:keep.sum()] = self.strengths[keep]
            self.birth_step[:keep.sum()] = self.birth_step[keep]
            self.usage[:keep.sum()] = self.usage[keep]
            self.n_patterns = keep.sum().item()
        return n_pruned

    def decay(self):
        self.strengths[:self.n_patterns] *= self.strength_decay
        self.step_count += 1

    def snapshot(self):
        return {"n_patterns": self.n_patterns,
                "total_writes": self.total_writes,
                "total_recalls": self.total_recalls,
                "recall_hits": self.recall_hits}


def compute_correlation_dim(bank, n_patterns, max_pairs=1000):
    """Compute ν (correlation dimension) of memory manifold."""
    if n_patterns < 10:
        return 0.0
    patterns = bank[:n_patterns]
    # Sample pairs
    n_pairs = min(max_pairs, n_patterns * (n_patterns - 1) // 2)
    idx1 = torch.randint(0, n_patterns, (n_pairs,))
    idx2 = torch.randint(0, n_patterns, (n_pairs,))
    # Avoid self-pairs
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    if len(idx1) < 10:
        return 0.0
    # Compute distances
    d = (patterns[idx1] - patterns[idx2]).norm(dim=1)
    # Correlation sum C(r) for different r
    r_values = torch.logspace(-2, 1, 20).to(d.device)
    C_r = []
    for r in r_values:
        C = (d < r).float().mean().item()
        C_r.append(max(C, 1e-10))
    # Fit slope of log(C(r)) vs log(r) in the scaling region
    log_r = torch.log(r_values[5:15]).cpu().numpy()
    log_C = np.log(np.array(C_r[5:15]))
    if len(log_r) < 2:
        return 0.0
    # Linear fit
    slope = np.polyfit(log_r, log_C, 1)[0]
    return max(slope, 0.0)


def compute_drift_rate(state_history, window=100):
    """Compute w (drift rate) from state trajectories."""
    if len(state_history) < window + 1:
        return 0.0
    # Compute rolling norm of state change
    diffs = []
    for i in range(len(state_history) - 1):
        d = (state_history[i+1] - state_history[i]).norm().item()
        diffs.append(d)
    diffs = np.array(diffs[-window:])
    return diffs.mean()


def run_cdt_sim():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CDT Memory Sim — Device: {device}")

    cfg = ZeusConfig(dim=64, attn_heads=2, vocab=8192, window=16, slow_dim=16)
    model = ZeusCore(cfg).to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params")

    import numpy as np
    ids_cache = ROOT / "corpus" / "data" / "train_ids.npy"
    train_ids = np.load(ids_cache).tolist()[:500_000]
    print(f"Corpus: {len(train_ids)} tokens")

    log_path = ROOT / "experiments" / "emergence_sim" / "cdt_memory.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(entry):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # Two conditions: exact vs coarse, both with dynamic capacity
    conditions = {
        "exact_dynamic": {"coarse": False, "consolidate": True, "dynamic_cap": True},
        "coarse_dynamic": {"coarse": True, "consolidate": True, "dynamic_cap": True},
    }

    results = {}
    for cond_name, cond_params in conditions.items():
        print(f"\n=== {cond_name} ===")
        log({"event": "start", "condition": cond_name, **cond_params})

        if cond_params["coarse"]:
            hcm = CoarseHCM(cfg.dim, max_patterns=512, recall_threshold=0.3,
                           top_k=4, write_surp_thresh=1.5, min_age=5,
                           n_clusters=32, device=device)
        else:
            hcm = HCM(cfg.dim, max_patterns=512, recall_threshold=0.3,
                     top_k=4, write_surp_thresh=1.5, min_age=5,
                     strength_decay=0.995, device=device)

        dyn = [p for n, p in model.named_parameters() if "readout" not in n and "hcm" not in n]
        lm = [p for n, p in model.named_parameters() if "readout" in n]
        opt = torch.optim.AdamW([{"params": dyn, "lr": 3e-4}, {"params": lm, "lr": 1e-3}])

        state_history = []
        t0 = time.time()
        bptt = 33
        teacher_p = 0.9
        w_action = 0.05

        for step in range(1, 1001):
            if step <= 300:
                curriculum_prob = 1.0
            elif step <= 800:
                curriculum_prob = 1.0 - (step - 300) / 500
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

            for t in range(T):
                pred_before = model.self_pred(model.S)
                logits, aux = model.step(nxt_input)
                target = torch.tensor(seg[t + 1], device=device)
                ce = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                surp = (model.S - pred_before.detach()).norm()
                loss_t = ce + 0.1 * (-surp) + 0.1 * aux["rent"] + aux["div"]

                surp_ratio_t = min(1.0, surp.item() / max(hcm.write_surp_thresh, 1.0))
                log_prob_remember = torch.log_softmax(logits, dim=-1)[cfg.remember_id]
                loss_t = loss_t + w_action * surp_ratio_t * (-log_prob_remember)
                loss_total = loss_total + loss_t / T
                ce_sum += ce.item()
                surp_sum += surp.item()
                action_logits_sum += float(torch.softmax(logits, dim=-1)[cfg.remember_id])

                pred_token = int(logits.argmax().item())
                surp_ratio = min(1.0, surp.item() / max(hcm.write_surp_thresh, 1.0))
                effective_teacher_p = teacher_p * (1.0 - 0.5 * surp_ratio)
                if random.random() < effective_teacher_p:
                    nxt_input = int(seg[t + 1])
                    if random.random() < curriculum_prob:
                        nxt_input = cfg.remember_id
                    elif surp.item() > hcm.write_surp_thresh:
                        nxt_input = cfg.remember_id
                else:
                    nxt_input = pred_token

                if nxt_input == cfg.remember_id:
                    wrote = hcm.write(model.S.detach().clone(), surp.item(),
                                      from_action=(pred_token == cfg.remember_id))
                    if wrote:
                        hcm_writes += 1
                    retrieved, sim_val, _, _, _ = hcm.read(model.S.detach())
                    if retrieved is not None:
                        model.hcm_pending = retrieved
                        hcm_reads += 1

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            hcm.decay()

            # Record state
            state_history.append(model.S.detach().clone())

            # Consolidation every 100 steps
            if cond_params["consolidate"] and step % 100 == 0 and step > 0:
                n_pruned = hcm.consolidate(model.S.detach())

            # Dynamic capacity: adjust max_patterns based on ν
            if cond_params["dynamic_cap"] and step % 200 == 0 and hcm.n_patterns > 10:
                if hasattr(hcm, 'centers'):
                    nu = compute_correlation_dim(hcm.centers, hcm.n_filled)
                else:
                    nu = compute_correlation_dim(hcm.patterns, hcm.n_patterns)
                w = compute_drift_rate(state_history[-200:])
                if nu > w * 1.5 and hcm.n_patterns > 50:
                    # Over capacity: prune weakest
                    scores = hcm.strengths[:hcm.n_patterns] * (hcm.usage[:hcm.n_patterns] + 1)
                    threshold = torch.quantile(scores, 0.3)
                    keep = scores > threshold
                    hcm.patterns[:keep.sum()] = hcm.patterns[keep]
                    hcm.strengths[:keep.sum()] = hcm.strengths[keep]
                    hcm.birth_step[:keep.sum()] = hcm.birth_step[keep]
                    hcm.usage[:keep.sum()] = hcm.usage[keep]
                    hcm.n_patterns = keep.sum().item()

            if step % 200 == 0:
                avg_ce = ce_sum / T
                avg_surp = surp_sum / T
                action_prob = action_logits_sum / T
                # Compute ν and w
                if hcm.n_patterns > 10:
                    if hasattr(hcm, 'centers'):
                        nu = compute_correlation_dim(hcm.centers, hcm.n_filled)
                    else:
                        nu = compute_correlation_dim(hcm.patterns, hcm.n_patterns)
                else:
                    nu = 0.0
                w = compute_drift_rate(state_history[-min(200, len(state_history)):]) if len(state_history) >= 10 else 0.0
                sps = step / (time.time() - t0)
                entry = {"step": step, "condition": cond_name,
                         "ce": round(avg_ce, 4), "surp": round(avg_surp, 4),
                         "persist": round(-aux["tau_mean_t"].item(), 4),
                         "hcm_n": hcm.n_patterns, "hcm_reads": hcm_reads,
                         "hcm_writes": hcm_writes,
                         "action_prob": round(action_prob, 6),
                         "nu": round(nu, 4), "w": round(w, 4),
                         "nu_w_ratio": round(nu / max(w, 1e-6), 4),
                         "curriculum_prob": round(curriculum_prob, 4),
                         "sps": round(sps, 3)}
                log(entry)
                print(f"  step={step:5d} ce={avg_ce:.3f} surp={avg_surp:.2f} "
                      f"n={hcm.n_patterns} nu={nu:.3f} w={w:.3f} "
                      f"nu/w={nu/max(w,1e-6):.2f} sps={sps:.3f}")

        # Final snapshot
        results[cond_name] = {
            "final_ce": ce_sum / T,
            "n_patterns": hcm.n_patterns,
            "total_writes": hcm.total_writes,
            "total_recalls": hcm.total_recalls,
            "recall_hits": hcm.recall_hits,
        }
        log({"event": "complete", "condition": cond_name, **results[cond_name]})

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Condition':<20} {'CE':>8} {'Writes':>8} {'Recalls':>8} {'Hits':>8}")
    for cond, r in results.items():
        print(f"{cond:<20} {r['final_ce']:>8.3f} {r['total_writes']:>8} {r['total_recalls']:>8} {r['recall_hits']:>8}")


if __name__ == "__main__":
    run_cdt_sim()
