"""HCM — Hierarchical Context Memory (M3b).

Minimal implementation for night4: resonance-based recall through the
predictive coding pathway. Patterns are S-vectors stored when surprisal
is high; recalled patterns re-enter as error signals.

Contract (architecture.md §5.7):
- Recall is perception: re-entry uses the identical input pathway.
- Writes are earned: surprisal-gated, not every step.
- Capacity managed: LRU + strength-weighted eviction.
"""
import torch
import torch.nn.functional as F


class HCM:
    def __init__(self, dim, max_patterns=512, recall_threshold=0.3,
                 top_k=4, write_surp_thresh=1.5, strength_decay=0.995,
                 min_age=10, device="cpu"):
        self.dim = dim
        self.max_patterns = max_patterns
        self.recall_threshold = recall_threshold
        self.top_k = top_k
        self.write_surp_thresh = write_surp_thresh
        self.strength_decay = strength_decay
        self.min_age = min_age
        self.device = device

        self.patterns = torch.zeros(max_patterns, dim, device=device)
        self.strengths = torch.zeros(max_patterns, device=device)
        self.birth_step = torch.zeros(max_patterns, dtype=torch.long, device=device)
        self.usage = torch.zeros(max_patterns, device=device)
        self.n_patterns = 0
        self.step_count = 0
        self.total_writes = 0
        self.total_recalls = 0
        self.recall_hits = 0
        self.action_writes = 0
        self.auto_writes = 0

    def _cosine_sim(self, query, bank):
        if bank.shape[0] == 0:
            return torch.zeros(0, device=query.device)
        q = F.normalize(query.unsqueeze(0), dim=-1)
        b = F.normalize(bank[:self.n_patterns], dim=-1)
        return (q @ b.t()).squeeze(0)

    @torch.no_grad()
    def write(self, pattern, surprisal, from_action=False):
        if surprisal < self.write_surp_thresh:
            return False
        if self.n_patterns >= self.max_patterns:
            scores = self.strengths[:self.n_patterns] * (self.usage[:self.n_patterns] + 1)
            victim = scores.argmin()
            self.patterns[victim] = pattern.clone()
            self.strengths[victim] = 1.0
            self.birth_step[victim] = self.step_count
            self.usage[victim] = 0
        else:
            self.patterns[self.n_patterns] = pattern.clone()
            self.strengths[self.n_patterns] = 1.0
            self.birth_step[self.n_patterns] = self.step_count
            self.usage[self.n_patterns] = 0
            self.n_patterns += 1
        self.total_writes += 1
        if from_action:
            self.action_writes += 1
        else:
            self.auto_writes += 1
        return True

    @torch.no_grad()
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

    @torch.no_grad()
    def decay(self):
        self.strengths[:self.n_patterns] *= self.strength_decay
        self.step_count += 1

    def state_dict(self):
        return {"patterns": self.patterns[:self.n_patterns].clone(),
                "strengths": self.strengths[:self.n_patterns].clone(),
                "birth_step": self.birth_step[:self.n_patterns].clone(),
                "usage": self.usage[:self.n_patterns].clone(),
                "n_patterns": self.n_patterns, "step_count": self.step_count,
                "total_writes": self.total_writes, "total_recalls": self.total_recalls,
                "recall_hits": self.recall_hits,
                "action_writes": self.action_writes, "auto_writes": self.auto_writes}

    def load_state_dict(self, d):
        n = d["n_patterns"]
        self.patterns[:n] = d["patterns"]
        self.strengths[:n] = d["strengths"]
        self.birth_step[:n] = d.get("birth_step", torch.zeros(n, dtype=torch.long))
        self.usage[:n] = d["usage"]
        self.n_patterns = n
        self.step_count = d.get("step_count", 0)
        self.total_writes = d.get("total_writes", 0)
        self.total_recalls = d.get("total_recalls", 0)
        self.recall_hits = d.get("recall_hits", 0)
        self.action_writes = d.get("action_writes", 0)
        self.auto_writes = d.get("auto_writes", 0)

    def snapshot(self):
        return {"n_patterns": self.n_patterns,
                "total_writes": self.total_writes,
                "action_writes": self.action_writes,
                "auto_writes": self.auto_writes,
                "total_recalls": self.total_recalls,
                "recall_hits": self.recall_hits,
                "avg_strength": round(float(self.strengths[:self.n_patterns].mean()), 3) if self.n_patterns > 0 else 0}
