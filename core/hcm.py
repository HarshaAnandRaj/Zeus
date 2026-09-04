"""HCM — Hierarchical Context Memory (CDT-consistent).

Coarse-grained memory architecture based on Configuration Drift Theory.
Exact recall is transient in high-D state space; similar-state recall persists.

Key insight: store patterns at coarse resolution (bin states into regions),
retrieve by similarity to region centers, consolidate by pruning stale patterns.

Contract (architecture.md §5.7):
- Recall is perception: re-entry uses the identical input pathway.
- Writes are earned: surprisal-gated, not every step.
- Capacity managed: drift-aware consolidation + strength-weighted eviction.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def text_is_clean(text, min_chars=24, max_junk_frac=0.25):
    """Gate for memory writes: reject markup/control-garbage that high-
    surprisal points often land on, so remembered passages are prose."""
    if text is None:
        return False
    if len(text.strip()) < min_chars:
        return False
    ctrl = sum(1 for ch in text if (ord(ch) < 32 or (0x80 <= ord(ch) < 0xA0)))
    if ctrl / max(len(text), 1) > 0.05:
        return False
    if text.count("\ufffd") > 0:      # replacement chars => decode garbage
        return False
    if "||" in text or "{{" in text or "}}" in text or "|-\n" in text:
        return False
    alpha = sum(1 for ch in text if ch.isalpha())
    if alpha / max(len(text), 1) < 0.55:
        return False
    words = text.split()
    if not words:
        return False
    junk_words = sum(1 for w in words if not any(ch.isalpha() for ch in w))
    return junk_words / len(words) <= max_junk_frac


class HCM:
    def __init__(self, dim, max_patterns=512, recall_threshold=0.3,
                 top_k=4, write_surp_thresh=1.5, strength_decay=0.995,
                 min_age=10, n_clusters=32, device="cpu", context_len=30,
                 write_min_chars=24):
        self.dim = dim
        self.max_patterns = max_patterns
        self.recall_threshold = recall_threshold
        self.top_k = top_k
        self.write_surp_thresh = write_surp_thresh
        self.strength_decay = strength_decay
        self.min_age = min_age
        self.device = device
        self.n_clusters = n_clusters
        # Passage length a memory carries: how much remembered text the voice
        # should be able to continue. Longer than the old 8-token fragments.
        self.context_len = context_len
        # Minimum decoded characters for a context to be considered a real
        # passage (rejects markup/control-junk writes at high-surprisal points).
        self.write_min_chars = write_min_chars

        # Random projection matrix for coarse-grained binning
        # Fixed at init — projects dim-D state to n_clusters-D region hash
        self.proj = torch.randn(dim, n_clusters, device=device) / math.sqrt(dim)
        self.region_patterns = torch.zeros(n_clusters, max_patterns // n_clusters, dim, device=device)
        self.region_counts = torch.zeros(n_clusters, dtype=torch.long, device=device)
        self.region_capacity = max_patterns // n_clusters

        # Pattern storage (associated with regions)
        self.patterns = torch.zeros(max_patterns, dim, device=device)
        # Target embedding: what the model should predict, not where the state was.
        # This is the KEY fix — recall injects prediction signal, not state noise.
        self.target_embed = torch.zeros(max_patterns, dim, device=device)
        self.region_id = torch.zeros(max_patterns, dtype=torch.long, device=device)
        self.strengths = torch.zeros(max_patterns, device=device)
        self.birth_step = torch.zeros(max_patterns, dtype=torch.long, device=device)
        self.usage = torch.zeros(max_patterns, device=device)
        # Anti-crutch: store the target token that was being predicted
        # when this pattern was written. Used to detect verbatim regurgitation.
        self.target_token = torch.zeros(max_patterns, dtype=torch.long, device=device)
        # Provenance is per retained memory, not just an aggregate counter.  A
        # bank can evict old entries, so action_writes alone cannot establish
        # that the memories still available for recall were self-initiated.
        self.action_origin = torch.zeros(max_patterns, dtype=torch.bool, device=device)
        # Token context window (passage before write) for n-gram matching
        self.context_tokens = torch.zeros(max_patterns, self.context_len, dtype=torch.long, device=device)
        # Recent token buffer for context tracking
        self._recent_tokens = []
        # Recall utility: running average of whether each recall helped reduce loss.
        # CDT principle: patterns that recurrently help stay in recurrent regime.
        self.utility = torch.zeros(max_patterns, device=device)
        self.utility_ema = 0.99  # EMA decay for utility tracking
        # Recall outcome tracking: measure loss before/after recall
        self._recall_losses_before = []
        self._recall_losses_after = []
        self._recall_pattern_ids = []
        self.n_patterns = 0
        self.step_count = 0
        self.total_writes = 0
        self.total_recalls = 0
        self.recall_hits = 0
        self.action_writes = 0
        self.auto_writes = 0

    def _assign_region(self, pattern):
        """Assign pattern to region via random projection + sign quantization.

        Projects the dim-D state onto n_clusters random directions,
        then quantizes each projection to {0, 1} via sign.
        This gives a deterministic region hash that分散s patterns across
        2^n_clusters possible regions (capped at n_clusters).
        """
        # Project: (dim,) @ (dim, n_clusters) -> (n_clusters,)
        projected = pattern @ self.proj
        # Sign quantization: each projection -> {0, 1}
        # Combine into a region index via binary encoding
        bits = (projected > 0).long()
        # Convert binary vector to integer region index
        powers = 2 ** torch.arange(self.n_clusters, device=pattern.device)
        region_hash = (bits * powers).sum().item()
        region = region_hash % self.n_clusters
        return region

    def _cosine_sim(self, query, bank):
        if bank.shape[0] == 0:
            return torch.zeros(0, device=query.device)
        q = F.normalize(query.unsqueeze(0), dim=-1)
        b = F.normalize(bank[:self.n_patterns], dim=-1)
        return (q @ b.t()).squeeze(0)

    @torch.no_grad()
    def write(self, pattern, surprisal, from_action=False, target_token=-1,
              target_embed=None, recent_tokens=None, quality_ok=True, iter_step=None):
        if surprisal < self.write_surp_thresh:
            return False
        if not quality_ok:
            return False
        pattern = pattern.detach().to(self.device)
        if target_embed is not None:
            target_embed = target_embed.detach().to(self.device)
        # Track recent tokens for context
        if target_token >= 0:
            self._recent_tokens.append(target_token)
            if len(self._recent_tokens) > self.context_len:
                self._recent_tokens = self._recent_tokens[-self.context_len:]
        # Assign to region
        region = self._assign_region(pattern)
        # Pad context to context_len. Prefer the caller-supplied REAL token
        # stream (a contiguous passage), falling back to the write-target trail.
        if recent_tokens:
            recent = [int(t) for t in recent_tokens][-self.context_len:]
        else:
            recent = self._recent_tokens
        ctx = torch.zeros(self.context_len, dtype=torch.long, device=self.device)
        if recent:
            ctx[-min(len(recent), self.context_len):] = torch.tensor(
                recent[-self.context_len:], device=self.device)
        if self.n_patterns >= self.max_patterns:
            # Evict weakest pattern
            scores = self.strengths[:self.n_patterns] * (self.usage[:self.n_patterns] + 1)
            victim = scores.argmin()
            self.patterns[victim] = pattern.clone()
            self.target_embed[victim] = target_embed.clone() if target_embed is not None else pattern.clone()
            self.region_id[victim] = region
            self.strengths[victim] = 1.0
            self.birth_step[victim] = self.step_count
            self.usage[victim] = 0
            self.target_token[victim] = target_token if target_token >= 0 else 0
            self.action_origin[victim] = bool(from_action)
            self.context_tokens[victim] = ctx
        else:
            self.patterns[self.n_patterns] = pattern.clone()
            self.target_embed[self.n_patterns] = target_embed.clone() if target_embed is not None else pattern.clone()
            self.region_id[self.n_patterns] = region
            self.strengths[self.n_patterns] = 1.0
            self.birth_step[self.n_patterns] = self.step_count
            self.usage[self.n_patterns] = 0
            self.target_token[self.n_patterns] = target_token if target_token >= 0 else 0
            self.action_origin[self.n_patterns] = bool(from_action)
            self.context_tokens[self.n_patterns] = ctx
            self.n_patterns += 1
        self.total_writes += 1
        if from_action:
            self.action_writes += 1
        else:
            self.auto_writes += 1
        return True

    @torch.no_grad()
    def read(self, query):
        """Retrieve by similarity to stored patterns (region-aware).

        Returns the TARGET EMBEDDING (what to predict), not the raw state
        (where the state was). This is the key insight — recall should
        inject prediction signal, not state noise.
        """
        if self.n_patterns == 0:
            return None, None, None, None, None
        query = query.detach().to(self.device)
        sim = self._cosine_sim(query, self.patterns)
        age = self.step_count - self.birth_step[:self.n_patterns]
        fresh = (age >= self.min_age) & (self.strengths[:self.n_patterns] > 0.1)
        if not fresh.any():
            return None, None, None, None, None
        sim_masked = sim.clone()
        sim_masked[~fresh] = -1.0
        topk_sim, topk_idx = sim_masked.topk(min(self.top_k, self.n_patterns))
        mask = topk_sim >= self.recall_threshold
        if not mask.any():
            return None, None, None, None, None
        valid_idx = topk_idx[mask]
        valid_sim = topk_sim[mask]
        self.strengths[valid_idx] += 1.0
        self.usage[valid_idx] = self.step_count
        self.total_recalls += 1
        self.recall_hits += 1
        weights = F.softmax(valid_sim, dim=0)
        # Return TARGET EMBEDDING, not raw state
        retrieved = (self.target_embed[valid_idx] * weights.unsqueeze(-1)).sum(0)
        stored_targets = self.target_token[valid_idx]
        # The remembered phrase: the top-1 recalled pattern's context tokens
        # (last 8 tokens before that memory was written). Discrete ids -> use the
        # single best match (can't average token ids meaningfully).
        best = int(valid_idx[valid_sim.argmax()])
        ctx = self.context_tokens[best].long()
        return retrieved, valid_sim.mean(), stored_targets, valid_idx, ctx

    @torch.no_grad()
    def record_target_match(self, pattern_ids, true_next_token):
        """Track whether a recalled pattern's stored target token matches the
        actual next token. Positive utility => the pattern helps prediction
        (stays in the recurrent regime); negative => noise (gets pruned)."""
        if pattern_ids is None or len(pattern_ids) == 0:
            return
        for pid in pattern_ids:
            pid = int(pid)
            if pid < self.n_patterns:
                helped = (int(self.target_token[pid].item()) == int(true_next_token))
                reward = 1.0 if helped else -0.1
                self.utility[pid] = self.utility_ema * self.utility[pid] + (1 - self.utility_ema) * reward

    @torch.no_grad()
    def record_recall_outcome(self, pattern_ids, loss_before, loss_after):
        """Record whether a recall helped reduce loss.

        Called after each recall with the loss measured before and after
        the recall was injected. Updates utility for each recalled pattern.

        CDT principle: patterns that recurrently help prediction stay
        in the recurrent regime. Patterns that don't help get pruned.
        """
        if pattern_ids is None or len(pattern_ids) == 0:
            return
        # Did loss go down? If yes, recall helped. If no, it was noise.
        loss_delta = loss_before - loss_after  # positive = helped
        # Update utility with EMA
        for pid in pattern_ids:
            pid = int(pid)
            if pid < self.n_patterns:
                # Reward: +1 if helped, -0.1 if didn't
                reward = 1.0 if loss_delta > 0.01 else -0.1
                self.utility[pid] = self.utility_ema * self.utility[pid] + (1 - self.utility_ema) * reward

    @torch.no_grad()
    def consolidate(self, current_state, drift_threshold=0.5):
        """Prune patterns using drift + utility (CDT recall-guided consolidation).

        Two signals determine which patterns survive:
        1. Drift: patterns far from current state are in transient regime
        2. Utility: patterns that never helped prediction are noise

        CDT principle: patterns that recurrently help AND stay close to
        current trajectories remain in the recurrent regime. Patterns
        that are stale OR useless get pruned.
        """
        if self.n_patterns == 0:
            return 0
        sims = self._cosine_sim(current_state, self.patterns)
        # Combined score: drift similarity + recall utility
        # Low similarity AND low utility → prune
        # High similarity OR high utility → keep
        combined_score = sims * 0.5 + self.utility[:self.n_patterns] * 0.5
        # BUG FIX: don't prune recently written patterns (< 200 steps old)
        # Give them time to be recalled before judging
        age = self.step_count - self.birth_step[:self.n_patterns]
        too_new = age < 200
        stale = (combined_score < drift_threshold) & (~too_new)
        n_pruned = stale.sum().item()
        if n_pruned > 0:
            keep = (~stale).nonzero(as_tuple=True)[0]
            n_keep = len(keep)
            self.patterns[:n_keep] = self.patterns[keep]
            self.target_embed[:n_keep] = self.target_embed[keep]
            self.region_id[:n_keep] = self.region_id[keep]
            self.strengths[:n_keep] = self.strengths[keep]
            self.birth_step[:n_keep] = self.birth_step[keep]
            self.usage[:n_keep] = self.usage[keep]
            self.target_token[:n_keep] = self.target_token[keep]
            self.action_origin[:n_keep] = self.action_origin[keep]
            self.context_tokens[:n_keep] = self.context_tokens[keep]
            self.utility[:n_keep] = self.utility[keep]
            self.n_patterns = n_keep
        return n_pruned

    @torch.no_grad()
    def compute_manifold_metrics(self, state_history=None):
        """Compute CDT metrics: ν (memory manifold dimension) and w (drift rate).

        ν: correlation dimension of stored patterns. Measures effective
           dimensionality of the memory manifold. Low ν = patterns cluster
           in few dimensions = recurrent regime. High ν = spread across
           many dimensions = transient regime.

        w: drift rate from state trajectory. Measures how fast the model's
           state wanders. High w = fast drift = harder to recall exact states.

        Phase boundary: ν ≤ w → recurrent (recall works)
                        ν > w → transient (recall fails)
        """
        result = {"nu": 0.0, "w": 0.0, "nu_w_ratio": 0.0}
        if self.n_patterns < 20:
            return result

        patterns = self.patterns[:self.n_patterns]

        # --- Compute ν (correlation dimension) ---
        # Sample pairs and compute pairwise distances
        n_pairs = min(2000, self.n_patterns * (self.n_patterns - 1) // 2)
        idx1 = torch.randint(0, self.n_patterns, (n_pairs,), device=self.device)
        idx2 = torch.randint(0, self.n_patterns, (n_pairs,), device=self.device)
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]
        if len(idx1) < 20:
            return result
        dists = (patterns[idx1] - patterns[idx2]).norm(dim=1)

        # Correlation sum C(r) at multiple scales
        r_min = dists.quantile(0.05).item()
        r_max = dists.quantile(0.95).item()
        if r_max <= r_min:
            return result
        r_values = torch.linspace(r_min, r_max, 15, device=self.device)
        log_r = torch.log(r_values).cpu().numpy()
        log_C = []
        for r in r_values:
            C = (dists < r).float().mean().item()
            log_C.append(np.log(max(C, 1e-10)))
        log_C = np.array(log_C)

        # Fit slope in the middle region (scaling region)
        mid = len(log_r) // 2
        lo, hi = max(mid - 3, 0), min(mid + 4, len(log_r))
        if hi - lo < 3:
            return result
        slope = np.polyfit(log_r[lo:hi], log_C[lo:hi], 1)[0]
        result["nu"] = round(max(slope, 0.0), 4)

        # --- Compute w (drift rate) ---
        if state_history is not None and len(state_history) >= 20:
            # Rolling average of state change norms
            diffs = []
            for i in range(max(len(state_history) - 100, 0), len(state_history) - 1):
                d = (state_history[i + 1] - state_history[i]).norm().item()
                diffs.append(d)
            result["w"] = round(np.mean(diffs), 4) if diffs else 0.0

        # Phase boundary ratio
        if result["w"] > 1e-6:
            result["nu_w_ratio"] = round(result["nu"] / result["w"], 4)

        return result

    @torch.no_grad()
    def decay(self):
        self.strengths[:self.n_patterns] *= self.strength_decay
        self.step_count += 1

    def state_dict(self):
        return {"patterns": self.patterns[:self.n_patterns].clone(),
                "target_embed": self.target_embed[:self.n_patterns].clone(),
                "region_id": self.region_id[:self.n_patterns].clone(),
                "strengths": self.strengths[:self.n_patterns].clone(),
                "birth_step": self.birth_step[:self.n_patterns].clone(),
                "usage": self.usage[:self.n_patterns].clone(),
                "target_token": self.target_token[:self.n_patterns].clone(),
                "action_origin": self.action_origin[:self.n_patterns].clone(),
                "context_tokens": self.context_tokens[:self.n_patterns].clone(),
                "utility": self.utility[:self.n_patterns].clone(),
                "proj": self.proj.clone(),
                "n_patterns": self.n_patterns, "step_count": self.step_count,
                "total_writes": self.total_writes, "total_recalls": self.total_recalls,
                "recall_hits": self.recall_hits,
                "action_writes": self.action_writes, "auto_writes": self.auto_writes}

    def load_state_dict(self, d):
        n = d["n_patterns"]
        self.patterns[:n] = d["patterns"]
        if "target_embed" in d:
            self.target_embed[:n] = d["target_embed"]
        self.region_id[:n] = d.get("region_id", torch.zeros(n, dtype=torch.long))
        self.strengths[:n] = d["strengths"]
        self.birth_step[:n] = d.get("birth_step", torch.zeros(n, dtype=torch.long))
        self.usage[:n] = d["usage"]
        if "target_token" in d:
            self.target_token[:n] = d["target_token"]
        self.action_origin[:n] = False
        if "action_origin" in d:
            self.action_origin[:n] = d["action_origin"].to(self.device, dtype=torch.bool)
        if "context_tokens" in d:
            saved = d["context_tokens"]
            if saved.shape[1] != self.context_len:
                # Migrate an older (shorter-context) HCM: zero-pad to new width.
                buf = torch.zeros(n, self.context_len, dtype=saved.dtype,
                                  device=saved.device)
                k = min(saved.shape[1], self.context_len)
                buf[:, -k:] = saved[:, -k:]
                self.context_tokens[:n] = buf
                self._recent_tokens = [int(t) for t in buf[-1] if int(t) != 0]
            else:
                self.context_tokens[:n] = saved
        if "utility" in d:
            self.utility[:n] = d["utility"]
        if "proj" in d:
            self.proj = d["proj"]
        self.n_patterns = n
        self.step_count = d.get("step_count", 0)
        self.total_writes = d.get("total_writes", 0)
        self.total_recalls = d.get("total_recalls", 0)
        self.recall_hits = d.get("recall_hits", 0)
        self.action_writes = d.get("action_writes", 0)
        self.auto_writes = d.get("auto_writes", 0)

    def compute_crutch_loss(self, logits, stored_targets, w_crutch=0.1):
        """Compute anti-crutch penalty.

        CDT principle: exact recall is transient (death), rhyme is recurrent (life).
        When the model recalls a pattern, it should generate *from* it,
        not *reproduce* the exact tokens that produced it.

        The penalty measures how much the model's predictions overlap
        with the stored target tokens. High overlap = regurgitation = penalty.
        """
        if stored_targets is None or len(stored_targets) == 0:
            return torch.tensor(0.0, device=logits.device)

        # Get model's top prediction
        pred_token = logits.argmax(dim=-1)  # (batch,) or scalar

        # Check overlap with stored targets
        # stored_targets: (n_matched_patterns,) — the tokens that produced each pattern
        # We penalize if the model's prediction matches any stored target
        pred_expanded = pred_token.unsqueeze(-1).expand_as(stored_targets.unsqueeze(0))
        matches = (pred_expanded == stored_targets.unsqueeze(0)).float()

        # Penalty proportional to fraction of stored targets matched
        # If model predicts the same token that N patterns were based on, penalize
        match_frac = matches.sum() / max(len(stored_targets), 1)

        # Scaled penalty: higher when more patterns share the same target
        penalty = w_crutch * match_frac
        return penalty

    def snapshot(self):
        region_dist = {}
        if self.n_patterns > 0:
            for r in self.region_id[:self.n_patterns].tolist():
                region_dist[r] = region_dist.get(r, 0) + 1
        return {"n_patterns": self.n_patterns,
                "n_regions": len(region_dist),
                "region_distribution": region_dist,
                "total_writes": self.total_writes,
                "action_writes": self.action_writes,
                "auto_writes": self.auto_writes,
                "total_recalls": self.total_recalls,
                "recall_hits": self.recall_hits,
                "avg_strength": round(float(self.strengths[:self.n_patterns].mean()), 3) if self.n_patterns > 0 else 0,
                "avg_utility": round(float(self.utility[:self.n_patterns].mean()), 4) if self.n_patterns > 0 else 0,
                "positive_utility_frac": round(float((self.utility[:self.n_patterns] > 0).float().mean()), 4) if self.n_patterns > 0 else 0}
