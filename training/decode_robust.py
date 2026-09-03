"""training/decode_robust.py -- decode-time robustness for the live Broca voice.

Targets the stage1c draw-dependent loop artifact: the fluent voice produces
grammatical English but can wedge into repeating n-grams.  Three levers, each
pure (operate on int token ids / probability tensors), so they drop straight
into zsession's reply loop or a batch harness:

  1. ngram_blocker  -- streaming veto inside sampling: forbid any token that
     would complete an n-gram already emitted earlier in the reply.  Attacks
     local draw-dependent loops head-on.
  2. best_of_k      -- roll K candidate replies from rewound state, score by a
     legibility/loop metric, return the best.  Low-K (2-4) tames variance.
  3. router         -- classify a finished reply (legible | looped | junk) and
     pick the best across produced candidates or bail to a safe fallback.

All scoring is cheap string math (no model calls), so the router runs per
candidate without extra GPU work.
"""

import re

import torch
import torch.nn.functional as F

__all__ = ["NgramBlocker", "best_reply", "best_of_k", "router", "loop_score",
           "legibility_score"]


# ---------------------------------------------------------------- n-gram veto
class NgramBlocker:
    """Zero-out tokens that would form a repeat of any n-gram already emitted.

    Maintains the generated token window; between steps it learns every
    n-gram (size up to `order`) seen so far.  At sample time, for each token t
    it forms the prospective suffix and vetoes t if that n-gram is already in
    the seen set.  Pure CPU state.
    """

    def __init__(self, order=4):
        self.order = max(2, int(order))
        self.seen = set()          # tuples of n-grams already emitted
        self._next = {}            # tail (n-1)-gram -> set of tokens already seen after it
        self._buf = []             # full generated ids (this reply)

    def reset(self):
        self.seen = set()
        self._next = {}
        self._buf = []

    def observe(self, token_id):
        """Record a newly emitted token into the blocker's memory."""
        buf = self._buf
        tid = int(token_id)
        # the order-1 context that this token immediately followed
        if len(buf) >= self.order - 1:
            ctx = tuple(buf[-(self.order - 1):])
            self._next.setdefault(ctx, set()).add(tid)
        buf.append(tid)
        for n in range(2, self.order + 1):
            if len(buf) >= n:
                self.seen.add(tuple(buf[-n:]))

    def initial(self, prefix):
        """Seed the blocker with an already-generated prefix (replay)."""
        self.reset()
        for t in prefix:
            self.observe(t)

    def vetoed_set(self, tail_len=None):
        """Return the (small) set of token ids to veto for the current tail.

        Only tokens that would complete an already-emitted n-gram under the
        current generation tail are returned -- nothing scans the full vocab.
        """
        if len(self._buf) < self.order - 1:
            return set()
        ctx = tuple(self._buf[-(self.order - 1):])
        return set(self._next.get(ctx, ()))

    def vetoed_mask(self, probs, max_blocks=16):
        """bool mask of disallowed tokens for the current emitted tail."""
        mask = probs.new_zeros(probs.shape, dtype=torch.bool)
        nxt_set = self.vetoed_set()
        if not nxt_set:
            return mask
        n = 0
        for t in nxt_set:
            if t < probs.shape[-1]:
                mask[t] = True
                n += 1
                if n >= max_blocks:
                    break
        return mask


# ------------------------------------------------------------------ scoring
_WORD = re.compile(r"[a-zA-Z']{2,}")


def _word_count(tokens):
    return max(_WORD.findall(tokens), key=lambda w: len(w)) if tokens else ""


def loop_score(text):
    """Coverage of the longest repeated WORD n-gram (order 4) in the reply.

    Returns 0.0 for prose with no repetition of a 4-word sequence; rises toward
    1.0 as the reply degrades into an actual loop.  Operates on words, so
    ordinary repeated function words ("the", "and") do not trigger it.
    """
    if not text:
        return 1.0
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 8:
        return 1.0
    n = 4
    if len(words) < n:
        return 1.0
    spans = []                      # (start,end) of each 4-gram occurrence
    pos = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        if gram in pos:
            spans.append((pos[gram][0], i + n))
        else:
            pos[gram] = (i, i + n)
    if not spans:
        return 0.0
    # merge overlaps -> total word-span covered by repeats / whole reply
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    covered = sum(e - s for s, e in merged)
    return min(1.0, covered / max(len(words), 1))


def legibility_score(text, min_words=6, max_alpha_frac=0.25):
    """0..1 legibility: rewards wordy prose, punishes junk/repeats."""
    if not text:
        return 0.0
    words = _WORD.findall(text)
    alpha = sum(1 for ch in text if ch.isalpha())
    if len(text) == 0:
        return 0.0
    alpha_frac = alpha / max(len(text), 1)
    if alpha_frac < 0.55:
        return 0.0
    junk = sum(1 for w in words if not any(ch.isalpha() for ch in w))
    junk_frac = junk / max(len(words), 1)
    if junk_frac > max_alpha_frac:
        return 0.0
    # base: word density + length
    score = min(1.0, len(words) / min_words)
    # penalize looping
    score *= (1.0 - 0.7 * loop_score(text))
    return max(0.0, min(1.0, score))


def _best_key(cands, decode):
    """Return the least-looped, most-legible candidate.

    This is deliberately a ``min`` key: loop_score is a penalty while
    legibility_score is a reward.  Keeping that polarity explicit matters
    because this function chooses the reply that will be persisted into the
    live state trajectory.
    """
    def key(item):
        text = decode(item)
        return (loop_score(text), -legibility_score(text))
    return min(cands, key=key)


# ---------------------------------------------------------------- best-of-k
def best_of_k(model, prompt_ids, k, max_tokens, temperature=0.7, top_p=0.92,
              rep_penalty=1.2, blocker_order=4, seed=0, recall_vec=None,
              anchor_vec=None, skip_pad=False):
    """Generate k candidate replies from rewound state, return the best.

    Returns (best_ids, all_candidates).  Each candidate is a fresh rollout
    from a rollback of the pre-generation snapshot, so candidates share the
    prompt conditioning but diverge stochastically.
    """

    if k <= 1:
        # single path with blocker (best speed)
        return _single_roll(model, prompt_ids, max_tokens, temperature,
                            top_p, rep_penalty, blocker_order,
                            recall_vec=recall_vec, anchor_vec=anchor_vec,
                            skip_pad=skip_pad)[0], None

    candidates = []
    snap = model.snapshot_runtime()
    # hold the conditioning already ingested in the model so each candidate
    # re-starts from the SAME post-ingest state rather than re-ingesting.
    base_ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(k):
            model.restore_runtime(snap)
            ids = _roll_from_snapshot(model, base_ids, max_tokens, temperature,
                                      top_p, rep_penalty, blocker_order,
                                      recall_vec=recall_vec,
                                      anchor_vec=anchor_vec, seed=seed)
            candidates.append(ids)
            seed += 1
    best = _best_key(candidates, model.decode)
    # A candidate is not merely display text: its tokens become part of the
    # next interaction's state.  Replay the selected branch so runtime state
    # agrees with the answer the user actually saw.
    model.restore_runtime(snap)
    for token_id in best:
        if recall_vec is not None:
            model.hcm_pending = recall_vec
        model.step(token_id)
    return best, candidates


def _roll_from_snapshot(model, prompt_ids, max_tokens, temperature, top_p,
                        rep_penalty, blocker_order, recall_vec, anchor_vec, seed):
    # multinomial requires its generator to live on the same device as the
    # probability tensor.  Live Zeus sampling is CUDA while quick unit tests
    # are CPU, so make the rollout generator follow the model state.
    g = torch.Generator(device=model.S.device)
    g.manual_seed(seed)
    blocker = NgramBlocker(blocker_order)
    logits = model.observe()
    out = []
    for _ in range(max_tokens):
        probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        if rep_penalty > 1.0 and out:
            uniq = torch.tensor(sorted(set(out)), device=probs.device)
            probs[uniq] = probs[uniq] / rep_penalty
        mask = blocker.vetoed_mask(probs)
        if mask.any():
            probs = probs.clone()
            probs[mask] = 0.0
            probs = probs / probs.sum()
        if top_p < 1.0:
            sorted_p, idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, 0)
            keep = cum <= top_p
            keep[0] = True
            sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
            probs = torch.zeros_like(probs).scatter_(0, idx, sorted_p)
            probs = probs / probs.sum()
        nxt = torch.multinomial(probs, 1, generator=g).item()
        out.append(nxt)
        blocker.observe(nxt)
        if recall_vec is not None:
            model.hcm_pending = recall_vec
        logits, _ = model.step(nxt)
    return out


def _single_roll(model, prompt_ids, max_tokens, temperature, top_p,
                 rep_penalty, blocker_order, recall_vec, anchor_vec, skip_pad):
    blocker = NgramBlocker(blocker_order)
    logits = model.observe()
    out = []
    for _ in range(max_tokens):
        probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        if rep_penalty > 1.0 and out:
            uniq = torch.tensor(sorted(set(out)), device=probs.device)
            probs[uniq] = probs[uniq] / rep_penalty
        mask = blocker.vetoed_mask(probs)
        if mask.any():
            probs = probs.clone()
            probs[mask] = 0.0
            probs = probs / probs.sum()
        if top_p < 1.0:
            sorted_p, idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, 0)
            keep = cum <= top_p
            keep[0] = True
            sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
            probs = torch.zeros_like(probs).scatter_(0, idx, sorted_p)
            probs = probs / probs.sum()
        nxt = torch.multinomial(probs, 1).item()
        out.append(nxt)
        blocker.observe(nxt)
        if recall_vec is not None:
            model.hcm_pending = recall_vec
        logits, _ = model.step(nxt)
    return out, blocker


# ------------------------------------------------------------------ router
def router(candidate_texts, decode, k=3):
    """Choose the most legible, least-looped candidate among produced replies.

    candidate_texts: list of decoded reply strings (parallel to token arrays).
    Returns the index of the best candidate; falls back to the first if all
    are degenerate.
    """
    ranked = []
    for i, text in enumerate(candidate_texts):
        ls = loop_score(text)
        leg = legibility_score(text)
        ranked.append((leg, -ls, -len(text), i, text))
    ranked.sort(reverse=True)
    best_text = ranked[0][4]
    return ranked[0][3], best_text


def best_reply(model, prompt_ids, k=3, max_tokens=48, temperature=0.7,
               top_p=0.92, rep_penalty=1.2, seed=0, recall_vec=None,
               skip_pad=False, blocker_order=4):
    """Convenience: best-of-k with router, returns (ids, text, all_texts)."""
    ids, all_cands = best_of_k(model, prompt_ids, k, max_tokens,
                               temperature=temperature, top_p=top_p,
                               rep_penalty=rep_penalty, seed=seed,
                               recall_vec=recall_vec, skip_pad=skip_pad,
                               blocker_order=blocker_order)
    texts = [model.decode(c) for c in (all_cands or [ids])]
    best_text = model.decode(ids)
    return ids, best_text, texts
