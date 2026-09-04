"""training/free_run_gate.py -- authoritative free-run voice evaluator.

Built as the trusted metric layer for the corpus probe and the strict voice
gate, replacing the decode-time word-scoring in training/decode_robust.py as
the *decision* gate (decode_robust metrics remain diagnostics only).

Two failure modes it must catch that word-based scoring missed:

  1. TOKEN-FRAGMENT loops (broca: "Geva Geva More Fire Gra", "ctirdctird"):
     sub-word repeats that word-level metrics rewrite into "legible" prose.
  2. STRUCTURAL-TEMPLATE loops (stage1c: the "|| LINEAR || Socorro || Kitt
     Peak || align=right | 5.3 km ||" wiki-table attractor): token n-gram
     diversity can reach 100% while the reply is unusable markup-junk.

Everything here is pure string/id math (no model calls), so it runs cheaply
per reply and is deterministic given the token list.

Core metric contract
--------------------
A reply is scored into:

  * rep_onset         first token index where ANY repeat of an n-gram (n in
                      2..order), OR a token-fragment repeat, OR a periodic
                      template starts. None (inf) if the reply is fully novel.
  * rep_span_frac     fraction of token positions covered by a repeated
                      (exact n-gram OR fragment) span.
  * frac_unique{2,3,4} normalized token n-gram novelty (may be 1.0 while the
                      reply is still a template loop -- NOT a pass signal).
  * alpha_frac        alphabetic character fraction.
  * markup_frac       table/markup/symbol-pipe fraction (|,=,{,},##,[[ etc).
  * symbol_frac       non-alpha non-space character fraction.
  * word_frac         word-like-token fraction (has >=2 alpha chars).
  * frag_frac         fraction of tokens that are fragment-shaped
                      (all-lower alpha, no word chars, repeats a prefix).
  * sustained_leg     minimum-over-windows word legibility, so a reply is
                      only "legible" if it stays legible through the whole
                      reply, not a good first few tokens.
  * periodic          detected template period (stride) or None.

Gate decision (GateResult)
--------------------------
  passes: All of
      * rep_onset is None or >= min_onset (default 8)
      * rep_span_frac <= max_rep_span (default 0.30)
      * sustained_leg >= min_sustained_leg (default 0.60)
      * alpha_frac >= min_alpha (default 0.60)
      * word_frac >= min_word_frac (default 0.55)
      * markup_frac <= max_markup (default 0.10)
      * symbol_frac <= max_symbol (default 0.06)
      * periodic is None (no clean template periodicity)
      * word-loop coverage <= max_loop_score (default 0.25)

Aggregation returns per-reply pass/fail plus a Wilson score 95% confidence
interval over the pass fraction.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import Counter

__all__ = [
    "analyze", "GateResult", "gate_decision", "passes_gate",
    "WilsonCI", "wilson", "load_known_words", "neolog_frac",
]

MIN_ALPHA = 0.60
MAX_MARKUP = 0.10
MAX_SYMBOL = 0.06
MIN_WORD_FRAC = 0.55
MIN_SUSTAINED_LEG = 0.60
MIN_ONSET = 8
MAX_REP_SPAN = 0.30
MAX_NEOLOG = 0.10
MAX_LOOP_SCORE = 0.25

_DEFAULT_KNOWN = pathlib.Path(__file__).resolve().parent / "data" / "known_words.json"
_KNOWN_CACHE = None

_WORD = re.compile(r"[a-zA-Z]{2,}")
_FRAGMENT = re.compile(r"^[a-z]{2,}$")   # lowercase alpha only, no word boundary
_MARKUP = re.compile(r"[|={}\[\]<>#*#\\/]")
_FRAGRE = "frag"


# ------------------------------------------------------------------ helpers
def _n_gram_repeat_onset(ids, order):
    """First index where any n-gram of size 2..order repeats exactly."""
    n = len(ids)
    for o in range(2, order + 1):
        seen = {}
        for i in range(n - o + 1):
            gram = tuple(ids[i:i + o])
            if gram in seen:
                return seen[gram] + o - 1
            seen[gram] = i
    return None


def _fragment_repeat_onset(ids):
    """First index with a repeating token-fragment run: same token repeated
    directly (frag frag ...) or a token that is a strict prefix repeat."""
    n = len(ids)
    if n < 2:
        return None
    # direct adjacent repeats (covers "frag frag" and char-repeat fragments)
    for i in range(1, n):
        if ids[i] == ids[i - 1]:
            return i
    # repeated short token sequences via token-level bigram
    seen2 = {}
    for i in range(n - 1):
        g = (ids[i], ids[i + 1])
        if g in seen2:
            return i
        seen2[g] = i
    return None


def _periodic_template(ids, min_period=2, max_period=12, min_repeats=3):
    """Detect a clean periodic template in the token-class sequence.

    Maps each token to a coarse class (alpha-word / fragment / markup / digit
    / other) and looks for a repeating period in that class string. Returns
    the stride if a period of length in [min_period, max_period] repeats
    >= min_repeats times cleanly, else None.
    """
    import string as _string
    n = len(ids)
    if n < (min_repeats + 1) * min_period:
        return None
    classes = []
    for t in ids:
        if isinstance(t, str):
            s = t
        else:
            s = str(t)
        if _FRAGMENT.match(s):
            c = "F"
        elif any(ch in _MARKUP_RAW for ch in s):
            c = "M"
        elif any(ch.isdigit() for ch in s):
            c = "D"
        elif _WORD.match(s):
            c = "W"
        else:
            c = "O"
        classes.append(c)
    cl_str = "".join(classes)
    for p in range(min_period, max_period + 1):
        # clean periodicity: whole suffix (from a pivot) repeats the period
        # pattern with high fidelity
        for start in range(0, min(p, n - p)):
            head = cl_str[start:start + p]
            if set(head) == {"M"} or len(set(head)) < 2:
                continue
            reps = 0
            pos = start + p
            while pos + p <= n and cl_str[pos:pos + p] == head:
                reps += 1
                pos += p
            if reps >= min_repeats and pos >= n - 2:
                return p
    return None


_MARKUP_RAW = set("|={}[]<>#*#\\/")


def load_known_words(path=None):
    """Load the known-English-word set (json list). Cached module-wide."""
    global _KNOWN_CACHE
    if _KNOWN_CACHE is not None:
        return _KNOWN_CACHE
    p = pathlib.Path(path) if path else _DEFAULT_KNOWN
    if not p.exists():
        _KNOWN_CACHE = frozenset()
        return _KNOWN_CACHE
    _KNOWN_CACHE = frozenset(json.loads(p.read_text(encoding="utf-8")))
    return _KNOWN_CACHE


def neolog_frac(ids, known=None):
    """Fraction of space-delimited words that are neologisms / invented --
    NOT in the known-English-word set. Flags the invented-lexicon sprawl that
    token/word novelty metrics miss (e.g. 'ottraz', 'inemanzanz')."""
    if known is None:
        known = load_known_words()
    if not known:
        return 0.0
    if isinstance(ids, str):
        text = ids
    else:
        text = " ".join(str(w) for w in ids)
    words = re.findall(r"[a-zA-Z']+", text)
    n = len(words)
    if n == 0:
        return 1.0
    bad = 0
    for w in words:
        wl = w.strip("'").lower()
        # skip single chars and purely-punctuation (not prose junk signals)
        if len(wl) <= 1:
            continue
        if re.match(r"^[a-z']+$", wl) and wl not in known:
            bad += 1
    return bad / n


def _sustained_legibility(words, window=8, step=4):
    """Min word-legibility across sliding windows; None if too short."""
    n = len(words)
    if n < 6:
        return None
    wins = []
    i = 0
    while i + window <= n:
        w = words[i:i + window]
        alpha = sum(1 for ch in " ".join(w) if ch.isalpha())
        wins.append(alpha / max(sum(len(x) for x in w), 1))
        i += step
    return (min(wins) if wins else None)


# ------------------------------------------------------------------ analyze
def analyze(ids, order=4):
    """Return a full metric dict for a token id / string sequence."""
    if isinstance(ids, str):
        words = ids.split()
        text = ids
    else:
        words = ids
        text = " ".join(str(w) for w in ids)

    n = len(ids)
    if n == 0:
        return {"n_tokens": 0, "rep_onset": None, "rep_span_frac": 1.0,
                "frac_unique2": 0.0, "frac_unique3": 0.0, "frac_unique4": 0.0,
                "alpha_frac": 0.0, "markup_frac": 0.0, "symbol_frac": 1.0,
                "word_frac": 0.0, "frag_frac": 1.0, "sustained_leg": 0.0,
                "periodic": None, "loop_score": 1.0}

    # --- repetition onset: earliest of exact n-gram and fragment ---
    onsets = [x for x in (_n_gram_repeat_onset(ids, order),
                          _fragment_repeat_onset(ids)) if x is not None]
    rep_onset = min(onsets) if onsets else None

    # --- repeat span fraction (exact n-gram unions) ---
    covered = set()
    for o in range(2, 5):
        c = Counter(tuple(ids[i:i + o]) for i in range(max(n - o + 1, 0)))
        for i in range(n - o + 1):
            if c[tuple(ids[i:i + o])] > 1:
                covered.update(range(i, i + o))
    rep_span_frac = len(covered) / max(n, 1)

    # --- normalized token n-gram novelty ---
    fu = {}
    for o in (2, 3, 4):
        s = {tuple(ids[i:i + o]) for i in range(max(n - o + 1, 0))}
        fu[f"frac_unique{o}"] = len(s) / max(max(n - o + 1, 1), 1)

    # --- char ratios on decoded text ---
    alpha = sum(1 for ch in text if ch.isalpha())
    alpha_frac = alpha / max(len(text), 1)
    markup_frac = sum(1 for ch in text if ch in _MARKUP_RAW) / max(len(text), 1)
    symbol = sum(1 for ch in text if not ch.isalpha() and not ch.isspace())
    symbol_frac = symbol / max(len(text), 1)

    # --- token shape fractions ---
    toks = [str(t) for t in ids]
    word_toks = [t for t in toks if _WORD.search(t)]
    frag_toks = [t for t in toks if _FRAGMENT.match(t)]
    word_frac = len(word_toks) / max(n, 1)
    frag_frac = len(frag_toks) / max(n, 1)

    sustained_leg = _sustained_legibility(toks)
    periodic = _periodic_template(toks)

    # word-based loop_score for compatibility/diagnostics only
    ws = re.findall(r"[a-z']+", text.lower())
    loop = 0.0
    if len(ws) >= 8:
        poses = {}
        spans = []
        for i in range(len(ws) - 3):
            g = tuple(ws[i:i + 4])
            if g in poses:
                spans.append((poses[g][0], i + 4))
            else:
                poses[g] = (i, i + 4)
        if spans:
            spans.sort()
            cov = 0
            cs, ce = spans[0]
            for s_, e_ in spans[1:]:
                if s_ <= ce:
                    ce = max(ce, e_)
                else:
                    cov += ce - cs
                    cs, ce = s_, e_
            cov += ce - cs
            loop = min(1.0, cov / max(len(ws), 1))

    return {"n_tokens": n, "rep_onset": rep_onset, "rep_span_frac": rep_span_frac,
            **fu, "alpha_frac": alpha_frac, "markup_frac": markup_frac,
            "symbol_frac": symbol_frac, "word_frac": word_frac,
            "frag_frac": frag_frac, "sustained_leg": sustained_leg,
            "periodic": periodic, "loop_score": loop,
            "neolog_frac": neolog_frac(ids)}


# ------------------------------------------------------------------ gate
class GateResult:
    __slots__ = ("metrics", "reasons")

    def __init__(self, metrics, reasons):
        self.metrics = metrics
        self.reasons = reasons

    @property
    def passes(self):
        return not self.reasons

    def __repr__(self):
        return (f"GateResult(passes={self.passes}, "
                f"onset={self.metrics['rep_onset']}, "
                f"rep_span={self.metrics['rep_span_frac']:.2f}, "
                f"leg={self.metrics['sustained_leg']}, "
                f"alpha={self.metrics['alpha_frac']:.2f}, "
                f"markup={self.metrics['markup_frac']:.2f}, "
                f"periodic={self.metrics['periodic']})"
                + (f" FAIL: {self.reasons}" if self.reasons else ""))


def passes_gate(m, min_onset=MIN_ONSET, max_rep_span=MAX_REP_SPAN,
                min_sustained_leg=MIN_SUSTAINED_LEG, min_alpha=MIN_ALPHA,
                min_word_frac=MIN_WORD_FRAC, max_markup=MAX_MARKUP,
                max_symbol=MAX_SYMBOL, max_neolog=MAX_NEOLOG,
                max_loop_score=MAX_LOOP_SCORE):
    """Return list of failed-reason strings (empty == pass)."""
    reasons = []
    if m["rep_onset"] is not None and m["rep_onset"] < min_onset:
        reasons.append(f"early_onset({m['rep_onset']}<{min_onset})")
    if m["rep_span_frac"] > max_rep_span:
        reasons.append(f"rep_span({m['rep_span_frac']:.2f}>{max_rep_span})")
    sl = m["sustained_leg"]
    if sl is None or sl < min_sustained_leg:
        reasons.append(f"sustained_leg({sl})")
    if m["alpha_frac"] < min_alpha:
        reasons.append(f"alpha({m['alpha_frac']:.2f}<{min_alpha})")
    if m["word_frac"] < min_word_frac:
        reasons.append(f"word_frac({m['word_frac']:.2f}<{min_word_frac})")
    if m["markup_frac"] > max_markup:
        reasons.append(f"markup({m['markup_frac']:.2f}>{max_markup})")
    if m["symbol_frac"] > max_symbol:
        reasons.append(f"symbol({m['symbol_frac']:.2f}>{max_symbol})")
    if m["loop_score"] > max_loop_score:
        reasons.append(f"word_loop({m['loop_score']:.2f}>{max_loop_score})")
    if m["periodic"] is not None:
        reasons.append(f"periodic(p={m['periodic']})")
    if m["neolog_frac"] > max_neolog:
        reasons.append(f"neolog({m['neolog_frac']:.2f}>{max_neolog})")
    return reasons


def gate_decision(ids, **kw):
    """Analyze ids then decide pass/fail; returns (GateResult, metrics)."""
    m = analyze(ids, order=kw.pop("order", 4))
    reasons = passes_gate(m, **kw)
    return GateResult(m, reasons), m


# ------------------------------------------------------------------ agg
def wilson(k, n, z=1.96):
    """Wilson score interval for pass fraction k/n."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


class WilsonCI:
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.p, self.lo, self.hi = wilson(k, n)

    def __repr__(self):
        return f"WilsonCI({self.k}/{self.n}, p={self.p:.3f} [{self.lo:.3f},{self.hi:.3f}])"


def aggregate(gate_results):
    """Aggregate GateResult objects into pass fraction + Wilson CI + reason histogram."""
    k = sum(1 for g in gate_results if g.passes)
    n = len(gate_results)
    reasons = Counter()
    for g in gate_results:
        for r in g.reasons:
            key = r.split("(")[0]
            reasons[key] += 1
    return {"pass_frac": k / n if n else 0.0, "n": n, "k": k,
            "ci": (wilson(k, n)[1], wilson(k, n)[2]), "reasons": dict(reasons)}
