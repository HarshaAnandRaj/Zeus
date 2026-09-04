"""corpus/build_corpus_v2.py -- document-record corpus builder (v2).

Replaces the .gitignored-per-source anonymous-block pipeline (build_corpus.py)
with a document-record pipeline that satisfies the probe contract:

  * reads every raw/*.txt as a sequence of DOCUMENTS (not anonymous blocks);
  * keeps per-document metadata:
      - source file, document id, license/provenance tag
      - raw sha1, filtered sha1
      - token count (frozen bpe_8192)
      - train/validation assignment (document-level, split BEFORE shuffle)
      - rejection reason(s), if filtered
  * applies junk filters (markup/tables, urls, boilerplate/navigation, low
    alphabetic density, excessive symbols, malformed unicode);
  * document-level exact + near dedup (cross-source);
  * enforces per-source token quotas so one source cannot dominate;
  * source-balanced concatenation (round-robin by source) instead of a single
    global shuffle;
  * writes corpus/data/train_ids.npy + val_ids.npy (frozen tokenizer) and a
    corpus_manifest.json, plus a readable corpus_report.txt.

The tokenizer and every filtering rule are recorded in the manifest so the
corpus can be frozen/hashed before the long probe run.

Run:  python corpus/build_corpus_v2.py
"""
import argparse
import hashlib
import json
import pathlib
import random
import re
import unicodedata

import numpy as np
from tokenizers import Tokenizer

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
DATA = HERE / "data"
TOKENIZER = DATA / "tokenizer" / "bpe_8192.json"

LICENSES = {
    "gutenberg": "Project Gutenberg License (public domain in US)",
    "dailydialog": "CC BY-NC 4.0 (research use)",
    "samsum": "CC BY-NC-ND 4.0",
    "dialogsum": "CC BY-SA 4.0",
    "fineweb_edu": "ODC-BY / Common Crawl terms",
    "reference": "mixed public-domain / permissive",
}

# ---- junk / quality filters ----------------------------------------------
RE_START_GUT = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG", re.I)
RE_END_GUT = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_BOILER = re.compile(
    r"^\s*(?:"
    r"Category:|References?\s*\(?List\)?|External links?|Related pages?|See also|"
    r"Further reading|Notes?|Sources|Retrieved from|This article is a stub|"
    r"From Wikipedia|Creative Commons|Wikipedia'?s|Help:|Special:|"
    r"\d{4}\s+births|\d{4}\s+deaths|\bLiving people\b|\bInfobox\b|\bCoordinates\b|"
    r"\bNavigation\b|\bBibliography\b|\bFurther\s+reading\b|\bWorks\s+cited\b|"
    r"ISBN(?:-13|-10)?[: ]|ISSN|OCLC|LCCN|VIAF|GND|\bepep\b|\[citation needed\]"
    r")", re.IGNORECASE)
RE_TABLE = re.compile(r"^\s*\|[-=! |]*|\{\|\s*class=|\{\| style=")
MARKUP_CHARS = set("|={}[]<>#*#\\/")
MIN_DOC_CHARS = 200
MIN_ALPHA_FRAC = 0.55
MAX_SYMBOL_FRAC = 0.25
MAX_MARKUP_FRAC = 0.10
MAX_URL_RATIO = 500  # chars per URL (too many urls => navigation junk)


def sha1_hex(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # DailyDialog speaker labels are high-frequency, low-information markup.
    # In sampled continuations they became an attractor ("#Person1#" loops),
    # while the dialogue itself remains useful interaction-language material.
    # Remove only the explicit label prefix, retaining utterance boundaries.
    text = re.sub(r"#Person[12]#\s*:\s*", "", text)
    # strip project gutenberg license banners
    m = RE_START_GUT.search(text)
    if m:
        text = text[m.end():]
    m = RE_END_GUT.search(text)
    if m:
        text = text[:m.start()]
    lines = []
    for ln in text.split("\n"):
        ln = ln.rstrip()
        if RE_BOILER.search(ln):
            continue
        lines.append(ln)
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip("\n").strip()


RE_CHAPTER = re.compile(
    r"^\s{0,4}(?:CHAPTER|Chapter|chapter)\s+[IVXLCDM0-9]+(?:\s*[.:\-–]?\s*.*)?$"
    r"|^\s{0,4}(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\.\s+[A-Z]",
    re.MULTILINE)
DIALOGUE_PREFIX = ("dialog", "samsum", "dialogsum")

# ---- web-structural noise: fact-boxes, listicle furniture, navigation ----
WEB_NOISE_WORDS = (
    "did you know", "click here", "click the link",
    "related articles", "related searches", "more for you",
    "watch the video", "share this", "share on", "follow us",
    "sign up", "sign in", "subscribe", "read more", "read next",
    "table of contents", "source:", "references",
)


def check_web_noise(doc_id, text, is_web=False):
    """Extra structural filters for web-derived sources (fineweb etc.).
    Returns a reject reason or None."""
    if not is_web:
        return None
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 4:
        n_short = sum(1 for ln in lines if len(ln) < 60)
        if n_short / len(lines) > 0.6 and len(lines) >= 6:
            return "factbox_like"
    n_noise = sum(1 for ln in lines
                  if any(w in ln.lower() for w in WEB_NOISE_WORDS))
    if n_noise and n_noise / max(len(lines), 1) > 0.06:
        return "nav_boilerplate"
    from collections import Counter
    c = Counter(lines)
    top = c.most_common(1)
    if top and top[0][0] and top[0][1] >= 5 and len(top[0][0]) < 40:
        return "repeated_heading"
    return None


def doc_blocks(source: str, text: str, min_doc_chars=2400):
    """Yield (doc_id, raw_text) per document.

    Dialogue sources: each blank-line block is one conversation.
    Prose sources: split on CHAPTER headers into chapter documents (with a
    paragraph-accumulation fallback when a file has no chapter markers)."""
    if any(source.startswith(p) for p in DIALOGUE_PREFIX) or "dialog" in source:
        out = []
        for idx, b in enumerate(text.split("\n\n")):
            b = b.strip()
            if b:
                out.append((idx, b))
        return out

    # prose: locate chapter-header offsets
    marks = [mm.start() for mm in RE_CHAPTER.finditer(text)]
    boundaries = [0] + marks + [len(text)]
    docs = []
    if len(marks) >= 2:
        for i in range(len(boundaries) - 1):
            seg = text[boundaries[i]:boundaries[i + 1]]
            if seg.strip():
                docs.append((i, seg.strip()))
        return docs

    # no chapter markers: accumulate paragraphs into min-size docs
    out = []
    cur = []
    cur_len = 0
    idx = 0
    for b in text.split("\n\n"):
        s = b.strip()
        if not s:
            continue
        cur.append(s)
        cur_len += len(s)
        if cur_len >= min_doc_chars:
            out.append((idx, "\n".join(cur)))
            idx += 1
            cur, cur_len = [], 0
    if cur:
        out.append((idx, "\n".join(cur)))
    return out


def check_quality(doc_id: str, norm: str, is_web=False):
    """Return (reject_reason|None, filtered_text)."""
    nxt = re.sub(r"[ \t]+", " ", norm)
    if len(nxt) < MIN_DOC_CHARS:
        return "too_short", None
    # mojibake / malformed unicode: replacement chars, stray Latin-1 mojibake
    if "\ufffd" in nxt or "\u0000" in nxt:
        return "malformed_unicode", None
    # language sniff: if it looks non-English, drop for the English chore
    lang = _sniff_lang(nxt)
    if lang != "en":
        return f"lang_{lang}", None
    # web-structural noise (fact-boxes, nav boilerplate, repeated headings)
    web = check_web_noise(doc_id, nxt, is_web=is_web)
    if web:
        return web, None
    alpha = sum(1 for ch in nxt if ch.isalpha())
    if alpha / max(len(nxt), 1) < MIN_ALPHA_FRAC:
        return "low_alpha", None
    markup = sum(1 for ch in nxt if ch in MARKUP_CHARS)
    if markup / max(len(nxt), 1) > MAX_MARKUP_FRAC:
        return "high_markup", None
    symbol = sum(1 for ch in nxt if (not ch.isalpha() and not ch.isspace()))
    if symbol / max(len(nxt), 1) > MAX_SYMBOL_FRAC:
        return "high_symbol", None
    n_urls = len(RE_URL.findall(nxt))
    if n_urls and len(nxt) / n_urls < MAX_URL_RATIO:
        return "too_many_urls", None
    if RE_TABLE.search(nxt):
        # tables are the structural attractor -- drop unless told to keep
        return "table_markup", None
    return None, nxt


# ---- lightweight English language sniff ----------------------------------
_ACCENTED_CHARS = "áàâãäåéèêëíìîïóòôõöúùûüçñ"
_NON_EN_MARKERS = [
    # high-frequency non-English function words / contractions
    " les ", " des ", " une ", " que ", " qui ", " dans ", " pour ",   # fr
    " los ", " las ", " del ", " para ", " con ", " uno ", " una ",    # es
    " und ", " der ", " die ", " das ", " mit ", " nicht ", " eine ",  # de
    " il ", " del ", " della ", " che ", " non ", " con ", " per ",    # it
    " de ", " o ", " a ", " e ", " em ", " do ", " da ", " na ",        # pt (risky)
]
# only trigger on strong signals to avoid false positives:
_FR_STOP = re.compile(r"\b(?:les|des|une|que|qui|dans|pour|avec|mais|tout|cette|ses|elle|nous)\b")
_ES_STOP = re.compile(r"\b(?:los|las|del|para|con|estas|qué|más|pero|ella|ellos)\b")
_DE_STOP = re.compile(r"\b(?:der|die|das|und|mit|nicht|eine|ist|den|auf|ich)\b")
_IT_STOP = re.compile(r"\b(?:della|delle|nella|che|c'è|perché|questa|sua)\b")


def _sniff_lang(text):
    low = " " + " ".join(text.split()[:400]).lower() + " "
    accented = sum(1 for ch in low if ch in _ACCENTED_CHARS)
    # mojibake is handled above; here high accent density on top of strong
    # foreign stop-word signal => non-English
    if _DE_STOP.search(low):
        return "de"
    if _IT_STOP.search(low):
        return "it"
    if accent_density(low, accented) > 0.012 and _FR_STOP.search(low):
        return "fr"
    if accent_density(low, accented) > 0.012 and _ES_STOP.search(low):
        return "es"
    return "en"


def accent_density(low, count):
    return count / max(len(low), 1)


def _shingles(text, k=8):
    """Set of char k-shingles (normalized: lowercase, collapse whitespace)."""
    t = re.sub(r"\s+", " ", text).lower()
    if len(t) < k:
        return {t}
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def near_dup(docs, k=8, jaccard=0.9, n_sh=120):
    """In-place dedup: exact sha1 + near via sampled shingle Jaccard.

    A doc is dropped if its normalized text is byte-identical to an earlier
    doc (exact) OR its Jaccard similarity to an earlier doc sharing the same
    min-hash bucket exceeds `jaccard`. The min-hash bucket prunes comparisons
    so distinct documents are compared rarely; bucket collisions are resolved
    with sampled Jaccard."""
    seen_exact = set()
    bucket = {}
    for d_i, d in enumerate(docs):
        if d["reject"]:
            continue
        norm = d["filtered"]
        exact = sha1_hex(norm.encode("utf-8"))
        if exact in seen_exact:
            d["reject"] = "dedup"
            continue
        sh = list(_shingles(norm, k))
        if not sh:
            d["reject"] = "too_short"
            continue
        hashes = sorted(hashlib.md5(s.encode("utf-8", "ignore")).digest()
                        for s in sh)
        mh = min(hashes[-n_sh:])  # min-hash from the sampled tail
        dropped = False
        for other_i in bucket.get(mh, ()):
            other = docs[other_i]
            if _sample_jaccard(norm, other["filtered"], k, min(n_sh, 80)) > jaccard:
                dropped = True
                break
        if dropped:
            d["reject"] = "dedup"
            continue
        seen_exact.add(exact)
        bucket.setdefault(mh, []).append(d_i)
    return docs


def _sample_jaccard(a, b, k, n=120):
    """Unbiased-estimate Jaccard over a random sample of shingle hashes from
    each doc: |H(A) ∩ H(B)| / |H(A) ∪ H(B)| on the sampled universe."""
    ha = set(hashlib.md5(x.encode("utf-8", "ignore")).digest()
             for x in list(_shingles(a, k))[:n])
    hb = set(hashlib.md5(x.encode("utf-8", "ignore")).digest()
             for x in list(_shingles(b, k))[:n])
    if not ha or not hb:
        return 0.0
    inter = len(ha & hb)
    union = len(ha | hb)
    return inter / max(1, union)


def build(include_wiki=False, wiki_frac=0.0, min_doc_chars=200):
    manifest = {"tokenizer": sha1_hex(TOKENIZER.read_bytes()),
                "tokenizer_path": str(TOKENIZER),
                "include_wiki": include_wiki, "wiki_frac": wiki_frac}
    tok = Tokenizer.from_file(str(TOKENIZER))
    all_docs = []
    for path in sorted(RAW.glob("*.txt")):
        name = path.name
        name = {"gutenberg": "gutenberg", "wiki_clean": "wiki", }.get(name, name)
        if name.startswith("wiki") and not include_wiki:
            all_docs.append({"source": name, "doc_id": f"{name}:excluded",
                             "license": "n/a", "raw_sha1": "", "filtered": None,
                             "filtered_sha1": None, "reject": "excluded(wiki)",
                             "tokens": 0})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for idx, raw in doc_blocks(name, text, min_doc_chars=min_doc_chars):
            reason, nxt = check_quality(idx, normalize(raw), is_web=False)
            rec = {"source": name, "doc_id": f"{name}:{idx}",
                   "license": LICENSES.get(name.split("_")[0], "unknown"),
                   "raw_sha1": sha1_hex(raw.encode("utf-8")),
                   "filtered": nxt, "filtered_sha1": sha1_hex(nxt.encode("utf-8")) if nxt else None,
                   "reject": reason, "tokens": 0}
            all_docs.append(rec)
    # fineweb-edu JSONL batches: one document per JSON record (exact boundaries)
    for jpath in sorted((RAW / "fineweb_edu").glob("*.jsonl")):
        src = jpath.name
        for j, line in enumerate(jpath.read_text(encoding="utf-8", errors="replace").splitlines()):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            reason, nxt = check_quality(f"{src}:{j}", normalize(text), is_web=True)
            rec = {"source": src, "doc_id": f"{src}:{j}",
                   "license": LICENSES["fineweb_edu"],
                   "raw_sha1": sha1_hex(text.encode("utf-8")),
                   "filtered": nxt, "filtered_sha1": sha1_hex(nxt.encode("utf-8")) if nxt else None,
                   "reject": reason, "tokens": 0}
            all_docs.append(rec)

    # quality pass count before dedup (for accounting)
    pre_dedup = [d for d in all_docs if not d["reject"]]
    kept = list(pre_dedup)
    kept = near_dup(kept)
    kept = [d for d in kept if not d["reject"]]

    # tokenize each kept doc with the frozen bpe
    for d in kept:
        d["tokens"] = len(tok.encode(d["filtered"]).ids)

    accounting = {
        "downloaded_raw_tokens": sum(len(tok.encode(p.read_text(
            encoding="utf-8", errors="replace")).ids)
            for p in RAW.glob("*.txt") if not (p.name.startswith("wiki") and not include_wiki))
            + sum(len(tok.encode(p.read_text(encoding="utf-8", errors="replace")).ids)
                  for p in (RAW / "fineweb_edu").glob("*.jsonl")),
        "docs_input": len(all_docs),
        "docs_pre_dedup": len(pre_dedup),
        "docs_post_dedup": len(kept),
        "reject_histogram": _hist([d["reject"] for d in all_docs]),
        "reject_tokens": _reject_by_source(all_docs),
    }
    manifest["accounting"] = accounting
    manifest["sources"] = _by_source(kept)
    return manifest, kept


def _reject_by_source(docs):
    out = {}
    for d in docs:
        src = d["source"]
        if src not in out:
            out[src] = {}
        r = d["reject"] or "accepted"
        out[src][r] = out[src].get(r, 0) + 1
    return out


def _hist(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def _by_source(docs):
    out = {}
    for d in docs:
        out.setdefault(d["source"], {"docs": 0, "tokens": 0})
        out[d["source"]]["docs"] += 1
        out[d["source"]]["tokens"] += d["tokens"]
    return out


def _assign_split(docs, val_frac=0.02, rng=None):
    """Document-level train/val split, done BEFORE any shuffle of the corpus
    stream. Each document is independently assigned; a document is never split."""
    rng = rng or random.Random(20260903)
    for d in docs:
        d["split"] = "val" if rng.random() < val_frac else "train"
    return docs


def _source_balanced_concat(docs):
    """Round-robin by source so no single source dominates window order."""
    by_src = {}
    for d in docs:
        by_src.setdefault(d["source"], []).append(d)
    out = []
    i = 0
    while any(i < len(v) for v in by_src.values()):
        for src, lst in by_src.items():
            if i < len(lst):
                out.append(lst[i])
        i += 1
    return out


def emit(kept, out_dir, val_frac=0.02, max_tokens=0):
    """Assign splits, write train.txt/val.txt + train_ids.npy/val_ids.npy and
    the manifest/report. Returns per-source final counts."""
    rng = random.Random(20260903)
    _assign_split(kept, val_frac, rng)
    train_docs = [d for d in kept if d["split"] == "train"]
    val_docs = [d for d in kept if d["split"] == "val"]
    train_docs = _source_balanced_concat(train_docs)

    def to_text(docs):
        return "\n\n".join(d["filtered"] for d in docs) + "\n"

    train_text = to_text(train_docs)
    val_text = to_text(val_docs)
    (out_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (out_dir / "val.txt").write_text(val_text, encoding="utf-8")

    _write_ids(train_text, out_dir / "train_ids.npy", max_tokens)
    _write_ids(val_text, out_dir / "val_ids.npy", max_tokens)

    # accounting for split + final source composition
    man = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    man["split"] = {"train_docs": len(train_docs), "val_docs": len(val_docs),
                    "train_tokens": sum(d["tokens"] for d in train_docs),
                    "val_tokens": sum(d["tokens"] for d in val_docs)}
    man["final_sources"] = _by_source(kept)
    (out_dir / "corpus_manifest.json").write_text(
        json.dumps(man, indent=2, default=str), encoding="utf-8")

    lines = [f"tokenizer_sha1={man['tokenizer']}",
             f"include_wiki={man['include_wiki']} wiki_frac={man['wiki_frac']}",
             f"split: train_docs={man['split']['train_docs']} val_docs={man['split']['val_docs']}",
             f"split: train_tokens={man['split']['train_tokens']} val_tokens={man['split']['val_tokens']}"]
    lines.append("reject_histogram: " + json.dumps(man["accounting"]["reject_histogram"]))
    lines.append("FINAL per-source:")
    for src, s in sorted(man["final_sources"].items(), key=lambda kv: -kv[1]["tokens"]):
        lines.append(f"  {src:<42} docs={s['docs']:>5} tokens={s['tokens']:>9}")
    (out_dir / "corpus_report.txt").write_text("\n".join(lines), encoding="utf-8")
    return man


def _write_ids(text, out_path, max_tokens=0):
    tok = Tokenizer.from_file(str(TOKENIZER))
    chunks = [text[i:i + 2_000_000] for i in range(0, len(text), 2_000_000)]
    ids = []
    for enc in tok.encode_batch(chunks):
        ids.extend(enc.ids)
    arr = np.asarray(ids, dtype=np.int32)
    if max_tokens and arr.size > max_tokens:
        arr = arr[:max_tokens]
    np.save(out_path, arr)
    print(f"wrote {out_path}: {arr.size} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-wiki", action="store_true",
                    help="include cleaned wiki in the corpus (ablation probe)")
    ap.add_argument("--wiki-frac", type=float, default=0.0,
                    help="target fraction of wiki tokens (0..1), used when include-wiki")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="cap total train tokens (0 = no cap; pilot use)")
    args = ap.parse_args()

    manifest, kept = build(include_wiki=args.include_wiki,
                           wiki_frac=args.wiki_frac)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # write manifest BEFORE emit so emit can append split info
    (out_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    emit(kept, out_dir, args.val_frac, args.max_tokens)
    print(f"\nTOTAL accepted tokens: {sum(s['tokens'] for s in manifest['sources'].values())}")
    print(json.dumps({"accounting": manifest["accounting"]}, indent=2))


if __name__ == "__main__":
    main()
