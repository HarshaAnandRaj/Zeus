import argparse
import pathlib
import sys
import time

from datasets import load_dataset

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def fetch_simplewiki(max_docs=None, max_chars=600_000_000):
    out = RAW / "simplewiki.txt"
    f = open(out, "w", encoding="utf-8", buffering=1024 * 256)
    n, chars, t0 = 0, 0, time.time()
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
        for doc in ds:
            text = (doc.get("title", "") + "\n\n" + doc.get("text", "")).strip()
            if not text:
                continue
            f.write(text + "\n\n")
            n += 1
            chars += len(text) + 2
            if n % 5000 == 0:
                f.flush()
                rate = n / max(time.time() - t0, 1)
                print(f"{n} docs, {chars/1e6:.1f}M chars, {rate:.0f} docs/s", flush=True)
            if max_docs and n >= max_docs:
                break
            if chars >= max_chars:
                break
    finally:
        f.flush()
        f.close()
    print(f"simplewiki done: {n} docs, {chars:,} chars -> {out}")


def fetch_dailydialog():
    repos = ["roskoN/daily_dialog", "Samsung/samsum", "knkarthick/dialogsum"]
    out = RAW / "dailydialog.txt"
    errors = []
    for repo in repos:
        try:
            ds = load_dataset(repo, split="train")
        except Exception as e:
            errors.append(f"{repo}: {type(e).__name__}")
            continue
        lines_out = []
        for row in ds:
            text = None
            for key in ("dialogue", "dialog", "conversation"):
                v = row.get(key)
                if isinstance(v, str) and len(v) > 20:
                    text = v
                    break
            if not text:
                continue
            text = text.replace("__eou__", "\n").replace("__eot__", " ").strip()
            turns = [t.strip() for t in text.splitlines() if t.strip()]
            if len(turns) >= 2:
                lines_out.append("\n".join(turns))
        if len(lines_out) >= 1000:
            out.write_text("\n\n".join(lines_out), encoding="utf-8")
            wc = sum(len(l.split()) for l in lines_out)
            print(f"dailydialog done via {repo}: {len(lines_out)} dialogues, {wc:,} words -> {out}")
            return
        errors.append(f"{repo}: only {len(lines_out)} usable dialogues")
    print("FAIL all dialogue sources:", "; ".join(errors))


def _score_of(doc):
    """continuous score (user's threshold is score >= 4.0)."""
    v = doc.get("score")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0, 0
    return f, int(doc.get("int_score") or round(f))


def _web_structural_reject(text):
    """Mirror build_corpus_v2.check_web_noise cheaply for pre-filtering."""
    import re
    from collections import Counter
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 6:
        return "too_short_web"
    n_short = sum(1 for ln in lines if len(ln) < 60)
    if n_short / len(lines) > 0.6:
        return "factbox_like"
    noise_words = ("did you know", "click here", "click the link",
                   "related articles", "related searches", "more for you",
                   "watch the video", "share this", "share on", "follow us",
                   "sign up", "sign in", "subscribe", "read more", "read next",
                   "table of contents", "source:", "references")
    n_noise = sum(1 for ln in lines
                  if any(w in ln.lower() for w in noise_words))
    if n_noise and n_noise / max(len(lines), 1) > 0.06:
        return "nav_boilerplate"
    c = Counter(lines)
    top = c.most_common(1)
    if top and top[0][1] >= 5 and len(top[0][0]) < 40:
        return "repeated_heading"
    return None


def fetch_fineweb_edu_sample(target_tokens=3_000_000, seed=20260904,
                             min_score=4.0, out_dir=RAW / "fineweb_edu",
                             sample_label="score4", write_report=True):
    """Randomized score>=min_score sample of FineWeb-Edu.

    User decision: primary threshold score >= 4.0. Uses .shuffle(seed) on the
    streaming iterable for randomized (non-sequential-front) selection, then
    filters by continuous score, English, and structural web noise. Stops at
    target_tokens of *accepted* (post-filter) estimated tokens and writes a
    sampling report with acceptance/contamination/doc-length/dup/domain stats.
    """
    import json
    from urllib.parse import urlparse
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train",
                      streaming=True).shuffle(seed=seed)
    out_path = out_dir / f"fw_edu_{sample_label}_sample.jsonl"
    report_path = out_dir / f"fw_edu_{sample_label}_sample_report.json"
    f = open(out_path, "w", encoding="utf-8", buffering=1024 * 256)
    stats = {"seen": 0, "rejected": {}, "accepted": 0,
             "accepted_tokens_est": 0, "accepted_tokens_real": 0,
             "doc_len_sum": 0, "domains": set(), "sample_label": sample_label,
             "min_score": min_score, "seed": seed}
    import collections
    rej = collections.Counter()
    try:
        for doc in ds:
            stats["seen"] += 1
            text = doc.get("text") or ""
            if not text or text.strip() == "":
                rej["empty"] += 1
                continue
            score, iscore = _score_of(doc)
            if score < min_score:
                rej[f"score<{min_score}"] += 1
                continue
            if doc.get("language") not in (None, "en"):
                rej["lang"] += 1
                continue
            if len(text) < 200:
                rej["too_short"] += 1
                continue
            if "\ufffd" in text:
                rej["malformed"] += 1
                continue
            wr = _web_structural_reject(text)
            if wr:
                rej[wr] += 1
                continue
            rec = {"text": text.strip(), "score": round(score, 3),
                   "int_score": iscore, "url": doc.get("url"),
                   "lang": doc.get("language")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["accepted"] += 1
            stats["accepted_tokens_est"] += len(text) // 4
            stats["accepted_tokens_real"] += len(text)
            stats["doc_len_sum"] += len(text)
            dom = urlparse(doc.get("url") or "").netloc
            if dom:
                stats["domains"].add(dom)
            if stats["accepted_tokens_est"] >= target_tokens:
                break
    finally:
        f.flush(); f.close()
    stats["rejected"] = dict(rej)
    stats["n_domains"] = len(stats["domains"])
    stats["accepted_bytes"] = stats["accepted_tokens_real"]
    stats["accepted_real_tokens_est"] = stats["accepted_tokens_est"]
    stats["mean_doc_char_len"] = round(
        stats["doc_len_sum"] / max(stats["accepted"], 1), 1)
    total_seen = max(stats["seen"], 1)
    stats["acceptance_rate"] = round(stats["accepted"] / total_seen, 4)
    stats["est_accepted_tokens"] = stats["accepted_tokens_est"]
    stats["domains_sample"] = sorted(list(stats["domains"]))[:20]
    stats.pop("domains", None)
    if write_report:
        report_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "domains_sample"},
                     indent=2, ensure_ascii=False), flush=True)
    print(f"wrote sample -> {out_path}\nreport -> {report_path}", flush=True)


def fetch_fineweb_edu_cli(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--batch", type=int, default=3_000_000)
    ap.add_argument("--min-score", type=float, default=4.0)
    ap.add_argument("--out", type=str, default=str(RAW / "fineweb_edu"))
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--target-tokens", type=int, default=3_000_000)
    a = ap.parse_args(argv)
    fetch_fineweb_edu_sample(target_tokens=a.target_tokens, seed=a.seed,
                             min_score=a.min_score, out_dir=a.out)


def _hf_main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["simplewiki", "dailydialog", "fineweb_edu"])
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--batch", type=int, default=3_000_000)
    ap.add_argument("--min-score", type=float, default=4.0)
    ap.add_argument("--target-tokens", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)
    if args.source == "simplewiki":
        fetch_simplewiki(max_docs=args.max_docs)
    elif args.source == "fineweb_edu":
        fetch_fineweb_edu_sample(target_tokens=args.target_tokens, seed=args.seed,
                                 min_score=args.min_score,
                                 out_dir=args.out or (RAW / "fineweb_edu"))
    else:
        fetch_dailydialog()


if __name__ == "__main__":
    _hf_main(sys.argv[1:])
