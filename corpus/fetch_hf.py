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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["simplewiki", "dailydialog"])
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()
    if args.source == "simplewiki":
        fetch_simplewiki(max_docs=args.max_docs)
    else:
        fetch_dailydialog()
