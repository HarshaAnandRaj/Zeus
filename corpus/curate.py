"""corpus/curate.py -- remove bad data from the raw corpus and rebalance.

Reads raw/simplewiki.txt (the 96% bulk), strips junk blocks/lines, and keeps a
random 35% of what remains into raw/wiki_clean.txt. The original file is renamed
to simplewiki_orig.txt so the full dump is never lost. Rerun build_corpus.py
afterwards to regenerate train.txt/val.txt/stats.json.
"""
import pathlib
import random
import re
import shutil

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
SRC = RAW / "_archived" / "simplewiki.txt"
OUT = RAW / "wiki_clean.txt"
KEEP = 0.22
SEED = 1337

JUNK_LINE = re.compile(
    r"^\s*(?:"
    r"Category:|References?\s*\(?List\)?|External links?|Related pages?|See also|"
    r"Further reading|Notes?|Sources|Retrieved from|Stub|This article is a stub|"
    r"(?:From Wikipedia|Creative Commons|Wikipedia'?s|Help:|Special:|Geographic coordinate)|"
    r"\d{4}\s+births|\d{4}\s+deaths|\bLiving people\b|\bInfobox\b|\bCoordinates\b|"
    r"ISBN(?:-13|-10)?[: ]|ISSN|OCLC|LCCN|VIAF|GND|Freebase|\bepep\b|\b[0-9A-F]{16,}\b|"
    r"\[citation needed\]|\(?citation needed\)?"
    r")", re.IGNORECASE)

LISTY = re.compile(r"^[ \t]*[A-Za-z][^.!?\n]{0,80},\s+\d{1,3}(?:,\s*\d{4})?[^\n]*$")


def clean_block(b: str) -> str:
    lines = b.split("\n")
    kept = []
    for ln in lines:
        if JUNK_LINE.search(ln):
            continue
        if len(ln) > 400 and sum(1 for ch in ln if not ch.isalpha() and ch not in " .,;:'\"-()?!") > len(ln) * 0.4:
            continue
        kept.append(ln)
    out = "\n".join(kept)
    return out.strip()


def main() -> None:
    if not SRC.exists():
        print("nothing to curate (no raw/_archived/simplewiki.txt)")
        return
    text = SRC.read_text(encoding="utf-8")
    blocks = [x.strip() for x in text.split("\n\n")]
    cleaned = []
    for b in blocks:
        if len(b) < 200:
            continue
        cb = clean_block(b)
        if len(cb) < 200:
            continue
        cleaned.append(cb)
    rng = random.Random(SEED)
    rng.shuffle(cleaned)
    kept = cleaned[: int(len(cleaned) * KEEP)]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n\n".join(kept) + "\n")
    print(f"wiki: {len(blocks)} raw -> {len(cleaned)} cleaned -> keep "
          f"{len(kept)} ({len(kept)/max(1, len(blocks))*100:.1f}% of raw) -> {OUT}")


if __name__ == "__main__":
    main()