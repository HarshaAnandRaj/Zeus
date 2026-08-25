import hashlib
import json
import pathlib
import random
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
DATA = HERE / "data"
DATA.mkdir(parents=True, exist_ok=True)


def normalize(block: str) -> str | None:
    text = unicodedata.normalize("NFKC", block)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or 32 <= ord(ch) < 0x110000)
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    text = text.strip("\n").strip()
    if len(text) < 200:
        return None
    return text


def main() -> None:
    blocks: list[str] = []
    stats_sources = {}
    for path in sorted(RAW.glob("*.txt")):
        raw_blocks = path.read_text(encoding="utf-8").split("\n\n")
        kept = []
        for b in raw_blocks:
            nb = normalize(b)
            if nb:
                kept.append(nb)
        digest = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
        h = hashlib.sha1()
        seen = set()
        uniq = []
        for kb in kept:
            key = hashlib.sha1(kb.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(kb)
        del h, digest
        blocks.extend(uniq)
        stats_sources[path.name] = {"blocks": len(uniq)}
    rng = random.Random(1337)
    rng.shuffle(blocks)
    n_val = max(100, int(len(blocks) * 0.01))
    val, train = blocks[:n_val], blocks[n_val:]
    train_text = "\n\n".join(train) + "\n"
    val_text = "\n\n".join(val) + "\n"
    (DATA / "train.txt").write_text(train_text, encoding="utf-8")
    (DATA / "val.txt").write_text(val_text, encoding="utf-8")
    def wc(t: str) -> int:
        return sum(len(ln.split()) for ln in t.splitlines())
    stats = {
        "sources": stats_sources,
        "blocks": len(blocks),
        "train_words": wc(train_text),
        "val_words": wc(val_text),
        "train_bytes": len(train_text.encode()),
        "val_bytes": len(val_text.encode()),
    }
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
