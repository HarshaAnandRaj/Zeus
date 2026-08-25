"""HCM proficiency probe (M1-M4). Operates on memory/hcm/ + session logs.
Gracefully reports INSUFFICIENT DATA until the model writes and sessions exist."""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
HCM = ROOT / "memory" / "hcm"
CATEGORIES = ("state", "people", "scratch")


def ensure_dirs():
    for c in CATEGORIES:
        (HCM / c).mkdir(parents=True, exist_ok=True)


def parse_front_matter(text):
    meta = {}
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta


def entry_stats():
    ensure_dirs()
    stats = {}
    for c in CATEGORIES:
        files = list((HCM / c).glob("*.md"))
        entries = []
        for f in files:
            meta = parse_front_matter(f.read_text(encoding="utf-8"))
            entries.append({"name": f.name, "meta": meta})
        stats[c] = {"count": len(files), "entries": entries}
    return stats


def session_logs():
    log_dirs = sorted((ROOT / "experiments").glob("*/session_log.jsonl"))
    return [str(p) for p in log_dirs]


def main(model=None):
    out = {"hcm": entry_stats(), "session_logs": session_logs()}
    total_entries = sum(v["count"] for v in out["hcm"].values())
    if total_entries == 0 or not out["session_logs"]:
        out["verdict"] = "INSUFFICIENT DATA (no HCM writes or no session logs yet)"
        out.update({"M1_write_discipline": None, "M2_recall_utility": None,
                    "M3_delayed_recall_Dmem": None, "M4_profile_fidelity": None})
    else:
        out["verdict"] = "DATA PRESENT - M-metrics pending trainer integration (P4)"
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
