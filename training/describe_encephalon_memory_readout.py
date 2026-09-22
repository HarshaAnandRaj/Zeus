"""Prespecified-anchor descriptions only; no qualification or regime votes."""
import json

import numpy as np

from training.encephalon_memory_readout import (
    ARMS, BODY_IDS, LAGS, LOCK, OUT, PROFILES, STARTS, canonical, digest,
    load_result, result_path, source_guard, write_atomic,
)


def body_row(body, start, lag):
    a = body["anchors"].get(str(start), {})
    if a.get("status") != "OK":
        return None
    g = a["growth"].get(str(lag), {})
    before = a["influence"].get("1", {})
    after = a["influence"].get(str(lag), {})
    if before.get("status") != "OK" or after.get("status") != "OK":
        return None
    ratios = {"hidden": [], "policy": []}
    baselines = {"hidden": [], "policy": []}
    for b, q in zip(before["directions"], after["directions"]):
        for k in ratios:
            v = b[f"{k}_sensitivity"]
            if v <= 1e-10 or b[f"{k}_check"] != "OK" or q[f"{k}_check"] not in ("OK", "NUMERICAL_FLOOR"):
                continue
            ratios[k].append(max(q[f"{k}_sensitivity"], 1e-10) / v)
            baselines[k].append(v)
    if not ratios["hidden"] or not ratios["policy"]:
        return None
    return dict(hidden=float(np.median(ratios["hidden"])),
                policy=float(np.median(ratios["policy"])),
                growth=g.get("maximum") if g.get("status") == "OK" else None,
                baseline_hidden=float(np.median(baselines["hidden"])),
                baseline_policy=float(np.median(baselines["policy"])))


def summarize(rows, profile, start, lag):
    lineages = []
    bodies = 0
    for r in rows:
        p = next(p for p in r["profiles"] if p["profile"] == profile)
        vals = [body_row(b, start, lag) for b in p["bodies"]]
        vals = [v for v in vals if v is not None]
        bodies += len(vals)
        if vals:
            lineages.append({k: float(np.median([v[k] for v in vals if v[k] is not None]))
                             for k in ("hidden", "policy", "baseline_hidden", "baseline_policy")})
            grown = [v["growth"] for v in vals if v["growth"] is not None]
            lineages[-1]["growth"] = float(np.median(grown)) if grown else None
    if not lineages:
        return dict(eligible_bodies=0, eligible_lineages=0)
    return dict(eligible_bodies=bodies, eligible_lineages=len(lineages),
                **{k+"_median": float(np.median([v[k] for v in lineages if v[k] is not None]))
                   for k in ("hidden", "policy", "growth", "baseline_hidden", "baseline_policy")
                   if any(v[k] is not None for v in lineages)})


def main():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    source_guard(lock)
    lock_hash = digest(LOCK)
    rows = {}
    for arm in ARMS:
        for update in (0, 1024, 2048):
            group = []
            for record in lock["records"]:
                if record["arm"] != arm or record["update"] != update:
                    continue
                result = load_result(result_path(record))
                assert result["status"] == "OK" and result["input_lock_sha256"] == lock_hash
                group.append(result)
            assert len(group) == 8
            for profile in PROFILES:
                for start in STARTS:
                    for lag in (16, 32, 64):
                        key = f"{arm}/{update}/{profile}/t{start}/lag{lag}"
                        rows[key] = summarize(group, profile, start, lag)
            print(f"described {arm} update {update}", flush=True)
    write_atomic(OUT / "descriptions.json", canonical(dict(input_lock_sha256=lock_hash, rows=rows)))


if __name__ == "__main__":
    main()
