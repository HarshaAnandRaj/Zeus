"""Stream frozen E1 readout receipts into a compact audit and lineage summary.

This postprocessing changes no registered statistic: primary_body and
cohort_summary are imported from the pre-readout frozen instrument.
"""
from collections import defaultdict
import json

import numpy as np

from training.encephalon_memory_readout import (
    ARMS, LOCK, OUT, canonical, cohort_summary, digest, load_result,
    primary_body, result_path, source_guard, write_atomic,
)


def main():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    source_guard(lock)
    lock_hash = digest(LOCK)
    receipts = []
    per_checkpoint = []
    final_rows = defaultdict(list)
    profile_rows = defaultdict(list)
    validation = []
    totals = defaultdict(lambda: dict(bodies=0, windows32=0, lags32=0, direction_checks=0,
                                      invalid_direction_checks=0, deaths=[], censored=0))
    for index, record in enumerate(lock["records"], 1):
        path = result_path(record)
        if not path.exists():
            receipts.append(dict(id=record["id"], status="MISSING"))
            continue
        result = load_result(path)
        assert result["id"] == record["id"] and result["input_lock_sha256"] == lock_hash
        receipts.append(dict(id=record["id"], status=result["status"], path=str(path),
                             sha256=digest(path), reason=result.get("reason")))
        if result["status"] != "OK":
            continue
        for profile in result["profiles"]:
            p = profile["profile"]
            bodies = profile["bodies"]
            compact = []
            primary = []
            for body in bodies:
                assert body["body_id"] in record["profiles"][0]["body_ids"]
                key = (result["arm"], p, result["update"])
                tally = totals[key]
                tally["bodies"] += 1
                tally["deaths"].append(body["ticks"])
                tally["censored"] += body["censored"]
                anchor0 = body["anchors"].get("0", {})
                tally["windows32"] += anchor0.get("growth", {}).get("32", {}).get("status") == "OK"
                tally["lags32"] += anchor0.get("influence", {}).get("32", {}).get("status") == "OK"
                if result["update"] == 2048 and p == "balanced" and body["body_id"] == 0:
                    validation.append(dict(arm=result["arm"], lineage=result["lineage"],
                                           finite_difference=body.get("finite_difference"),
                                           qr=body.get("qr")))
                for item in anchor0.get("influence", {}).get("32", {}).get("directions", []):
                    tally["direction_checks"] += 1
                    tally["invalid_direction_checks"] += (
                        item["hidden_check"] == "NONLINEAR_OR_UNRESOLVED"
                        or item["policy_check"] == "NONLINEAR_OR_UNRESOLVED")
                primary.append(primary_body(body))
                compact.append({k: v for k, v in body.items() if k != "trace"})
            per_checkpoint.append(dict(id=result["id"], arm=result["arm"],
                                       lineage=result["lineage"], update=result["update"],
                                       profile=p, body_ids=[b["body_id"] for b in bodies],
                                       ticks=[b["ticks"] for b in bodies],
                                       censored=[b["censored"] for b in bodies],
                                       trace_sha256=[b["trace_sha256"] for b in bodies],
                                       primary=primary))
            if result["update"] == 2048:
                profile_rows[(result["arm"], p)].append(dict(lineage=result["lineage"], bodies=compact))
                if p == "balanced":
                    final_rows[result["arm"]].append(dict(lineage=result["lineage"],
                                                          profiles=[dict(profile=p, bodies=compact)]))
        if index % 100 == 0:
            print(f"audited {index}/816", flush=True)
    assert len(receipts) == 816
    cohorts = {arm: cohort_summary(arm, final_rows[arm]) if len(final_rows[arm]) == 8
               else dict(status="VOID_OR_MISSING", present=len(final_rows[arm])) for arm in ARMS}
    profiles = {}
    for (arm, profile), rows in profile_rows.items():
        bodies = [b for r in rows for b in r["bodies"]]
        med = [primary_body(b) for b in bodies]
        med = [m for m in med if m is not None]
        profiles[f"{arm}/{profile}"] = dict(n=len(bodies), eligible=len(med),
            death_tick_median=float(np.median([b["ticks"] for b in bodies])),
            growth32_median=float(np.median([m["growth"] for m in med])) if med else None,
            hidden_ratio32_median=float(np.median([m["hidden"] for m in med])) if med else None,
            policy_ratio32_median=float(np.median([m["policy"] for m in med])) if med else None)
    exposure = {}
    for (arm, profile, update), v in totals.items():
        exposure[f"{arm}/{profile}/{update}"] = dict(
            bodies=v["bodies"], windows32=v["windows32"], lags32=v["lags32"],
            direction_checks=v["direction_checks"],
            invalid_direction_checks=v["invalid_direction_checks"],
            death_tick_median=float(np.median(v["deaths"])), censored=v["censored"])
    errors = [e for row in validation for check in (row["finite_difference"] or {}).values()
              if check["status"] == "OK" for e in check["directions"]]
    qr = [row["qr"] for row in validation if row["qr"] and row["qr"]["status"] == "OK"]
    instrument = dict(final_validation_fits=len(validation), finite_difference_directions=len(errors),
                      max_relative_error=max((e["relative"] or 0) for e in errors),
                      qr_eligible=len(qr),
                      qr_top_rate_range=[min(q["top_rate"] for q in qr), max(q["top_rate"] for q in qr)] if qr else None,
                      qr_directional_max_range=[min(q["directional_max"] for q in qr),
                                                max(q["directional_max"] for q in qr)] if qr else None)
    count = dict(planned=816, complete=sum(r["status"] == "OK" for r in receipts),
                 void=sum(r["status"] == "VOID" for r in receipts),
                 missing=sum(r["status"] == "MISSING" for r in receipts))
    report = dict(version=lock["version"], protocol_commit=lock["protocol_commit"],
                  input_lock_sha256=lock_hash, count=count, instrument=instrument,
                  cohorts=cohorts, profiles=profiles, exposure=exposure,
                  per_checkpoint=per_checkpoint, validation=validation, receipts=receipts)
    write_atomic(OUT / "summary.json", canonical(report))
    print(json.dumps(count, indent=2), flush=True)
    print(json.dumps({arm: cohorts[arm]["status"] for arm in ARMS}, indent=2), flush=True)


if __name__ == "__main__":
    main()
