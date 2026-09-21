"""Temporal HEGH campaign runner: port / dev / val / formal(gated) modes.

Blinding (§J): dev/val run at unit prices only — no W/N trajectories exist
before the formal freeze, so calibration cannot observe any Wide/Narrow
outcome. Candidate selection is the mechanical first-qualifier rule. Formal
maps and W/N trajectories are generated exclusively in formal mode, which
refuses to run without --authorize-formal.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import gzip
import hashlib
import json
import math
import pathlib
import platform
import subprocess
import sys
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from training import temporal_hegh_contract as C
from training import temporal_hegh_world as W
from training import temporal_hegh_analysis as A

MODULES = ["training/temporal_hegh_contract.py", "training/temporal_hegh_world.py",
           "training/temporal_hegh_analysis.py", "training/audit_temporal_hegh.py",
           "training/run_temporal_hegh.py", "training/test_temporal_hegh.py",
           "docs/temporal_hegh_design_20260921.md"]


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=1) + "\n", encoding="utf-8")


def dump_gzip(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode("utf-8")
    with open(path, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as z:
            z.write(raw)


def load_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def manifest(mode, params):
    return dict(contract=C.VERSION, mode=mode, params=params,
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                commit=git_commit(),
                sources={m: sha_file(ROOT / m) for m in MODULES},
                python=platform.python_version(), numpy=np.__version__,
                platform=platform.platform())


def r_units_of(s):
    return int(Fraction(s) * 8 * C.Q)


def k_units_of(k):
    return k * C.UNIT


# ---------------- port-compatibility (§D) ----------------

def hegh0_rule(costs, reserve, bonus):
    """Float one-decision rule, HEGH-0 spec: requested task, known prices."""
    cutoff = 1.0 + bonus - 0.1
    limit = reserve if reserve < cutoff else cutoff
    wins = 0
    for c in costs:
        if c < limit and c < reserve:
            wins += 1
    return wins


def run_port(outdir):
    outdir = pathlib.Path(outdir)
    per_map = []
    for shard in range(8):
        data = load_gzip(ROOT / "zeus_sandbox/universe/reports"
                         / f"hegh0_20260920_{shard * 16:04d}.json.gz")
        for m in data["maps"]:
            raw = np.frombuffer(base64.b64decode(m["raw"]["data"]), dtype="<f8")
            raw = raw.reshape(m["raw"]["shape"])[:, :32]
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            dist = np.linalg.norm(raw[1:] / norms[1:] - raw[0] / norms[0], axis=1)
            wide = dist / float(dist.mean())
            w, err = W.apportion(wide)
            assert err <= 2e-12, (m["index"], err)
            ported = [v / C.Q for v in w]
            narrow = [1.0 + (p - 1.0) / 8.0 for p in ported]
            counts = {}
            for name, prices in (("wide", ported), ("narrow", narrow)):
                for reserve in (1.1, 0.9):
                    counts[(name, reserve)] = hegh0_rule(prices, reserve, 0.2)
                    margin = min(min(abs(p - reserve) for p in prices),
                                 min(abs(p - 1.1) for p in prices))
                    assert margin > 1e-9, (m["index"], name, reserve, margin)
            per_map.append(dict(index=m["index"], counts={f"{k[0]}/{k[1]}": v for k, v in counts.items()}))
    totals = {}
    for row in per_map:
        for k, v in row["counts"].items():
            totals[k] = totals.get(k, 0) + v
    expected = {"wide/1.1": 3565, "narrow/1.1": 4096, "wide/0.9": 555, "narrow/0.9": 0}
    ok = all(totals[k] == v for k, v in expected.items())
    receipt = dict(maps=len(per_map), totals=totals, expected=expected,
                   verdict="PASS" if ok else "IMPLEMENTATION_FAIL")
    dump_json(outdir / "port.json", receipt)
    print(json.dumps(receipt))
    return 0 if ok else 1


# ---------------- shared execution ----------------

def run_primary_cell(charges, schedules, r_units, k_units, horizon_blocks, n):
    """Returns (S_rows, P_rows) of metric dicts."""
    out_s, out_p = [], []
    for eco in range(n):
        s_ev, p_ev = schedules[eco]
        out_s.append(W.run_trajectory(charges, s_ev[:horizon_blocks * C.BLOCK],
                                      r_units, k_units, "primary"))
        out_p.append(W.run_trajectory(charges, p_ev[:horizon_blocks * C.BLOCK],
                                      r_units, k_units, "primary"))
    return out_s, out_p


def survival_rate(rows):
    return sum(1 for r in rows if r["survived"]) / len(rows)


def mean_m(rows):
    return sum(r["m"] for r in rows) / len(rows)


def deprivation_checks(r_units, k_units):
    """Returns dict of control verdicts (unit prices)."""
    k_price = k_units // C.UNIT
    wait_charges = tuple([10 ** 18] * C.NBINS)  # never eligible: pure wait
    w = W.run_trajectory(wait_charges, [W.NULL] * 4 * k_price, r_units, k_units, "primary")
    wait_ok = (not w["survived"]) and w["tau"] == 4 * k_price
    r_price = Fraction(r_units, C.UNIT)
    bound = math.ceil(float(k_price + 32 * r_price) / 0.25)
    d = W.run_trajectory(W.unit_charges(), [W.NULL] * (bound + 64), r_units, k_units, "primary")
    starve_ok = (not d["survived"]) and d["tau"] <= bound
    return dict(wait_death_tick=w["tau"], wait_expect=4 * k_price, wait_ok=wait_ok,
                starve_bound=bound, starve_tick=d["tau"], starve_ok=starve_ok)


# ---------------- dev calibration ----------------

def dev_candidate(r_str, k, schedules, cache):
    r_units, k_units = r_units_of(r_str), k_units_of(k)
    key = (r_str, k)
    if key in cache:
        return cache[key]
    unit = W.unit_charges()
    ps, pp = run_primary_cell(unit, schedules, r_units, k_units, C.H // C.BLOCK, C.N_DEV)
    # twins: sequential rerun, byte-identical metrics required
    ps2, pp2 = run_primary_cell(unit, schedules, r_units, k_units, C.H // C.BLOCK, C.N_DEV)
    assert [json.dumps(x, sort_keys=True) for x in ps] == [json.dumps(x, sort_keys=True) for x in ps2]
    assert [json.dumps(x, sort_keys=True) for x in pp] == [json.dumps(x, sort_keys=True) for x in pp2]
    rs, rp = [], []
    for s_ev, p_ev in schedules:
        for ev, out in ((s_ev, rs), (p_ev, rp)):
            seg = ev[:8192]
            lists = W.offer_times(seg + ev[8192:8192 + C.BLOCK])
            out.append(W.run_trajectory(unit, seg, r_units, k_units, "reference", lists))
    rs_twin, rp_twin = [], []
    for s_ev, p_ev in schedules:
        for ev, out in ((s_ev, rs_twin), (p_ev, rp_twin)):
            out.append(W.run_trajectory(unit, ev[:8192], r_units, k_units, "reference",
                                        W.offer_times(ev[:8192 + C.BLOCK])))
    assert [json.dumps(x, sort_keys=True) for x in rs + rp] == \
           [json.dumps(x, sort_keys=True) for x in rs_twin + rp_twin]
    result = dict(r=r_str, k=k,
                  primary_S_survival=survival_rate(ps), primary_P_survival=survival_rate(pp),
                  pooled_mean_m=sum(r["m"] for r in ps + pp) / (2 * C.N_DEV),
                  ref_S_survival=survival_rate(rs), ref_P_survival=survival_rate(rp),
                  ref_S_joint=sum(1 for r in rs if r["survived"] and A.renewed_frac(r) >= 0.9) / C.N_DEV,
                  ref_P_joint=sum(1 for r in rp if r["survived"] and A.renewed_frac(r) >= 0.9) / C.N_DEV,
                  deprivation=deprivation_checks(r_units, k_units))
    result["gates"] = dict(
        primary_S=C.DEV_SURV_LO <= result["primary_S_survival"] <= C.DEV_SURV_HI,
        primary_P=C.DEV_SURV_LO <= result["primary_P_survival"] <= C.DEV_SURV_HI,
        pooled_m=result["pooled_mean_m"] >= C.DEV_MEAN_M,
        ref_S=result["ref_S_survival"] >= C.REF_SURVIVAL_CAL and result["ref_S_joint"] >= C.REF_JOINT_CAL,
        ref_P=result["ref_P_survival"] >= C.REF_SURVIVAL_CAL and result["ref_P_joint"] >= C.REF_JOINT_CAL,
        deprivation=result["deprivation"]["wait_ok"] and result["deprivation"]["starve_ok"])
    result["qualifies"] = all(result["gates"].values())
    cache[key] = result
    return result


def run_dev(outdir):
    outdir = pathlib.Path(outdir)
    mani = manifest("dev", dict(r_grid=C.R_GRID, k_grid=C.K_GRID, n=C.N_DEV))
    dump_json(outdir / "manifest.json", mani)
    schedules = []
    for eco in range(C.N_DEV):
        s_blocks, p_blocks = W.gen_schedule(C.BASE_DEV, eco, C.BLOCKS_EXTENDED)
        schedules.append((W.flatten(s_blocks), W.flatten(p_blocks)))
    dump_json(outdir / "schedule_hashes.json",
              [hashlib.sha256(json.dumps(s).encode()).hexdigest() for s in schedules])
    grid, cache, selected = [], {}, None
    for r_str in sorted(C.R_GRID, key=float):
        for k in sorted(C.K_GRID):
            row = dev_candidate(r_str, k, schedules, cache)
            grid.append({kk: vv for kk, vv in row.items()})
            print(f"DEV r={r_str} K={k} qual={row['qualifies']} "
                  f"S={row['primary_S_survival']:.3f} P={row['primary_P_survival']:.3f} "
                  f"M={row['pooled_mean_m']:.3f} ref={row['ref_S_survival']:.3f}/{row['ref_P_survival']:.3f} "
                  f"joint={row['ref_S_joint']:.3f}/{row['ref_P_joint']:.3f}", flush=True)
            if row["qualifies"] and selected is None:
                selected = dict(r=r_str, k=k)
    verdict = "CALIBRATION_PASS" if selected else "CALIBRATION_FAIL"
    dump_json(outdir / "dev_grid.json", grid)
    dump_json(outdir / "dev_selection.json", dict(selected=selected, verdict=verdict))
    print(json.dumps(dict(selected=selected, verdict=verdict)))
    return 0 if selected else 1


# ---------------- validation ----------------

def encode_records(charges, events, r_units, k_units, mode, offer_lists=None):
    out = W.run_trajectory(charges, events, r_units, k_units, mode, offer_lists, keep_records=True)
    live = []
    for rec in out.pop("records"):
        if rec["action"] == "dead":
            continue
        live.append([rec["tick"], rec["event"],
                     0 if rec["action"] == "wait" else 1, rec.get("bin", -1),
                     rec["actual_c"], rec["gain"], rec["intended_m"], rec["actual_m"], rec["b"]])
    dead_start = None
    if not out["survived"]:
        dead_start = out["tau"]
    return out, dict(live=live, dead_start=dead_start)


def run_val(outdir, r_str, k):
    from training import audit_temporal_hegh as AU
    outdir = pathlib.Path(outdir)
    r_units, k_units = r_units_of(r_str), k_units_of(k)
    mani = manifest("validation", dict(r=r_str, k=k, n=C.N_VAL))
    dump_json(outdir / "manifest.json", mani)
    unit = W.unit_charges()
    metrics_S, metrics_P, refS, refP, shards = [], [], [], [], {}
    for eco in range(C.N_VAL):
        s_blocks, p_blocks = W.gen_schedule(C.BASE_VAL, eco, C.BLOCKS_EXTENDED)
        s_ev, p_ev = W.flatten(s_blocks), W.flatten(p_blocks)
        row = dict(eco=eco, cells={})
        for temporal, ev in (("S", s_ev), ("P", p_ev)):
            seg = ev[:C.H]
            lists = W.offer_times(ev[:C.H + C.BLOCK])
            m, rec = encode_records(unit, seg, r_units, k_units, "primary")
            row["cells"][f"primary_{temporal}"] = dict(metrics=m, records=rec)
            rseg = ev[:8192]
            rlists = W.offer_times(ev[:8192 + C.BLOCK])
            rm, rrec = encode_records(unit, rseg, r_units, k_units, "reference", rlists)
            rm["renewed_frac"] = A.renewed_frac(rm)
            row["cells"][f"ref_{temporal}"] = dict(metrics=rm, records=rrec)
        metrics_S.append(row["cells"]["primary_S"]["metrics"])
        metrics_P.append(row["cells"]["primary_P"]["metrics"])
        refS.append(dict(survived=row["cells"]["ref_S"]["metrics"]["survived"],
                         renewed_frac=row["cells"]["ref_S"]["metrics"]["renewed_frac"]))
        refP.append(dict(survived=row["cells"]["ref_P"]["metrics"]["survived"],
                         renewed_frac=row["cells"]["ref_P"]["metrics"]["renewed_frac"]))
        shards.setdefault(eco // 8, []).append(row)
        if len(shards[eco // 8]) == 8 or eco == C.N_VAL - 1:
            dump_gzip(outdir / f"val_{eco // 8:04d}.json.gz", shards[eco // 8])
    surv_S, surv_P = survival_rate(metrics_S), survival_rate(metrics_P)
    pooled_m = sum(r["m"] for r in metrics_S + metrics_P) / (2 * C.N_VAL)
    ref_gate = A.reference_gate(dict(S=refS, P=refP), C.REF_SURVIVAL_VAL, C.REF_JOINT_VAL)
    gates = dict(primary_S=C.VAL_SURV_LO <= surv_S <= C.VAL_SURV_HI,
                 primary_P=C.VAL_SURV_LO <= surv_P <= C.VAL_SURV_HI,
                 pooled_m=pooled_m >= C.VAL_MEAN_M,
                 ref_S=ref_gate["S"]["verdict"] == "PASS",
                 ref_P=ref_gate["P"]["verdict"] == "PASS",
                 deprivation=all(v for v in deprivation_checks(r_units, k_units).values()
                                 if isinstance(v, bool)))
    ok = all(gates.values())
    # independent audit: replay every validation trajectory from stored inputs
    for eco in range(C.N_VAL):
        s_blocks, p_blocks = W.gen_schedule(C.BASE_VAL, eco, C.BLOCKS_EXTENDED)
        for temporal, blocks in (("S", s_blocks), ("P", p_blocks)):
            ev = W.flatten(blocks)
            for cell, mode, horizon in (("primary", "primary", C.H), ("ref", "reference", 8192)):
                seg = ev[:horizon]
                lists = W.offer_times(ev[:horizon + C.BLOCK])
                rep = AU.replay_trajectory(unit, seg, r_units, k_units, mode, lists)
                stored = load_gzip(outdir / f"val_{eco // 8:04d}.json.gz")[eco % 8][
                    "cells"][f"{cell}_{temporal}"]["metrics"]
                for key in ("m", "qb", "u", "tau", "survived", "eta", "rho"):
                    a, b = rep[key], stored[key]
                    if isinstance(a, float):
                        assert abs(a - b) <= 1e-10, (eco, temporal, cell, key, a, b)
                    else:
                        assert a == b, (eco, temporal, cell, key, a, b)
                for key in ("accepted", "rejected", "collected", "assimilated",
                                "travel", "maintenance", "unassimilated"):
                    assert rep["led"][key] == stored[key], (eco, temporal, cell, key)
                assert rep["led"]["renewed"] == stored["renewed_assimilated"], (eco, key)
    verdict = "VALIDATION_PASS" if ok else "VALIDATION_FAIL"
    dump_json(outdir / "validation.json",
              dict(r=r_str, k=k, primary_S_survival=surv_S, primary_P_survival=surv_P,
                   pooled_mean_m=pooled_m, ref=ref_gate, gates=gates,
                   audit_worlds=C.N_VAL * 4, verdict=verdict))
    print(json.dumps(dict(r=r_str, k=k, gates=gates, verdict=verdict)))
    return 0 if ok else 1


# ---------------- formal (gated: needs --authorize-formal) ----------------

FORMAL_CELLS = ("WS", "NS", "WP", "NP", "refS", "refP")


def formal_eco_inputs(eco):
    raw_slice, wide, narrow = W.geometry_prices(C.BASE_FORMAL, eco)
    w, err = W.apportion(wide)
    assert err <= 2e-12
    wide_c, narrow_c = W.charges_from_w(w)
    s_blocks, p_blocks = W.gen_schedule(C.BASE_FORMAL, eco, C.BLOCKS_FORMAL)
    return dict(raw_slice=raw_slice, wide=wide, narrow=narrow, w=w,
                wide_c=wide_c, narrow_c=narrow_c, s_blocks=s_blocks, p_blocks=p_blocks)


def run_formal_eco(eco, r_str, k, inputs):
    from training import audit_temporal_hegh as AU
    r_units, k_units = r_units_of(r_str), k_units_of(k)
    s_ev = W.flatten(inputs["s_blocks"])[:C.H]
    p_ev = W.flatten(inputs["p_blocks"])[:C.H]
    cells = {}
    order = W.exec_order(C.BASE_FORMAL, eco)
    names = ["WS", "NS", "WP", "NP", "refS", "refP"]
    jobs = dict(WS=(inputs["wide_c"], s_ev, "primary", None),
                NS=(inputs["narrow_c"], s_ev, "primary", None),
                WP=(inputs["wide_c"], p_ev, "primary", None),
                NP=(inputs["narrow_c"], p_ev, "primary", None))
    for name in [names[i] for i in order]:
        if name in jobs:
            charges, ev, mode, _ = jobs[name]
            m, rec = encode_records(charges, ev, r_units, k_units, mode)
        else:
            temporal = name[3:]
            ev = s_ev if temporal == "S" else p_ev
            lists = W.offer_times(ev + W.flatten(
                inputs["s_blocks"] if temporal == "S" else inputs["p_blocks"])[C.H:C.H + C.BLOCK])
            m, rec = encode_records(W.unit_charges(), ev, r_units, k_units, "reference", lists)
            m["renewed_frac"] = A.renewed_frac(m)
        cells[name] = dict(metrics=m, records=rec)
    eco_in = dict(
        raw_b64=base64.b64encode(np.asarray(inputs["raw_slice"], dtype="<f8").tobytes()).decode("ascii"),
        wide_bits=[AU.floathex(v) for v in inputs["wide"]],
        narrow_bits=[AU.floathex(v) for v in inputs["narrow"]],
        w=list(inputs["w"]),
        wide_c=list(inputs["wide_c"]), narrow_c=list(inputs["narrow_c"]),
        s_b64=base64.b64encode(bytes(W.flatten(inputs["s_blocks"]))).decode("ascii"),
        p_b64=base64.b64encode(bytes(W.flatten(inputs["p_blocks"]))).decode("ascii"),
        exec_order=order)
    return eco_in, cells


def run_formal_twin(outdir, r_str, k, twin):
    outdir = pathlib.Path(outdir)
    per_eco, ref_rows = [], dict(S=[], P=[])
    for eco in range(C.N_FORMAL):
        inputs = formal_eco_inputs(eco)
        eco_in, cells = run_formal_eco(eco, r_str, k, inputs)
        per_eco.append((eco_in, cells))
        for temporal in ("S", "P"):
            ref_rows[temporal].append(dict(
                survived=cells[f"ref{temporal}"]["metrics"]["survived"],
                renewed_frac=cells[f"ref{temporal}"]["metrics"]["renewed_frac"]))
        if len(per_eco) == 8 or eco == C.N_FORMAL - 1:
            first = eco - len(per_eco) + 1
            dump_gzip(outdir / f"twin_{twin}" / f"in_{first // 8:04d}.json.gz",
                      [dict(eco=first + j, inputs=e[0]) for j, e in enumerate(per_eco)])
            dump_gzip(outdir / f"twin_{twin}" / f"ex_{first // 8:04d}.json.gz",
                      [dict(eco=first + j, cells=e[1]) for j, e in enumerate(per_eco)])
            for shard in (outdir / f"twin_{twin}" / f"in_{first // 8:04d}.json.gz",
                          outdir / f"twin_{twin}" / f"ex_{first // 8:04d}.json.gz"):
                assert shard.stat().st_size <= 90 * 1024 * 1024, shard
            per_eco = []
        if (eco + 1) % 1024 == 0:
            print(f"formal twin {twin}: {eco + 1}/{C.N_FORMAL}", flush=True)
    dump_json(outdir / f"twin_{twin}_ref.json", ref_rows)
    return ref_rows


def run_formal(outdir, r_str, k, authorize=False):
    from training import audit_temporal_hegh as AU
    outdir = pathlib.Path(outdir)
    if not authorize:
        print("FORMAL REFUSED: re-run with --authorize-formal after validation PASS. "
              "No formal map or W/N trajectory was generated.")
        return 2
    assert r_str and k, "--r/--k required (from validation selection)"
    r_units, k_units = r_units_of(r_str), k_units_of(k)
    mani = manifest("formal", dict(r=r_str, k=k, n=C.N_FORMAL))
    dump_json(outdir / "manifest.json", mani)
    ref_a = run_formal_twin(outdir, r_str, k, "a")
    # twin B in a fresh process
    cmd = [sys.executable, "training/run_temporal_hegh.py", "formal-twin",
           "--out", str(outdir), "--r", r_str, "--k", str(k)]
    subprocess.run(cmd, cwd=ROOT, check=True)
    nshards = (C.N_FORMAL + 7) // 8
    for s in range(nshards):
        ha = sha_file(outdir / "twin_a" / f"ex_{s:04d}.json.gz")
        hb = sha_file(outdir / "twin_b" / f"ex_{s:04d}.json.gz")
        assert ha == hb, f"execution twin mismatch shard {s}"
    # canonical analysis on twin A (single shard pass)
    per_eco = []
    all_m, deaths = [], 0
    total = 0
    for s in range(nshards):
        for row in load_gzip(outdir / "twin_a" / f"ex_{s:04d}.json.gz"):
            cells = row["cells"]
            m = {("W", "S"): cells["WS"]["metrics"],
                 ("N", "S"): cells["NS"]["metrics"],
                 ("W", "P"): cells["WP"]["metrics"],
                 ("N", "P"): cells["NP"]["metrics"]}
            per_eco.append(A.ecology_contrasts(m))
            for cell in ("WS", "NS", "WP", "NP"):
                all_m.append(cells[cell]["metrics"]["m"])
                deaths += cells[cell]["metrics"]["tau"] < C.H
                total += 1
    table = A.decide_intervals(per_eco)
    claim = A.interaction_claim(table)
    panel = A.panel_equivalence(table)
    pooled_surv = 1 - deaths / total
    pooled_m = sum(all_m) / len(all_m)
    nontrivial = A.nontrivial_exposure(pooled_surv, pooled_m)
    ref_gate = A.reference_gate(ref_a, C.REF_SURVIVAL_VAL, C.REF_JOINT_VAL)
    # independent audit over every world (twin A inputs)
    for s in range(nshards):
        in_rows = {r["eco"]: r["inputs"] for r in load_gzip(outdir / "twin_a" / f"in_{s:04d}.json.gz")}
        ex_rows = {r["eco"]: r["cells"] for r in load_gzip(outdir / "twin_a" / f"ex_{s:04d}.json.gz")}
        for eco, eco_in in in_rows.items():
            raw = np.frombuffer(base64.b64decode(eco_in["raw_b64"]), dtype="<f8").reshape(33, 32)
            wide, narrow = AU.audit_geometry(raw.tolist(), eco_in["wide_bits"], eco_in["narrow_bits"])
            w = AU.audit_apportion(eco_in["wide_bits"], eco_in["w"])
            assert list(w) == eco_in["w"]
            s_ev = list(base64.b64decode(eco_in["s_b64"]))
            p_ev = list(base64.b64decode(eco_in["p_b64"]))
            assert s_ev == W.flatten(W.gen_schedule(C.BASE_FORMAL, eco, C.BLOCKS_FORMAL)[0])
            assert p_ev == W.flatten(W.gen_schedule(C.BASE_FORMAL, eco, C.BLOCKS_FORMAL)[1])
            cells = ex_rows[eco]
            jobs = dict(WS=(eco_in["wide_c"], s_ev), NS=(eco_in["narrow_c"], s_ev),
                        WP=(eco_in["wide_c"], p_ev), NP=(eco_in["narrow_c"], p_ev))
            for cell, (charges, ev) in jobs.items():
                rep = AU.replay_trajectory(charges, ev[:C.H], r_units_of(r_str),
                                           k_units_of(k), "primary", None)
                stored = cells[cell]["metrics"]
                for key in ("m", "qb", "u"):
                    assert abs(rep[key] - stored[key]) <= 1e-10, (eco, cell, key)
                assert rep["tau"] == stored["tau"] and rep["survived"] == stored["survived"]
                assert k_units + rep["led"]["assimilated"] == \
                    rep["final_b"] + rep["led"]["travel"] + rep["led"]["maintenance"]
    audit_table = AU.audit_intervals(per_eco)
    for name in table:
        a, b, c_ = audit_table[name]
        assert abs(a - table[name]["mean"]) <= 1e-10
        assert abs(b - table[name]["lower"]) <= 1e-10
        assert abs(c_ - table[name]["upper"]) <= 1e-10
    verdict = ("H_INTERACTION_PASS" if claim == "PASS" and nontrivial == "QUALIFIED"
               and all(v["verdict"] == "PASS" for v in ref_gate.values())
               else ("ASSAY_FEASIBILITY_FAIL" if nontrivial != "QUALIFIED"
                     or any(v["verdict"] != "PASS" for v in ref_gate.values())
                     else f"H_INTERACTION_{claim}"))
    dump_json(outdir / "formal.json",
              dict(r=r_str, k=k, claim=claim, panel=panel, nontrivial=nontrivial,
                   ref=ref_gate, pooled_survival=pooled_surv, pooled_mean_m=pooled_m,
                   intervals=table, verdict=verdict))
    print(json.dumps(dict(claim=claim, panel=panel, nontrivial=nontrivial,
                          ref={k_: v["verdict"] for k_, v in ref_gate.items()},
                          verdict=verdict)))
    return 0


def run_formal_twin_only(outdir, r_str, k):
    outdir = pathlib.Path(outdir)
    assert (outdir / "manifest.json").exists(), "twin B requires twin A manifest"
    run_formal_twin(outdir, r_str, k, "b")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["port", "dev", "val", "formal", "formal-twin"])
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--r", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--authorize-formal", action="store_true")
    args = parser.parse_args()
    if args.mode == "port":
        raise SystemExit(run_port(args.out))
    if args.mode == "dev":
        raise SystemExit(run_dev(args.out))
    if args.mode == "val":
        assert args.r and args.k, "--r/--k required (from dev selection)"
        raise SystemExit(run_val(args.out, args.r, args.k))
    if args.mode == "formal-twin":
        raise SystemExit(run_formal_twin_only(args.out, args.r, args.k))
    raise SystemExit(run_formal(args.out, args.r, args.k, args.authorize_formal))


if __name__ == "__main__":
    main()
