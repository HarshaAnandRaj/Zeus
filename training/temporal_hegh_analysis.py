"""Temporal HEGH analysis: paired contrasts, simultaneous empirical-Bernstein
intervals (§I), gate verdicts (§H), PASS/FAIL rules (§L).
"""
from __future__ import annotations

import math

from training import temporal_hegh_contract as C

OUTCOMES = ("m", "qb", "u")


def bernstein(values, r_width):
    """Simultaneous bounded-mean interval (spec §I)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        var = 0.0
    else:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = math.sqrt(2 * var * C.BERN_L / n) + 7 * r_width * C.BERN_L / (3 * (n - 1))
    return dict(n=n, mean=mean, var=var, half=half,
                lower=mean - half, upper=mean + half)


def ecology_contrasts(metrics):
    """metrics: {('W'|'N', 'S'|'P'): {m, qb, u}}. Returns 15 named scalars."""
    out = {}
    for o in OUTCOMES:
        ws = metrics[("W", "S")][o]
        ns = metrics[("N", "S")][o]
        wp = metrics[("W", "P")][o]
        np_ = metrics[("N", "P")][o]
        out[f"D_S/{o}"] = ws - ns
        out[f"D_P/{o}"] = wp - np_
        out[f"T_W/{o}"] = ws - wp
        out[f"T_N/{o}"] = ns - np_
        out[f"I/{o}"] = (ws - ns) - (wp - np_)
    return out


def decide_intervals(per_eco):
    """per_eco: list of ecology_contrasts dicts. 15 intervals + statuses.

    EFFECT: interval wholly outside [-0.05, +0.05] (strict; boundary ties
    earn neither claim). EQUIVALENT: wholly inside. Else UNRESOLVED.
    """
    table = {}
    for name in sorted(per_eco[0]):
        kind = name.split("/")[0]
        width = C.RANGE_INTERACTION if kind == "I" else C.RANGE_SIMPLE
        iv = bernstein([e[name] for e in per_eco], width)
        m = C.MARGIN
        if iv["lower"] > m or iv["upper"] < -m:
            status = "EFFECT"
        elif iv["lower"] > -m and iv["upper"] < m:
            status = "EQUIVALENT"
        else:
            status = "UNRESOLVED"
        iv.update(status=status, margin=m)
        table[name] = iv
    return table


def interaction_claim(table):
    """Primary H_interaction annotation (spec §L.5)."""
    if table["I/m"]["status"] == "EFFECT":
        return "PASS"
    if all(table[f"I/{o}"]["status"] == "EQUIVALENT" for o in OUTCOMES):
        return "EQUIVALENT_ON_PRIMARY"
    return "PRECISION_UNRESOLVED"


def panel_equivalence(table):
    if all(v["status"] == "EQUIVALENT" for v in table.values()):
        return "EQUIVALENT_ON_PANEL"
    return None


def nontrivial_exposure(pooled_survival, pooled_mean_m):
    ok = (C.POOLED_LO <= pooled_survival <= C.POOLED_HI
          and pooled_mean_m >= C.POOLED_MEAN_M)
    return "QUALIFIED" if ok else "OUTCOME_DEGENERATE"


def reference_gate(ref_cells, surv_min, joint_min):
    """ref_cells: {temporal: [{survived, renewed_frac}]}. Per-cell verdicts."""
    out = {}
    for temporal, rows in ref_cells.items():
        n = len(rows)
        surv = sum(1 for r in rows if r["survived"]) / n
        joint = sum(1 for r in rows if r["survived"] and r["renewed_frac"] >= 0.9) / n
        out[temporal] = dict(survival=surv, joint=joint,
                             verdict="PASS" if surv >= surv_min and joint >= joint_min else "FAIL")
    return out


def renewed_frac(metrics):
    denom = metrics["assimilated"]
    if denom == 0:
        return 0.0
    return metrics["renewed_assimilated"] / denom
