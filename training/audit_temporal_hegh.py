"""Independent Temporal HEGH audit. Uses only the contract + stored artifacts.
Never imports training.temporal_hegh_world or training.temporal_hegh_analysis
(enforced by test with those modules blocked). All math rewritten separately.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from fractions import Fraction

from training import temporal_hegh_contract as C

TOL = 1e-10


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scalar_norm(xs):
    return math.sqrt(sum(float(x) * float(x) for x in xs))


def scalar_dist(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def audit_geometry(raw_slice, wide_bits, narrow_bits):
    """raw_slice: 33x32 stored floats. wide/narrow_bits: 32 hex float64 each.
    Returns (wide_floats, narrow_floats) validated within 1e-10."""
    assert len(raw_slice) == 33 and all(len(r) == 32 for r in raw_slice)
    norms = [scalar_norm(r) for r in raw_slice]
    unit = [[x / n for x in r] for r, n in zip(raw_slice, norms)]
    dists = [scalar_dist(unit[0], unit[i]) for i in range(1, 33)]
    mean = sum(dists) / 32
    assert mean > 0, "degenerate geometry: zero mean distance"
    wide = [d / mean for d in dists]
    narrow = [1.0 + (p - 1.0) / 8.0 for p in wide]
    for got, hexbits in zip(wide, wide_bits):
        exact = struct.unpack("<d", bytes.fromhex(hexbits))[0]
        assert abs(got - exact) <= TOL, (got, exact)
    for got, hexbits in zip(narrow, narrow_bits):
        exact = struct.unpack("<d", bytes.fromhex(hexbits))[0]
        assert abs(got - exact) <= TOL, (got, exact)
    return wide, narrow


def audit_apportion(wide_bits, w_stored):
    """Exact largest-remainder reconstruction from stored binary64 bits."""
    fracs = [Fraction(struct.unpack("<d", bytes.fromhex(h))[0]) for h in wide_bits]
    total = sum(fracs)
    scaled = [f * 32 / total * C.Q for f in fracs]
    floors = [int(s // 1) for s in scaled]
    rema = [s - f for s, f in zip(scaled, floors)]
    leftover = 32 * C.Q - sum(floors)
    order = sorted(range(32), key=lambda i: (rema[i], -i), reverse=True)
    w = list(floors)
    for k in range(leftover):
        w[order[k]] += 1
    assert tuple(w) == tuple(w_stored)
    maxerr = max(abs(Fraction(v, C.Q) - f * 32 / total) for v, f in zip(w, fracs))
    assert float(maxerr) <= 2e-12
    return tuple(w)


def replay_trajectory(charges, events, r_units, k_units, mode, offer_lists):
    """Independent transition/policy reimplementation (plain loops, no bisect,
    no bitmask helpers from the primary module)."""
    n = len(events)
    order = sorted(range(32), key=lambda i: (charges[i], i))
    stock = [1] * 32
    renewed = [0] * 32
    ptrs = [0] * 32
    b = k_units
    alive = True
    tau = None
    sum_b = 0
    eta = rho = None
    led = dict(accepted=0, rejected=0, collected=0, assimilated=0, travel=0,
               maintenance=0, unassimilated=0, renewed=0)
    for t in range(n):
        e = events[t]
        if e != 255:
            if stock[e]:
                led["rejected"] += 1
            else:
                stock[e] = 1
                renewed[e] = 1
                led["accepted"] += 1
        if not alive:
            continue
        pick = -1
        if mode == "primary":
            for i in order:
                if stock[i] and charges[i] < b and r_units > charges[i]:
                    pick = i
                    break
            if pick >= 0 and b - charges[pick] + r_units > k_units:
                pick = -1
        elif mode == "reference":
            if b > C.UNIT and b - C.UNIT + r_units <= k_units:
                best = None
                for i in range(32):
                    if not stock[i]:
                        continue
                    nxt = None
                    lst = offer_lists[i]
                    k = ptrs[i]
                    while k < len(lst) and lst[k] <= t:
                        k += 1
                    ptrs[i] = k
                    if k < len(lst):
                        nxt = lst[k]
                    if nxt is not None and (best is None or nxt < best[0]
                                            or (nxt == best[0] and i < best[1])):
                        best = (nxt, i)
                if best is not None:
                    pick = best[1]
        else:
            raise ValueError("unknown mode")
        if pick < 0:
            pay = C.M_UNITS if C.M_UNITS < b else b
            b -= pay
            led["maintenance"] += pay
        else:
            ci = charges[pick]
            assert stock[pick] and b > ci, "audit encountered fatal transaction"
            stock[pick] = 0
            was = renewed[pick]
            renewed[pick] = 0
            b -= ci
            led["travel"] += ci
            room = k_units - b
            gain = r_units if r_units < room else room
            led["unassimilated"] += r_units - gain
            b += gain
            led["assimilated"] += gain
            led["collected"] += 1
            if was:
                led["renewed"] += gain
            pay = C.M_UNITS if C.M_UNITS < b else b
            b -= pay
            led["maintenance"] += pay
        sum_b += b
        s = t + 1
        if b == 0:
            alive = False
            tau = s
        if eta is None and b <= k_units // 2:
            eta = s
        if eta is not None and rho is None and alive and b >= 3 * k_units // 4:
            rho = s
    h = n
    m = (tau if tau is not None else h) / h
    # conservation + stock identities on the replayed books
    assert k_units + led["assimilated"] == b + led["travel"] + led["maintenance"]
    return dict(m=m, qb=sum_b / (h * k_units), u=0 if eta is None or rho is not None else 1,
                tau=tau if tau is not None else h, survived=tau is None, eta=eta, rho=rho,
                final_b=b, led=led)


def mean_var(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return mean, var


def audit_intervals(per_eco):
    """Independent 15-contrast recomputation. Returns table for comparison."""
    import math as _math
    table = {}
    for name in sorted(per_eco[0]):
        kind = name.split("/")[0]
        width = C.RANGE_INTERACTION if kind == "I" else C.RANGE_SIMPLE
        vals = [e[name] for e in per_eco]
        mean, var = mean_var(vals)
        half = _math.sqrt(2 * var * C.BERN_L / len(vals)) + 7 * width * C.BERN_L / (3 * (len(vals) - 1))
        table[name] = (mean, mean - half, mean + half)
    return table


def floathex(x):
    return struct.pack("<d", float(x)).hex()
