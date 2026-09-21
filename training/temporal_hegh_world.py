"""Temporal HEGH-1 world: maps, integer prices, event schedules, transitions,
primary/reference policies, trajectory runner with exact ledgers. No learning.
Frozen HEGH-0 sources are neither imported nor modified.
"""
from __future__ import annotations

import bisect
from fractions import Fraction

import numpy as np

from training import temporal_hegh_contract as C

NULL = 255


def rng_for(base, eco, stream, block):
    return np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([base, eco, stream, block])))


def geometry_prices(base, eco):
    """Actual HEGH-0 construction, fresh SeedSequence streams (spec §D)."""
    rng = rng_for(base, eco, C.STREAM_MAP, 0)
    raw = rng.standard_normal((C.NBINS + 1, 2048))
    coords = raw[:, :C.NBINS]
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    unit = coords / norms
    dist = np.linalg.norm(unit[1:] - unit[0], axis=1)
    mean = float(dist.mean())
    wide = dist / mean
    narrow = 1.0 + (wide - 1.0) / 8.0
    return raw[:, :C.NBINS].copy(), wide, narrow


def apportion(prices):
    """Exact largest-remainder apportionment to sum 32Q (spec §D).

    Treats binary64 inputs as exact rationals, normalizes to sum 32,
    scales by Q, floors, and deals leftover units to largest remainders
    (ties by bin ID). Returns (w tuple, max |w_i/Q - p_i|).
    """
    fracs = [Fraction(float(p)) for p in prices]
    total = sum(fracs)
    scaled = [f * 32 / total * C.Q for f in fracs]
    floors = [int(s // 1) for s in scaled]
    rema = [s - f for s, f in zip(scaled, floors)]
    leftover = 32 * C.Q - sum(floors)
    # largest remainder first; exact ties broken by smaller bin ID:
    # sorting (remainder, -id) descending puts larger -id (= smaller id) first.
    order = sorted(range(C.NBINS), key=lambda i: (rema[i], -i), reverse=True)
    w = list(floors)
    for k in range(leftover):
        w[order[k]] += 1
    assert sum(w) == 32 * C.Q and all(v > 0 for v in w)
    maxerr = max(abs(Fraction(v, C.Q) - f * 32 / total) for v, f in zip(w, fracs))
    return tuple(w), float(maxerr)


def charges_from_w(w):
    wide = tuple(8 * v for v in w)
    narrow = tuple(7 * C.Q + v for v in w)
    assert sum(wide) == 32 * 8 * C.Q and sum(narrow) == 32 * 8 * C.Q
    return wide, narrow


def gen_block_tokens(base, eco, block):
    """One block's S and P event streams (lists of 64: bin id or NULL)."""
    pi = [int(x) for x in rng_for(base, eco, C.STREAM_SITE, block).permutation(C.NBINS)]
    offset = int(rng_for(base, eco, C.STREAM_ROT, block).integers(0, 64))
    tokperm = [int(x) for x in rng_for(base, eco, C.STREAM_TOK, block).permutation(64)]
    base_tokens = [(0, b) for b in pi] + [(1, n) for n in range(32)]
    s_order = base_tokens[offset:] + base_tokens[:offset]
    p_order = [base_tokens[i] for i in tokperm]
    to_ev = lambda order: [int(b) if kind == 0 else NULL for kind, b in order]
    return to_ev(s_order), to_ev(p_order)


def gen_schedule(base, eco, nblocks):
    s_blocks, p_blocks = [], []
    for b in range(nblocks):
        s, p = gen_block_tokens(base, eco, b)
        s_blocks.append(s)
        p_blocks.append(p)
    return s_blocks, p_blocks


def flatten(blocks):
    return [e for block in blocks for e in block]


def offer_times(events):
    """Per-bin sorted tick lists of offers."""
    per = [[] for _ in range(C.NBINS)]
    for t, e in enumerate(events):
        if e != NULL:
            per[e].append(t)
    return per


def new_state(k_units):
    return dict(b=k_units, stock=(1 << C.NBINS) - 1, origin=0, alive=True,
                tau=None, sum_b=0, eta=None, rho=None, ptrs=[0] * C.NBINS,
                accepted=0, rejected=0, collected=0, assimil=0, travel=0,
                maint=0, unassim=0, ren_assimil=0, a_accepted=0, a_rejected=0,
                a_collected=0, a_assimil=0, a_travel=0, a_maint=0,
                a_unassim=0, a_ren_assimil=0)


def apply_event(st, e):
    """Step 1: exogenous offer (applies even after death)."""
    if e == NULL:
        return
    if st["stock"] >> e & 1:
        st["rejected"] += 1
        if st["alive"]:
            st["a_rejected"] += 1
    else:
        st["stock"] |= 1 << e
        st["origin"] |= 1 << e
        st["accepted"] += 1
        if st["alive"]:
            st["a_accepted"] += 1


def choose_primary(st, charges, r_units, k_units, order):
    pick = -1
    for i in order:
        if st["stock"] >> i & 1 and charges[i] < st["b"] and r_units > charges[i]:
            pick = i
            break
    if pick >= 0 and st["b"] - charges[pick] + r_units > k_units:
        pick = -1  # headroom: wait, never switch bins
    return pick


def choose_reference(st, t, r_units, k_units, offer_lists):
    pick = -1
    if st["b"] > C.UNIT and st["b"] - C.UNIT + r_units <= k_units:
        best = None
        for i in range(C.NBINS):
            if not (st["stock"] >> i & 1):
                continue
            lst = offer_lists[i]
            k = bisect.bisect_right(lst, t, lo=st["ptrs"][i])
            st["ptrs"][i] = k
            if k < len(lst) and (best is None or lst[k] < best[0]
                                 or (lst[k] == best[0] and i < best[1])):
                best = (lst[k], i)
        if best is not None:
            pick = best[1]
    return pick


def transact(st, pick, charges, r_units, k_units):
    """Steps 4-5: harvest/wait + maintenance. force_pick exposes the fatal
    guard to tests; policies must never trigger it."""
    b = st["b"]
    if pick < 0:
        pay = C.M_UNITS if C.M_UNITS < b else b
        st["b"] = b - pay
        st["maint"] += pay
        st["a_maint"] += pay
        return dict(action="wait", intended_c=0, actual_c=0, gain=0,
                    intended_m=C.M_UNITS, actual_m=pay)
    ci = charges[pick]
    if not (st["stock"] >> pick & 1) or not b > ci:
        # fatal guard: travel exhausts energy, no food, no maintenance
        pay_actual = ci if ci < b else b
        st["b"] = b - pay_actual
        st["travel"] += pay_actual
        st["a_travel"] += pay_actual
        return dict(action="fatal", intended_c=ci, actual_c=pay_actual, gain=0,
                    intended_m=0, actual_m=0)
    st["stock"] &= ~(1 << pick)
    was_ren = (st["origin"] >> pick) & 1
    st["origin"] &= ~(1 << pick)
    b -= ci
    st["travel"] += ci
    st["a_travel"] += ci
    gain = r_units if r_units < k_units - b else k_units - b
    st["unassim"] += r_units - gain
    st["a_unassim"] += r_units - gain
    b += gain
    st["assimil"] += gain
    st["a_assimil"] += gain
    st["collected"] += 1
    st["a_collected"] += 1
    if was_ren:
        st["ren_assimil"] += gain
        st["a_ren_assimil"] += gain
    pay = C.M_UNITS if C.M_UNITS < b else b
    st["b"] = b - pay
    st["maint"] += pay
    st["a_maint"] += pay
    return dict(action="harvest", intended_c=ci, actual_c=ci, gain=gain,
                intended_m=C.M_UNITS, actual_m=pay)


def step_once(st, t, e, charges, r_units, k_units, mode, order, offer_lists=None):
    """One tick in spec §B order. Returns per-tick record dict."""
    apply_event(st, e)
    if not st["alive"]:
        return dict(tick=t, event=e, action="dead", intended_c=0, actual_c=0,
                    gain=0, intended_m=0, actual_m=0, b=0)
    if mode == "primary":
        pick = choose_primary(st, charges, r_units, k_units, order)
    elif mode == "reference":
        pick = choose_reference(st, t, r_units, k_units, offer_lists)
    else:
        raise ValueError("unknown mode")
    rec = transact(st, pick, charges, r_units, k_units)
    rec.update(tick=t, event=e, bin=pick)
    st["sum_b"] += st["b"]
    s = t + 1
    if st["b"] == 0:
        st["alive"] = False
        st["tau"] = s
    if st["eta"] is None and st["b"] <= k_units // 2:
        st["eta"] = s
    if st["eta"] is not None and st["rho"] is None and st["alive"] \
            and st["b"] >= 3 * k_units // 4:
        st["rho"] = s
    rec["b"] = st["b"]
    return rec


def finalize(st, n, k_units):
    h = n
    tau = st["tau"]
    m = (tau if tau is not None else h) / h
    qb = st["sum_b"] / (h * k_units)
    remaining = bin(st["stock"]).count("1")
    if st["eta"] is None:
        u = 0
    else:
        u = 0 if st["rho"] is not None else 1
    return dict(m=m, qb=qb, u=u, tau=tau if tau is not None else h,
                survived=tau is None, eta=st["eta"], rho=st["rho"], final_b=st["b"],
                accepted=st["accepted"], rejected=st["rejected"], collected=st["collected"],
                remaining=remaining,
                assimilated=st["assimil"], travel=st["travel"], maintenance=st["maint"],
                unassimilated=st["unassim"], renewed_assimilated=st["ren_assimil"],
                alive=dict(accepted=st["a_accepted"], rejected=st["a_rejected"],
                           collected=st["a_collected"], assimilated=st["a_assimil"],
                           travel=st["a_travel"], maintenance=st["a_maint"],
                           unassimilated=st["a_unassim"],
                           renewed_assimilated=st["a_ren_assimil"]))


def run_trajectory(charges, events, r_units, k_units, mode, offer_lists=None,
                   keep_records=False):
    """Run one cell trajectory (spec §B). keep_records returns per-tick
    records (dead suffix RLE-compressible downstream)."""
    n = len(events)
    order = sorted(range(C.NBINS), key=lambda i: (charges[i], i))
    st = new_state(k_units)
    records = [] if keep_records else None
    for t in range(n):
        rec = step_once(st, t, events[t], charges, r_units, k_units, mode,
                        order, offer_lists)
        if keep_records:
            records.append(rec)
    out = finalize(st, n, k_units)
    if keep_records:
        out["records"] = records
    return out


def unit_charges():
    return tuple([C.UNIT] * C.NBINS)


def exec_order(base, eco):
    return [int(x) for x in rng_for(base, eco, C.STREAM_ORDER, 0).permutation(6)]
