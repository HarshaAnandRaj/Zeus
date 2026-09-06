"""ChiClock v2: novelty-weighted experiential time with rarefaction-calibrated
glass alarm. Minted cell = +1.0 (one lived generation), known territory =
revisit_credit. Alarm is CONDITIONAL: minting must stall WHILE motion continues
at >= gate fraction of long-run baseline — mature residents (declining minting
matching coverage) never fire it, movers-that-discover-nothing do. Chao1
estimator reports remaining-discovery headroom."""
import json


class ChiClock:
    def __init__(self, res=1.0, revisit_credit=0.05, window=500,
                 motion_ratio_gate=0.5):
        self.res = res
        self.revisit_credit = revisit_credit
        self.window = window
        self.motion_ratio_gate = motion_ratio_gate
        self.counts = {}
        self.chi = 0.0
        self.minted = 0
        self.revisits = 0
        self.last_mint_step = 0
        self._win = []
        self.motion_base = None

    def update(self, s, step, moved=None):
        key = tuple(int(round(float(v) / self.res)) for v in s.tolist())
        cnt = self.counts.get(key, 0)
        mint = cnt == 0
        self.counts[key] = cnt + 1
        dchi = 1.0 if mint else self.revisit_credit
        self.chi += dchi
        if mint:
            self.minted += 1
            self.last_mint_step = step
        else:
            self.revisits += 1
        if moved is not None:
            self._win.append((1.0 if mint else 0.0, float(moved)))
            if len(self._win) > self.window:
                self._win.pop(0)
            w = self.motion_base
            self.motion_base = float(moved) if w is None else 0.99 * w + 0.01 * float(moved)
        return dchi

    def rarefaction(self):
        f1 = sum(1 for c in self.counts.values() if c == 1)
        f2 = sum(1 for c in self.counts.values() if c == 2)
        s_obs = len(self.counts)
        est = s_obs + (f1 * f1) / max(2 * max(f2, 1), 1)
        return {"observed_cells": s_obs, "singletons": f1, "doubletons": f2,
                "sd_ratio": round(f1 / max(f2, 1), 3),
                "chao1_estimated_total": round(est, 1),
                "coverage": round(s_obs / max(est, 1.0), 4)}

    def glass_alarm(self):
        """Three regimes, separated:
        young explorer   — mints present in window           -> no alarm
        mature resident  — minting declined, motion continues -> no alarm (this is health)
        glass            — zero minting AND motion >= gate*baseline -> ALARM"""
        if len(self._win) < self.window:
            if len(self.counts) < 25:
                return False
            return all(not m for m, _ in self._win)
        rec = self._win[-self.window:]
        mints = sum(m for m, _ in rec)
        motion = sum(v for _, v in rec) / len(rec)
        if mints > 0:
            return False
        if self.motion_base is None:
            return False
        return motion >= self.motion_ratio_gate * self.motion_base

    def snapshot(self):
        rf = self.rarefaction()
        return {"chi": round(self.chi, 2), "minted": self.minted,
                "revisits": self.revisits, "last_mint_step": self.last_mint_step,
                "cells_visited": rf["observed_cells"],
                "singletons": rf["singletons"], "doubletons": rf["doubletons"],
                "sd_ratio": rf["sd_ratio"],
                "coverage": rf["coverage"]}

    def state_dict(self):
        """Full restart state; unlike snapshot(), this preserves geometry."""
        return {
            "counts": dict(self.counts),
            "chi": self.chi,
            "minted": self.minted,
            "revisits": self.revisits,
            "last_mint_step": self.last_mint_step,
            "_win": list(self._win),
            "motion_base": self.motion_base,
        }

    def load_state_dict(self, state):
        if not state:
            return False
        self.counts = {tuple(k): v for k, v in state.get("counts", {}).items()}
        self.chi = state.get("chi", 0.0)
        self.minted = state.get("minted", 0)
        self.revisits = state.get("revisits", 0)
        self.last_mint_step = state.get("last_mint_step", 0)
        self._win = [tuple(row) for row in state.get("_win", [])]
        self.motion_base = state.get("motion_base")
        return True

    def load(self, st):
        if not st:
            return
        self.chi = st.get("chi", 0.0)
        self.minted = st.get("minted", 0)
        self.revisits = st.get("revisits", 0)
        self.last_mint_step = st.get("last_mint_step", 0)

    def dump_state(self, path):
        state = self.state_dict()
        payload = {**state,
                   "counts": {json.dumps(list(k)): v for k, v in self.counts.items()},
                   "_win": [list(t) for t in self._win]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load_counts(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            payload["counts"] = {tuple(json.loads(k)): v
                                 for k, v in payload["counts"].items()}
            self.load_state_dict(payload)
        except FileNotFoundError:
            pass
