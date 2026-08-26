"""ChiClock: novelty-weighted experiential time (two-currency doctrine).
dχ = 1.0 when S enters a coarse-cell never visited (minted configuration);
dχ = revisit_credit otherwise. Motion alone earns nothing — the jitter-trap
reads flat, the explorer reads endlessly. χ is Zeus's generation counter:
population-genetics formulas defined per-generation port natively once this
is the unit. A stalled χ while the system keeps stepping is the F-glass
signature: death measured in the subject's own time."""
import json


class ChiClock:
    def __init__(self, res=1.0, revisit_credit=0.05, stall_steps=2500):
        self.res = res
        self.revisit_credit = revisit_credit
        self.stall_steps = stall_steps
        self.visited = {}
        self.chi = 0.0
        self.minted = 0
        self.revisits = 0
        self.last_mint_step = 0

    def update(self, s, step):
        key = tuple(int(round(float(v) / self.res)) for v in s.tolist())
        if key not in self.visited:
            self.visited[key] = step
            self.minted += 1
            self.chi += 1.0
            self.last_mint_step = step
            return 1.0
        self.revisits += 1
        self.chi += self.revisit_credit
        return self.revisit_credit

    def stalled(self, step):
        return (step - self.last_mint_step) > self.stall_steps

    def snapshot(self):
        return {"chi": round(self.chi, 2), "minted": self.minted,
                "revisits": self.revisits,
                "last_mint_step": self.last_mint_step}

    def load(self, st):
        if not st:
            return
        self.chi = st.get("chi", 0.0)
        self.minted = st.get("minted", 0)
        self.revisits = st.get("revisits", 0)
        self.last_mint_step = st.get("last_mint_step", 0)
        self.visited = {tuple(k): v for k, v in st.get("visited", {}).items()}

    def dump_visited(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({json.dumps(list(k)): v for k, v in self.visited.items()}, f)
