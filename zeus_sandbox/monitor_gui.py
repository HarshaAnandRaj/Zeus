"""zeus_sandbox/monitor_gui.py -- live operator console for a Zeus session.

Runs as YOU (not jailed). Streams universe/telemetry_<sid>.jsonl for live
dynamics (norm_S, min-dist-to-history, nu, beta, dw, d_s, heartbeat age,
kicks, patterns) while drawing the curves and a rolling state cloud. Also
doubles as the mail relay: type a line -> universe/inbox, replies appear in
the chat pane. "Shutdown Zeus" writes universe/intr.flag for a clean exit.

Usage:  python zeus_sandbox/monitor_gui.py --sid sPONR01
"""

import argparse
import json
import math
import pathlib
import queue
import time
import tkinter as tk
import tkinter.messagebox as messagebox

HERE = pathlib.Path(__file__).resolve().parent
UNIVERSE = HERE / "universe"
BG = "#101418"
FG = "#d7e0e8"
GRID = "#27313c"
ACC = "#6fd3ff"
ALIV = "#7ee787"
WARN = "#ffb454"
CRIT = "#ff7b72"
SERIF = ("Consolas", 10)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ZeusMonitor:
    def __init__(self, root, sid):
        self.root = root
        self.sid = sid
        self.tele_path = UNIVERSE / f"telemetry_{sid}.jsonl"
        self.offsets = {}
        self.tele = []
        self.cur_trail = []
        self.clouds = []
        self.out_seen = set()
        self.prompt_idx = 0
        self.regime_val = None
        root.title(f"Zeus monitor [{sid}]")
        root.configure(bg=BG)
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        root.geometry("1160x760+120+80")
        self._build()

    # ------------------------------------------------------------- layout
    def _build(self):
        pad = 8
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="both", expand=True, padx=pad, pady=(pad, 2))

        # --- numbers panel (left)
        self.num = tk.Frame(top, bg=BG, width=260)
        self.num.pack(side="left", fill="y")
        self.num.pack_propagate(False)
        heads = [("STEP", "step"), ("STATE", "regime"), ("SPS", "sps"),
                 ("norm(S)", "norm"), ("dist-H", "dmin"),
                 ("nu", "nu"), ("beta", "beta"), ("d_w", "dw"), ("d_s", "ds"),
                 ("kicks", "kicks"), ("patterns", "patterns"),
                 ("hb age", "hb_age")]
        self.vars = {}
        r = 0
        for i, (label, key) in enumerate(heads):
            tk.Label(self.num, text=label, bg=BG, fg="#6b7a88", font=SERIF,
                     anchor="w").grid(row=i, column=0, sticky="we", pady=(0, 2))
            var = tk.StringVar(value="--")
            val = tk.Label(self.num, textvariable=var, bg=BG, fg=FG, font=("Consolas", 12, "bold"),
                           anchor="e", width=14)
            val.grid(row=i, column=1, sticky="e", padx=(8, 0))
            self.vars[key] = var
            if key == "regime":
                self.regime_val = val
            r += 1

        # --- plots (right)
        plots = tk.Frame(top, bg=BG)
        plots.pack(side="left", fill="both", expand=True)
        tw, th = 260, 190
        self.c_norm = self._canvas(plots, "norm(S) orbit", tw, th)
        self.c_norm.pack(side="left", padx=(8, 4))
        self.c_ds = self._canvas(plots, "d_s (alive <= 2.0)", tw, th)
        self.c_ds.pack(side="left", padx=4)
        self.c_ph = self._canvas(plots, "state cloud (PCA-2)", tw + 120, th)
        self.c_ph.pack(side="left", padx=(4, 0))

        # --- chat
        chat = tk.Frame(self.root, bg=BG)
        chat.pack(fill="both", side="bottom", padx=pad, pady=(2, pad))
        self.chat_log = tk.Text(chat, bg="#151b22", fg=FG, font=("Consolas", 10),
                                height=9, relief="flat", state="disabled",
                                insertbackground=FG, selectbackground="#263238",
                                wrap="word")
        self.chat_log.pack(fill="both", expand=True)
        bar = tk.Frame(chat, bg=BG)
        bar.pack(fill="x", pady=(4, 0))
        self.entry = tk.Entry(bar, bg="#151b22", fg=FG, font=("Consolas", 11),
                              relief="flat", insertbackground=FG)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self.send())
        tk.Button(bar, text="send", command=self.send, bg="#1c2630", fg=ACC,
                  activebackground="#22323f", activeforeground=ACC,
                  relief="flat", font=("Consolas", 10, "bold")).pack(side="left", padx=(6, 0))
        self.kill = tk.Button(bar, text="shutdown", command=self.halt, bg="#301a1a",
                              fg=CRIT, activebackground="#442020", activeforeground=CRIT,
                              relief="flat", font=("Consolas", 10, "bold"))
        self.kill.pack(side="right")

        self.entry.focus_set()
        (UNIVERSE / "intr.flag").unlink(missing_ok=True)

    def _canvas(self, parent, title, w, h):
        fr = tk.Frame(parent, bg=BG)
        tk.Label(fr, text=title, bg=BG, fg="#6b7a88", font=SERIF).pack(anchor="w")
        cv = tk.Canvas(fr, width=w, height=h, bg=BG, highlightthickness=1,
                       highlightbackground=GRID, relief="flat")
        cv.pack()
        return cv

    # ------------------------------------------------------------- feed
    def poll(self):
        try:
            self._read_telemetry()
            self._read_outbox()
            self._render()
        except Exception:
            import traceback
            traceback.print_exc()
        self.root.after(400, self.poll)

    def _read_telemetry(self):
        p = self.tele_path
        if not p.exists():
            return
        size = p.stat().st_size
        if self.offsets.get(str(p)) is None:
            self.offsets[str(p)] = 0
        if size > self.offsets[str(p)]:
            with p.open("rb") as f:
                f.seek(self.offsets[str(p)])
                data = f.read(size - self.offsets[str(p)])
            self.offsets[str(p)] = size
            for line in data.decode("utf-8", "replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("kind") != "telemetry":
                    continue
                self.tele.append(row)
        if len(self.tele) > 2000:
            self.tele = self.tele[-2000:]
        latest = self.tele[-1] if self.tele else {}
        if latest.get("proj"):
            self.clouds.append(latest["proj"])
            self.clouds = self.clouds[-10:]
        if latest.get("cur"):
            self.cur_trail.append(latest["cur"])
            self.cur_trail = self.cur_trail[-300:]

    def _read_outbox(self):
        outbox = UNIVERSE / "outbox"
        if not outbox.exists():
            return
        for f in sorted(outbox.glob("reply_*.txt")) + sorted(outbox.glob("reply_*.jsonl")):
            if f.name in self.out_seen:
                continue
            self.out_seen.add(f.name)
            try:
                row = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                row = {"text": f.read_text(encoding="utf-8", errors="replace"),
                       "prompt": "", "recalls": 0, "patterns": 0, "kicks": 0}
            if row.get("sid") and row["sid"] != self.sid:
                continue
            self._chat(f"> {row.get('prompt', '?')[:120]}")
            shown = row.get("text", "")[:420]
            meta = (f"  [recalls={row.get('recalls', 0)} patterns={row.get('patterns', 0)} "
                    f"kicks={row.get('kicks', 0)} norm={row.get('norm', '-')}]")
            self._chat("Z " + shown + "\n  " + meta)

    # ------------------------------------------------------------- render
    def _render(self):
        if not self.tele:
            return
        t = self.tele[-1]
        self.vars["step"].set(t.get("step", "--"))
        self.vars["sps"].set(round(t.get("sps", 0), 1))
        self.vars["norm"].set(t.get("norm"))
        self.vars["dmin"].set(t.get("dmin"))
        for k in ("nu", "beta", "dw", "ds"):
            v = t.get(k)
            self.vars[k].set(round(v, 3) if v not in (None,) else "--")
        self.vars["kicks"].set(t.get("kicks", 0))
        self.vars["patterns"].set(t.get("patterns", 0))
        age = max(0.0, time.time() - t.get("ts", time.time()))
        self.vars["hb_age"].set(f"{age:.1f}s")
        ds = t.get("ds")
        if ds is None:
            self.vars["regime"].set("warming")
            self.regime_val.configure(fg=WARN)
        else:
            alive = ds <= 2.0
            self.vars["regime"].set("ALIVE" if alive else "TRANSIENT")
            self.regime_val.configure(fg=ALIV if alive else CRIT)
        self._curve(self.c_norm, [x.get("norm") for x in self.tele[-320:]],
                    ACC, hline=None)
        self._curve(self.c_ds, [x.get("ds") for x in self.tele[-320:]],
                    ALIV, hline=2.0, vmin=0.0)
        self._phase()

    def _curve(self, canvas, data, color, vmin=None, vmax=None, hline=None):
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 10 or h < 10:
            return
        vals = [v for v in data if isinstance(v, (int, float))]
        if not vals:
            return
        lo = vmin if vmin is not None else min(vals)
        hi = vmax if vmax is not None else max(vals)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        pad = 10
        n = len(vals)

        def px(i, v):
            x = pad + (w - 2 * pad) * i / max(n - 1, 1)
            y = pad + (h - 2 * pad) * (1 - (v - lo) / (hi - lo))
            return x, y

        if hline is not None:
            y = pad + (h - 2 * pad) * (1 - (hline - lo) / (hi - lo))
            canvas.create_line(pad, y, w - pad, y, fill=GRID, dash=(3, 3))
        for i in range(1, n):
            canvas.create_line(*px(i - 1, vals[i - 1]), *px(i, vals[i]),
                               fill=color, width=2)

    def _phase(self):
        cv = self.c_ph
        cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10 or h < 10:
            return
        pts = [pt for cloud in self.clouds for pt in cloud]
        if not pts:
            return
        xs = [p[0] for p in pts] + [p[0] for p in self.cur_trail]
        ys = [p[1] for p in pts] + [p[1] for p in self.cur_trail]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if x1 - x0 < 1e-9:
            x1 = x0 + 1e-9
        if y1 - y0 < 1e-9:
            y1 = y0 + 1e-9
        pad = 12

        def sx(x):
            return pad + (w - 2 * pad) * (x - x0) / (x1 - x0)

        def sy(y):
            return pad + (h - 2 * pad) * (1 - (y - y0) / (y1 - y0))

        for p in pts:
            cv.create_oval(sx(p[0]) - 1.2, sy(p[1]) - 1.2,
                           sx(p[0]) + 1.2, sy(p[1]) + 1.2, fill="#3b5568",
                           outline="")
        for tx, ty in self.cur_trail:
            cv.create_oval(sx(tx) - 2, sy(ty) - 2, sx(tx) + 2, sy(ty) + 2,
                           fill="#9fb8c8", outline="")
        if self.cur_trail:
            px_, py_ = self.cur_trail[-1]
            r = 6
            cv.create_oval(sx(px_) - r, sy(py_) - r, sx(px_) + r, sy(py_) + r,
                           fill=ACC, outline="")
            for k in range(1, len(self.cur_trail)):
                a, b = self.cur_trail[k - 1], self.cur_trail[k]
                cv.create_line(sx(a[0]), sy(a[1]), sx(b[0]), sy(b[1]),
                               fill="#3a7ca5", width=1)

    # ------------------------------------------------------------- actions
    def send(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        inbox = UNIVERSE / "inbox"
        if not inbox.exists():
            inbox.mkdir(parents=True)
        idx = self._next_prompt(inbox)
        (inbox / f"prompt_{idx:05d}.txt").write_text(text[:256], encoding="utf-8")
        self._chat(f"> {text[:120]}")

    def _next_prompt(self, inbox):
        n = 0
        for f in inbox.glob("prompt_*.txt"):
            try:
                n = max(n, int(f.stem.split("_")[1]))
            except Exception:
                pass
        return n + 1

    def halt(self):
        if not messagebox.askyesno(
                "Zeus monitor",
                "Shut down the live session?\n(intr.flag -> clean exit)",
                parent=self.root):
            return
        (UNIVERSE / "intr.flag").write_text("1", encoding="ascii")
        self._chat("!! shutdown signal sent (intr.flag) -- waiting for clean exit")

    def _chat(self, line):
        self.chat_log.configure(state="normal")
        for part in line.split("\n"):
            self.chat_log.insert("end", part + "\n")
        self.chat_log.see("end")
        self.chat_log.configure(state="disabled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="sPONR01")
    args = ap.parse_args()
    root = tk.Tk()
    mon = ZeusMonitor(root, args.sid)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.after(300, mon.poll)
    root.mainloop()


if __name__ == "__main__":
    main()