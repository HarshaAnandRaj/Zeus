"""
ZeRun watcher — a tiny dependency-free TUI to follow a Zeus training run live.

Run it in YOUR OWN terminal (separate from opencode):

    python tools/watch.py runs/proof_heartbeat
    python tools/watch.py runs/lm_pretrain          # pretrain logs
    python tools/watch.py runs/proof_heartbeat 1.5 3000   # interval + total steps

It redraws a dashboard every <interval> seconds from the run's train.log /
pretrain.log: current step, latest eval metrics, sparklines of val_ce / d_s /
train_ce, and the most recent raw lines. Ctrl-C to quit.

Detects train vs pretrain logs automatically.
"""
import sys
import os
import time
import json

BARS = " .:-=+*#"  # ASCII-only so it works in any console encoding
CLEAR = "\033[2J\033[H"


def safe(s):
    return s.encode("ascii", "replace").decode("ascii")


def spark(data, width=46):
    if not data:
        return "(no data yet)"
    lo, hi = min(data), max(data)
    rng = (hi - lo) or 1.0
    sample = data[-width:]
    return "".join(BARS[min(7, int((v - lo) / rng * 7 + 0.5))] for v in sample)


def find_log(run):
    if run.endswith(".log"):
        return run
    for name in ("train.log", "pretrain.log"):
        p = os.path.join(run, name)
        if os.path.exists(p):
            # for pretrain, prefer the line-buffered progress.log if present
            if name == "pretrain.log":
                pp = os.path.join(run, "progress.log")
                if os.path.exists(pp):
                    return pp
            return p
    return None


def is_pretrain(log):
    return "pretrain" in log or log.endswith("progress.log")


def parse_train(lines):
    eval_ce, eval_ds, train_ce = [], [], []
    last_step = 0
    eval_step = 0
    sps = 0.0
    last = {}
    for ln in lines:
        s = ln.strip()
        if not s.startswith("{"):
            continue
        try:
            d = json.loads(s)
        except Exception:
            continue
        st = d.get("step")
        if isinstance(st, (int, float)):
            last_step = st
        if "steps_per_s" in d:
            sps = d["steps_per_s"]
        if "val_ce_nats" in d:
            last = d
            eval_step = st
            eval_ce.append(d["val_ce_nats"])
            if "d_s" in d:
                eval_ds.append(d["d_s"])
        elif "ce" in d and isinstance(d["ce"], (int, float)):
            train_ce.append(d["ce"])
    return last_step, eval_step, sps, eval_ce, eval_ds, train_ce, last


def parse_pretrain(lines):
    header = ""
    epochs = []
    prog = []  # (step, ce) from mid-epoch progress lines
    for ln in lines:
        s = ln.strip()
        if s.startswith("pretrain:"):
            header = s
        elif "avg_ce=" in s:
            try:
                ce = float(s.split("avg_ce=")[1].split()[0])
            except Exception:
                continue
            if "epoch" in s and "/" in s.split("avg_ce=")[0] and "step" not in s:
                epochs.append(ce)
            elif "step" in s:
                try:
                    step = int(s.split("step")[1].split("/")[0].strip())
                except Exception:
                    step = len(prog)
                prog.append((step, ce))
    return header, epochs, prog


def dash_train(log, total):
    with open(log, "r", errors="replace") as f:
        lines = f.readlines()[-3000:]
    step, eval_step, sps, eval_ce, eval_ds, train_ce, last = parse_train(lines)
    pct = f" ({100.0*step/total:.0f}%)" if total else ""
    out = []
    out.append(f"===== Zeus run watcher  (train)  =====  log: {log}")
    out.append(f"step: {step}{pct}   sps: {sps:.3f}")
    if last:
        vc = last.get("val_ce_nats")
        ds = last.get("d_s")
        h = last.get("health") or {}
        rho = h.get("rho_exact")
        reg = h.get("state_regime")
        floor = last.get("floor_L1"); floor2 = last.get("floor_L2")
        hb = last.get("hb_fire")
        out.append("-" * 52)
        out.append(f"last eval @ {eval_step}")
        out.append(f"  val_ce : {vc:.3f}  (L1={floor}, L2={floor2})  "
                   f"{'LM-OK' if (vc is not None and floor2 and vc < floor2) else 'weak'}")
        out.append(f"  d_s    : {ds:.3f}  (<=2 alive)   rho_exact: {rho}")
        out.append(f"  hb_fire: {hb}  kicks={last.get('hb_kicks')}  "
                   f"margin={last.get('margin_outer')}")
        if reg:
            out.append(f"  regime : {reg}")
    out.append("-" * 52)
    out.append(f"val_ce  : {spark(eval_ce)}")
    out.append(f"d_s     : {spark(eval_ds)}")
    out.append(f"train_ce: {spark(train_ce)}")
    out.append("-" * 52)
    out.append("recent raw:")
    for ln in lines[-6:]:
        out.append("  " + safe(ln.strip())[:110])
    return "\n".join(out)


def dash_pretrain(log):
    with open(log, "r", errors="replace") as f:
        lines = f.readlines()[-400:]
    header, epochs, prog = parse_pretrain(lines)
    out = []
    out.append(f"===== Zeus run watcher  (pretrain)  =====  log: {log}")
    if header:
        out.append(header)
    out.append("-" * 52)
    if prog:
        cur_step, cur_ce = prog[-1]
        out.append(f"progress: step {cur_step}   cur_avg_ce: {cur_ce:.3f}")
        out.append(f"avg_ce trend: {spark([c for _, c in prog])}")
    if epochs:
        out.append(f"epochs done: {len(epochs)}   final avg_ce: {epochs[-1]:.3f}")
    elif not prog:
        out.append("waiting for first step...")
    out.append("-" * 52)
    out.append("recent raw:")
    for ln in lines[-6:]:
        out.append("  " + safe(ln.strip())[:110])
    return "\n".join(out)


def main():
    if len(sys.argv) < 2:
        print("usage: watch.py <run_dir_or_log> [interval_s] [total_steps]")
        sys.exit(1)
    run = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    total = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    log = find_log(run)
    if not log:
        print(f"no train.log / pretrain.log found in {run}")
        sys.exit(1)
    try:
        while True:
            if is_pretrain(log):
                text = dash_pretrain(log)
            else:
                text = dash_train(log, total)
            print(CLEAR + text + f"\n\n[watching {log} | Ctrl-C to quit]")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
