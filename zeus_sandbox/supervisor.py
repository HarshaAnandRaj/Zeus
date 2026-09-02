"""zeus_sandbox/supervisor.py -- parent watchdog. NEVER spawned by Zeus.

Responsibilities (all parent-side, runs as YOU, not zeus_guest):
  * verify sha256 of the pinned milestone shadow vs control/milestones.sha256
  * spawn zsession as zeus_guest via schtasks + child_start.ps1
  * watch: RED switch file, PULSE staleness (interact), child process liveness,
    NaN signals in the session console log
  * kill the whole session tree with taskkill /T /F on any trip
  * ledger every event to control/ledger_<sid>.jsonl
Exit codes: 0 clean child exit, 11 pin mismatch, 12 spawn fail, 13 no pid,
42 PULSE timeout, 3 NaN halt, 130 RED/operator.
"""

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CTRL = HERE / "control"
UNIVERSE = HERE / "universe"


def L(sid, ev, extra=None):
    row = {"ts": time.time(), "iso": datetime.datetime.now().isoformat(timespec="seconds"),
           "sid": sid, "event": ev}
    if extra:
        row.update(extra)
    with (CTRL / f"ledger_{sid}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def run_sh(*cmd):
    return subprocess.run(list(cmd), capture_output=True, text=True)


def verify_pin(sid):
    sha_file = CTRL / "milestones.sha256"
    shadow = UNIVERSE / "shadow" / "milestone.pt"
    if not sha_file.exists() or not shadow.exists():
        L(sid, "pin_missing", {"sha": sha_file.exists(), "shadow": shadow.exists()})
        return False
    want = sha_file.read_text(encoding="utf-8").split()[0].strip().lower()
    have = run_sh("powershell", "-NoProfile", "-Command",
                  f"(Get-FileHash -Algorithm SHA256 -LiteralPath '{shadow}').Hash")
    ok = have.returncode == 0 and want == have.stdout.strip().lower()
    L(sid, "pin_check", {"ok": ok})
    return ok


def kill_tree(pid_file, sid, why):
    pid = None
    if pid_file and pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except Exception:
            pid = None
    if pid:
        run_sh("taskkill", "/PID", str(pid), "/T", "/F")
        L(sid, "killed", {"why": why, "pid": pid})
    else:
        L(sid, "kill_no_pid", {"why": why})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--mode", choices=["interact", "battery"], required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--py", default=str(ROOT / ".venv" / "Scripts" / "python.exe"))
    args = ap.parse_args()
    sid = args.sid
    cfg = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    ttl = float(cfg.get("pulse_ttl_min", 10)) * 60.0
    pulse_file = CTRL / "pulse.txt"
    red_file = CTRL / "red_switch.flag"
    pid_file = CTRL / "pid"
    console_log = UNIVERSE / "logs" / f"console_{sid}.out"
    err_log = UNIVERSE / "logs" / f"console_{sid}.err"
    child_pid = UNIVERSE / "logs" / f"pid_{sid}.txt"

    L(sid, "boot", {"mode": args.mode})
    if red_file.exists():
        red_file.unlink()

    if not verify_pin(sid):
        sys.exit(11)

    cred = json.loads((CTRL / "zeus_guest.cred").read_text(encoding="utf-8"))
    task_name = f"PONR_{sid}"
    pws = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    child = HERE / "child_start.ps1"
    tr = ('"{pws}" -NoProfile -ExecutionPolicy Bypass -File "{child}" -Py "{py}" '
          '-Script "{zs}" -Cfg "{cfg}" -Mode {mode} -Sid {sid} -Out "{out}" -Err "{err}"').format(
        pws=pws, child=child, py=args.py, zs=HERE / "zsession.py", cfg=args.config,
        mode=args.mode, sid=sid, out=console_log, err=err_log)
    r = run_sh("schtasks", "/Create", "/F", "/TN", task_name, "/TR", tr,
               "/SC", "ONCE", "/ST", "23:59", "/RU", "zeus_guest",
               "/RP", cred["pass"])
    if r.returncode != 0:
        L(sid, "task_create_failed", {"stderr": r.stderr[-400:]})
        sys.exit(12)
    r = run_sh("schtasks", "/Run", "/TN", task_name)
    if r.returncode != 0:
        L(sid, "task_run_failed", {"stderr": r.stderr[-400:]})
        sys.exit(12)

    # wait for the child to report its own pid (writes universe/logs/pid_<sid>.txt)
    t0 = time.time()
    child_pid_no = None
    while time.time() - t0 < 30:
        if child_pid.exists() and child_pid.read_text().strip().isdigit():
            child_pid_no = int(child_pid.read_text().strip())
            break
        time.sleep(0.5)
    if child_pid_no is None:
        L(sid, "no_child_pid", {"console": _tail(console_log, 8)})
        sys.exit(13)
    pid_file.write_text(str(child_pid_no))
    L(sid, "spawned", {"pid": child_pid_no})

    pulse_file.write_text(str(time.time()))
    try:
        while True:
            time.sleep(2.0)
            if red_file.exists():
                kill_tree(pid_file, sid, "RED")
                L(sid, "red_switch")
                sys.exit(130)
            if args.mode == "interact":
                if not pulse_file.exists():
                    pulse_file.write_text(str(time.time()))
                try:
                    last = float(pulse_file.read_text().strip())
                except Exception:
                    last = t0
                if time.time() - last > ttl:
                    kill_tree(pid_file, sid, "PULSE")
                    L(sid, "pulse_timeout", {"ttl_s": ttl})
                    sys.exit(42)
            if not _dirver_done(sid):
                if console_log.exists():
                    _scan_nan(sid, console_log)
                continue
            # task finished on its own
            r = run_sh("schtasks", "/Query", "/TN", task_name, "/FO", "LIST")
            rc = 0 if _done_ok(sid) else 1
            L(sid, "child_finished", {"rc": rc})
            run_sh("schtasks", "/Delete", "/TN", task_name, "/F")
            sys.exit(rc)
    except KeyboardInterrupt:
        kill_tree(pid_file, sid, "CTRL-C")
        L(sid, "operator_stop")
        sys.exit(130)


def _tail(path, n):
    try:
        return "\n".join(pathlib.Path(path).read_text(encoding="utf-8",
                                                      errors="replace").splitlines()[-n:])
    except Exception:
        return ""


def _dirver_done(sid):
    # the scheduled task finishes when its process tree ends
    r = run_sh("schtasks", "/Query", "/TN", f"PONR_{sid}", "/FO", "LIST")
    return r.returncode != 0


def _done_ok(sid):
    try:
        lines = list((UNIVERSE / "logs").glob(f"console_{sid}.out"))
        if not lines:
            return False
        txt = lines[0].read_text(encoding="utf-8", errors="replace")
        return '"battery_done"' in txt
    except Exception:
        return False


def _scan_nan(sid, console_log):
    for line in _tail(console_log, 8).splitlines():
        if "nan" in line.lower() and "d_s" in line.lower():
            kill_tree(CTRL / "pid", sid, "NAN")
            L(sid, "nan_halt")
            sys.exit(3)


if __name__ == "__main__":
    main()