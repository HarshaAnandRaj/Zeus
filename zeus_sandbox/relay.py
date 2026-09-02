"""zeus_sandbox/relay.py -- operator-side I/O forwarder (runs as YOU, not zeus_guest).

You type into THIS console; it drops prompts into universe/inbox/ and prints
Zeus's replies + vitals from universe/outbox/ as they appear. Zeus never sees
your terminal or your network; it only ever reads/writes its own universe dir.
"""

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
INBOX = HERE / "universe" / "inbox"
OUTBOX = HERE / "universe" / "outbox"

sys.stdout.reconfigure(encoding="utf-8")


def main():
    n = 0
    seen = set()
    print("> relay up. type a line; CTRL-C / EOF to exit.", flush=True)
    while True:
        try:
            line = sys.stdin.readline()
        except (KeyboardInterrupt, EOFError):
            break
        if not line:
            break
        line = line.rstrip("\n\r")
        if not line.strip():
            line = "hello"
        n += 1
        (INBOX / f"prompt_{n:05d}.txt").write_text(line, encoding="utf-8")
        time.sleep(0.05)
        for f in sorted(OUTBOX.glob("reply_*.jsonl")):
            if f.name in seen:
                continue
            seen.add(f.name)
            try:
                row = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            print(f"\n-- prompt: {row.get('prompt','')}")
            print(f"-- reply : {row.get('text','')}")
            print(f"-- vitals: norm={row.get('norm')} kicks={row.get('kicks')} "
                  f"recalls={row.get('recalls')} patterns={row.get('patterns')}")
            sys.stdout.flush()
    print("\n> relay down.", flush=True)


if __name__ == "__main__":
    main()