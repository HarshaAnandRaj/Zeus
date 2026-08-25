"""Relaunch wrapper: keeps training alive across crashes, resumes from newest checkpoint."""
import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def complete(save_dir: pathlib.Path, target: int):
    final = save_dir / "zeus.pt"
    if not final.exists():
        return False
    try:
        import torch
        p = torch.load(final, map_location="meta", weights_only=False)
        return int(p.get("step", -1)) >= target
    except Exception:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("train_args", nargs="*", default=[])
    args = ap.parse_args()
    sd = ROOT / args.save_dir
    sd.mkdir(parents=True, exist_ok=True)
    rlog = open(sd / "relaunch.log", "a", encoding="utf-8")

    def rprint(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        rlog.write(line + "\n")
        rlog.flush()

    newest = sorted(sd.glob("zeus_step*.pt"))
    if newest:
        args.train_args = [a for a in args.train_args]
        seed_note = f"resuming from {newest[-1].name}"
    else:
        seed_note = "fresh start"
    rprint(f"wrapper up ({seed_note}), target step {args.target}")

    attempt = 0
    while attempt < 60:
        if complete(sd, args.target):
            rprint("COMPLETE marker reached - wrapper exiting")
            break
        attempt += 1
        cmd = [sys.executable, str(ROOT / "training" / "train.py"),
               "--save_dir", args.save_dir, "--steps", str(args.target),
               "--resume", "auto"] + args.train_args
        rprint(f"attempt {attempt}: launching trainer")
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        rprint(f"trainer exited rc={rc}")
        time.sleep(10)
    rprint("wrapper done")


if __name__ == "__main__":
    main()
