"""Inner-monologue sampler: the mouth reading the self.

Draws a long S-driven token stream from a Zeus checkpoint and decodes it.
Two modes:
  --mode self  : step(None); mouth imagines purely from the self-trajectory.
  --mode loop  : self-sustained inner speech (sampled token fed back as input).
"""
import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusCore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--mode", default="loop", choices=["self", "loop"])
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ZeusCore.load(args.ckpt, device)
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    step = payload.get("step", "?")

    g = torch.Generator().manual_seed(args.seed)
    model.reset_state(noise=0.1, generator=g)
    model.eval()

    out = []
    prev = None
    for _ in range(args.steps):
        logits, _ = model.step(prev)
        probs = torch.softmax(logits / max(args.temperature, 1e-4), dim=-1)
        tok = torch.multinomial(probs.cpu(), 1, generator=g).item()
        out.append(tok)
        prev = tok if args.mode == "loop" else None

    text = model.decode(out)
    print(f"--- zeus.pt step {step} | mode={args.mode} | temp={args.temperature} | steps={args.steps}")
    print(text)


if __name__ == "__main__":
    main()