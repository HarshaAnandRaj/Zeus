"""Per-position CE over random train windows (mimics driven_pass 'ce').
Shows how much of the ~11-12 reading is the cross-window-context first tokens."""
import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusCore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/proof_coupled/zeus_step2500.pt")
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--bptt", type=int, default=33)
    args = ap.parse_args()

    train = np.load(str(ROOT / "corpus" / "data" / "train_ids.npy")).astype(np.int64)
    model = ZeusCore().to("cpu")
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    pos_ce = np.zeros(args.bptt - 1)
    pos_cnt = np.zeros(args.bptt - 1)
    rng = np.random.RandomState(0)
    with torch.no_grad():
        model.reset_state(noise=0.05)
        for _ in range(args.windows):
            off = rng.randint(0, len(train) - args.bptt - 1)
            seg = train[off:off + args.bptt]
            nxt = int(seg[0])
            for t in range(args.bptt - 1):
                logits, _ = model.step(nxt)
                target = int(seg[t + 1])
                ce = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target])).item()
                pos_ce[t] += ce; pos_cnt[t] += 1
                nxt = target
    pos_ce /= np.maximum(pos_cnt, 1)
    for i in range(len(pos_ce)):
        bar = "#" * int(pos_ce[i])
        print(f"tok{t:>2}  ce={pos_ce[i]:6.2f}  {bar}" if False else
              f"tok {i:>2}  ce={pos_ce[i]:6.2f}  {bar}")
    print(f"position-mean: first5={pos_ce[:5].mean():6.2f}  last10={pos_ce[-10:].mean():6.2f}  window-mean={pos_ce.mean():6.2f} (cf. log 'ce' ~ 9-12)")


if __name__ == "__main__":
    main()