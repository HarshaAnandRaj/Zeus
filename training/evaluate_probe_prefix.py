"""Measure deploy-shaped short- and long-prefix CE for a probe checkpoint.

Dense CE is useful for numerical health but mostly scores full real contexts.
Zeus starts replies from short, right-aligned, zero-filled prompts, so this
script reports that exact next-token distribution separately.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusConfig, ZeusCore
from training.probe_train import readout_logits


def prefix_ce(readout, emb, cfg, ids_arr, *, low, high, batches, batch_size, seed, device):
    if low < 0 or high >= cfg.ctx_window or low > high:
        raise ValueError("prefix range must be within [0, ctx_window - 1]")
    rng = np.random.default_rng(seed)
    total = 0.0
    readout.eval(); emb.eval()
    with torch.no_grad():
        for _ in range(batches):
            starts = rng.integers(0, len(ids_arr) - cfg.ctx_window - 1, size=batch_size)
            rows = np.stack([ids_arr[s:s + cfg.ctx_window + 1] for s in starts])
            ids = torch.from_numpy(rows.astype(np.int64, copy=False)).to(device)
            prefix = torch.from_numpy(rng.integers(low, high + 1, size=batch_size)).to(device)
            layout = torch.zeros(batch_size, cfg.ctx_window, cfg.dim, device=device)
            targets = torch.empty(batch_size, dtype=torch.long, device=device)
            for row, length in enumerate(prefix.tolist()):
                layout[row, cfg.ctx_window - length:] = emb(ids[row, :length])
                targets[row] = ids[row, length]
            logits = readout_logits(readout, emb, layout, cfg,
                                    last_token_present=prefix > 0)[:, -1]
            total += float(F.cross_entropy(logits.float(), targets).item())
    return total / batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base_checkpoint", default="zeus_sandbox/universe/shadow/milestone.pt")
    ap.add_argument("--val_ids", required=True)
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = dict(base["config"])
    cfg_dict.update(saved["readout_config"])
    cfg = ZeusConfig(**cfg_dict)
    device = torch.device(args.device)
    model = ZeusCore(cfg, tokenizer_path=base.get("tokenizer")).to(device)
    model.readout.load_state_dict(saved["readout"])
    model.embed.load_state_dict(saved["embedding"])
    val_ids = np.load(args.val_ids, mmap_mode="r")
    report = {
        "kind": "probe_prefix_ce", "created_unix": int(time.time()),
        "checkpoint": str(args.checkpoint), "step": int(saved["global_step"]),
        "val_ids": str(args.val_ids), "batches": args.batches, "batch": args.batch,
        "short_prefix_ce": round(prefix_ce(model.readout, model.embed, cfg, val_ids,
                                            low=3, high=16, batches=args.batches,
                                            batch_size=args.batch, seed=args.seed,
                                            device=device), 5),
        "blank_prefix_ce": round(prefix_ce(model.readout, model.embed, cfg, val_ids,
                                            low=0, high=2, batches=args.batches,
                                            batch_size=args.batch, seed=args.seed + 2,
                                            device=device), 5),
        "long_prefix_ce": round(prefix_ce(model.readout, model.embed, cfg, val_ids,
                                           low=48, high=63, batches=args.batches,
                                           batch_size=args.batch, seed=args.seed + 1,
                                           device=device), 5),
    }
    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.checkpoint).with_name("prefix_ce.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
