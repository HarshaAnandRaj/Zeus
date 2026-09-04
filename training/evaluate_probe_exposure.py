"""Measure short-prefix teacher-forced versus self-generated continuation CE.

This is an evaluation-only diagnostic.  It asks whether a mouth that looks
good on real-token histories remains accurate after it has consumed its own
tokens from the same deployment-shaped short prefixes.  It does not choose or
validate a training intervention.
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


def bootstrap_mean_ci(values, *, seed, resamples=2000):
    """Deterministic 95% bootstrap interval for one value per trajectory."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty trajectory set")
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[draw].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


@torch.no_grad()
def exposure_metrics(readout, emb, cfg, ids_arr, *, low, high, rollout_tokens,
                     batches, batch_size, seed, device):
    """Compare target CE after real versus greedy self-generated histories."""
    if low < 1 or high < low:
        raise ValueError("prefix range must be positive and ordered")
    if rollout_tokens < 1 or high + rollout_tokens >= cfg.ctx_window:
        raise ValueError("prefix range and rollout length do not fit the context window")
    rng = np.random.default_rng(seed)
    teacher_total = rollout_total = 0.0
    teacher_trajectories, rollout_trajectories = [], []
    generated_matches = positions = 0
    readout.eval(); emb.eval()
    for _ in range(batches):
        starts = rng.integers(0, len(ids_arr) - cfg.ctx_window - 1, size=batch_size)
        rows = np.stack([ids_arr[s:s + cfg.ctx_window + 1] for s in starts])
        ids = torch.from_numpy(rows.astype(np.int64, copy=False)).to(device)
        prefixes = torch.from_numpy(rng.integers(low, high + 1, size=batch_size)).to(device)
        generated = [[] for _ in range(batch_size)]
        teacher_by_row = torch.zeros(batch_size, device=device)
        rollout_by_row = torch.zeros(batch_size, device=device)
        for offset in range(rollout_tokens):
            teacher_layout = torch.zeros(batch_size, cfg.ctx_window, cfg.dim, device=device)
            rollout_layout = torch.zeros_like(teacher_layout)
            targets = torch.empty(batch_size, dtype=torch.long, device=device)
            for row, prefix_len in enumerate(prefixes.tolist()):
                teacher_context = emb(ids[row, :prefix_len + offset])
                rollout_context = teacher_context[:prefix_len]
                if generated[row]:
                    rollout_context = torch.cat([rollout_context, torch.stack(generated[row])], dim=0)
                teacher_layout[row, cfg.ctx_window - teacher_context.shape[0]:] = teacher_context
                rollout_layout[row, cfg.ctx_window - rollout_context.shape[0]:] = rollout_context
                targets[row] = ids[row, prefix_len + offset]
            teacher_logits = readout_logits(readout, emb, teacher_layout, cfg)[:, -1]
            rollout_logits = readout_logits(readout, emb, rollout_layout, cfg)[:, -1]
            teacher_loss = F.cross_entropy(teacher_logits.float(), targets, reduction="none")
            rollout_loss = F.cross_entropy(rollout_logits.float(), targets, reduction="none")
            teacher_total += float(teacher_loss.sum())
            rollout_total += float(rollout_loss.sum())
            teacher_by_row += teacher_loss
            rollout_by_row += rollout_loss
            next_ids = rollout_logits.argmax(dim=-1)
            generated_matches += int((next_ids == targets).sum())
            positions += batch_size
            next_emb = emb(next_ids)
            for row in range(batch_size):
                generated[row].append(next_emb[row])
        teacher_trajectories.extend((teacher_by_row / rollout_tokens).cpu().tolist())
        rollout_trajectories.extend((rollout_by_row / rollout_tokens).cpu().tolist())
    teacher_ce = teacher_total / positions
    rollout_ce = rollout_total / positions
    gaps = np.asarray(rollout_trajectories) - np.asarray(teacher_trajectories)
    return {"teacher_forced_ce": teacher_ce, "self_generated_ce": rollout_ce,
            "exposure_gap": rollout_ce - teacher_ce,
            "greedy_target_match": generated_matches / positions,
            "trajectory_n": len(teacher_trajectories),
            "teacher_forced_ci": bootstrap_mean_ci(teacher_trajectories, seed=seed + 11),
            "self_generated_ci": bootstrap_mean_ci(rollout_trajectories, seed=seed + 12),
            "exposure_gap_ci": bootstrap_mean_ci(gaps, seed=seed + 13)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base_checkpoint", default="zeus_sandbox/universe/shadow/milestone.pt")
    ap.add_argument("--val_ids", required=True)
    ap.add_argument("--low", type=int, default=3)
    ap.add_argument("--high", type=int, default=16)
    ap.add_argument("--rollout_tokens", type=int, default=8)
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260904)
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
    report = {"kind": "probe_exposure_gap", "created_unix": int(time.time()),
              "checkpoint": str(args.checkpoint), "step": int(saved["global_step"]),
              "val_ids": str(args.val_ids), "prefix_range": [args.low, args.high],
              "rollout_tokens": args.rollout_tokens, "batches": args.batches,
              "batch": args.batch,
              **{k: ([round(x, 5) for x in v] if isinstance(v, list)
                      else round(v, 5) if isinstance(v, float) else v)
                  for k, v in exposure_metrics(
                      model.readout, model.embed, cfg, val_ids, low=args.low, high=args.high,
                      rollout_tokens=args.rollout_tokens, batches=args.batches, batch_size=args.batch,
                      seed=args.seed, device=device).items()}}
    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.checkpoint).with_name("exposure_gap.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
