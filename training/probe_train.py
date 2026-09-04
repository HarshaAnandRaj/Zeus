"""Resumable clean-corpus readout probe for Zeus.

This is deliberately a substrate experiment, not a self-organization pass. It
trains only the token mouth on the exact right-aligned deployment distribution,
while preserving an optimizer-and-RNG-complete checkpoint so a long run can be
resumed without silently changing the experiment.

The corresponding free-run/ownership evaluation is
``training/evaluate_probe_voice.py`` after assembling the emitted readout and
embedding weights into a Zeus checkpoint.
"""

import argparse
import json
import math
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusConfig, ZeusCore


def _json_default(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(type(value).__name__)


def capture_rng(np_rng):
    return {
        "python": random.getstate(),
        "numpy": np_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    # The training payload may be loaded with map_location=cuda. RNG streams
    # are control data, not model buffers: PyTorch requires CPU ByteTensors
    # when restoring them.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
    rng = np.random.default_rng()
    rng.bit_generator.state = state["numpy"]
    return rng


def make_rollout_generator(device, *, seed, state=None):
    """Create the dedicated device-local sampler used by raw rollout exposure."""
    generator = torch.Generator(device=device)
    if state is None:
        generator.manual_seed(seed)
    else:
        generator.set_state(state.cpu())
    return generator


def atomic_torch_save(payload, path, *, max_replace_attempts=12, retry_seconds=0.25):
    """Make checkpoints observable only after every tensor record is written.

    Windows can transiently deny a rename while an indexer, viewer, or stale
    reader releases a checkpoint handle.  The temporary payload is complete at
    that point; retry the atomic replacement rather than killing a multi-hour
    run or falling back to an in-place write.
    """
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    if max_replace_attempts < 1:
        raise ValueError("max_replace_attempts must be positive")
    for attempt in range(max_replace_attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt + 1 >= max_replace_attempts:
                raise
            time.sleep(max(0.0, retry_seconds))


def should_rollout(completed_steps, every, start):
    """Whether the next update is a rollout after an explicit warm phase."""
    return every > 0 and completed_steps >= start and (
        (completed_steps - start + 1) % every == 0
    )


def sample_visible_lengths(np_rng, batch, width, short_prefix_prob=0.0,
                           short_prefix_max=16, blank_prefix_prob=0.0,
                           blank_prefix_max=2):
    """Sample deployment contexts, including an opt-in blank-initiation mode."""
    if not 0.0 <= short_prefix_prob <= 1.0 or not 0.0 <= blank_prefix_prob <= 1.0:
        raise ValueError("prefix probabilities must be in [0, 1]")
    if short_prefix_prob + blank_prefix_prob > 1.0:
        raise ValueError("short_prefix_prob + blank_prefix_prob must not exceed 1")
    if not 3 <= short_prefix_max < width:
        raise ValueError("short_prefix_max must be within [3, ctx_window - 1]")
    if not 0 <= blank_prefix_max < 3:
        raise ValueError("blank_prefix_max must be within [0, 2]")
    visible = np_rng.integers(3, width, size=batch)
    # Preserve the v5 random stream exactly when both options are disabled.
    if short_prefix_prob or blank_prefix_prob:
        choice = np_rng.random(batch)
        choose_blank = choice < blank_prefix_prob
        choose_short = ((choice >= blank_prefix_prob) &
                        (choice < blank_prefix_prob + short_prefix_prob))
        blank_count = int(choose_blank.sum())
        if blank_count:
            visible[choose_blank] = np_rng.integers(0, blank_prefix_max + 1,
                                                     size=blank_count)
        count = int(choose_short.sum())
        if count:
            visible[choose_short] = np_rng.integers(3, short_prefix_max + 1, size=count)
    return visible


def raw_freq_weights(targets, counts, alpha, wmax=8.0):
    """Unnormalized rare-target weights (see target_weights)."""
    c = counts.to(targets.device)[targets].clamp_min(1).to(torch.float32)
    ref = float(torch.median(counts.to(torch.float32)))
    return torch.clamp((ref / c) ** float(alpha), min=1.0, max=float(wmax))


def target_weights(targets, counts, alpha, wmax=8.0):
    """Rare-target upweighting for the lexical-tail diagnosis.

    w = clip((median_count / count)^alpha, 1, wmax): frequent types are never
    downweighted (floor 1), rare types are boosted, self-normalized to mean 1
    so the effective learning-rate scale is preserved.  alpha <= 0 disables
    (returns None) and the caller falls back to plain mean CE.
    """
    if alpha <= 0:
        return None
    w = raw_freq_weights(targets, counts, alpha, wmax)
    return w / w.mean().clamp_min(1e-9)


def wordfinal_raw(global_idx, mask, weight):
    """Raw word-validity weights: `weight` where the target completes a word.

    mask is a bool array aligned with the corpus ids (position i word-final
    iff token i+1 begins with G); global_idx gives each target's corpus
    position.  weight <= 1 disables the pressure (all ones).
    """
    if weight <= 1.0:
        return None
    m = torch.from_numpy(np.asarray(mask, dtype=bool)).to(global_idx.device)
    hit = m[global_idx.to(torch.long).clamp_max(m.numel() - 1)]
    return torch.where(hit, torch.full((), float(weight), device=global_idx.device),
                       torch.ones((), device=global_idx.device))


def weighted_ce(logits, targets, weights):
    ce = F.cross_entropy(logits.float(), targets, reduction="none")
    if weights is None:
        return ce.mean()
    return (ce * weights).mean()


def combine_weights(targets, gidx, freq, wpos):
    """Multiply rare-target and word-validity pressures, normalized to mean 1."""
    w = None
    if freq is not None:
        w = raw_freq_weights(targets, *freq)
    if wpos is not None and gidx is not None:
        mask, weight = wpos
        wf = wordfinal_raw(gidx, mask, weight)
        if wf is not None:
            w = wf if w is None else w * wf
    if w is None:
        return None
    return w / w.mean().clamp_min(1e-9)


def readout_logits(readout, emb, e_in, cfg, *, last_token_present=True):
    """The exact self-source token path used by the deploy-shaped trainer."""
    batch, width, dim = e_in.shape
    x = e_in.transpose(0, 1).contiguous() + readout.ctx_pos[:width].unsqueeze(1)
    causal = torch.triu(torch.ones(width, width, device=e_in.device, dtype=torch.bool), diagonal=1)
    if getattr(cfg, "cross_attn", False):
        for layer in readout.ctx_tf:
            x = layer(x, causal, None)  # token source is its own context during pretraining
    else:
        x = readout.ctx_tf(x, mask=causal)
    h = x.transpose(0, 1)
    s_zero = torch.zeros(batch, width, dim, device=e_in.device)
    # ``deploy_self_source`` zeros the live state but does not remove the
    # readout's state projection.  With a zero state it is a learned, static
    # logit prior (LayerNorm can also have a learned bias).  Omitting it here
    # silently trained a different logit distribution from the one Zeus uses
    # in ``observe``.  Keep this algebraically identical to
    # CoupledReadout.forward so an isolated mouth checkpoint is deploy-valid.
    s_n = readout.ln(s_zero)
    s_dir = F.normalize(s_n, dim=-1)
    e_path = (readout.e_proj(e_in) +
              readout.gate_gain * readout.gate(torch.cat([s_n, e_in], dim=-1)))
    if isinstance(last_token_present, torch.Tensor):
        present = last_token_present.to(device=e_in.device, dtype=e_path.dtype)
        if present.dim() == 1:
            present = present[:, None, None]
        e_path = e_path * present
    elif not last_token_present:
        e_path = torch.zeros_like(e_path)
    return (readout.s_scale * readout.s_proj(s_dir) +
            readout.ctx_gain * readout.ctx_head(h) + e_path)


def right_aligned_loss(readout, emb, cfg, ids, np_rng, *, short_prefix_prob=0.0,
                       short_prefix_max=16, blank_prefix_prob=0.0,
                       blank_prefix_max=2, freq=None, gstarts=None, wpos=None):
    """Predict a real token from a right-aligned, zero-filled prefix."""
    device, width, dim = ids.device, cfg.ctx_window, cfg.dim
    batch = ids.shape[0]
    visible = torch.from_numpy(sample_visible_lengths(
        np_rng, batch, width, short_prefix_prob, short_prefix_max,
        blank_prefix_prob, blank_prefix_max,
    )).to(device)
    layouts = torch.zeros(batch, width, dim, device=device)
    targets = torch.empty(batch, dtype=torch.long, device=device)
    gidx = torch.empty(batch, dtype=torch.long, device=device)
    for row, r in enumerate(visible.tolist()):
        layouts[row, width - r:] = emb(ids[row, :r])
        targets[row] = ids[row, r]
        gidx[row] = int(gstarts[row]) + r if gstarts is not None else 0
    logits = readout_logits(readout, emb, layouts, cfg,
                            last_token_present=visible > 0)[:, -1]
    weights = combine_weights(targets, gidx, freq, wpos)
    return weighted_ce(logits, targets, weights)


def rollout_loss(readout, emb, cfg, ids, np_rng, *, rollout_tokens, min_prefix,
                 sampling="greedy", generator=None, freq=None, gstarts=None, wpos=None):
    """Train recovery from the mouth's *own* sampled history.

    Dense/teacher-forced CE can look good while every small sampling error moves
    the mouth off its training distribution.  This intentionally unrolls a
    short deployment-shaped continuation: generated tokens become the
    next right-aligned context, while every position is still supervised by its
    held-out corpus continuation.  The discrete feedback is detached by the
    argmax, exactly as it is at runtime.
    """
    if rollout_tokens < 1:
        raise ValueError("rollout_tokens must be positive")
    if sampling not in {"greedy", "raw"}:
        raise ValueError("rollout sampling must be 'greedy' or 'raw'")
    if sampling == "raw" and generator is None:
        raise ValueError("raw rollout sampling requires a generator")
    width, dim = cfg.ctx_window, cfg.dim
    if min_prefix < 3 or min_prefix + rollout_tokens >= width:
        raise ValueError("min_prefix and rollout_tokens do not fit the context window")
    batch = ids.shape[0]
    # Randomizing prefix length prevents the generator from becoming good only
    # at one artificial prompt position.  Keep room for every target token.
    max_prefix = width - rollout_tokens - 1
    prefixes = torch.from_numpy(
        np_rng.integers(min_prefix, max_prefix + 1, size=batch)
    ).to(ids.device)
    generated = [[] for _ in range(batch)]
    losses = []
    for offset in range(rollout_tokens):
        layout = torch.zeros(batch, width, dim, device=ids.device)
        for row, prefix_len in enumerate(prefixes.tolist()):
            context = emb(ids[row, :prefix_len])
            if generated[row]:
                context = torch.cat([context, torch.stack(generated[row])], dim=0)
            layout[row, width - context.shape[0]:] = context
        logits = readout_logits(readout, emb, layout, cfg)[:, -1]
        targets = torch.stack([ids[row, int(prefixes[row]) + offset]
                               for row in range(batch)])
        if gstarts is not None:
            gidx = torch.stack([torch.tensor(int(gstarts[row]) + int(prefixes[row]) + offset)
                                for row in range(batch)]).to(ids.device)
        else:
            gidx = None
        weights = combine_weights(targets, gidx, freq, wpos)
        losses.append(weighted_ce(logits, targets, weights))
        # Runtime sampling is not differentiable.  The raw mode deliberately
        # uses exactly the strict evaluator's unmodified distribution
        # (temperature/top-p/repetition penalty all 1); its own CUDA/CPU
        # generator is checkpointed separately from global Torch RNG.
        if sampling == "greedy":
            nxt = logits.detach().argmax(dim=-1)
        elif sampling == "raw":
            probs = F.softmax(logits.detach().float(), dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        nxt_emb = emb(nxt)
        for row in range(batch):
            generated[row].append(nxt_emb[row])
    return torch.stack(losses).mean()


def dense_loss(readout, emb, cfg, ids, freq=None, gstarts=None, wpos=None):
    width = cfg.ctx_window
    e_in = emb(ids[:, :width])
    logits = readout_logits(readout, emb, e_in, cfg)
    targets = ids[:, 1:width + 1].reshape(-1).long()
    if gstarts is not None:
        grid = (torch.from_numpy(np.asarray(gstarts)).to(ids.device)[:, None]
                + torch.arange(1, width + 1, device=ids.device)[None, :]).reshape(-1)
    else:
        grid = None
    weights = combine_weights(targets, grid, freq, wpos)
    return weighted_ce(logits.reshape(-1, logits.shape[-1]), targets, weights)


@torch.no_grad()
def validate(readout, emb, cfg, val_ids, np_rng, device, batches=24, batch_size=32):
    readout.eval()
    total = 0.0
    for _ in range(batches):
        starts = np_rng.integers(0, len(val_ids) - cfg.ctx_window - 1, size=batch_size)
        rows = np.stack([val_ids[s:s + cfg.ctx_window + 1] for s in starts])
        ids = torch.from_numpy(rows.astype(np.int64, copy=False)).to(device)
        total += float(dense_loss(readout, emb, cfg, ids).item())
    readout.train()
    return total / batches


def save_checkpoint(path, *, readout, emb, optimizer, scaler, np_rng, rollout_generator,
                    args, cfg, step, history):
    payload = {
        "version": 1,
        "global_step": step,
        "readout": readout.state_dict(),
        "embedding": emb.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng(np_rng),
        "rollout_generator_state": rollout_generator.get_state().cpu(),
        "args": vars(args),
        "readout_config": {k: getattr(cfg, k) for k in
                           ("readout_layers", "readout_ffn_mult", "readout_heads", "ctx_anchor", "cross_attn", "vocab", "dim", "ctx_window")},
        "history": history,
    }
    atomic_torch_save(payload, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ids", default="runs/probe_pilot/train_ids.npy")
    ap.add_argument("--val_ids", default="runs/probe_pilot/val_ids.npy")
    ap.add_argument("--base_checkpoint", default="zeus_sandbox/universe/shadow/milestone.pt")
    ap.add_argument("--init_checkpoint", default="",
                    help="optional probe checkpoint supplying only mouth weights; optimizer/RNG restart fresh")
    ap.add_argument("--save_dir", default="runs/probe_v2_primary")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=0, help="optional smoke-run cap")
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--dense_weight", type=float, default=0.15)
    ap.add_argument("--freq_weight_alpha", type=float, default=0.0,
                    help="rare-target upweight exponent; 0 disables (plain mean CE)")
    ap.add_argument("--freq_weight_max", type=float, default=8.0,
                    help="clip for rare-target upweighting")
    ap.add_argument("--wordfinal_weight", type=float, default=1.0,
                    help="gradient multiplier on word-completing targets; 1 disables")
    ap.add_argument("--wordfinal_mask", default="",
                    help=".npy bool mask aligned with train_ids (required when weight > 1)")
    ap.add_argument("--short_prefix_prob", type=float, default=0.0,
                    help="fraction of right-aligned updates sampled from short reply-start prefixes")
    ap.add_argument("--short_prefix_max", type=int, default=16,
                    help="largest visible-token prefix when short-prefix sampling is selected")
    ap.add_argument("--blank_prefix_prob", type=float, default=0.0,
                    help="fraction of updates sampled from true blank through two-token starts")
    ap.add_argument("--blank_prefix_max", type=int, default=2,
                    help="largest visible-token prefix in blank-initiation sampling")
    ap.add_argument("--rollout_every", type=int, default=4,
                    help="replace one dense step every N steps with self-generated rollout CE; 0 disables")
    ap.add_argument("--rollout_start", type=int, default=0,
                    help="number of fresh-run teacher-forced updates before rollout exposure starts")
    ap.add_argument("--rollout_batch", type=int, default=2,
                    help="batch size for the serial autoregressive rollout step")
    ap.add_argument("--rollout_tokens", type=int, default=8,
                    help="number of self-generated tokens exposed per rollout update")
    ap.add_argument("--rollout_min_prefix", type=int, default=8,
                    help="smallest real-token prefix used by a rollout update")
    ap.add_argument("--rollout_sampling", choices=("greedy", "raw"), default="greedy",
                    help="self-history token rule; raw is unmodified strict-runtime sampling")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true",
                    help="opt into CUDA FP16 autocast after a stable smoke test")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    train_path, val_path, base_path = map(pathlib.Path, (args.train_ids, args.val_ids, args.base_checkpoint))
    if not train_path.exists() or not val_path.exists() or not base_path.exists():
        raise FileNotFoundError("train_ids, val_ids, and base_checkpoint must all exist")
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, log_path = save_dir / "checkpoint.pt", save_dir / "progress.jsonl"
    existing_run_config = None
    prior_config_path = save_dir / "run_config.json"
    if ckpt_path.exists() and not args.fresh and prior_config_path.exists():
        try:
            existing_run_config = json.loads(prior_config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            # Checkpoint configuration remains authoritative for safe resume;
            # a damaged human-readable run config must not block recovery.
            existing_run_config = None
    device = torch.device(args.device)

    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    rollout_generator = make_rollout_generator(device, seed=args.seed + 10_003)

    base = torch.load(base_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(base["config"])
    # The planned probe is the deploy-shaped, 6-layer cross-attention voice.
    cfg_dict.update({"readout_layers": 6, "readout_ffn_mult": 2,
                     "readout_heads": 12, "cross_attn": True,
                     "ctx_anchor": False})
    cfg = ZeusConfig(**cfg_dict)
    model = ZeusCore(cfg, tokenizer_path=base.get("tokenizer")).to(device)
    readout, emb = model.readout, model.embed
    readout.train(); emb.train()
    optimizer = torch.optim.AdamW(list(readout.parameters()) + list(emb.parameters()), lr=args.lr)
    # The initial 6-layer probe smoke produced finite loss but NaN FP16
    # gradients.  Default to trustworthy FP32; AMP is opt-in only after it
    # proves stable for the exact configuration being launched.
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    global_step, history = 0, []

    run_config = {"args": vars(args), "readout_config": {k: getattr(cfg, k) for k in
                  ("readout_layers", "readout_ffn_mult", "readout_heads", "ctx_anchor", "cross_attn", "vocab", "dim", "ctx_window")},
                  "base_checkpoint": str(base_path.resolve()), "initialization": None}

    if ckpt_path.exists() and not args.fresh and args.init_checkpoint:
        raise RuntimeError("local resume checkpoint and --init_checkpoint are mutually exclusive")
    if ckpt_path.exists() and not args.fresh:
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        if saved.get("readout_config") != run_config["readout_config"]:
            raise RuntimeError("resume checkpoint readout configuration does not match this run")
        readout.load_state_dict(saved["readout"])
        emb.load_state_dict(saved["embedding"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        np_rng = restore_rng(saved["rng"])
        rollout_generator = make_rollout_generator(
            device, seed=args.seed + 10_003, state=saved.get("rollout_generator_state"))
        global_step, history = int(saved["global_step"]), list(saved.get("history", []))
        if existing_run_config is not None:
            run_config["initialization"] = existing_run_config.get("initialization")
        print(f"resumed at step {global_step}", flush=True)
    elif args.init_checkpoint:
        init_path = pathlib.Path(args.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"initialization checkpoint does not exist: {init_path}")
        source = torch.load(init_path, map_location=device, weights_only=False)
        if source.get("readout_config") != run_config["readout_config"]:
            raise RuntimeError("initialization checkpoint readout configuration does not match this run")
        readout.load_state_dict(source["readout"])
        emb.load_state_dict(source["embedding"])
        # This is a curriculum boundary rather than a mathematically exact
        # resume: avoid carrying Adam moments/RNG sampled for the prior,
        # teacher-forced objective into the rollout-exposure phase.
        run_config["initialization"] = {
            "checkpoint": str(init_path.resolve()),
            "source_global_step": int(source.get("global_step", 0)),
            "optimizer_and_rng": "fresh",
        }
        print(f"initialized mouth from {init_path} at source step {source.get('global_step', 0)}", flush=True)

    (save_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=_json_default), encoding="utf-8")

    train_ids = np.load(train_path, mmap_mode="r")
    val_ids = np.load(val_path, mmap_mode="r")
    freq = None
    if args.freq_weight_alpha > 0:
        counts = torch.bincount(
            torch.from_numpy(np.asarray(train_ids).astype(np.int64)),
            minlength=cfg.vocab).to(torch.float64)
        freq = (counts, args.freq_weight_alpha, args.freq_weight_max)
        print(json.dumps({"kind": "freq_weights",
                          "alpha": args.freq_weight_alpha,
                          "max": args.freq_weight_max,
                          "median_count": float(torch.median(counts.to(torch.float32))),
                          "types_le100": int((counts <= 100).sum()),
                          "vocab": len(counts)}),
              flush=True)
    wpos = None
    if args.wordfinal_weight > 1.0:
        if not args.wordfinal_mask:
            raise ValueError("--wordfinal_mask is required when --wordfinal_weight > 1")
        mask = np.load(args.wordfinal_mask, mmap_mode="r")
        if len(mask) != len(train_ids):
            raise ValueError("wordfinal mask length must match train_ids")
        wpos = (mask, args.wordfinal_weight)
        print(json.dumps({"kind": "wordfinal_weights",
                          "weight": args.wordfinal_weight,
                          "mask": str(args.wordfinal_mask),
                          "wordfinal_frac": float(np.mean(np.asarray(mask))) }),
              flush=True)
    if min(len(train_ids), len(val_ids)) <= cfg.ctx_window + 1:
        raise ValueError("corpus is too short for the configured context window")
    stop_at = min(args.steps, args.max_steps) if args.max_steps else args.steps
    if global_step >= stop_at:
        print(f"already complete at step {global_step}/{stop_at}", flush=True)
        return

    t0 = time.time()
    print(json.dumps({"kind": "probe_start", "train_tokens": len(train_ids),
                      "val_tokens": len(val_ids), "target_steps": stop_at,
                      "device": str(device), "config": run_config["readout_config"]}), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        while global_step < stop_at:
            is_rollout = should_rollout(global_step, args.rollout_every, args.rollout_start)
            this_batch = args.rollout_batch if is_rollout else args.batch
            starts = np_rng.integers(0, len(train_ids) - cfg.ctx_window - 1, size=this_batch)
            rows = np.stack([train_ids[s:s + cfg.ctx_window + 1] for s in starts])
            ids = torch.from_numpy(rows.astype(np.int64, copy=False)).to(device)
            lr_scale = min(1.0, (global_step + 1) / max(1, args.warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if is_rollout:
                    loss = rollout_loss(readout, emb, cfg, ids, np_rng,
                                        rollout_tokens=args.rollout_tokens,
                                        min_prefix=args.rollout_min_prefix,
                                        sampling=args.rollout_sampling,
                                        generator=rollout_generator,
                                        freq=freq, gstarts=starts, wpos=wpos)
                else:
                    loss = right_aligned_loss(readout, emb, cfg, ids, np_rng,
                                              short_prefix_prob=args.short_prefix_prob,
                                              short_prefix_max=args.short_prefix_max,
                                              blank_prefix_prob=args.blank_prefix_prob,
                                              blank_prefix_max=args.blank_prefix_max,
                                              freq=freq, gstarts=starts, wpos=wpos)
                    loss = loss + args.dense_weight * dense_loss(readout, emb, cfg, ids, freq, starts, wpos)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = float(torch.nn.utils.clip_grad_norm_(list(readout.parameters()) + list(emb.parameters()), 1.0).item())
            if not math.isfinite(gnorm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "non-finite gradient norm; refusing to advance or checkpoint this run. "
                    "Use FP32 or repair the unstable configuration before retrying."
                )
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if global_step % args.eval_every == 0 or global_step == stop_at:
                val = validate(readout, emb, cfg, val_ids, np_rng, device)
                row = {"step": global_step, "train_loss": round(float(loss.item()), 5),
                       "update": "rollout" if is_rollout else "teacher_forced",
                       "val_dense_ce": round(val, 5), "grad_norm": round(gnorm, 5),
                       "seconds": round(time.time() - t0, 2)}
                history.append(row)
                log.write(json.dumps(row) + "\n"); log.flush()
                print(json.dumps({"kind": "probe_eval", **row}), flush=True)
            if global_step % args.save_every == 0 or global_step == stop_at:
                save_checkpoint(ckpt_path, readout=readout, emb=emb, optimizer=optimizer,
                                scaler=scaler, np_rng=np_rng, rollout_generator=rollout_generator,
                                args=args, cfg=cfg,
                                step=global_step, history=history)
                atomic_torch_save(readout.state_dict(), save_dir / "readout.pt")
                atomic_torch_save(emb.state_dict(), save_dir / "emb.pt")
    print(json.dumps({"kind": "probe_done", "step": global_step, "save_dir": str(save_dir)}), flush=True)


if __name__ == "__main__":
    main()
