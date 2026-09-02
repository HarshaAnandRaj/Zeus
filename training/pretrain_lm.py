"""
Standalone next-token LM pretraining for the Zeus readout.

Trains ONLY the token path of CoupledReadout (e_proj + Transformer over the
recent-token window + ctx_head + gate) on the raw corpus, with the autonomous
state S held at zero. This produces a reliable "language voice" (bigram floor
~4.47 nats and below) that the coupled model can then reuse, so LM competence
no longer has to fight the S-dynamics gradient.

Optional anchor scheme (--ctx_anchor): a soft-prefix token (a projected
external vector, e.g. the HCM recall target) is prepended to each causal
window so memory has a direct channel into the LM voice. During pretraining
anchors are random vocabulary embeddings so the net learns the column while
keeping its own LM competence. This mirrors the deployment call in
CoupledReadout.forward exactly (inference reads logits from the last slot).

After pretraining, load the saved weights into ZeusCore via train.py
--lm_pretrain (train.py reads readout_knobs from run_config.json).
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core.model as m

READOUT_KNOBS = ("readout_layers", "readout_ffn_mult", "readout_heads", "ctx_anchor", "cross_attn")


def forward_loss(ro, emb, inp, tgt_tok, anchor, causal, cfg, device, fp32=False):
    """Inp (B, W) real tokens, tgt_tok (B, W) next-token ids.

    Anchor scheme: sequence = [ctx_anchor(embed of a random IN-WINDOW token),
    r_0 .. r_{W-2}], targets = inp (slot k predicts r_k). Anchors are real
    corpus token embeddings so the Transformer learns to read the soft-prefix
    column (same distribution as deployment recall targets) without leaking
    future tokens. Plain scheme keeps the legacy inp -> tgt_tok alignment.
    """
    e_seq = None
    with torch.autocast("cuda", dtype=torch.float16,
                        enabled=str(device).startswith("cuda") and not fp32):
        e_seq = emb(inp)
        if anchor:
            pos = torch.randint(0, e_seq.shape[1], (e_seq.shape[0],), device=device)
            a = ro.ctx_anchor(e_seq[torch.arange(e_seq.shape[0], device=device), pos])
            e_in = torch.cat([a.unsqueeze(1), e_seq[:, :-1]], dim=1)
            target = inp
        else:
            e_in = e_seq
            target = tgt_tok
        x = e_in.transpose(0, 1) + ro.ctx_pos[: e_in.shape[1]].unsqueeze(1)
        if getattr(cfg, "cross_attn", False):
            # Broca stage-1 pretrain: no brain yet (S=0 -> H empty), so the
            # cross-attn sublayer serializes its own token source (kv=None falls
            # back to self), learning the "render a source into language" skill.
            for layer in ro.ctx_tf:
                x = layer(x, causal, None)
            h = x.transpose(0, 1)
        else:
            h = ro.ctx_tf(x, mask=causal).transpose(0, 1)
        s_n = torch.zeros(inp.shape[0], e_in.shape[1], cfg.dim, device=device)
        logits = (
            ro.ctx_gain * ro.ctx_head(h)
            + ro.e_proj(e_in)
            + ro.gate_gain * ro.gate(torch.cat([s_n, e_in], dim=-1))
        )
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/data/train_ids.npy")
    ap.add_argument("--val_ids", default="corpus/data/val_ids.npy")
    ap.add_argument("--tokens", type=int, default=600000,
                    help="number of corpus tokens to pretend on (0 = whole corpus)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=256)  # unused (W fixed)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save_dir", default="runs/lm_pretrain")
    ap.add_argument("--norm_bound", type=float, default=40.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--readout_layers", type=int, default=2)
    ap.add_argument("--readout_ffn_mult", type=int, default=2)
    ap.add_argument("--readout_heads", type=int, default=12)
    ap.add_argument("--ctx_anchor", action="store_true",
                    help="train the anchor-prefix scheme (mirrors deployment with recall anchors)")
    ap.add_argument("--cross_attn", action="store_true",
                    help="build the Broca-layer readout: token voice cross-attends to a source. "
                         "Stage-1 pretrain trains the cross-attn sublayer on its own token source "
                         "(the 'render a source into language' skill) before coupling real brain H.")
    ap.add_argument("--fresh", action="store_true", help="ignore state.json resume")
    ap.add_argument("--warmup_steps", type=int, default=600)
    ap.add_argument("--clip", type=float, default=1.0, help="max grad norm")
    ap.add_argument("--fp32", action="store_true", help="disable fp16 autocast")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    prog_f = open(os.path.join(args.save_dir, "progress.log"), "a", buffering=1)
    device = torch.device(args.device)

    cfg = m.ZeusConfig()
    cfg.norm_bound = args.norm_bound
    cfg.readout_layers = args.readout_layers
    cfg.readout_ffn_mult = args.readout_ffn_mult
    cfg.readout_heads = args.readout_heads
    cfg.ctx_anchor = args.ctx_anchor
    cfg.cross_attn = args.cross_attn
    net = m.ZeusCore(cfg).to(device)
    emb = net.embed
    ro = net.readout
    V = cfg.vocab
    emb.train()
    ro.train()

    opt = torch.optim.Adam(
        list(ro.parameters()) + list(emb.parameters()), lr=args.lr
    )

    ids = np.load(args.corpus).astype(np.int64)
    if args.tokens and args.tokens > 0:
        ids = ids[: args.tokens]
    val_ids = np.load(args.val_ids).astype(np.int64) if os.path.exists(args.val_ids) else ids[:100000]
    W = cfg.ctx_window
    n_win = max(1, (len(ids) - 1) // W)
    n_val = max(1, (len(val_ids) - 1) // W)

    # resume
    start_ep, gs = 0, 0
    state_path = os.path.join(args.save_dir, "state.json")
    if not args.fresh and os.path.exists(state_path):
        try:
            st = json.loads(open(state_path, encoding="utf-8").read())
            start_ep = int(st.get("epoch", 0))
            gs = int(st.get("global_step", 0))
            ep_save = start_ep
            ro_p, em_p = os.path.join(args.save_dir, f"readout_ep{ep_save}.pt"), \
                os.path.join(args.save_dir, f"emb_ep{ep_save}.pt")
            if os.path.exists(ro_p) and os.path.exists(em_p):
                ro.load_state_dict(torch.load(ro_p, map_location=device, weights_only=True))
                emb.load_state_dict(torch.load(em_p, map_location=device, weights_only=True))
                print(f"resumed from epoch {ep_save} (global step {gs})", flush=True)
                start_ep = ep_save  # re-run that epoch from a fresh shuffle
            else:
                start_ep, gs = 0, 0
        except Exception as exc:
            _err = print
            print(f"resume failed ({exc!r}), starting fresh", flush=True)
            start_ep, gs = 0, 0

    run_cfg = {k: getattr(cfg, k) for k in READOUT_KNOBS}
    run_cfg["vocab"] = V
    run_cfg["dim"] = cfg.dim
    with open(os.path.join(args.save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    causal_a = torch.triu(torch.ones(W, W, dtype=torch.bool, device=device), diagonal=1)
    causal_p = torch.triu(torch.ones(W, W, dtype=torch.bool, device=device), diagonal=1)

    seq = torch.from_numpy(ids[: n_win * W + 1]).to(device)
    inp_data = seq[:-1].reshape(n_win, W)
    tgt_data = seq[1:].reshape(n_win, W)
    del seq
    vseq = torch.from_numpy(val_ids[: n_val * W + 1]).to(device)
    v_inp = vseq[:-1].reshape(n_val, W)
    v_tgt = vseq[1:].reshape(n_val, W)
    del vseq

    def val_ce(use_anchor):
        total, cnt = 0.0, 0
        with torch.no_grad():
            for st in range(0, n_val, 256):
                i = v_inp[st:st + 256]
                t = v_tgt[st:st + 256]
                if len(i) == 0:
                    break
                loss = forward_loss(ro, emb, i, t, use_anchor, causal_a if use_anchor else causal_p, cfg, device, args.fp32)
                total += loss.item() * len(i)
                cnt += len(i)
        return total / max(1, cnt)

    def save_epoch(ep):
        torch.save(ro.state_dict(), os.path.join(args.save_dir, f"readout_ep{ep}.pt"))
        torch.save(emb.state_dict(), os.path.join(args.save_dir, f"emb_ep{ep}.pt"))
        torch.save(ro.state_dict(), os.path.join(args.save_dir, "readout.pt"))
        torch.save(emb.state_dict(), os.path.join(args.save_dir, "emb.pt"))
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"epoch": ep, "global_step": gs}, f)

    t0 = time.time()
    print(f"pretrain: {len(ids)} tokens, W={W}, {n_win} windows/epoch, {args.epochs} epochs, "
          f"batch={args.batch}, lr={args.lr}, layers={cfg.readout_layers}, ffn_x{cfg.readout_ffn_mult}, "
          f"anchor={cfg.ctx_anchor}, fp32={args.fp32}, clip={args.clip}, device={device}", flush=True)
    prog_f.write(f"# pretrain start knobs={json.dumps(run_cfg)}\n")

    def lr_at(global_step):
        if global_step < args.warmup_steps:
            return args.lr * (global_step + 1) / max(1, args.warmup_steps)
        total = args.epochs * n_win // args.batch
        frac = min(1.0, (global_step - args.warmup_steps) / max(1, total - args.warmup_steps))
        return args.lr * 0.5 * (1.0 + math_cos(max(0.0, min(1.0, frac))))

    for ep in range(start_ep, args.epochs):
        perm = np.random.RandomState(1337 + ep).permutation(n_win)
        running_ce = 0.0
        n_batch = 0
        done = 0
        for st in range(0, n_win, args.batch):
            idxs = perm[st:st + args.batch]
            if len(idxs) == 0:
                break
            inp = inp_data[idxs]
            tgt = tgt_data[idxs]
            for pg in opt.param_groups:
                pg["lr"] = lr_at(gs)
            loss = forward_loss(ro, emb, inp, tgt, cfg.ctx_anchor,
                                causal_a if cfg.ctx_anchor else causal_p, cfg, device, args.fp32)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                list(ro.parameters()) + list(emb.parameters()), args.clip
            )
            opt.step()
            running_ce += loss.item()
            n_batch += 1
            done += len(idxs)
            gs += 1
            if n_batch % 200 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()
            if n_batch % args.log_every == 0:
                msg = (f"epoch {ep+1}/{args.epochs}  step {done}/{n_win}  "
                       f"avg_ce={running_ce/n_batch:.3f}  gn={gnorm:.2f}  "
                       f"({time.time()-t0:.0f}s)")
                print(msg, flush=True)
                prog_f.write(msg + "\n")
        avg_ce = running_ce / max(1, n_batch)
        va = val_ce(True) if cfg.ctx_anchor else float("nan")
        vp = val_ce(False)
        elapsed = time.time() - t0
        msg = (f"epoch {ep+1}/{args.epochs}  avg_ce={avg_ce:.3f}  "
               f"val_ce_anchor={va:.3f}  val_ce_plain={vp:.3f}  ({elapsed:.0f}s)")
        print(msg, flush=True)
        prog_f.write(msg + "\n")
        save_epoch(ep + 1)
    prog_f.close()
    print(f"saved readout+emb -> {args.save_dir}", flush=True)


def math_cos(x):
    import math
    return math.cos(math.pi * x)


if __name__ == "__main__":
    main()