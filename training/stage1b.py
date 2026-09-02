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

import core.model as m

READOUT_KNOBS = ("readout_layers", "readout_ffn_mult", "readout_heads", "ctx_anchor", "cross_attn")


def readout_logits(ro, emb, ids, cfg, device, fp16=True, last_only=False):
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and str(device).startswith("cuda")):
        e_in = emb(ids)
        W = e_in.shape[1]
        x = e_in.transpose(0, 1) + ro.ctx_pos[:W].unsqueeze(1)
        causal = torch.triu(torch.ones(W, W, device=ids.device, dtype=torch.bool), diagonal=1)
        for layer in ro.ctx_tf:
            x = layer(x, causal, None)
        if last_only:
            h = x.transpose(0, 1)[:, -1:]
            e = e_in[:, -1:]
        else:
            h = x.transpose(0, 1)
            e = e_in
        s_n = torch.zeros(h.shape[0], h.shape[1], cfg.dim, device=ids.device)
        logits = (ro.ctx_gain * ro.ctx_head(h) + ro.e_proj(e)
                  + ro.gate_gain * ro.gate(torch.cat([s_n, e], dim=-1)))
        return logits


def readout_emb_logits(ro, e_in, cfg, device, fp16=True):
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and str(device).startswith("cuda")):
        N, W, d = e_in.shape
        x = e_in.transpose(0, 1).contiguous() + ro.ctx_pos[:W].unsqueeze(1)
        causal = torch.triu(torch.ones(W, W, device=e_in.device, dtype=torch.bool), diagonal=1)
        for layer in ro.ctx_tf:
            x = layer(x, causal, None)
        h = x.transpose(0, 1)[:, -1:]
        e = e_in[:, -1:]
        s_n = torch.zeros(N, 1, d, device=e_in.device)
        logits = (ro.ctx_gain * ro.ctx_head(h) + ro.e_proj(e)
                  + ro.gate_gain * ro.gate(torch.cat([s_n, e], dim=-1)))
        return logits[:, 0]


def rollout_loss(ro, emb, ids_all, pos_all, R, G, cfg, device, fp16=True):
    B = ids_all.shape[0]
    W = cfg.ctx_window
    d = cfg.dim
    prefix = ids_all[:, :R]
    cont = ids_all[:, R: R + G]
    p_emb = emb(prefix)
    c_emb = emb(cont)
    layouts = torch.zeros((B, G, W, d), device=device)
    for g in range(G):
        layouts[:, g, W - R - g: W - g] = p_emb
        layouts[:, g, W - g:] = c_emb[:, :g]
    logits = readout_emb_logits(ro, layouts.reshape(B * G, W, d), cfg, device, fp16)
    return F.cross_entropy(logits, cont.reshape(-1))


def dense_loss(ro, emb, inp, tgt_tok, cfg, device, fp16=True):
    logits = readout_logits(ro, emb, inp, cfg, device, fp16)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_tok.reshape(-1))


@torch.no_grad()
def sample(ro, emb, tok, prompt, steps, temp=0.72, top_p=0.92, rp=1.15, device="cpu", fp16=True):
    cfg = ro.cfg
    W = cfg.ctx_window
    pt = tok.encode(prompt).ids
    if not pt:
        pt = [tok.token_to_id(".")]
    R = len(pt)
    prefix = torch.tensor(pt, dtype=torch.long, device=device).unsqueeze(0)
    p_emb = emb(prefix)
    d = cfg.dim
    out = []
    reply = []
    for g in range(steps):
        e_in = torch.zeros((1, W, d), device=device)
        e_in[0, W - R - min(g, W - R): W - min(g, W - R)] = p_emb[0]
        if g > 0:
            e_in[0, W - g:] = torch.stack(reply, dim=0)[:min(g, W)]
        logits = readout_emb_logits(ro, e_in, cfg, device, fp16)[0].float()
        probs = F.softmax(logits / temp, dim=-1)
        if rp > 1.0 and out:
            uniq = torch.tensor(sorted(set(out)), device=probs.device)
            probs[uniq] = probs[uniq] / rp
        probs = probs / probs.sum()
        if top_p < 1.0:
            sp, idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, 0)
            keep = cum <= top_p
            keep[0] = True
            sp = torch.where(keep, sp, torch.zeros_like(sp))
            probs = torch.zeros_like(probs).scatter_(0, idx, sp)
            probs = probs / probs.sum()
        nxt = torch.multinomial(probs, 1).item()
        out.append(nxt)
        reply.append(emb(torch.tensor(nxt, device=device)))
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_dir", default="runs/broca_pretrain")
    ap.add_argument("--save_dir", default="runs/broca_stage1b")
    ap.add_argument("--corpus", default="corpus/data/train_ids.npy")
    ap.add_argument("--val_ids", default="corpus/data/val_ids.npy")
    ap.add_argument("--tokens", type=int, default=160000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--rollout_batch", type=int, default=6)
    ap.add_argument("--rollout_every", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    knobs = json.loads(open(f"{args.init_dir}/run_config.json", encoding="utf-8").read())
    cfg = m.ZeusConfig()
    for k in READOUT_KNOBS:
        setattr(cfg, k, knobs.get(k, getattr(cfg, k)))
    net = m.ZeusCore(cfg).to(device)
    emb, ro = net.embed, net.readout
    ro.load_state_dict(torch.load(f"{args.init_dir}/readout.pt", map_location=device, weights_only=True), strict=False)
    emb.load_state_dict(torch.load(f"{args.init_dir}/emb.pt", map_location=device, weights_only=True), strict=False)
    emb.train(); ro.train()
    opt = torch.optim.Adam(list(ro.parameters()) + list(emb.parameters()), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    prog_f = open(os.path.join(args.save_dir, "progress.log"), "a", buffering=1)
    ids_arr = np.load(args.corpus).astype(np.int64)
    if args.tokens and args.tokens > 0:
        ids_arr = ids_arr[: args.tokens]
    val_arr = np.load(args.val_ids).astype(np.int64)[: 40000]
    W = cfg.ctx_window
    n_win = max(1, (len(ids_arr) - 1) // W)
    seq = torch.from_numpy(ids_arr[: n_win * W + 1]).to(device)
    inp_data = seq[:-1].reshape(n_win, W)
    tgt_data = seq[1:].reshape(n_win, W)
    del seq
    N = len(ids_arr)
    cpu_ids = ids_arr

    start_ep, gs = 0, 0
    state_path = os.path.join(args.save_dir, "state.json")
    if os.path.exists(state_path):
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
                start_ep = ep_save
        except Exception as exc:
            print(f"resume failed ({exc!r}), starting fresh", flush=True)
            start_ep, gs = 0, 0

    run_cfg = {k: getattr(cfg, k) for k in READOUT_KNOBS}
    run_cfg["vocab"] = cfg.vocab
    run_cfg["dim"] = cfg.dim
    with open(os.path.join(args.save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    def dense_batch(idx):
        inp = inp_data[idx]
        tgt = tgt_data[idx]
        return inp, tgt

    def rollout_batch(bs):
        R = random.randint(4, 40)
        P = random.randint(0, min(20, W - R - 8))
        G = W - R - P
        base = np.random.randint(0, max(1, N - W - 1), size=bs)
        rows = np.stack([cpu_ids[b: b + R + G + 1] for b in base])
        return torch.from_numpy(rows).to(device), R, G

    def val_dense():
        total, cnt = 0.0, 0
        with torch.no_grad():
            vseq = torch.from_numpy(val_arr[: (len(val_arr) - 1) // W * W + 1]).to(device)
            v_inp = vseq[:-1].reshape(-1, W)
            v_tgt = vseq[1:].reshape(-1, W)
            for st in range(0, v_inp.shape[0], 256):
                i, t = v_inp[st: st + 256], v_tgt[st: st + 256]
                total += dense_loss(ro, emb, i, t, cfg, device, not args.fp32).item() * len(i)
                cnt += len(i)
            v_inp = None; v_tgt = None; vseq = None
        return total / max(1, cnt)

    def val_rollout(bs=24):
        ro.eval(); emb.eval()
        rr = 16
        gg = 48
        base = np.random.RandomState(7).randint(0, max(1, len(val_arr) - W - 1), size=bs)
        rows = np.stack([val_arr[b: b + rr + gg + 1] for b in base])
        ids_all = torch.from_numpy(rows).to(device)
        prefix = ids_all[:, :rr]
        cont = ids_all[:, rr: rr + gg]
        p_emb = emb(prefix).detach()
        d = cfg.dim
        reply = []
        with torch.no_grad():
            for g in range(gg):
                e_in = torch.zeros((bs, W, d), device=device)
                e_in[:, W - rr - min(g, W - rr): W - min(g, W - rr)] = p_emb
                if g > 0:
                    e_in[:, W - g:] = torch.stack(reply, dim=1)[:, :min(g, W)]
                logits = readout_emb_logits(ro, e_in, cfg, device, not args.fp32).float()
                ce = F.cross_entropy(logits, cont[:, g]).item()
                nxt = logits.argmax(-1)
                reply.append(emb(nxt).detach())
                if g == 0:
                    T0 = ce
        ro.train(); emb.train()
        return T0, ce

    def save_epoch(ep):
        torch.save(ro.state_dict(), os.path.join(args.save_dir, f"readout_ep{ep}.pt"))
        torch.save(emb.state_dict(), os.path.join(args.save_dir, f"emb_ep{ep}.pt"))
        torch.save(ro.state_dict(), os.path.join(args.save_dir, "readout.pt"))
        torch.save(emb.state_dict(), os.path.join(args.save_dir, "emb.pt"))
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"epoch": ep, "global_step": gs}, f)

    def lr_at(step):
        total = args.epochs * (n_win // args.batch)
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        frac = min(1.0, (step - args.warmup_steps) / max(1, total - args.warmup_steps))
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * frac))

    t0 = time.time()
    print(f"stage1b init={args.init_dir} W={W} n_win={n_win} batch={args.batch} "
          f"rollout={args.rollout_batch} every={args.rollout_every} lr={args.lr} "
          f"epochs={args.epochs} fp32={args.fp32}", flush=True)
    prog_f.write(f"# stage1b start knobs={json.dumps(run_cfg)}\n")

    val_d = val_dense() if start_ep == 0 else None
    val_r0, val_rf = val_rollout() if start_ep == 0 else (None, None)
    if val_d is not None:
        print(f"baseline  val_dense_ce={val_d:.3f}  val_rollout_ce@0={val_r0:.3f} "
              f"rollout_ce@G={val_rf:.3f}", flush=True)

    for ep in range(start_ep, args.epochs):
        perm = np.random.RandomState(1337 + ep).permutation(n_win)
        run_loss, run_roll, nb, done = 0.0, 0.0, 0, 0
        step_in_epoch = 0
        for st in range(0, n_win, args.batch):
            idxs = perm[st: st + args.batch]
            if len(idxs) == 0:
                break
            for pg in opt.param_groups:
                pg["lr"] = lr_at(gs)
            if step_in_epoch % args.rollout_every == args.rollout_every - 1:
                ids_all, R, G = rollout_batch(args.rollout_batch)
                loss = rollout_loss(ro, emb, ids_all, None, R, G, cfg, device, not args.fp32)
                run_roll += loss.item()
            else:
                inp, tgt = dense_batch(idxs)
                loss = dense_loss(ro, emb, inp, tgt, cfg, device, not args.fp32)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                list(ro.parameters()) + list(emb.parameters()), 1.0)
            opt.step()
            run_loss += loss.item()
            nb += 1
            done += len(idxs)
            gs += 1
            step_in_epoch += 1
            if done % (args.batch * 10) == 0:
                torch.cuda.empty_cache()
            if done % (args.batch * 20) == 0:
                msg = (f"epoch {ep+1}/{args.epochs}  step {done}/{n_win}  "
                       f"avg_ce={run_loss/nb:.3f}  gn={gnorm:.2f}  ({time.time()-t0:.0f}s)")
                print(msg, flush=True)
                prog_f.write(msg + "\n")
        avg = run_loss / max(1, nb)
        vd = val_dense()
        vr0, vrf = val_rollout()
        msg = (f"epoch {ep+1}/{args.epochs}  avg_ce={avg:.3f}  val_dense_ce={vd:.3f}  "
               f"rollout_ce@0={vr0:.3f}  rollout_ce@G={vrf:.3f}  ({time.time()-t0:.0f}s)")
        print(msg, flush=True)
        prog_f.write(msg + "\n")
        save_epoch(ep + 1)
        emb.eval(); ro.eval()
        with torch.no_grad():
            for p in ["hello", "once upon a time the old librarian climbed the stairs",
                      "tell me about the past"]:
                print("  SAMPLE:", p, "->", sample(ro, emb, net.tokenizer, p, 40,
                                                    device=device, fp16=not args.fp32)[:120], flush=True)
        emb.train(); ro.train()
    save_epoch(args.epochs)
    prog_f.close()
    print(f"saved stage1b -> {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()