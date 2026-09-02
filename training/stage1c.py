import argparse, json, math, os, pathlib, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusConfig, CoupledReadout

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=320000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--warmup_steps", type=int, default=80)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--save_dir", type=str, default="runs/broca_stage1c")
    ap.add_argument("--init_dir", type=str, default="runs/broca_pretrain")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_cfg = dict(readout_layers=6, readout_ffn_mult=2, readout_heads=12,
                   ctx_anchor=False, cross_attn=True, vocab=8192, dim=768, window=32)
    if os.path.exists(os.path.join(args.init_dir, "run_config.json")):
        with open(os.path.join(args.init_dir, "run_config.json"), encoding="utf-8") as f:
            merged = json.load(f)
        merged.update(run_cfg)
        run_cfg = merged
    cfg = ZeusConfig(**run_cfg)
    cfg.vocab = 8192

    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb = torch.nn.Embedding(cfg.vocab, cfg.dim).to(device)
    emb.load_state_dict(torch.load(os.path.join(args.init_dir, "emb.pt"),
                                   map_location=device, weights_only=True))

    ro = CoupledReadout(cfg).to(device)
    ro_state = torch.load(os.path.join(args.init_dir, "readout.pt"),
                          map_location=device, weights_only=True)
    ro_state.pop("ctx_tf", None)
    try:
        ro.load_state_dict(ro_state, strict=False)
    except Exception:
        pass
    ro.init_from = "fresh-tf-warm-emb"
    ro.cfg = cfg

    os.makedirs(args.save_dir, exist_ok=True)
    state_path = os.path.join(args.save_dir, "state.json")
    prog_path = os.path.join(args.save_dir, "progress.log")
    with open(os.path.join(args.save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f)

    corpus_dir = "corpus/data"
    val_path = os.path.join(corpus_dir, "val_ids.npy")
    train_path = os.path.join(corpus_dir, "train_ids.npy")

    W = cfg.ctx_window
    d = cfg.dim


    def load_arr(p):
        return np.load(p).astype(np.int64)


    train_arr = np.load(train_path)[: 4000000]
    val_arr = np.load(val_path)[: 40000]

    n_train = args.tokens
    n_win = max(1, (n_train - 1) // (W - 1))


    def dense_batch(idxs):
        pos_all = np.random.randint(0, max(1, len(train_arr) - W - 1), size=len(idxs))
        rows = np.stack([train_arr[p: p + W + 1] for p in pos_all])
        ids_all = torch.from_numpy(rows).to(device)
        return ids_all


    def readout_emb_logits(e_in):
        N, Wl, dl = e_in.shape
        x = e_in.transpose(0, 1).contiguous() + ro.ctx_pos[:Wl].unsqueeze(1)
        causal = torch.triu(torch.ones(Wl, Wl, device=e_in.device, dtype=torch.bool), diagonal=1)
        for layer in ro.ctx_tf:
            x = layer(x, causal, None)
        h = x.transpose(0, 1)[:, -1:]
        e = e_in[:, -1:]
        s_n = torch.zeros(N, 1, dl, device=e_in.device)
        logits = (ro.ctx_gain * ro.ctx_head(h) + ro.e_proj(e) + ro.gate_gain * ro.gate(torch.cat([s_n, e], dim=-1)))
        return logits[:, 0]


    def right_align_loss(ids_all):
        B = ids_all.shape[0]
        Rnd = np.random.randint(3, W, size=len(ids_all))
        targets = torch.empty(B, dtype=torch.long, device=device)
        layouts = torch.zeros((B, W, d), device=device)
        for i in range(B):
            R = int(Rnd[i])
            targets[i] = ids_all[i, R]
            layouts[i, W - R:] = emb(ids_all[i, :R])
        logits = readout_emb_logits(layouts)
        return F.cross_entropy(logits.float(), targets.long())


    def dense_loss(ids_all):
        B = ids_all.shape[0]
        inp = ids_all[:, :W]
        tgt = ids_all[:, 1: W + 1]
        layouts = emb(inp)
        x = layouts.transpose(0, 1).contiguous() + ro.ctx_pos[:W].unsqueeze(1)
        causal = torch.triu(torch.ones(W, W, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in ro.ctx_tf:
            x = layer(x, causal, None)
        h = x.transpose(0, 1)
        e_in = layouts
        s_n = torch.zeros((B, W, d), device=device)
        logits = (ro.ctx_gain * ro.ctx_head(h) + ro.e_proj(e_in)
                  + ro.gate_gain * ro.gate(torch.cat([s_n, e_in], dim=-1)))
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1).long())


    @torch.no_grad()
    def val_right(bs=24):
        ro.eval()
        rr = 16
        gg = 48
        base = np.random.RandomState(7).randint(0, max(1, len(val_arr) - W - 1), size=bs)
        rows = np.stack([val_arr[b: b + rr + gg + 1] for b in base])
        ids_all = torch.from_numpy(rows).to(device)
        prefix = ids_all[:, :rr]
        cont = ids_all[:, rr: rr + gg]
        p_emb = emb(prefix)
        reply = []
        for g in range(gg):
            e_in = torch.zeros((bs, W, d), device=device)
            e_in[:, W - rr - min(g, W - rr): W - min(g, W - rr)] = p_emb
            if g > 0:
                e_in[:, W - g:] = torch.stack(reply, dim=1)[:, :min(g, W)]
            logits = readout_emb_logits(e_in).float()
            ce = F.cross_entropy(logits.float(), cont[:, g].long()).item()
            nxt = logits.argmax(-1)
            reply.append(emb(nxt))
            if g == 0:
                ce0 = ce
        ro.train()
        return ce0, ce


    @torch.no_grad()
    def val_dense(bs=32):
        ro.eval()
        base = np.random.RandomState(11).randint(0, max(1, len(val_arr) - W - 1), size=bs)
        rows = np.stack([val_arr[b: b + W + 1] for b in base])
        ids_all = torch.from_numpy(rows).to(device)
        v = dense_loss(ids_all).item()
        ro.train()
        return v


        # ---- flops guard ----
    with torch.no_grad():
        _ = readout_emb_logits(torch.zeros(1, W, d, device=device))

    opt = torch.optim.AdamW(list(ro.parameters()) + list(emb.parameters()), lr=args.lr)
    start_ep, gs = 0, 0
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            st = json.load(f)
        start_ep, gs = st.get("epoch", 0), st.get("global_step", 0)
        if gs > 0:
            try:
                ro.load_state_dict(torch.load(os.path.join(args.save_dir, "readout_latest.pt"),
                                              map_location=device, weights_only=True))
                emb.load_state_dict(torch.load(os.path.join(args.save_dir, "emb_latest.pt"),
                                               map_location=device, weights_only=True))
            except Exception:
                pass
        print(f"resumed from epoch {start_ep} (global step {gs})")

    def save_epoch(ep):
        torch.save(ro.state_dict(), os.path.join(args.save_dir, f"readout_ep{ep}.pt"))
        torch.save(emb.state_dict(), os.path.join(args.save_dir, f"emb_ep{ep}.pt"))
        torch.save(ro.state_dict(), os.path.join(args.save_dir, "readout_latest.pt"))
        torch.save(emb.state_dict(), os.path.join(args.save_dir, "emb_latest.pt"))
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"epoch": ep, "global_step": gs}, f)

    def lr_at(step):
        total = args.epochs * (n_win // args.batch)
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        frac = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac)))

    t0 = time.time()
    print(f"stage1c fresh-readout deploy-shaped W={W} n_win={n_win} batch={args.batch} "
          f"lr={args.lr} epochs={args.epochs} fp32={args.fp32}", flush=True)
    with open(prog_path, "a", encoding="utf-8") as pf:
        pf.write("# stage1c start knobs=" + json.dumps(run_cfg) + "\n")

    ce0b, cefb = val_right()
    print(f"baseline  val_right ce0={ce0b:.3f} ceG={cefb:.3f}", flush=True)

    n_batches = n_win // args.batch + 1
    for ep in range(start_ep, args.epochs):
        perm = np.random.RandomState(1337 + ep).permutation(n_win)
        run_loss, nb, done = 0.0, 0, 0
        for st in range(0, n_win, args.batch):
            idxs = perm[st: st + args.batch]
            if len(idxs) == 0:
                break
            for pg in opt.param_groups:
                pg["lr"] = lr_at(gs)
            ids_all = dense_batch(idxs)
            loss = None
            for _ in range(args.samples):
                loss_ = right_align_loss(ids_all)
                loss_ = loss_ / args.samples
                if loss is None:
                    loss = loss_
                else:
                    loss = loss + loss_
            loss = loss + 0.15 * dense_loss(ids_all)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                list(ro.parameters()) + list(emb.parameters()), 1.0)
            opt.step()
            run_loss += loss.item()
            nb += 1
            done += len(idxs)
            gs += 1
            if done % (args.batch * 10) == 0:
                torch.cuda.empty_cache()
            if done % (args.batch * 20) == 0:
                msg = (f"epoch {ep+1}/{args.epochs}  step {done}/{n_win}  "
                       f"avg={run_loss/nb:.3f}  gn={gnorm:.2f}  ({time.time()-t0:.0f}s)")
                print(msg, flush=True)
                with open(prog_path, "a", encoding="utf-8") as pf:
                    pf.write(msg + "\n")
        ce0, cef = val_right()
        vd = val_dense()
        msg = (f"epoch {ep+1}/{args.epochs}  avg={run_loss/nb:.3f}  val_dense={vd:.3f}  "
               f"val_right ce0={ce0:.3f} ceG={cef:.3f}  ({time.time()-t0:.0f}s)")
        print(msg, flush=True)
        with open(prog_path, "a", encoding="utf-8") as pf:
            pf.write(msg + "\n")
        save_epoch(ep + 1)

    print(f"saved stage1c -> {args.save_dir}")


if __name__ == "__main__":
    main()
