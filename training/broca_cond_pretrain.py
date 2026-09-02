import argparse, json, pathlib, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core.model as m


def cond_forward(ro, emb, src_ids, tgt_seq_ctx, causal_p, cfg, device):
    """Conditional (encoder-decoder) Broca pass.
    src_ids: (B, S) external source tokens -> KV for cross-attn.
    tgt_seq_ctx: (B, W) query + target window (last W tokens); predict next-token
                 over the window, cross-attending to the external source.
    """
    with torch.autocast("cuda", dtype=torch.float16,
                        enabled=str(device).startswith("cuda")):
        # External source embeddings (the "memory/context" content)
        src_e = emb(src_ids)                     # (B,S,dim)
        src_x = src_e.transpose(0, 1)            # (S,B,dim)
        # Query window embeddings
        qe = emb(tgt_seq_ctx)                    # (B,W,dim)
        x = qe.transpose(0, 1) + ro.ctx_pos[: qe.shape[1]].unsqueeze(1)  # (W,B,dim)
        causal = torch.triu(torch.ones(qe.shape[1], qe.shape[1], device=device,
                                       dtype=torch.bool), diagonal=1)
        for layer in ro.ctx_tf:
            x = layer(x, causal, src_x)          # cross-attn reads external source
        h = x.transpose(0, 1)                    # (B,W,dim)
        s_n = torch.zeros(tgt_seq_ctx.shape[0], qe.shape[1], cfg.dim, device=device)
        logits = (ro.ctx_gain * ro.ctx_head(h)
                  + ro.e_proj(qe)
                  + ro.gate_gain * ro.gate(torch.cat([s_n, qe], dim=-1)))
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/data/train_ids.npy")
    ap.add_argument("--val_ids", default="corpus/data/val_ids.npy")
    ap.add_argument("--tokens", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--readout_layers", type=int, default=4)
    ap.add_argument("--save_dir", default="runs/broca_cond_smoke")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = m.ZeusConfig()
    cfg.readout_layers = args.readout_layers
    cfg.readout_heads = 12
    cfg.readout_ffn_mult = 2
    cfg.cross_attn = True
    net = m.ZeusCore(cfg).to(args.device)
    ro, emb = net.readout, net.embed
    ro.train(); emb.train()
    opt = torch.optim.Adam(list(ro.parameters()) + list(emb.parameters()), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)

    ids = np.load(args.corpus).astype(np.int64)[:args.tokens]
    val = np.load(args.val_ids).astype(np.int64)[:8000]
    W = cfg.ctx_window
    # Half the window is external source, half is the query/target continuation
    S = W // 2
    H = W - S

    def windows(arr):
        n = (len(arr) - 1) // W
        blocks = arr[: n * W].reshape(-1, W)     # each block = 64 consecutive tokens
        src = blocks[:, :S]                       # external source: [0:32]
        ctx = blocks[:, S - 1: W - 1]             # query window:   [31:63]
        tgt = blocks[:, S: W]                     # targets:        [32:64]
        return src, ctx, tgt

    def make_batches(arr, shuffle):
        src, ctx, tgt = windows(arr)
        n = src.shape[0]
        perm = np.random.RandomState(0).permutation(n) if shuffle else np.arange(n)
        for st in range(0, n, args.batch):
            idx = perm[st:st + args.batch]
            if len(idx) == 0:
                break
            yield (torch.tensor(src[idx], device=args.device, dtype=torch.long),
                   torch.tensor(ctx[idx], device=args.device, dtype=torch.long),
                   torch.tensor(tgt[idx], device=args.device, dtype=torch.long))

    t0 = time.time()
    for ep in range(args.epochs):
        run = 0.0; nb = 0
        for src_b, ctx_b, tgt_b in make_batches(ids, shuffle=True):
            logits = cond_forward(ro, emb, src_b, ctx_b, None, cfg, args.device)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_b.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(ro.parameters()) + list(emb.parameters()), 1.0)
            opt.step()
            run += loss.item(); nb += 1
        # val
        v = 0.0; nv = 0
        ro.eval(); emb.eval()
        with torch.no_grad():
            for src_b, ctx_b, tgt_b in make_batches(val, shuffle=False):
                logits = cond_forward(ro, emb, src_b, ctx_b, None, cfg, args.device)
                v += F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_b.reshape(-1)).item(); nv += 1
        ro.train(); emb.train()
        print(f"epoch {ep+1}/{args.epochs}  train_ce={run/nb:.3f}  val_ce={v/nv:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    torch.save(ro.state_dict(), f"{args.save_dir}/readout.pt")
    torch.save(emb.state_dict(), f"{args.save_dir}/emb.pt")
    json.dump(vars(cfg), open(f"{args.save_dir}/run_config.json", "w"))
    print("saved cond-pretrain ->", args.save_dir)


if __name__ == "__main__":
    import os
    main()
