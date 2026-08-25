import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusCore, ZeusConfig
from tokenizers import Tokenizer

TOK = ROOT / "corpus" / "data" / "tokenizer" / "bpe_8192.json"
IDS_CACHE = ROOT / "corpus" / "data" / "train_ids.npy"
VAL_IDS_CACHE = ROOT / "corpus" / "data" / "val_ids.npy"


def get_ids(cache, text_path):
    if cache.exists():
        return np.load(cache)
    tok = Tokenizer.from_file(str(TOK))
    text = pathlib.Path(text_path).read_text(encoding="utf-8")
    chunks = [text[i:i + 2_000_000] for i in range(0, len(text), 2_000_000)]
    ids = []
    for enc in tok.encode_batch(chunks):
        ids.extend(enc.ids)
    arr = np.asarray(ids, dtype=np.int32)
    np.save(cache, arr)
    return arr


def dyn_lm_params(model):
    lm_keys = ("embed", "readout")
    dyn, lm = [], []
    for n, p in model.named_parameters():
        (lm if n.startswith(lm_keys) else dyn).append(p)
    return dyn, lm


class TeacherController:
    def __init__(self, p=1.0, lo=0.2, hi=1.0, rise=0.02, fall=0.005, margin=0.03):
        self.p, self.lo, self.hi = p, lo, hi
        self.rise, self.fall, self.margin = rise, fall, margin
        self.conf_ema = None
        self.conf_best = None
        self.block = []

    def observe(self, ce):
        self.block.append(ce)
        if len(self.block) >= 500:
            mean = sum(self.block) / len(self.block)
            self.conf_ema = mean if self.conf_ema is None else 0.9 * self.conf_ema + 0.1 * mean
            self.conf_best = mean if self.conf_best is None else min(self.conf_best, mean)
            degraded = self.conf_ema > self.conf_best + self.margin
            if degraded:
                self.p = min(self.hi, self.p + self.rise)
            else:
                self.p = max(self.lo, self.p - self.fall)
            self.block = []
            return {"teacher_p": round(self.p, 4), "conf_ema": round(self.conf_ema, 4),
                    "conf_best": round(self.conf_best, 4), "event": True}
        return None

    def state(self):
        return {"p": self.p, "ema": self.conf_ema, "best": self.conf_best}

    def load(self, st):
        if st:
            self.p, self.conf_ema, self.conf_best = st["p"], st["ema"], st["best"]


def driven_pass(model, ids_seg, teacher_p, w_persist, w_surp):
    model.reset_state(noise=0.05)
    ce_sum, surp_sum, rent_sum, div_sum = 0.0, 0.0, 0.0, 0.0
    nxt_input = int(ids_seg[0])
    T = len(ids_seg) - 1
    loss_total = 0.0
    for t in range(T):
        pred_before = model.self_pred(model.S)
        logits, aux = model.step(nxt_input)
        target = torch.tensor(ids_seg[t + 1], device=model.S.device)
        ce = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        surp = (model.S - pred_before.detach()).norm()
        persist = -aux["tau_mean_t"]
        loss_t = ce + w_surp * (-surp) + w_persist * persist + 0.1 * aux["rent"] + aux["div"]
        loss_total = loss_total + loss_t / T
        ce_sum += ce.item()
        surp_sum += surp.item()
        rent_sum += float(aux["rent"].item())
        div_sum += float(aux["div"].item())
        if random.random() < teacher_p:
            nxt_input = int(ids_seg[t + 1])
        else:
            nxt_input = int(logits.argmax().item())
    loss_total.backward()
    return {"ce": ce_sum / T, "surp": surp_sum / T, "rent": rent_sum / T,
            "div": div_sum / T, "persist": float(persist.item())}


def self_pass(model, steps=24, w_var=0.5):
    model.reset_state(noise=0.2)
    total = 0.0
    traj = []
    for _ in range(steps):
        pred = model.self_pred(model.S)
        model.step(None)
        total = total + F.mse_loss(pred, model.S.detach()) / steps
        traj.append(model.S.detach().clone())
    traj = torch.stack(traj)
    T = traj.shape[0]
    var_pen = torch.relu(traj[T // 2:].var(dim=0).mean() - 0.3)
    (total + w_var * var_pen).backward()
    return {"self_mse": float(total.item()),
            "var_floor": float(torch.exp(-10.0 * var_pen).item()),
            "rms": round(float((traj - traj.mean(0)).norm(dim=1).mean()), 4),
            "max_norm": round(float(traj.norm(dim=1).max().item()), 3)}


@torch.no_grad()
def val_ce(model, val_ids, chunks=6, seg=48):
    model.eval()
    total, n = 0.0, 0
    for c in range(chunks):
        seg_ids = val_ids[c * 512:(c * 512) + seg + 1]
        if len(seg_ids) < seg + 1:
            break
        model.reset_state(noise=0.0)
        for t in range(len(seg_ids) - 1):
            logits, _ = model.step(int(seg_ids[t]))
            total += F.cross_entropy(logits.unsqueeze(0),
                                     torch.tensor(int(seg_ids[t + 1]), device=model.S.device).unsqueeze(0)).item()
            n += 1
    model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bptt", type=int, default=33)
    ap.add_argument("--self_ratio", type=float, default=0.4)
    ap.add_argument("--w_persist", type=float, default=0.1)
    ap.add_argument("--w_surp", type=float, default=0.1)
    ap.add_argument("--lr_dyn", type=float, default=3e-4)
    ap.add_argument("--lr_lm", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    save_dir = ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train.log"

    def log(obj):
        obj["t"] = time.strftime("%H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    device = args.device
    torch.manual_seed(1337)
    random.seed(1337)

    train_ids = get_ids(IDS_CACHE, ROOT / "corpus" / "data" / "train.txt")
    val_ids = get_ids(VAL_IDS_CACHE, ROOT / "corpus" / "data" / "val.txt")

    model = ZeusCore().to(device)
    model.train()
    dyn, lm = dyn_lm_params(model)
    opt = torch.optim.AdamW([{"params": dyn, "lr": args.lr_dyn}, {"params": lm, "lr": args.lr_lm}])
    ctrl = TeacherController()

    start = 0
    ckpts = sorted(save_dir.glob("zeus_step*.pt"))
    if args.resume == "auto" and ckpts:
        latest = ckpts[-1]
        payload = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["opt"])
        ctrl.load(payload.get("controller"))
        start = payload["step"]
        log({"event": "resume", "from_step": start, "ckpt": latest.name})
    elif not ckpts:
        log({"event": "fresh_start"})

    t0 = time.time()
    for step in range(start + 1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        if random.random() < args.self_ratio:
            m = self_pass(model)
            entry = {"step": step, **{k: round(v, 5) for k, v in m.items()}}
        else:
            off = random.randint(0, len(train_ids) - args.bptt - 1)
            seg = train_ids[off:off + args.bptt].tolist()
            m = driven_pass(model, seg, ctrl.p, args.w_persist, args.w_surp)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            event = ctrl.observe(m["ce"])
            entry = {"step": step, **{k: round(v, 5) for k, v in m.items()}, **(event or {})}
        opt.step()
        if step % 25 == 0:
            sps = step / (time.time() - t0)
            entry["steps_per_s"] = round(sps, 3)
            log(entry)
        if step % args.eval_every == 0:
            v = val_ce(model, val_ids)
            log({"step": step, "val_ce_nats": round(v, 4), "floor_L1": 7.10, "floor_L2": 4.47})
        if step % args.ckpt_every == 0 or step == args.steps:
            model.save(save_dir, step, extra={"controller": ctrl.state(), "opt": opt.state_dict()})
            log({"event": "ckpt", "step": step})
    log({"event": "COMPLETE", "step": args.steps})


if __name__ == "__main__":
    main()
