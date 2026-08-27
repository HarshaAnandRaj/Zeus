import json
import math
import pathlib
import re
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TOKENIZER = pathlib.Path(__file__).resolve().parents[1] / "corpus" / "data" / "tokenizer" / "bpe_8192.json"


@dataclass
class ZeusConfig:
    dim: int = 768
    vocab: int = 8192
    experts: int = 8
    expert_hidden: int = 1536
    tau_min: float = 0.5
    tau_max: float = 8.0
    dt: float = 1.0
    substeps: int = 4
    rho_max: float = 0.85
    window: int = 32
    attn_heads: int = 4
    slow_dim: int = 64
    slow_keep: float = 0.9
    w_rent: float = 0.1
    rent_target: float = 1.0
    w_diverse: float = 0.02
    k_repulse: float = 1.0
    repulse_sigma: float = 0.5
    repulse_skip: int = 2
    repulse_adaptive: bool = True
    repulse_q: float = 0.3
    repulse_smin: float = 0.25
    repulse_smax: float = 16.0
    k_wall: float = 2.0
    wall_margin: float = 6.0
    n_actions: int = 2

    @property
    def remember_id(self):
        return self.vocab - 1

    @property
    def note_id(self):
        return self.vocab - 2


class SpectralClampedLinear(nn.Module):
    def __init__(self, dim, rho_max):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.rho_max = rho_max
        self.register_buffer("u", torch.randn(dim), persistent=False)
        self.register_buffer("v", torch.randn(dim), persistent=False)

    def effective_weight(self):
        w = self.weight
        with torch.no_grad():
            for _ in range(5):
                nv = F.normalize(w.t() @ self.u, dim=0)
                nu = F.normalize(w @ nv, dim=0)
                self.u.copy_(nu)
                self.v.copy_(nv)
            sigma = (self.u * (w @ self.v)).sum()
        scale = torch.clamp(self.rho_max / (sigma + 1e-8), max=1.0)
        return w * scale

    def forward(self, x):
        w = self.effective_weight()
        if x.dim() == 1:
            return F.linear(x.unsqueeze(0), w, self.bias).squeeze(0)
        return F.linear(x, w, self.bias)


class PathwayLayer(nn.Module):
    def __init__(self, cfg: ZeusConfig):
        super().__init__()
        self.cfg = cfg
        E, d, hd = cfg.experts, cfg.dim, cfg.expert_hidden
        self.w1 = nn.Parameter(torch.randn(E, d, hd) / math.sqrt(d))
        self.b1 = nn.Parameter(torch.zeros(E, hd))
        self.w2 = nn.Parameter(torch.zeros(E, hd, d))
        self.b2 = nn.Parameter(torch.zeros(E, d))
        self.router = nn.Linear(2 * cfg.dim + 4, cfg.experts)
        self.w_pm = nn.Linear(cfg.dim, cfg.dim)
        nn.init.normal_(self.w_pm.weight, std=0.02)

    def forward(self, s, h, tau_stats):
        r_in = torch.cat([s, h, tau_stats], dim=-1)
        logits = self.router(r_in)
        g = F.softmax(logits, dim=-1)
        u = F.gelu(torch.einsum('eih,i->eh', self.w1, h) + self.b1)
        outs = torch.einsum('ehd,eh->ed', self.w2, u) + self.b2
        m = (outs * g.unsqueeze(-1)).sum(0)
        entropy = -(g * torch.log(g + 1e-9)).sum()
        rent_loss = F.relu(self.cfg.rent_target - entropy).mean()
        div_loss = self._diversity(outs)
        return m, g, rent_loss, div_loss

    def _diversity(self, outs):
        if self.cfg.w_diverse <= 0:
            return torch.zeros(())
        flat = F.normalize(outs, dim=-1)
        sim = flat @ flat.t()
        n = sim.shape[0]
        off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
        return off.mean()


class AttentiveReadout(nn.Module):
    def __init__(self, cfg: ZeusConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.dim, cfg.dim)
        self.attn = nn.MultiheadAttention(cfg.dim, cfg.attn_heads, batch_first=True)
        self.ln = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim, 2 * cfg.dim), nn.GELU(), nn.Linear(2 * cfg.dim, cfg.vocab))

    def forward(self, s, history):
        if isinstance(history, torch.Tensor):
            h = history.unsqueeze(0) if history.dim() == 2 else history
        else:
            h = torch.stack(history, dim=0).unsqueeze(0)  # (1, W, dim)
        if s.dim() == 1:
            q = self.q_proj(s).view(1, 1, -1)
            att, _ = self.attn(q, h, h)
            x = self.ln(s + att.view(-1))
            return self.mlp(x)
        s2 = s.unsqueeze(1)                               # (B, 1, dim)
        q = self.q_proj(s).unsqueeze(1)                  # (B, 1, dim)
        att, _ = self.attn(q, h, h)
        x = self.ln(s2 + att)
        return self.mlp(x).squeeze(1)                    # (B, vocab)


class ZeusCore(nn.Module):
    def __init__(self, cfg: ZeusConfig = None, tokenizer_path=None):
        super().__init__()
        self.cfg = cfg or ZeusConfig()
        tok_path = pathlib.Path(tokenizer_path or DEFAULT_TOKENIZER)
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.cfg.vocab = self.tokenizer.get_vocab_size()
        c = self.cfg
        self.embed = nn.Embedding(c.vocab, c.dim)
        self.self_pred = nn.Sequential(nn.Linear(c.dim, 1024), nn.Tanh(), nn.Linear(1024, c.dim))
        self.err_gate = nn.Linear(c.dim, 1)
        self.tau_net = nn.Sequential(nn.Linear(c.dim, 128), nn.Tanh(), nn.Linear(128, c.dim))
        self.rec = SpectralClampedLinear(c.dim, c.rho_max)
        self.err_proj = nn.Linear(c.dim, c.dim, bias=False)
        nn.init.normal_(self.err_proj.weight, std=0.02)
        self.pathways = PathwayLayer(c)
        self.readout = AttentiveReadout(c)
        self.w_slow = nn.Linear(c.slow_dim, c.dim)
        self.register_buffer("S", torch.zeros(c.dim), persistent=False)
        self.register_buffer("slow", torch.zeros(c.slow_dim), persistent=False)
        self.register_buffer("H", torch.zeros(c.window, c.dim), persistent=True)
        self._hptr = 0
        self.hcm = None
        self.hcm_pending = None

    # ---- state management ----
    def reset_state(self, noise=0.0, generator=None):
        c = self.cfg
        if noise > 0:
            g = generator or torch.default_generator
            self.S = (torch.randn(c.dim, generator=g) * noise).to(self.S.device)
            self.slow = (torch.randn(c.slow_dim, generator=g) * noise * 0.1).to(self.slow.device)
        else:
            self.S = torch.zeros(c.dim, device=self.S.device)
            self.slow = torch.zeros(c.slow_dim, device=self.slow.device)
        self.H.copy_(self.S.unsqueeze(0).expand(c.window, -1))
        self._hptr = 0

    def export_state(self):
        return {"S": self.S.detach().clone(), "slow": self.slow.detach().clone(),
                "H": self.H.detach().clone()}

    def import_state(self, st):
        self.S = st["S"].to(self.S.device)
        self.slow = st["slow"].to(self.S.device)
        self.H.copy_(st["H"].to(self.S.device))

    # ---- single token ----
    def step(self, token_id=None, embed_override=None, freeze_dynamics=False,
             temperature_tau=True, pin_mask=None, pin_tau_min=2.0):
        c = self.cfg
        with torch.no_grad() if not self.training else torch.enable_grad():
            if embed_override is not None:
                e = embed_override
            elif token_id is not None:
                e = self.embed(torch.tensor(token_id, device=self.S.device))
            else:
                e = None
            anticipate = self.self_pred(self.S)
            if e is None:
                u = torch.zeros_like(self.S)
            else:
                err = e - anticipate
                surprise = torch.sigmoid(self.err_gate(err)).squeeze(-1)
                u = err * surprise
            tau = self.tau_net(self.S)
            if temperature_tau:
                tau = c.tau_min + F.softplus(tau)
                tau = torch.clamp(tau, c.tau_min, c.tau_max)
            tau_pre_override = tau.detach().clone()
            if pin_mask is not None:
                tau = torch.where(pin_mask, torch.clamp(tau, pin_tau_min, c.tau_max), tau)
            tau_stats = torch.stack([tau.mean(), tau.std(unbiased=False), tau.min(), tau.max()])
            h = torch.tanh(self.rec(self.S) + self.err_proj(u))
            m, g, rent, div = self.pathways(self.S, h, tau_stats)
            m = 3.0 * torch.tanh(m / 3.0)
            if freeze_dynamics:
                S_new = self.S.clone()
            else:
                drive = -self.S / tau + h + self.w_slow(self.slow) * 0.1 + m
                if self.hcm_pending is not None:
                    err_hcm = self.hcm_pending - anticipate
                    drive = drive + err_hcm
                    self.hcm_pending = None
                if c.k_repulse > 0:
                    j = torch.arange(c.window, device=self.H.device)
                    age = (self._hptr - 1 - j) % c.window + 1
                    Hs = self.H[age > getattr(c, "repulse_skip", 2)]
                    diff = self.S.unsqueeze(0) - Hs
                    dist = (diff ** 2).sum(-1).sqrt()
                    if c.repulse_adaptive:
                        sigma = torch.quantile(dist.detach(), c.repulse_q)
                        sigma = sigma.clamp(c.repulse_smin, c.repulse_smax)
                    else:
                        sigma = c.repulse_sigma
                    wgt = torch.exp(-(dist ** 2) / (sigma ** 2))
                    drive = drive + c.k_repulse * (wgt.unsqueeze(-1) * diff).sum(0)
                if c.k_wall > 0:
                    over = self.S.abs() - c.wall_margin
                    drive = drive - c.k_wall * F.relu(over) * torch.sign(self.S)
                S_new = self.S + (c.dt / c.substeps) * drive
            S_new = torch.clamp(S_new, -8.0, 8.0)
            slow_new = torch.tanh(c.slow_keep * self.slow + 0.05 * torch.tanh(S_new)[: c.slow_dim])
            self.S = S_new
            self.slow = slow_new
            ptr = self._hptr
            Hn = self.H.clone()
            Hn[ptr] = S_new.detach()
            self.H = Hn
            self._hptr = (ptr + 1) % c.window
            aux = {"rent": rent, "div": div, "g": g.detach(), "tau_mean": tau_stats[0].item(),
                   "tau_mean_t": tau.mean(), "tau_pre_override": tau_pre_override}
            return self.readout(self.S, self.H), aux

    # ---- generation loops ----
    @torch.no_grad()
    def ingest(self, ids):
        for i in ids:
            self.step(i)

    @torch.no_grad()
    def observe(self):
        return self.readout(self.S, self.H)

    @torch.no_grad()
    def reply(self, prompt_ids, max_tokens=48, temperature=0.7, generator=None):
        out = []
        self.ingest(prompt_ids)
        logits = self.observe()
        for _ in range(max_tokens):
            probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
            gen = generator.to(self.S.device) if generator is not None else None
            nxt = torch.multinomial(probs.to(self.S.device), 1, generator=gen).item()
            out.append(nxt)
            logits, _ = self.step(nxt)
        return out

    @torch.no_grad()
    def free_roll(self, steps, generator=None, freeze_dynamics=False):
        traj = []
        for _ in range(steps):
            logits, _ = self.step(None, freeze_dynamics=freeze_dynamics)
            traj.append(self.S.detach().clone())
        return torch.stack(traj)

    # ---- persistence ----
    def save(self, ckpt_dir, step, extra=None):
        p = pathlib.Path(ckpt_dir)
        p.mkdir(parents=True, exist_ok=True)
        payload = {"config": asdict(self.cfg), "step": step,
                   "model": self.state_dict(), "tokenizer": str(DEFAULT_TOKENIZER)}
        if extra:
            payload.update(extra)
        torch.save(payload, p / f"zeus_step{step}.pt")
        torch.save(payload, p / "zeus.pt")

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = ZeusConfig(**payload["config"])
        model = cls(cfg, tokenizer_path=payload.get("tokenizer"))
        sd = payload["model"]
        if "pathways.w1" not in sd:
            E, d, hd = cfg.experts, cfg.dim, cfg.expert_hidden
            w1 = torch.zeros(E, d, hd)
            b1 = torch.zeros(E, hd)
            w2 = torch.zeros(E, hd, d)
            b2 = torch.zeros(E, d)
            keep = {}
            pat = re.compile(r"pathways\.experts\.(\d+)\.net\.(0|2)\.(weight|bias)")
            for k, v in sd.items():
                mt = pat.match(k)
                if mt is None:
                    keep[k] = v
                    continue
                e, layer, kind = int(mt.group(1)), int(mt.group(2)), mt.group(3)
                if layer == 0 and kind == "weight":
                    w1[e] = v.t()
                elif layer == 0:
                    b1[e] = v
                elif kind == "weight":
                    w2[e] = v.t()
                else:
                    b2[e] = v
            keep["pathways.w1"] = w1
            keep["pathways.b1"] = b1
            keep["pathways.w2"] = w2
            keep["pathways.b2"] = b2
            if "H" not in keep:
                keep["H"] = torch.zeros(cfg.window, cfg.dim)
            sd = keep
        model.load_state_dict(sd)
        model.reset_state(0.0)
        model.to(device)
        model.eval()
        return model

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
