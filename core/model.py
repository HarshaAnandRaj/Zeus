import json
import math
import pathlib
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


class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class PathwayLayer(nn.Module):
    def __init__(self, cfg: ZeusConfig):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleList([Expert(cfg.dim, cfg.expert_hidden) for _ in range(cfg.experts)])
        self.router = nn.Linear(2 * cfg.dim + 4, cfg.experts)
        self.w_pm = nn.Linear(cfg.dim, cfg.dim)
        nn.init.normal_(self.w_pm.weight, std=0.02)

    def forward(self, s, h, tau_stats):
        r_in = torch.cat([s, h, tau_stats], dim=-1)
        logits = self.router(r_in)
        g = F.softmax(logits, dim=-1)
        outs = torch.stack([e(h) for e in self.experts], dim=-2)
        m = (outs * g.unsqueeze(-1)).sum(dim=-2)
        entropy = -(g * torch.log(g + 1e-9)).sum(dim=-1)
        rent_loss = F.relu(self.cfg.rent_target - entropy).mean()
        div_loss = self._diversity(outs)
        return m, g, rent_loss, div_loss

    def _diversity(self, outs):
        if self.cfg.w_diverse <= 0 or outs.shape[0] == 0:
            return torch.zeros(())
        flat = outs.reshape(-1, outs.shape[-1])
        flat = F.normalize(flat, dim=-1)
        sim = flat @ flat.t()
        n = flat.shape[0]
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
        self.history: list = []

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
        self.history = []

    def export_state(self):
        return {"S": self.S.detach().clone(), "slow": self.slow.detach().clone(),
                "history": [t.detach().clone() for t in self.history]}

    def import_state(self, st):
        self.S = st["S"].to(self.S.device)
        self.slow = st["slow"].to(self.slow.device)
        self.history = [t.to(self.S.device) for t in st["history"]]

    # ---- single token ----
    def step(self, token_id=None, embed_override=None, freeze_dynamics=False, temperature_tau=True):
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
            tau_stats = torch.stack([tau.mean(), tau.std(unbiased=False), tau.min(), tau.max()])
            h = torch.tanh(self.rec(self.S) + self.err_proj(u))
            m, g, rent, div = self.pathways(self.S, h, tau_stats)
            m = 3.0 * torch.tanh(m / 3.0)
            if freeze_dynamics:
                S_new = self.S.clone()
            else:
                drive = -self.S / tau + h + self.w_slow(self.slow) * 0.1 + m
                if c.k_repulse > 0 and len(self.history) > 0:
                    Hs = torch.stack(list(self.history)[-c.window:])
                    diff = self.S.unsqueeze(0) - Hs
                    wgt = torch.exp(-(diff ** 2).sum(-1) / (c.repulse_sigma ** 2))
                    drive = drive + c.k_repulse * (wgt.unsqueeze(-1) * diff).sum(0)
                S_new = self.S + (c.dt / c.substeps) * drive
            S_new = torch.clamp(S_new, -8.0, 8.0)
            slow_new = torch.tanh(c.slow_keep * self.slow + 0.05 * torch.tanh(S_new)[: c.slow_dim])
            self.S = S_new
            self.slow = slow_new
            self.history.append(self.S if self.training else self.S.detach().clone())
            if len(self.history) > c.window:
                self.history.pop(0)
            aux = {"rent": rent, "div": div, "g": g.detach(), "tau_mean": tau_stats[0].item(),
                   "tau_mean_t": tau.mean()}
            return self.readout(self.S, self.history), aux

    # ---- generation loops ----
    @torch.no_grad()
    def ingest(self, ids):
        for i in ids:
            self.step(i)

    @torch.no_grad()
    def observe(self):
        return self.readout(self.S, self.history)

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
        model.load_state_dict(payload["model"])
        model.to(device)
        model.eval()
        return model

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
