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
    tau_min: float = 4.0
    tau_max: float = 18.0
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
    k_repulse: float = 2.0
    repulse_sigma: float = 0.5
    repulse_skip: int = 2
    repulse_adaptive: bool = True
    repulse_q: float = 0.3
    repulse_smin: float = 0.25
    repulse_smax: float = 16.0
    k_wall: float = 2.0
    wall_margin: float = 6.0
    w_recall: float = 0.25
    ctx_window: int = 64
    n_actions: int = 2
    readout_layers: int = 2
    readout_ffn_mult: int = 2
    readout_heads: int = 12
    ctx_anchor: bool = False
    cross_attn: bool = False          # Broca-layer: token transformer cross-attends to H (brain trajectory)
    # ---- virtual heartbeat (viability controller; model-specific, opt-in) ----
    # A state-triggered, externally-driven perturbation loop. Per the canonical
    # theorem file it is a viability-control mechanism for one specified model,
    # not a universal persistence organ: it can hold operation inside a target
    # set while active and says nothing about autonomy (stops => relapse).
    # Disabled by default; enabled by train.py --heartbeat.
    hb_enabled: bool = False
    hb_target_ds: float = 1.8        # fire when d_s (2*nu/d_w) rises past this (outer wall = 2)
    hb_inner_rho: float = 0.05      # ...or rho_exact climbs past this (inner wall = lock-in)
    hb_gain: float = 1.5            # proportional gain: amp = hb_gain * deficit
    hb_amp_max: float = 1.5         # cap on kick magnitude (drive-add)
    hb_horizon: float = 300.0       # predictive: fire if steps-to-death (ttl) < this
    hb_min_gap: int = 3             # min eval-steps between kicks
    hb_hold: int = 3                # steps a single kick persists
    hb_amp_floor: float = 0.05      # minimum kick magnitude when firing

    @property
    def remember_id(self):
        return 0

    @property
    def note_id(self):
        return 1


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


class BrocaCrossAttention(nn.Module):
    """Cross-attention: the token stream (voice) reads the brain trajectory H.

    Query = token hidden states (ongoing serialization), Key/Value = projected
    H (the brain's authored state trajectory). This is the biologically-faithful
    'Broca layer': the voice renders what the brain has decided, attending to the
    abstract intent trajectory as its source document. H is detached so the mouth
    (readout) can never reshape the brain (no afferent attractor), mirroring the
    existing brain/mouth rule.
    """
    def __init__(self, dim, heads):
        super().__init__()
        # LayerNorm on the KV source so the bridge reads a well-scaled intent
        # trajectory: H (brain states, rms ~70+) is a different scale than the
        # token embeddings the cross-attn was stage-1 pretrained on (~1). Without
        # this normalization the scale mismatch makes the bridge spew garbage and
        # CE blows up; with it the trained wq/wk/wv assumptions carry over.
        self.kv_ln = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

    def forward(self, q, kv_source):
        # q: (Lq, B, dim), kv_source: (Lk, B, dim)
        B = q.shape[1]
        h, d = self.heads, q.shape[-1] // self.heads
        kv = (q if kv_source is None else kv_source).detach()
        kv = self.kv_ln(kv)
        q_ = self.wq(q)
        k_ = self.wk(kv)
        v_ = self.wv(kv)
        Lq, Lk = q.shape[0], kv.shape[0]
        q_ = q_.reshape(Lq, B, h, d).permute(1, 2, 0, 3)   # (B, h, Lq, d)
        k_ = k_.reshape(Lk, B, h, d).permute(1, 2, 0, 3)   # (B, h, Lk, d)
        v_ = v_.reshape(Lk, B, h, d).permute(1, 2, 0, 3)
        attn = torch.softmax((q_ @ k_.transpose(-1, -2)) * self.scale, dim=-1)
        out = attn @ v_                                     # (B, h, Lq, d)
        out = out.permute(2, 0, 1, 3).reshape(Lq, B, h * d)
        return self.out(out)


class BrocaLayer(nn.Module):
    """Pre-norm transformer layer with self-attention + cross-attention-on-H."""
    def __init__(self, dim, heads, ffn_mult):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=False)
        self.ln1b = nn.LayerNorm(dim)
        self.cross = BrocaCrossAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, x, causal, h_src):
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=causal)[0]
        x = x + self.cross(self.ln1b(x), h_src)
        x = x + self.ffn(self.ln2(x))
        return x


class CoupledReadout(nn.Module):
    def __init__(self, cfg: ZeusConfig):
        super().__init__()
        self.cfg = cfg
        self.ln = nn.LayerNorm(cfg.dim)
        # LM signal must come from the TOKEN stream (transformer + e_proj), NOT from S,
        # because val_ce resets S. S is kept as a *small* modulation so the observer
        # coupling C != 0, but it can no longer dominate the logits (which made the
        # model lean on S and left the transformer untrained -> val_ce useless).
        self.s_scale = 0.1
        self.gate_gain = 0.4
        self.ctx_gain = 2.0
        self.s_proj = nn.Linear(cfg.dim, cfg.vocab)
        self.e_proj = nn.Linear(cfg.dim, cfg.vocab)
        # Transformer over the recent-token window: the LM "voice". It only READS the
        # state/token stream (observer), never writes S, so the autonomous self-dynamics
        # (and thus CDT life/death) are preserved. S coupling stays via s_proj + gate.
        self.ctx_pos = nn.Parameter(torch.zeros(cfg.ctx_window, cfg.dim))
        if getattr(cfg, "cross_attn", False):
            self.ctx_tf = nn.ModuleList(
                [BrocaLayer(cfg.dim, cfg.readout_heads, cfg.readout_ffn_mult)
                 for _ in range(cfg.readout_layers)]
            )
            self.ctx_head = nn.Linear(cfg.dim, cfg.vocab)
        else:
            self.ctx_tf = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=cfg.dim, nhead=cfg.readout_heads,
                    dim_feedforward=cfg.dim * cfg.readout_ffn_mult,
                    dropout=0.0, batch_first=False,
                ),
                num_layers=cfg.readout_layers,
            )
            self.ctx_head = nn.Linear(cfg.dim, cfg.vocab)
        self.ctx_anchor = nn.Linear(cfg.dim, cfg.dim) if cfg.ctx_anchor else None
        if self.ctx_anchor is not None:
            nn.init.normal_(self.ctx_anchor.weight, std=0.02)
            nn.init.zeros_(self.ctx_anchor.bias)
        self.gate = nn.Sequential(
            nn.Linear(2 * cfg.dim, cfg.dim),
            nn.Tanh(),
            nn.Linear(cfg.dim, cfg.vocab),
        )

    def forward(self, s, history=None, e=None, e_ctx=None, anchor=None):
        if s.dim() == 1:
            s = s.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        s_n = self.ln(s)
        s_dir = F.normalize(s_n, dim=-1)
        out = self.s_scale * self.s_proj(s_dir)
        if e is not None:
            if e.dim() == 1:
                e = e.unsqueeze(0)
            out = out + self.e_proj(e)
            out = out + self.gate_gain * self.gate(torch.cat([s_n, e], dim=-1))
        # Clean recent-token context (decouples LM competence from the alive/repelled
        # state S). The observer still reads S (s_proj/gate) so C != 0, but the LM
        # signal comes from the token stream, which carries the corpus structure.
        if e_ctx is not None and e_ctx.shape[0] > 0:
            L = e_ctx.shape[0]
            if anchor is not None and self.ctx_anchor is not None:
                a = anchor if anchor.dim() == 1 else anchor.reshape(-1)
                a_emb = self.ctx_anchor(a)
                e_in = torch.cat([a_emb.unsqueeze(0), e_ctx[1:]], dim=0)
            else:
                e_in = e_ctx
            x = e_in.unsqueeze(1) + self.ctx_pos[:L].unsqueeze(1)  # (L, 1, dim)
            causal = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
            )
            if getattr(self.cfg, "cross_attn", False):
                # Broca layer: the token stream (voice) cross-attends to the brain
                # trajectory H — the abstract intent document it must render. When
                # history is None it self-sources (kv=q), matching stage-1 pretrain.
                h_src = history.unsqueeze(1) if history is not None and history.dim() == 2 else history
                for layer in self.ctx_tf:
                    x = layer(x, causal, h_src)
                h = x
            else:
                h = self.ctx_tf(x, mask=causal)  # (L, 1, dim)
            out = out + self.ctx_gain * self.ctx_head(h[-1])
        if squeeze:
            out = out.squeeze(0)
        return out


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
        self.readout = CoupledReadout(c)
        self.w_slow = nn.Linear(c.slow_dim, c.dim)
        # Physical interface. These modules are intentionally separate from
        # the token mouth: a future sensorimotor phase must learn how bodily
        # conditions enter S and how actions are selected from S. They are not
        # a rule-based controller and old checkpoints load them fresh.
        self.body_proj = nn.Sequential(nn.Linear(5, c.dim), nn.Tanh(), nn.Linear(c.dim, c.dim))
        self.action_head = nn.Sequential(
            nn.Linear(c.dim + 5, c.dim), nn.Tanh(), nn.Linear(c.dim, 6),
        )
        self.register_buffer("S", torch.zeros(c.dim), persistent=False)
        self.register_buffer("slow", torch.zeros(c.slow_dim), persistent=False)
        self.register_buffer("E_hist", torch.zeros(c.ctx_window, c.dim), persistent=False)
        self.register_buffer("H", torch.zeros(c.window, c.dim), persistent=True)
        self._hptr = 0
        self.hcm = None
        self.hcm_pending = None
        self.anchor_vec = None
        self.last_e = None
        self.heartbeat_pending = None
        # Deploy knob: when True, the readout's cross-attn self-sources (history=None,
        # = stage-1 plain continuation), so memory content arrives via the token
        # stream instead of the brain trajectory H (Option 3 routing).
        self.deploy_self_source = False

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
        self.last_e = None
        self.heartbeat_pending = None
        self.anchor_vec = None
        self.hcm_pending = None

    def export_state(self):
        return {"S": self.S.detach().clone(), "slow": self.slow.detach().clone(),
                "H": self.H.detach().clone()}

    def import_state(self, st):
        self.S = st["S"].to(self.S.device)
        self.slow = st["slow"].to(self.S.device)
        self.H.copy_(st["H"].to(self.S.device))

    def snapshot_runtime(self):
        hb = self.heartbeat_pending
        return {"S": self.S.detach().clone(), "slow": self.slow.detach().clone(),
                "E_hist": self.E_hist.detach().clone(), "H": self.H.detach().clone(),
                "hptr": self._hptr,
                "last_e": None if self.last_e is None else self.last_e.detach().clone(),
                "heartbeat": (None if hb is None
                              else {"vec": hb["vec"].detach().clone(),
                                    "steps": hb["steps"]}),
                "hcm_pending": self.hcm_pending,
                "anchor_vec": self.anchor_vec}

    def restore_runtime(self, st):
        self.S.copy_(st["S"])
        self.slow.copy_(st["slow"])
        self.E_hist.copy_(st["E_hist"])
        self.H.copy_(st["H"])
        self._hptr = st["hptr"]
        self.last_e = st["last_e"]
        self.heartbeat_pending = st["heartbeat"]
        self.hcm_pending = st["hcm_pending"]
        self.anchor_vec = st.get("anchor_vec")

    # ---- virtual heartbeat (external viability control) -----------------------
    # External actor: knocks S off an approaching attractor WITHOUT changing the
    # intrinsic dynamics (external input is necessary only once an internal
    # no-escape set is established; otherwise an internal subsystem with an
    # admissible exit can rescue). amp scales with the deficit and the kick is
    # on-manifold (random unit direction added to the drive).
    def request_heartbeat(self, amp, direction=None, generator=None):
        g = generator or torch.default_generator
        if direction is None:
            v = torch.randn(self.cfg.dim, generator=g, device=self.S.device)
        else:
            v = direction.to(self.S.device)
        v = v / (v.norm() + 1e-8)
        self.heartbeat_pending = {"vec": amp * v, "steps": self.cfg.hb_hold}

    # ---- single token ----
    def step(self, token_id=None, embed_override=None, freeze_dynamics=False,
             temperature_tau=True, pin_mask=None, pin_tau_min=2.0,
             record_token_context=True):
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
                    drive = drive + c.w_recall * err_hcm
                    self.hcm_pending = None
                if self.heartbeat_pending is not None and self.heartbeat_pending["steps"] > 0:
                    drive = drive + self.heartbeat_pending["vec"]
                    self.heartbeat_pending["steps"] -= 1
                    if self.heartbeat_pending["steps"] <= 0:
                        self.heartbeat_pending = None
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
            if e is not None and record_token_context:
                self.last_e = e.detach()
                self.E_hist = torch.roll(self.E_hist, shifts=-1, dims=0)
                self.E_hist[-1] = e.detach()
            # Brain/mouth rule: the readout (mouth) READS the self-state S (forward
            # coupling, observer C != 0) but its gradient is STOPPED at S, so the
            # mouth can never drive / reshape the brain (no afferent attractor).
            hist = None if self.deploy_self_source else self.H
            read_s = (torch.zeros_like(self.S)
                      if self.deploy_self_source else self.S.detach())
            return self.readout(read_s, hist, e, self.E_hist, self.anchor_vec), aux

    # ---- generation loops ----
    @torch.no_grad()
    def ingest(self, ids):
        for i in ids:
            self.step(i)

    @torch.no_grad()
    def sense_body(self, observation):
        """Let physical observation perturb S without entering token history."""
        obs = torch.as_tensor(observation, dtype=self.S.dtype, device=self.S.device)
        if obs.numel() != 5:
            raise ValueError("body observation must contain exactly five values")
        return self.step(embed_override=self.body_proj(obs.reshape(5)),
                         record_token_context=False)

    def policy_logits(self, observation):
        """Differentiable sensorimotor policy readout.

        This is deliberately separate from ``select_action``: training may
        optimize only this policy surface while the recurrent core and mouth
        remain frozen.  It does not itself impose an action preference.
        """
        obs = torch.as_tensor(observation, dtype=self.S.dtype, device=self.S.device)
        if obs.numel() != 5:
            raise ValueError("body observation must contain exactly five values")
        return self.action_head(torch.cat([self.S, obs.reshape(5)], dim=0))

    @torch.no_grad()
    def action_logits(self, observation):
        """Inference wrapper for the learnable sensorimotor policy."""
        return self.policy_logits(observation)

    @torch.no_grad()
    def select_action(self, observation, *, generator=None):
        probs = F.softmax(self.action_logits(observation), dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator).item())

    # Stage-1 trained over dense windows of real embeddings; a zero-filled E_hist
    # (after reset_state) degenerates a short-prompt reply. Fill the window with a
    # real token's embedding so the voice always reads a dense context.
    @torch.no_grad()
    def pad_window(self, token_id=0):
        pad_e = self.embed(torch.tensor(token_id, device=self.S.device)).detach()
        self.E_hist.copy_(pad_e.unsqueeze(0).expand(self.E_hist.shape[0], -1))

    @torch.no_grad()
    def observe(self):
        hist = None if self.deploy_self_source else self.H
        read_s = torch.zeros_like(self.S) if self.deploy_self_source else self.S
        return self.readout(read_s, hist, self.last_e, self.E_hist, self.anchor_vec)

    @torch.no_grad()
    def reply(self, prompt_ids, max_tokens=48, temperature=0.7, generator=None):
        out = []
        self.ingest(prompt_ids)
        logits = self.observe()
        for _ in range(max_tokens):
            probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
            nxt = torch.multinomial(probs.to(self.S.device), 1, generator=generator).item()
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
        incompatible = model.load_state_dict(sd, strict=False)
        # The sensorimotor modules were added after the original Zeus
        # checkpoints.  They may be fresh, but every other missing or
        # unexpected parameter is a genuine compatibility failure rather than
        # something to silently initialize.
        fresh_sensorimotor = {
            "body_proj.0.weight", "body_proj.0.bias",
            "body_proj.2.weight", "body_proj.2.bias",
            "action_head.0.weight", "action_head.0.bias",
            "action_head.2.weight", "action_head.2.bias",
        }
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if not missing.issubset(fresh_sensorimotor) or unexpected:
            raise RuntimeError(
                "checkpoint compatibility failure: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        model.reset_state(0.0)
        model.to(device)
        model.eval()
        return model

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
