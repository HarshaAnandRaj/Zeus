"""Isolated TAG1 read adapter. No production HCM/ZeusCore changes."""
from dataclasses import dataclass
import hashlib
import json

import torch
import torch.nn.functional as F


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                            separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass
class Recall:
    vector: torch.Tensor
    indices: torch.Tensor
    similarities: torch.Tensor
    weights: torch.Tensor
    strength: torch.Tensor
    age: torch.Tensor
    origin: torch.Tensor


@torch.no_grad()
def immutable_read(bank, query):
    """Same valid HCM similarity selection, without reinforcing metadata."""
    n = bank.n_patterns
    if not n:
        return None
    age = bank.step_count - bank.birth_step[:n]
    eligible = (age >= bank.min_age) & (bank.strengths[:n] > 0.1)
    sim = bank._cosine_sim(query.detach().to(bank.device), bank.patterns)
    values, indices = sim.masked_fill(~eligible, -1.0).topk(min(bank.top_k, n))
    valid = (values >= bank.recall_threshold) & eligible[indices]
    indices, values = indices[valid], values[valid]
    if not indices.numel():
        return None
    weights = F.softmax(values, dim=0)
    return Recall(
        (bank.target_embed[indices] * weights[:, None]).sum(0),
        indices, values, weights,
        (bank.strengths[indices] / 20).clamp(0, 1),
        (age[indices].float() / 1000).clamp(0, 1),
        bank.action_origin[indices].float() * 2 - 1,
    )


class RecallTag(torch.nn.Module):
    def __init__(self, dim, seed=20260912):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.src_emb = torch.nn.Parameter(torch.randn(dim, generator=generator) * 0.02)
        self.gate = torch.nn.Parameter(torch.randn(3, generator=generator) * 0.02)

    def forward(self, recalled):
        if recalled is None:
            return None, None
        weights = recalled.weights
        s = (weights * recalled.strength).sum()
        a = (weights * recalled.age).sum()
        o = (weights * recalled.origin).sum()
        alpha = torch.sigmoid(self.gate[0] * s + self.gate[1] * a + self.gate[2])
        return alpha * (recalled.vector + o * self.src_emb), alpha


def snapshot(model):
    # Runtime snapshot alone omits the spectral power-iteration buffers.
    return (model.snapshot_runtime(), model.rec.u.detach().clone(),
            model.rec.v.detach().clone())


def restore(model, saved):
    with torch.no_grad():
        # Assign detached storage: copy_ into a live graph is not a reset.
        model.S = saved[0]["S"].clone()
        model.slow = saved[0]["slow"].clone()
        model.H = saved[0]["H"].clone()
        model.E_hist = saved[0]["E_hist"].clone()
        model.restore_runtime(saved[0])
        model.rec.u.copy_(saved[1])
        model.rec.v.copy_(saved[2])


def runtime_equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(runtime_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(runtime_equal(a, b) for a, b in zip(left, right))
    return left == right


def gradient_probe(model, seed=20260912):
    """Synthetic tokens/vectors, actual model path, no corpus measurement."""
    saved = snapshot(model)
    modes = [(module, module.training) for module in model.modules()]
    trainable = [p for p in model.parameters() if p.requires_grad]
    if trainable:
        raise ValueError("gradient probe requires all model parameters frozen")
    before = tensor_hash(dict(model.named_parameters()))
    tag = RecallTag(model.cfg.dim, seed)
    generator = torch.Generator().manual_seed(seed + 1)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    try:
        model.reset_state(0)
        model.E_hist.zero_()
        # Nonzero synthetic starting state avoids a degenerate zero-distance
        # mechanics setup. No checkpoint memory or corpus token is scored.
        model.S = torch.randn(model.cfg.dim, generator=generator) * 0.1
        model.H.copy_(model.S.expand_as(model.H))
        recalled = Recall(torch.randn(model.cfg.dim, generator=generator),
                          torch.tensor([0]), torch.tensor([1.]),
                          torch.tensor([1.]), torch.tensor([0.4]),
                          torch.tensor([0.2]), torch.tensor([1.]))
        model.hcm_pending, alpha = tag(recalled)
        logits, _ = model.step(2)
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([3]))
        connected = bool(loss.requires_grad)
        grads = (torch.autograd.grad(loss, tuple(tag.parameters()), allow_unused=True)
                 if connected else [None] * len(tuple(tag.parameters())))
        norms = {name: None if grad is None else float(grad.norm())
                 for (name, _), grad in zip(tag.named_parameters(), grads)}
        finite = bool(torch.isfinite(loss)) and all(
            grad is None or bool(torch.isfinite(grad).all()) for grad in grads)
        nonzero = any(grad is not None and bool((grad != 0).any()) for grad in grads)
        result = {"synthetic_only": True, "ce_requires_grad": connected,
                  "tag_gradient_norms": norms, "finite": finite,
                  "nonzero_gradient": nonzero,
                  "tag_parameter_count": sum(p.numel() for p in tag.parameters()),
                  "alpha": float(alpha.detach()),
                  "model_parameters_before": before,
                  "model_parameters_after": tensor_hash(dict(model.named_parameters()))}
    finally:
        restore(model, saved)
        for module, mode in modes:
            module.training = mode
    result["runtime_restored"] = runtime_equal(saved, snapshot(model))
    return result
