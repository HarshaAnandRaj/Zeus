"""Shared, side-effect-contained HCM recall counterfactual metrics."""

import torch
import torch.nn.functional as F

from core.hcm import HCM


def clone_hcm(hcm):
    """Copy retrieval state so counterfactual reads cannot mutate the bank."""
    probe = HCM(hcm.dim, max_patterns=hcm.max_patterns,
                recall_threshold=hcm.recall_threshold, top_k=hcm.top_k,
                write_surp_thresh=hcm.write_surp_thresh,
                strength_decay=hcm.strength_decay, min_age=hcm.min_age,
                n_clusters=hcm.n_clusters, device=hcm.device,
                context_len=hcm.context_len, write_min_chars=hcm.write_min_chars)
    probe.load_state_dict(hcm.state_dict())
    return probe


def configure_memory_state(model, hcm, index):
    """Restore the observable state/context attached to one retained memory."""
    model.reset_state(0.0)
    pattern = hcm.patterns[index].to(model.S.device)
    model.S.copy_(pattern)
    model.slow.zero_()
    model.H.copy_(pattern.unsqueeze(0).expand_as(model.H))
    model._hptr = 0
    context = [int(x) for x in hcm.context_tokens[index].tolist() if int(x) != 0]
    model.E_hist.zero_()
    if context:
        context = context[-model.cfg.ctx_window:]
        e = model.embed(torch.tensor(context, device=model.S.device)).detach()
        model.E_hist[-len(context):] = e
        model.last_e = e[-1].clone()
    else:
        model.last_e = torch.zeros(model.cfg.dim, device=model.S.device)
    # This evaluates continuous recall, never text-prefix continuation.
    model.deploy_self_source = False


@torch.no_grad()
def target_logprob_after_step(model, hcm, index, pending):
    configure_memory_state(model, hcm, index)
    model.hcm_pending = None if pending is None else pending.to(model.S.device).clone()
    logits, _ = model.step(None)
    target = int(hcm.target_token[index].item())
    return float(F.log_softmax(logits.float(), dim=-1)[target].item())


def summarize(rows):
    n = len(rows)
    if not n:
        return {"n": 0, "pass": False}
    gains = torch.tensor([r["matched_minus_none"] for r in rows])
    selective = torch.tensor([r["matched_minus_wrong"] for r in rows])
    positive = float((gains > 0.02).float().mean().item())
    selective_positive = float((selective > 0.02).float().mean().item())
    return {
        "n": n,
        "mean_matched_minus_none": round(float(gains.mean().item()), 5),
        "median_matched_minus_none": round(float(gains.median().item()), 5),
        "positive_gain_frac": round(positive, 3),
        "mean_matched_minus_wrong": round(float(selective.mean().item()), 5),
        "selective_gain_frac": round(selective_positive, 3),
        # Conservative diagnostic floor. This is a necessary, not sufficient,
        # condition for functional memory: the surrounding battery also
        # requires action provenance and a legible free-running voice.
        "pass": bool(n >= 8 and gains.mean().item() >= 0.02 and positive >= 0.6
                     and selective.mean().item() >= 0.02 and selective_positive >= 0.6),
    }


@torch.no_grad()
def evaluate_recall_counterfactual(model, hcm, limit=24):
    """Return matched/no/wrong causal effects while restoring model runtime."""
    if hcm is None or hcm.n_patterns == 0:
        return {"summary": {"n": 0, "pass": False}, "rows": []}
    saved_runtime = model.snapshot_runtime()
    saved_self_source = model.deploy_self_source
    try:
        source = clone_hcm(hcm)
        eligible = [i for i in range(source.n_patterns)
                    if int(source.step_count - source.birth_step[i]) >= source.min_age]
        chosen = eligible[:max(limit, 0)]
        rows = []
        for offset, index in enumerate(chosen):
            read_bank = clone_hcm(source)
            got = read_bank.read(read_bank.patterns[index])
            matched = got[0] if got is not None and got[0] is not None else None
            recalled_ids = [] if got is None or len(got) < 4 or got[3] is None else [
                int(x) for x in got[3].tolist()
            ]
            wrong_index = chosen[(offset + 1) % len(chosen)] if len(chosen) > 1 else index
            no = target_logprob_after_step(model, source, index, None)
            with_match = target_logprob_after_step(model, source, index, matched)
            wrong = target_logprob_after_step(model, source, index,
                                              source.target_embed[wrong_index])
            rows.append({
                "memory_id": int(index), "target_token": int(source.target_token[index]),
                "recalled_ids": recalled_ids, "wrong_memory_id": int(wrong_index),
                "logp_no_recall": round(no, 5), "logp_matched": round(with_match, 5),
                "logp_wrong": round(wrong, 5),
                "matched_minus_none": round(with_match - no, 5),
                "matched_minus_wrong": round(with_match - wrong, 5),
                "action_origin": bool(source.action_origin[index].item()),
            })
        return {"summary": summarize(rows), "rows": rows}
    finally:
        model.restore_runtime(saved_runtime)
        model.deploy_self_source = saved_self_source
