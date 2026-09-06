"""SEL1 selection and statistics; no production model or HCM mutation."""
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from core.hcm import HCM
from training.tagged_recall import snapshot, restore, runtime_equal, tensor_hash


def canonical_hash(value):
    """Hash nested metadata and tensors independently of pickle identity."""
    def convert(x):
        if torch.is_tensor(x):
            return {"tensor_sha256": tensor_hash({"value": x})}
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(convert(value), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def new_bank(dim, capacity=128):
    return HCM(dim, max_patterns=capacity, n_clusters=32, top_k=1,
               recall_threshold=.8, min_age=10, context_len=30)


def subset_bank(pool, ids, capacity=64):
    """Copy selected entries without losing the source bank's clock or metadata."""
    state = pool.state_dict()
    index = torch.tensor(ids, dtype=torch.long)
    fields = {"patterns", "target_embed", "region_id", "strengths", "birth_step",
              "usage", "target_token", "action_origin", "context_tokens", "utility"}
    for name in fields:
        state[name] = state[name][index].clone()
    state["n_patterns"] = len(ids)
    bank = new_bank(pool.dim, capacity)
    bank.load_state_dict(state)
    return bank


def eligible_ids(bank, query):
    n = bank.n_patterns
    if not n:
        return [], []
    similarities = bank._cosine_sim(query, bank.patterns)
    valid = ((bank.step_count - bank.birth_step[:n] >= bank.min_age) &
             (bank.strengths[:n] > .1) & (similarities >= bank.recall_threshold))
    ids = torch.where(valid)[0]
    return ids.tolist(), similarities[ids].tolist()


@torch.no_grad()
def ce_step(model, token, target, pending=None):
    model.hcm_pending = pending
    logits, _ = model.step(int(token))
    loss = float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([int(target)])))
    if not np.isfinite(loss):
        raise RuntimeError("nonfinite CE")
    return loss


@torch.no_grad()
def counterfactual_step(model, token, target, vectors):
    """Measure acute interventions; carry forward only the no-recall world."""
    before = snapshot(model)
    none = ce_step(model, token, target)
    after = snapshot(model)
    losses = []
    for vector in vectors:
        restore(model, before)
        if not runtime_equal(snapshot(model), before):
            raise RuntimeError("pre-intervention restoration failed")
        losses.append(ce_step(model, token, target, vector))
    restore(model, after)
    if not runtime_equal(snapshot(model), after):
        raise RuntimeError("post-intervention restoration failed")
    return none, losses


def block_utility(rows, n):
    grouped = [dict() for _ in range(n)]
    for row in rows:
        index = row["memory_id"]
        if index is not None:
            grouped[index].setdefault(row["block"], []).append(row["utility"])
    return [{key: float(np.mean(values)) for key, values in group.items()}
            for group in grouped]


def select_entries(rows, n, *, capacity=64, minimum_blocks=8, resamples=2000):
    evidence = block_utility(rows, n)
    estimates = []
    for index, group in enumerate(evidence):
        values = np.array(list(group.values()), dtype=np.float64)
        ci = None
        if len(values) >= minimum_blocks:
            rng = np.random.Generator(np.random.PCG64(20261000 + index))
            draws = rng.integers(0, len(values), (resamples, len(values)))
            ci = np.quantile(values[draws].mean(1), [.025, .975], method="linear").tolist()
        estimates.append({"memory_id": index, "blocks": len(values),
                          "mean": float(values.mean()) if len(values) else None,
                          "ci": ci, "consolidated": bool(ci and ci[0] > .02)})
    selected = sorted([x["memory_id"] for x in estimates if x["consolidated"]],
                      key=lambda i: (-estimates[i]["ci"][0], i))[:capacity]
    for row in estimates:
        row["consolidated"] = row["memory_id"] in selected
    return selected, estimates


def paired_ci(values, draws):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()),
            "ci": np.quantile(values[draws].mean(1), [.025, .975],
                              method="linear").tolist()}


def fraction_bootstrap(rows, n, n_blocks, draws):
    """Equal entry weighting; resample blocks, not correlated token hits."""
    groups = block_utility(rows, n)
    sums = np.zeros((n_blocks, n), dtype=np.float64)
    counts = np.zeros_like(sums)
    for index, group in enumerate(groups):
        for block, value in group.items():
            sums[block, index] = value
            counts[block, index] = 1
    total_counts = counts.sum(0)
    means = np.divide(sums.sum(0), total_counts, out=np.zeros(n), where=total_counts > 0)
    positive = (means > .02) & (total_counts > 0)
    fractions = []
    # Multiplicity weights preserve the paired block resampling while avoiding
    # an enormous bootstrap x block x memory array.
    for chunk in np.array_split(draws, max(1, len(draws) // 200)):
        weights = np.zeros((len(chunk), n_blocks))
        for k, draw in enumerate(chunk):
            weights[k] = np.bincount(draw, minlength=n_blocks)
        sampled_n = weights @ counts
        sampled_mean = np.divide(weights @ sums, sampled_n,
                                 out=np.zeros_like(sampled_n), where=sampled_n > 0)
        fractions.extend((((sampled_mean > .02) & (sampled_n > 0)).mean(1)).tolist())
    return {"mean": float(positive.mean()), "coverage_blocks": total_counts.astype(int).tolist(),
            "entry_means": means.tolist(),
            "ci": np.quantile(fractions, [.025, .975], method="linear").tolist()}, np.array(fractions)


def adjudicate_endpoint(dense, acute, selected_n, pool_n, resamples=10000):
    arms = list(dense)
    n_blocks = len(dense["selected"])
    rng = np.random.Generator(np.random.PCG64(20260928))
    draws = rng.integers(0, n_blocks, (resamples, n_blocks))
    selected_ce = np.array(dense["selected"])
    gains = {arm: paired_ci(np.array(dense[arm]) - selected_ce, draws)
             for arm in arms if arm != "selected"}
    retained, retained_draw = fraction_bootstrap(acute["selected"], selected_n, n_blocks, draws)
    pool, pool_draw = fraction_bootstrap(acute["pool"], pool_n, n_blocks, draws)
    difference = {"mean": retained["mean"] - pool["mean"],
                  "ci": np.quantile(retained_draw - pool_draw, [.025, .975], method="linear").tolist()}
    coverage = (min(retained["coverage_blocks"]) >= 8 and
                np.mean(np.array(pool["coverage_blocks"]) >= 8) >= .8)
    bars = {"selected_count": selected_n >= 8, "coverage": bool(coverage),
            "dense_gains": all(g["ci"][0] > .02 for g in gains.values()),
            "positive_fraction_gain": difference["ci"][0] > 0,
            "positive_fraction_floor": retained["ci"][0] >= .6}
    point_fail = (any(g["mean"] <= .02 for g in gains.values()) or
                  difference["mean"] <= 0 or retained["mean"] < .6)
    verdict = "PASS" if all(bars.values()) else ("FAIL" if point_fail else "UNDECIDED")
    return {"verdict": verdict, "bars": bars, "dense_gain_nats": gains,
            "dense_mean_ce": {arm: float(np.mean(values)) for arm, values in dense.items()},
            "retained_positive_fraction": retained, "pool_positive_fraction": pool,
            "fraction_difference": difference}
