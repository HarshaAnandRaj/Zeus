"""Registered SEL1 phases. Run generation and inheritance in separate processes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from core.hcm import text_is_clean
from core.model import ZeusConfig, ZeusCore
from training.qualify_tagged_recall import source_checks
from training.tagged_recall import immutable_read, snapshot, restore, tensor_hash
from training.utility_memory import (adjudicate_endpoint, canonical_hash, ce_step,
                                    counterfactual_step, eligible_ids, new_bank,
                                    select_entries, subset_bank)


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def setup():
    manifest = json.loads((ROOT / "docs/registrations/sel1_manifest.json").read_text())
    identities = source_checks(manifest)
    if str(torch.__version__) != "2.5.1+cu121" or np.__version__ != "2.5.2":
        raise RuntimeError("runtime identity mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260920)
    parent = torch.load(ROOT / manifest["checkpoint"], map_location="cpu", weights_only=False)
    if parent["step"] != 8000:
        raise RuntimeError("parent step mismatch")
    model = ZeusCore(ZeusConfig(**parent["config"]))
    model.load_state_dict(parent["model"], strict=True)
    model.requires_grad_(False).eval()
    model.deploy_self_source = False
    model.hcm_pending = model.heartbeat_pending = model.anchor_vec = None
    initial = snapshot(model)
    train = np.load(ROOT / "runs/probe_pilot_v3_nomarkers_b/train_ids.npy", mmap_mode="r")
    val = np.load(ROOT / "runs/probe_pilot_v3_nomarkers_b/val_ids.npy", mmap_mode="r")
    return model, initial, train, val, identities


def block_ids(train, val):
    train_draw = np.random.Generator(np.random.PCG64(20260921)).choice(len(train) // 97, 416, replace=False)
    val_draw = np.random.Generator(np.random.PCG64(20260922)).choice(len(val) // 97, 128, replace=False)
    return {"generate": train_draw[:128].tolist(), "select": train_draw[128:384].tolist(),
            "calibrate": train_draw[384:].tolist(), "endpoint": val_draw.tolist()}


def block_tokens(data, block):
    return [int(x) for x in data[block * 97:(block + 1) * 97]]


@torch.no_grad()
def start_block(model, initial, tokens):
    restore(model, initial)
    model.reset_state(0)
    model.E_hist.zero_()
    model.hcm_pending = model.heartbeat_pending = model.anchor_vec = None
    for token in tokens[:64]:
        model.step(token, emit_readout=False)


def progress(stage, block, total):
    if block % 8 == 0 or block == total:
        print(json.dumps({"phase": stage, "blocks": block, "total": total}), flush=True)


@torch.no_grad()
def generation(model, initial, train, draw):
    pool, legacy = new_bank(model.cfg.dim), new_bank(model.cfg.dim, 64)
    legacy.proj.copy_(pool.proj)
    proposals, legacy_ids = [], []
    for ordinal, block in enumerate(draw):
        tokens = block_tokens(train, block)
        start_block(model, initial, tokens)
        generation_losses = []
        for j in range(64, 95):
            generation_losses.append(ce_step(model, tokens[j], tokens[j + 1]))
        key = model.S.clone()
        loss = ce_step(model, tokens[95], tokens[96])
        generation_losses.append(loss)
        context = tokens[66:96]
        clean = text_is_clean(model.tokenizer.decode(context), min_chars=24)
        accepted = loss > 1.5 and clean
        clock = (ordinal + 1) * 97
        pool.step_count = legacy.step_count = clock
        index = pool.n_patterns if accepted else None
        row = {"generation_block": block, "position": 95, "target": tokens[96],
               "input": tokens[95], "context": context, "ce": loss,
               "clean": clean, "accepted": accepted, "memory_id": index,
               "provenance": "observed_token", "action_origin": False,
               "scored_inputs": tokens[64:96], "scored_targets": tokens[65:97],
               "scored_ce": generation_losses}
        proposals.append(row)
        if accepted:
            target_embed = model.embed(torch.tensor(tokens[96])).detach()
            kwargs = dict(surprisal=loss, target_token=tokens[96], target_embed=target_embed,
                          recent_tokens=context, quality_ok=clean, from_action=False)
            victim = (legacy.n_patterns if legacy.n_patterns < 64 else
                      int((legacy.strengths[:64] * (legacy.usage[:64] + 1)).argmin()))
            assert pool.write(key, **kwargs) and legacy.write(key, **kwargs)
            if victim == len(legacy_ids):
                legacy_ids.append(index)
            else:
                legacy_ids[victim] = index
        progress("generation", ordinal + 1, len(draw))
    pool.step_count += 10
    legacy.step_count += 10
    return pool, legacy, proposals, legacy_ids


@torch.no_grad()
def utility_sample(model, initial, data, draw, banks, seeds, stage):
    rngs = {name: np.random.Generator(np.random.PCG64(seeds[name])) for name in banks}
    rows = {name: [] for name in banks}
    hashes = {name: canonical_hash(bank.state_dict()) for name, bank in banks.items()}
    for ordinal, block in enumerate(draw):
        tokens = block_tokens(data, block)
        start_block(model, initial, tokens)
        for j in range(64, 96):
            choices, vectors = [], []
            for name, bank in banks.items():
                ids, similarities = eligible_ids(bank, model.S)
                choice = int(rngs[name].choice(ids)) if ids else None
                choices.append((name, choice, ids, similarities))
                vectors.append(None if choice is None else bank.target_embed[choice])
            none, losses = counterfactual_step(model, tokens[j], tokens[j + 1], vectors)
            for (name, choice, ids, sims), loss in zip(choices, losses):
                rows[name].append({"block": ordinal, "corpus_block": block, "position": j,
                                   "input": tokens[j], "target": tokens[j + 1],
                                   "memory_id": choice, "eligible_ids": ids,
                                   "similarities": sims, "ce_none": none, "ce_recall": loss,
                                   "utility": none - loss})
        progress(stage, ordinal + 1, len(draw))
    if hashes != {name: canonical_hash(bank.state_dict()) for name, bank in banks.items()}:
        raise RuntimeError("utility read mutated bank")
    return rows


def generate_phase(out):
    model, initial, train, val, identities = setup()
    model_before = tensor_hash(dict(model.named_parameters()))
    draw = block_ids(train, val)
    import hashlib
    draw_hashes = {name: hashlib.sha256(np.asarray([block_tokens(val if name == "endpoint" else train, b)
                                                   for b in blocks], dtype=np.int32).tobytes()).hexdigest()
                   for name, blocks in draw.items()}
    # Only byte identities are read for held-out draws, never model scores.
    pool, legacy, proposals, legacy_ids = generation(model, initial, train, draw["generate"])
    report = {"stage": "generation", "identities": identities, "draw": draw,
              "draw_token_hashes": draw_hashes, "proposals": proposals,
              "candidate_count": pool.n_patterns, "legacy_ids": legacy_ids}
    selected, estimates, rows = [], [], []
    if pool.n_patterns >= 64:
        rows = utility_sample(model, initial, train, draw["select"], {"pool": pool},
                              {"pool": 20260924}, "selection")["pool"]
        selected, estimates = select_entries(rows, pool.n_patterns)
    report.update({"selected_ids": selected, "estimates": estimates,
                   "selection_rows": rows, "endpoint_examples_exposed": 0,
                   "selection_ready": len(selected) >= 8,
                   "verdict": None if len(selected) >= 8 else "FAIL",
                   "reason": (None if len(selected) >= 8 else
                              "fewer than 64 candidates" if pool.n_patterns < 64 else
                              "fewer than 8 entries meet registered causal-utility lower bound")})
    selected_bank = subset_bank(pool, selected)
    for k, original_id in enumerate(selected):
        selected_bank.utility[k] = estimates[original_id]["mean"]
    rng = np.random.Generator(np.random.PCG64(20260923))
    permutation = rng.permutation(pool.n_patterns).tolist()
    banks = {"selected": selected_bank, "legacy": legacy,
             "recency": subset_bank(pool, list(range(max(0, pool.n_patterns - 64), pool.n_patterns))),
             "random": subset_bank(pool, permutation[:64]),
             "random_matched_count": subset_bank(pool, permutation[:len(selected)]),
             "none": subset_bank(pool, [])}
    payload = {"banks": {name: bank.state_dict() for name, bank in banks.items()},
               "pool": pool.state_dict(), "report": report, "random_order": permutation}
    payload_hash = canonical_hash(payload)
    model_after = tensor_hash(dict(model.named_parameters()))
    if model_before != model_after:
        raise RuntimeError("model parameter mutation")
    with (out / "inheritance.pt").open("xb") as stream:
        torch.save(payload, stream)
    reload_payload = torch.load(out / "inheritance.pt", map_location="cpu", weights_only=False)
    if canonical_hash(reload_payload) != payload_hash:
        raise RuntimeError("generation serialization failed")
    report = dict(report, payload_hash=payload_hash, model_parameter_hash=model_before,
                  bank_hashes={name: canonical_hash(state) for name, state in payload["banks"].items()})
    save_json(out / "generation.json", report)
    print(json.dumps({"phase": "generation_complete", "candidate_count": pool.n_patterns,
                      "selected_count": len(selected), "verdict": report["verdict"]}), flush=True)


@torch.no_grad()
def dense_evaluate(model, initial, data, draw, banks, stage):
    rows, means = {name: [] for name in banks}, {name: [] for name in banks}
    for ordinal, block in enumerate(draw):
        tokens = block_tokens(data, block)
        for name, bank in banks.items():
            start_block(model, initial, tokens)
            losses = []
            for j in range(64, 96):
                got = immutable_read(bank, model.S)
                loss = ce_step(model, tokens[j], tokens[j + 1], None if got is None else got.vector)
                losses.append(loss)
                rows[name].append({"block": ordinal, "corpus_block": block, "position": j,
                                   "input": tokens[j], "target": tokens[j + 1], "ce": loss,
                                   "memory_ids": [] if got is None else got.indices.tolist(),
                                   "similarities": [] if got is None else got.similarities.tolist()})
            means[name].append(float(np.mean(losses)))
        progress(stage, ordinal + 1, len(draw))
    return rows, means


def evaluate_phase(out):
    model, initial, train, val, identities = setup()
    model_before = tensor_hash(dict(model.named_parameters()))
    gen = json.loads((out / "generation.json").read_text())
    payload = torch.load(out / "inheritance.pt", map_location="cpu", weights_only=False)
    if gen["payload_hash"] != canonical_hash(payload) or gen["identities"] != identities:
        raise RuntimeError("inherited identity mismatch")
    if not gen["selection_ready"]:
        save_json(out / "evaluation.json", {"verdict": "FAIL", "stage": "selection",
                  "reason": gen["reason"], "endpoint_examples_exposed": 0,
                  "payload_hash": gen["payload_hash"], "reload_verified": True})
        print(json.dumps({"phase": "inheritance_check", "verdict": "FAIL", "reason": gen["reason"]}))
        return
    banks = {}
    for name, state in payload["banks"].items():
        banks[name] = new_bank(model.cfg.dim, 64)
        banks[name].load_state_dict(state)
        if canonical_hash(banks[name].state_dict()) != gen["bank_hashes"][name]:
            raise RuntimeError("bank reload mismatch")
    n = banks["selected"].n_patterns
    order = np.random.Generator(np.random.PCG64(20260925)).permutation(n)
    donor = np.empty(n, dtype=np.int64)
    donor[order] = np.roll(order, 1)
    permuted = subset_bank(banks["selected"], list(range(n)))
    permuted.target_embed[:n] = banks["selected"].target_embed[torch.tensor(donor)]
    banks["permuted"] = permuted
    before_banks = {name: canonical_hash(bank.state_dict()) for name, bank in banks.items()}
    calibration, cal_means = dense_evaluate(model, initial, train, gen["draw"]["calibrate"], banks, "calibration")
    save_json(out / "calibration.json", {"rows": calibration, "block_means": cal_means})
    dense, means = dense_evaluate(model, initial, val, gen["draw"]["endpoint"], banks, "endpoint_dense")
    pool = new_bank(model.cfg.dim)
    pool.load_state_dict(payload["pool"])
    acute = utility_sample(model, initial, val, gen["draw"]["endpoint"],
                           {"selected": banks["selected"], "pool": pool},
                           {"selected": 20260926, "pool": 20260927}, "endpoint_acute")
    result = adjudicate_endpoint(means, acute, n, pool.n_patterns)
    if before_banks != {name: canonical_hash(bank.state_dict()) for name, bank in banks.items()}:
        raise RuntimeError("inherited bank mutated")
    if model_before != tensor_hash(dict(model.named_parameters())):
        raise RuntimeError("model parameters mutated")
    result.update({"payload_hash": gen["payload_hash"], "reload_verified": True,
                   "dense_rows": dense, "acute_rows": acute, "permutation": donor.tolist(),
                   "endpoint_examples_exposed": 128 * 32,
                   "model_parameter_hash": model_before, "bank_hashes": before_banks})
    save_json(out / "evaluation.json", result)
    print(json.dumps({"phase": "evaluation_complete", "verdict": result["verdict"],
                      "bars": result["bars"]}), flush=True)


def finalize(base):
    paths = ["generation.json", "evaluation.json"]
    left, right = {}, {}
    for name in paths:
        left[name] = json.loads((base / "twin_a" / name).read_text())
        right[name] = json.loads((base / "twin_b" / name).read_text())
    identical = left == right
    generation_report, evaluation = left["generation.json"], left["evaluation.json"]
    if generation_report["selection_ready"]:
        identical = identical and ((base / "twin_a/calibration.json").read_bytes() ==
                                   (base / "twin_b/calibration.json").read_bytes())
    verdict = evaluation["verdict"] if identical else "VOID"
    report = {"kind": "SEL1_utility_selection_inheritance", "verdict": verdict,
              "exact_twins": identical, "generation": generation_report,
              "evaluation": evaluation,
              "consequence": ("Audit full retention phase exit before doctrine or higher pillars."
                              if verdict == "PASS" else
                              "Close this fixed continuous-HCM selector attempt; a new registered substrate is required."),
              "emergence_grading": {
                  "designed_setup": "Utility selection is engineered; any benefit is functional engineering.",
                  "unprogrammed_setpoint": "Entry identities were not chosen by humans; thresholds and objective were designed.",
                  "selection_artifact": "Selection utility is not evidence; independent inherited endpoint is required.",
                  "theory_predicted": "No CDT inference; fixed-model memory interventions target prediction only.",
                  "scope": "Predictive memory substrate; no higher pillar or consciousness claim.",
                  "current_audit": "All frozen bars and twin integrity required; no point-estimate override."}}
    save_json(ROOT / "zeus_sandbox/universe/reports/sel1_utility_inheritance_20260906.json", report)
    print(json.dumps({"verdict": verdict, "exact_twins": identical,
                      "candidate_count": generation_report["candidate_count"],
                      "selected_count": len(generation_report["selected_ids"])}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["generate", "evaluate", "finalize"])
    ap.add_argument("--twin", choices=["twin_a", "twin_b"])
    args = ap.parse_args()
    base = ROOT / "runs/sel1_20260906"
    if args.phase == "finalize":
        finalize(base)
        return
    if args.twin is None:
        ap.error("--twin is required")
    out = base / args.twin
    if args.phase == "generate":
        out.mkdir(parents=True, exist_ok=False)
        generate_phase(out)
    else:
        if (out / "evaluation.json").exists() or (out / "calibration.json").exists():
            raise RuntimeError("evaluation already started; do not overwrite")
        evaluate_phase(out)


if __name__ == "__main__":
    main()
