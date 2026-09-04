"""tools/hcm_provenance.py -- human-readable HCM memory provenance log.

Zeus's memory bank (HCM) persists raw tensors (state patterns, target
embeddings, region ids, counters).  This tool renders that bank into records an
operator can actually read: for every live pattern it decodes the remembered
context passage, the target token, and its lifecycle metadata.

For each memory:
    id            -- index into the pattern bank (0..n-1)
    region        -- the hashed cluster the state projected into
    context       -- decoded text remembered at write time (the passage)
    target_token  -- the predicted next token the memory points at
    birth_step    -- model step when written
    strength      -- recall accumulator (raised on successful recall)
    usage         -- last step at which it was recalled
    utility       -- EMA of whether recall helped prediction (+ = helps,
                     negative = hurt / prune candidate)

Negative prediction-error traces: the runtime tracks loss_before/after per
recall in-memory only (`_recall_losses_*`), never persisted in state_dict.  The
persisted proxy for 'this memory hurt' is a negative `utility`; those patterns
are surfaced under `negative_evidence` and are consolidation/prune candidates.

Output: zeus_sandbox/universe/reports/hcm_provenance_<timestamp>.json
        (+ a compact TSV alongside for quick eyeballing)

Run:
  .venv\\Scripts\\python tools/hcm_provenance.py
  .venv\\Scripts\\python tools/hcm_provenance.py --hcm PATH --checkpoint PATH
"""
import argparse
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_ARGS = sys.argv[1:]
sys.argv = ["zsession.py", "--mode", "interact"]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
import zsession  # noqa: E402

DEFAULT_HCM = str(ROOT / "zeus_sandbox" / "universe" / "sessions" / "sPONR01" /
                  "hcm.pt")


def load_hcm_state(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    sd = payload.get("hcm") if isinstance(payload, dict) and "hcm" in payload \
        else payload
    return sd


def decode_ctx(model, context_row):
    """Decode a pattern's remembered context (right-aligned, zero-padded)."""
    toks = [int(t) for t in context_row.tolist() if int(t) != 0]
    if not toks:
        return ""
    try:
        return model.decode(toks)
    except Exception as exc:  # noqa: BLE001
        return f"<decode-error: {exc}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hcm", default=DEFAULT_HCM)
    ap.add_argument("--checkpoint", default=str(ROOT / zsession.CONFIG["ckpt"]))
    ap.add_argument("--out", default=str(ROOT / "zeus_sandbox" / "universe" /
                                        "reports" /
                                        f"hcm_provenance_{int(time.time())}"))
    ap.add_argument("--max-chars", type=int, default=180)
    args = ap.parse_args(SCRIPT_ARGS)

    sd = load_hcm_state(args.hcm)
    n = int(sd["n_patterns"])
    print(f"loading model for tokenizer ({args.checkpoint}) ...")
    model, _ = zsession.load_safe(args.checkpoint)
    model = model.to(zsession.DEVICE).eval()

    # HCM constructor to rebuild region/utility semantics from raw tensors.
    from core.hcm import HCM
    hcm = HCM(model.cfg.dim, recall_threshold=zsession.RECALL_THRESHOLD,
              device="cpu")
    hcm.load_state_dict(sd)

    records = []
    neg_evidence = []
    positive = 0
    retained_action_origin = 0
    for i in range(n):
        ctx = decode_ctx(model, hcm.context_tokens[i])[: args.max_chars]
        util = float(hcm.utility[i].item())
        is_pos = util >= 0.0
        action_origin = bool(hcm.action_origin[i].item())
        positive += int(is_pos)
        retained_action_origin += int(action_origin)
        rec = {
            "id": i,
            "region": int(hcm.region_id[i].item()),
            "context": ctx,
            "context_tokens": [int(t) for t in hcm.context_tokens[i].tolist()
                               if int(t) != 0],
            "target_token": int(hcm.target_token[i].item()),
            "birth_step": int(hcm.birth_step[i].item()),
            "strength": round(float(hcm.strengths[i].item()), 4),
            "usage": int(hcm.usage[i].item()),
            "utility": round(util, 4),
            "utility_positive": is_pos,
            "action_origin": action_origin,
        }
        records.append(rec)
        if util < 0:
            neg_evidence.append(rec)

    counters = {
        "n_patterns": n,
        "step_count": int(sd.get("step_count", 0)),
        "total_writes": int(sd.get("total_writes", 0)),
        "total_recalls": int(sd.get("total_recalls", 0)),
        "recall_hits": int(sd.get("recall_hits", 0)),
        "action_writes": int(sd.get("action_writes", 0)),
        "auto_writes": int(sd.get("auto_writes", 0)),
        "retained_action_origin": retained_action_origin,
        "positive_utility": positive,
        "negative_utility": len(neg_evidence),
    }

    # Distribution of region occupancy for the provenance summary.
    region_counts = {}
    for r in records:
        region_counts[str(r["region"])] = region_counts.get(str(r["region"]), 0) + 1

    report = {
        "kind": "hcm_provenance",
        "created_unix": int(time.time()),
        "source_hcm": str(args.hcm),
        "note": (
            "context = remembered passage decoded via the voice tokenizer; "
            "action_origin marks a retained memory written after a model-emitted "
            "action; missing legacy provenance loads false. Negative utility marks "
            "memories that hurt prediction (prune candidates). Loss before/after "
            "recall deltas are runtime-only and not persisted, so utility is the "
            "persisted negative-evidence proxy."
        ),
        "counters": counters,
        "region_counts": {k: region_counts[k] for k in sorted(region_counts,
                                                              key=lambda x: int(x))},
        "negative_evidence": neg_evidence[: min(len(neg_evidence), 200)],
        "memories": records,
    }
    out = pathlib.Path(args.out + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    # Compact TSV for quick eyeballing (skip the long token lists).
    tsv = pathlib.Path(args.out + ".tsv")
    with tsv.open("w", encoding="utf-8") as f:
        f.write("id\tregion\tbirth\tstrength\tusage\tutility\taction_origin\tcontext\n")
        for r in records:
            f.write(f"{r['id']}\t{r['region']}\t{r['birth_step']}\t"
                    f"{r['strength']}\t{r['usage']}\t{r['utility']}\t{int(r['action_origin'])}\t"
                    f"{r['context']}\n")

    print(json.dumps({"counters": counters, "negative_utility": len(neg_evidence),
                      "region_counts": region_counts}, indent=2, ensure_ascii=False))
    print(f"\nwrote {out} and {tsv}")


if __name__ == "__main__":
    main()
