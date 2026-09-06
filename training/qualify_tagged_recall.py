"""Registered TAG1 qualification; refuses endpoint work on an invalid instrument."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from core.hcm import HCM
from core.model import ZeusConfig, ZeusCore
from training.tagged_recall import RecallTag, gradient_probe, immutable_read, tensor_hash


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def committed(path):
    # Text normalization avoids false differences from Git's CRLF conversion.
    expected = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)
    actual = (ROOT / path).read_bytes()
    return expected.replace(b"\r\n", b"\n") == actual.replace(b"\r\n", b"\n")


def source_checks(manifest):
    checks = {}
    for path, expected in manifest["artifacts"].items():
        checks[path] = {"expected": expected, "actual": sha256(ROOT / path)}
    for path in manifest["committed"]:
        if not committed(path):
            raise RuntimeError(f"not committed at HEAD: {path}")
    if any(row["actual"] != row["expected"] for row in checks.values()):
        raise RuntimeError("source identity mismatch; no compute licensed")
    return checks


def bank_qualification(bank, seed=20260915):
    n = bank.n_patterns
    ages = bank.step_count - bank.birth_step[:n]
    eligible = (ages >= bank.min_age) & (bank.strengths[:n] > 0.1)
    origins = bank.action_origin[:n]
    norm_s = (bank.strengths[:n] / 20).clamp(0, 1)
    norm_a = (ages.float() / 1000).clamp(0, 1)
    rng = np.random.Generator(np.random.PCG64(seed))
    maps = {key: rng.permutation(n).tolist() for key in ["source", "age", "strength"]}
    values = {"source": origins, "age": norm_a, "strength": norm_s}
    changed = {key: int((value != value[maps[key]]).sum())
               for key, value in values.items()}
    source_counts = {"action_origin_true": int((origins & eligible).sum()),
                     "action_origin_false": int((~origins & eligible).sum())}
    checks = {"eligible_count": int(eligible.sum()) >= 64,
              "source_identifiable": min(source_counts.values()) >= 32,
              "strength_variation": bool(norm_s[eligible].numel() > 1 and
                                          norm_s[eligible].var() > 0),
              "age_variation": bool(norm_a[eligible].numel() > 1 and
                                     norm_a[eligible].var() > 0),
              **{key + "_permutation_changes_values": count > 0
                 for key, count in changed.items()}}
    rows = [{"memory_id": i, "eligible": bool(eligible[i]),
             "action_origin": bool(origins[i]), "strength": float(bank.strengths[i]),
             "birth_step": int(bank.birth_step[i]), "age": int(ages[i]),
             "strength_norm": float(norm_s[i]), "age_norm": float(norm_a[i])}
            for i in range(n)]
    return {"n_patterns": n, "eligible_count": int(eligible.sum()),
            "source_counts_eligible": source_counts, "checks": checks,
            "permutations": maps, "changed_values": changed, "rows": rows}


def qualify(manifest):
    identities = source_checks(manifest)
    if str(torch.__version__) != "2.5.1+cu121" or np.__version__ != "2.5.2":
        raise RuntimeError("runtime differs from registration")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260912)
    checkpoint = torch.load(ROOT / manifest["checkpoint"], map_location="cpu",
                            weights_only=False)
    if checkpoint["step"] != 8000:
        raise RuntimeError("wrong parent step")
    model = ZeusCore(ZeusConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    bank = HCM(model.cfg.dim, max_patterns=512, recall_threshold=0.8, top_k=1,
               min_age=10, n_clusters=32, context_len=30)
    if "action_origin" not in checkpoint["hcm"]:
        raise RuntimeError("missing saved provenance cannot be imputed")
    bank.load_state_dict(checkpoint["hcm"])
    state = bank.state_dict()
    before = tensor_hash({k: v for k, v in state.items() if torch.is_tensor(v)})
    metadata = bank_qualification(bank)
    # Mechanics reads never score stored targets or registered corpus examples.
    hits = 0
    for row in metadata["rows"]:
        if row["eligible"]:
            hits += immutable_read(bank, bank.patterns[row["memory_id"]]) is not None
    after = tensor_hash({k: v for k, v in bank.state_dict().items() if torch.is_tensor(v)})
    scalars_unchanged = all(value == bank.state_dict()[key] for key, value in state.items()
                            if not torch.is_tensor(value))
    gradient = gradient_probe(model)
    empty = HCM(model.cfg.dim, max_patterns=512, n_clusters=32)
    empty_ok = RecallTag(model.cfg.dim)(immutable_read(empty, torch.zeros(model.cfg.dim))) == (None, None)
    checks = dict(metadata["checks"])
    checks.update({"bank_immutable": before == after and scalars_unchanged,
                   "model_parameters_frozen": gradient["model_parameters_before"] ==
                                              gradient["model_parameters_after"],
                   "gradient_connected": gradient["ce_requires_grad"] and
                                         gradient["nonzero_gradient"] and gradient["finite"],
                   "tag_parameter_count": gradient["tag_parameter_count"] == 771,
                   "runtime_restored": gradient["runtime_restored"],
                   "empty_bank_no_injection": empty_ok})
    return {"kind": "TAG1_instrument_qualification", "identities": identities,
            "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
                        "device": "cpu", "threads": 1, "seed": 20260912},
            "bank_before": before, "bank_after": after, "read_hits": hits,
            "bank": metadata, "gradient": gradient, "checks": checks,
            "qualified": all(checks.values()),
            "verdict": None if all(checks.values()) else "VOID",
            "failed_checks": [key for key, value in checks.items() if not value],
            "training_steps": 0, "endpoint_examples_exposed": 0,
            "scope": "instrument qualification only; not trained twins or a function verdict"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    allowed = (ROOT / "runs/tag1_r_20260906").resolve()
    if not out.is_relative_to(allowed) or out.exists():
        raise RuntimeError("use a new file within registered isolated run directory")
    manifest = json.loads((ROOT / "docs/registrations/tag1_manifest.json").read_text())
    report = qualify(manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"out": str(out), "verdict": report["verdict"],
                      "failed_checks": report["failed_checks"]}))


if __name__ == "__main__":
    main()
