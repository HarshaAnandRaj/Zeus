"""Transparent order-only supplement to the preserved E1-A audit exception."""
import hashlib
import json
import subprocess
import numpy as np
import torch

from training import encephalon_e1_contract as K
from training.audit_encephalon_e1 import decide, portable, replay
from training.encephalon_e1 import deterministic, read_checkpoint, tree_hash
from training.run_encephalon_e0 import encoded, read, save, sha, source_sha
from training.run_encephalon_e1 import verify_manifest, job_directory, load_gzip, save_gzip

SOURCES = ("training/audit_encephalon_e1_report_order.py",
           "training/test_encephalon_e1_report_order.py",
           "docs/encephalon_e1_report_order_erratum_20260920.md")


def canonical(report):
    expected = {(r, l, p, x) for r in K.CONFIG["routes"] for l in range(K.CONFIG["lineages"])
                for p in K.CONFIG["profiles"] for x in K.CONFIG["controls"]}
    cells = report["cells"]
    key = lambda row: (row["route"], row["lineage"], row["profile"], row["control"])
    assert len(cells) == len(expected) and {key(row) for row in cells} == expected
    return report | dict(cells=sorted(cells, key=key))


def compare(reconstructed, raw):
    a, b = canonical(reconstructed), canonical(raw)
    assert a == b, "report differs beyond cell order"
    return dict(all_144_cells_exact_by_identity=True, all_decisions_and_intervals_exact=True,
                original_order_matched=reconstructed == raw,
                canonical_report_sha256=hashlib.sha256(encoded(a)).hexdigest(),
                reconstructed_order_sha256=hashlib.sha256(encoded(reconstructed)).hexdigest(),
                runner_order_sha256=hashlib.sha256(encoded(raw)).hexdigest())


def seal_supplement():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=K.ROOT, text=True).strip()
    hashes = {}
    for name in SOURCES:
        data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=K.ROOT)
        hashes[name] = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        assert source_sha(K.ROOT / name) == hashes[name], "uncommitted supplemental source"
    return dict(commit=commit, sources=hashes)


def run():
    deterministic()
    manifest, supplement = verify_manifest(), seal_supplement()
    assert not K.REPORT.exists() and not K.ARCHIVE.exists()
    packets, jobs, numerical = [], [], []
    for lineage in range(K.CONFIG["lineages"]):
        paired_initial = []
        for route in K.CONFIG["routes"]:
            directories = [job_directory(route, lineage, twin) for twin in K.CONFIG["twins"]]
            snapshots = []
            for directory in directories:
                initial = read_checkpoint(directory / "checkpoint_000000.pt")
                final = read_checkpoint(directory / f"checkpoint_{K.CONFIG['updates']:06d}.pt")
                assert initial["manifest_sha256"] == final["manifest_sha256"] == sha(K.OUT / "manifest.json")
                assert final["update"] == K.CONFIG["updates"] and final["config"] == K.CONFIG
                assert final["lineage"] == lineage and final["route"] == route and not final["development"]
                assert tree_hash(initial["model"]) != tree_hash(final["model"])
                assert len(final["history"]) == K.CONFIG["updates"]
                for name in ("context", "sense", "gate", "actor", "value", "consequence"):
                    assert any(h["module_gradient_norms"][name] > 0 for h in final["history"])
                    assert any(not torch.equal(initial["model"][key], final["model"][key])
                               for key in final["model"] if key.startswith(name + "."))
                snapshots.append((initial, final))
            for index in (0, 1): assert tree_hash(snapshots[0][index]) == tree_hash(snapshots[1][index]), "whole-state twin mismatch"
            assert sha(directories[0] / "endpoints.json.gz") == sha(directories[1] / "endpoints.json.gz"), "endpoint twin mismatch"
            initial, final = snapshots[0]
            paired_initial.append(tree_hash(initial["model"]))
            rows = load_gzip(directories[0] / "endpoints.json.gz")
            for row in rows:
                model = initial["model"] if row["control"] == "untrained" else final["model"]
                numerical.append(replay(row, model))
            packets.extend(rows)
            jobs.append(dict(route=route, lineage=lineage, initial=portable(initial), final=portable(final),
                             initial_sha256=tree_hash(initial), final_sha256=tree_hash(final),
                             endpoint_sha256=sha(directories[0] / "endpoints.json.gz"), endpoints=rows))
            print(f"ORDER-SUPPLEMENT REPLAYED {route} {lineage}", flush=True)
        assert len(set(paired_initial)) == 1, "initial parameter matching failed"
    decision = decide(packets)
    proof = compare(decision, read(K.OUT / "raw_verdict.json"))
    assert not proof["original_order_matched"], "the registered reporting defect was not reproduced"
    error = dict(original_auditor="training/audit_encephalon_e1.py", original_source_commit=manifest["commit"],
                 error="AssertionError: decision == read(K.OUT / 'raw_verdict.json')",
                 cause="Auditor appends lineage then route; runner appends route then lineage. List ordering differs, identities and values do not.",
                 original_auditor_completed=False, raw_verdict_sha256=sha(K.OUT / "raw_verdict.json"), proof=proof)
    verify_manifest()
    assert seal_supplement() == supplement
    archive = dict(manifest=manifest, supplemental_verification=supplement, preserved_original_error=error,
                   jobs=jobs, numerical=numerical, verdict=decision)
    save_gzip(K.ARCHIVE, archive)
    report = dict(version=K.VERSION, evidence_verdict="PASS", evidence_verification="post-campaign order-only supplement; original audit error retained",
                  preserved_original_error=error, supplemental_verification=supplement, **decision,
                  frozen_commit=manifest["commit"], manifest_sha256=sha(K.OUT / "manifest.json"),
                  archive=str(K.ARCHIVE.relative_to(K.ROOT)), archive_sha256=sha(K.ARCHIVE),
                  exact_training_and_endpoint_twins=True, twin_bodies=2 * sum(x["bodies"] for x in numerical),
                  twin_steps=2 * sum(x["steps"] for x in numerical),
                  maximum_neural_replay_error=max(x["maximum_neural_error"] for x in numerical),
                  minimum_cdf_boundary_margin=min(x["minimum_cdf_boundary_margin"] for x in numerical))
    save(K.OUT / "original_audit_error.json", error)
    save(K.OUT / "audit_report_order_supplement.json", report)
    save(K.REPORT, report)
    print(json.dumps({k: report[k] for k in ("evidence_verdict", "e1_verdict", "maximum_neural_replay_error", "twin_bodies", "twin_steps")}), flush=True)


if __name__ == "__main__": run()
