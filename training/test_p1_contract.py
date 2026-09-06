import unittest
from unittest.mock import patch

from training.p1_contract import (
    CONTRACT_VERSION,
    EXPOSURE_DRAW,
    RAW_CONDITIONS,
    adjudicate,
    canonical_sha256,
    intervals_overlap,
)


class _Decision:
    def __init__(self, passes):
        self.passes = passes


def _fake_gate(words):
    return _Decision(words[0] == "PASS"), {}


def _expression(source, passes):
    samples = []
    for i in range(30):
        first = "PASS" if i < passes else "FAIL"
        text = " ".join([first] + ["word"] * 47)
        samples.append({"sample_id": f"pair-{i:02d}", "text": text,
                        "strict_gate_pass": i < passes})
    report = {
        "kind": "p1_expression_windows",
        "contract_version": CONTRACT_VERSION,
        "source": source,
        "draw_id": "fixed-draw-1",
        "window": {"unit": "words", "size": 48},
        "samples": samples,
    }
    if source == "model":
        report["conditions"] = dict(RAW_CONDITIONS)
    return report


def _review(model, reject=None):
    judgments = []
    for sample in model["samples"]:
        if sample["strict_gate_pass"]:
            judgments.append({"sample_id": sample["sample_id"],
                              "coherent_prose": sample["sample_id"] != reject})
    return {
        "kind": "p1_human_review",
        "contract_version": CONTRACT_VERSION,
        "draw_id": model["draw_id"],
        "reviewer": "human-judge",
        "model_report_sha256": canonical_sha256(model),
        "judgments": judgments,
    }


def _exposure(upper=2.18):
    return {"kind": "probe_exposure_gap", **EXPOSURE_DRAW,
            "exposure_gap_ci": [2.0, upper]}


class P1ContractTests(unittest.TestCase):
    def test_interval_overlap_is_inclusive(self):
        self.assertTrue(intervals_overlap([0.2, 0.4], [0.4, 0.6]))
        self.assertFalse(intervals_overlap([0.2, 0.39], [0.4, 0.6]))

    @patch("training.p1_contract.gate_decision", side_effect=_fake_gate)
    def test_all_four_bars_can_pass(self, _):
        model = _expression("model", 10)
        baseline = _expression("real_prose", 19)
        result = adjudicate(model, baseline, _review(model), _exposure())
        self.assertTrue(result["evidence_valid"], result["errors"])
        self.assertTrue(result["overall_pass"], result)

    @patch("training.p1_contract.gate_decision", side_effect=_fake_gate)
    def test_human_rejection_is_binding(self, _):
        model = _expression("model", 10)
        result = adjudicate(model, _expression("real_prose", 19),
                            _review(model, "pair-03"), _exposure())
        self.assertFalse(result["bars"]["direct_reading"]["pass"])
        self.assertFalse(result["overall_pass"])

    @patch("training.p1_contract.gate_decision", side_effect=_fake_gate)
    def test_recovery_and_decode_conditions_are_binding(self, _):
        model = _expression("model", 10)
        model["conditions"]["sampling"] = "greedy"
        review = _review(model)
        result = adjudicate(model, _expression("real_prose", 19), review,
                            _exposure(2.18001))
        self.assertFalse(result["bars"]["no_crutches"]["pass"])
        self.assertFalse(result["bars"]["recovery"]["pass"])

    @patch("training.p1_contract.gate_decision", side_effect=_fake_gate)
    def test_old_15_sample_report_cannot_enter(self, _):
        model = _expression("model", 10)
        model["samples"] = model["samples"][:15]
        review = _review(model)
        result = adjudicate(model, _expression("real_prose", 19), review, _exposure())
        self.assertFalse(result["evidence_valid"])
        self.assertTrue(any("exactly 30" in error for error in result["errors"]))

    @patch("training.p1_contract.gate_decision", side_effect=_fake_gate)
    def test_fixed_draw_provenance_is_required(self, _):
        model = _expression("model", 10)
        exposure = _exposure()
        del exposure["val_ids_sha256"]
        result = adjudicate(model, _expression("real_prose", 19),
                            _review(model), exposure)
        self.assertFalse(result["evidence_valid"])
        self.assertTrue(any("val_ids_sha256" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
