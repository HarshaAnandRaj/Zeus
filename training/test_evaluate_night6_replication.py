import json
import pathlib
import tempfile
import unittest

from training.evaluate_night6_replication import adjudicate


def _write_run(root, name, values, *, complete=True, alarm_steps=()):
    run = root / name
    run.mkdir()
    rows = []
    for step, value in zip((7500, 7750, 8000), values):
        rows.append({"step": step, "val_ce_nats": value})
    rows.extend({"step": step, "event": "CHI_GLASS_ALARM"} for step in alarm_steps)
    if complete:
        rows.append({"event": "COMPLETE", "step": 8000})
        (run / "zeus_step8000.pt").write_bytes(b"checkpoint")
    (run / "train.log").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return run


def _audit(selective=0.03):
    return {"kind": "hcm_causal_recall_audit", "step": 8000,
            "summary": {"n": 24, "mean_matched_minus_wrong": selective}}


class Night6ReplicationTests(unittest.TestCase):
    def test_all_registered_bars_can_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mem = _write_run(root, "mem", [6.8, 6.9, 7.0])
            control = _write_run(root, "control", [7.2, 7.3, 7.4])
            result = adjudicate(mem, control, _audit())
        self.assertTrue(result["evidence_valid"], result["errors"])
        self.assertTrue(result["overall_pass"], result)

    def test_l1_and_control_comparison_are_independently_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mem = _write_run(root, "mem", [7.2, 7.3, 7.4])
            control = _write_run(root, "control", [7.0, 7.1, 7.2])
            result = adjudicate(mem, control, _audit())
        self.assertFalse(result["bars"]["mem_below_l1"]["pass"])
        self.assertFalse(result["bars"]["mem_below_no_mem"]["pass"])
        self.assertFalse(result["overall_pass"])

    def test_missing_endpoint_is_invalid_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mem = _write_run(root, "mem", [6.8, 6.9, 7.0], complete=False)
            control = _write_run(root, "control", [7.2, 7.3, 7.4])
            result = adjudicate(mem, control, _audit())
        self.assertFalse(result["evidence_valid"])
        self.assertTrue(any("COMPLETE" in error for error in result["errors"]))

    def test_binding_mem_ce_failure_completes_fail_decision_without_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mem = _write_run(root, "mem", [30.0, 31.0, 32.0])
            control = _write_run(root, "control", [7.2, 7.3, 7.4], complete=False)
            result = adjudicate(mem, control, _audit())
        self.assertFalse(result["evidence_valid"])
        self.assertTrue(result["failure_proven"])
        self.assertEqual(result["decision_status"], "fail")

    def test_second_legacy_resume_invalidates_arm_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mem = _write_run(root, "mem", [6.8, 6.9, 7.0])
            control = _write_run(root, "control", [7.2, 7.3, 7.4])
            log = control / "train.log"
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            rows[0:0] = [{"event": "resume", "from_step": 2000},
                         {"event": "resume", "from_step": 2500}]
            log.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                           encoding="utf-8")
            result = adjudicate(mem, control, _audit())
        self.assertFalse(result["evidence_valid"])
        self.assertTrue(any("non-exact resume" in error for error in result["errors"]))

    def test_persistent_glass_alarms_fail_collapse_bar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mem = _write_run(root, "mem", [6.8, 6.9, 7.0],
                             alarm_steps=(7900, 7925))
            control = _write_run(root, "control", [7.2, 7.3, 7.4])
            result = adjudicate(mem, control, _audit())
        self.assertFalse(result["bars"]["no_collapse"]["pass"])

    def test_selective_recall_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mem = _write_run(root, "mem", [6.8, 6.9, 7.0])
            control = _write_run(root, "control", [7.2, 7.3, 7.4])
            result = adjudicate(mem, control, _audit(0.0))
        self.assertFalse(result["bars"]["selective_recall"]["pass"])


if __name__ == "__main__":
    unittest.main()
