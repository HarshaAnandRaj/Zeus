"""Ensure the order-only supplement rejects every semantic report change."""
import copy
import unittest
from training import encephalon_e1_contract as K
from training.audit_encephalon_e1_report_order import compare


class OrderProof(unittest.TestCase):
    def test_order_only_and_semantic_corruption(self):
        rows = [dict(route=r, lineage=l, profile=p, control=x, survived=0)
                for r in K.CONFIG["routes"] for l in range(K.CONFIG["lineages"])
                for p in K.CONFIG["profiles"] for x in K.CONFIG["controls"]]
        raw = dict(cells=rows, e1_verdict="FAIL", contrasts={"test": {"lower": .1}})
        reordered = copy.deepcopy(raw); reordered["cells"].reverse()
        self.assertFalse(compare(reordered, raw)["original_order_matched"])
        for change in ("value", "missing", "duplicate", "decision", "interval"):
            broken = copy.deepcopy(reordered)
            if change == "value": broken["cells"][0]["survived"] = 1
            if change == "missing": broken["cells"].pop()
            if change == "duplicate": broken["cells"][0] = copy.deepcopy(broken["cells"][1])
            if change == "decision": broken["e1_verdict"] = "PASS"
            if change == "interval": broken["contrasts"]["test"]["lower"] = .100001
            with self.assertRaises(AssertionError): compare(broken, raw)


if __name__ == "__main__": unittest.main()
