import pathlib
import sys
import unittest
from unittest import mock

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "zeus_sandbox"))
_SCRIPT_ARGS = sys.argv[:]
sys.argv = ["zsession.py", "--mode", "interact"]
import zsession  # noqa: E402
sys.argv = _SCRIPT_ARGS


class _DummyModel:
    def reset_state(self, *_args, **_kwargs):
        pass

    def decode(self, ids):
        return "answer token " + str(ids[0]) + " remains readable now"


class _BestOfKDummy:
    class _Cfg:
        ctx_anchor = False

    cfg = _Cfg()

    def encode(self, _prompt):
        return []

    def ingest(self, _tokens):
        pass


class ZSessionSamplingTests(unittest.TestCase):
    def test_same_decode_seed_is_used_across_prompt_counterfactuals(self):
        # The fake mouth has no prompt dependence.  Its first sampled token
        # must therefore match for the same seed across prompts; without the
        # explicit decode re-seed this test becomes a sampling-noise false
        # positive.
        def sampled_reply(*_args, **_kwargs):
            return [int(torch.randint(1_000_000, (1,)).item())], 0

        with mock.patch.object(zsession, "clone_model", side_effect=lambda _m: _DummyModel()), \
                mock.patch.object(zsession, "seeded_generator", side_effect=lambda s: s), \
                mock.patch.object(zsession, "reply_ids", side_effect=sampled_reply):
            result = zsession.p1_prompt_dependence(_DummyModel(), ["prompt A", "prompt B"], [7, 11])
        self.assertEqual(result["samples"]["prompt A|7"], result["samples"]["prompt B|7"])

    def test_self_source_mode_cannot_pass_state_causal_gate(self):
        self.assertFalse(zsession.p4_pass(False, 1.0, True, 0.0))
        self.assertFalse(zsession.p4_pass(True, 0.2, False, 0.0))
        self.assertTrue(zsession.p4_pass(True, 0.2, True, 0.0))

    def test_decode_overrides_reach_best_of_k_path(self):
        with mock.patch.object(zsession, "best_of_k", return_value=([3], None)) as best:
            ids, _ = zsession.reply_ids(_BestOfKDummy(), None, "", best_k=2,
                                         temperature=1.0, top_p=1.0, rep_penalty=1.0)
        self.assertEqual(ids, [3])
        self.assertEqual(best.call_args.kwargs["temperature"], 1.0)
        self.assertEqual(best.call_args.kwargs["top_p"], 1.0)
        self.assertEqual(best.call_args.kwargs["rep_penalty"], 1.0)


if __name__ == "__main__":
    unittest.main()
