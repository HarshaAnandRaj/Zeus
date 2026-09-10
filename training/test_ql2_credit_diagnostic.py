"""Check diagnostic branches against the actual optimizer update."""
import copy
import unittest
import torch
from core.quality_agent import QualityAgent, QualitySession
from core.lifetime_world_v2 import QualityInterface
from training import ql2_contract as K
from training.persistent_learning import SequenceBatch, update_segment
from training.diagnose_ql2_credit import branch, policy


class CreditDiagnosticTests(unittest.TestCase):
    def test_joint_branch_matches_actual_update_with_existing_moments(self):
        torch.set_num_threads(1); torch.manual_seed(11)
        model = QualityAgent(); session = QualitySession(model); session.begin_episode()
        opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01, foreach=False, fused=False)
        world = QualityInterface(K.start_world(K.TRAIN_BASE, 'curriculum', 0))
        rng = torch.Generator().manual_seed(17)
        for iteration in range(2):
            rows = []
            for _ in range(8):
                action, _ = session.act(world.observation(), generator=rng)
                effect = world.step(action)
                rows.append(session.record_outcome(effect.after, reward=K.reward(effect), terminated=effect.terminated))
            batch = SequenceBatch.from_transitions(rows)
            state_before = copy.deepcopy(model.state_dict())
            opt_before = copy.deepcopy(opt.state_dict())
            expected = branch(model, opt, batch, 'joint')
            for k, value in model.state_dict().items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, state_before[k]))
            from training.run_quality_learning import tree_hash
            self.assertEqual(tree_hash(opt.state_dict()), tree_hash(opt_before))
            update_segment(session, opt, batch, K.SETTINGS, max_grad_norm=1.)
            self.assertTrue(torch.equal(policy(model, batch), expected))
            session.refresh_state()


if __name__ == '__main__':
    unittest.main()
