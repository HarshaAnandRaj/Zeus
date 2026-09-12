import unittest
import torch
from core.quality_agent import QualityAgent, QualitySession, model_hash
from core.lifetime_world_v2 import QualityInterface
from training import ql2_contract as K
from training.persistent_learning import SequenceBatch
from training.diagnose_ql2_credit_split import split


class SplitTests(unittest.TestCase):
    def test_single_transition_has_no_other_transition_gradient(self):
        torch.set_num_threads(1); torch.manual_seed(31)
        model=QualityAgent(); session=QualitySession(model);session.begin_episode()
        world=QualityInterface(K.start_world(K.TRAIN_BASE,'curriculum',0))
        action,_=session.act(world.observation(),generator=torch.Generator().manual_seed(3))
        effect=world.step(action)
        row=session.record_outcome(effect.after,reward=1.,terminated=False)
        batch=SequenceBatch.from_transitions([row])
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01,foreach=False,fused=False)
        identity=model_hash(model)
        result=split(model,optimizer,batch,[],['feeding'])['steps'][0]
        self.assertGreater(result['own_gradient_slope'],0)
        self.assertAlmostEqual(result['other_transitions_slope'],0,places=5)
        self.assertGreater(result['probability_delta']['plain_sgd'],0)
        self.assertEqual(identity,model_hash(model));self.assertEqual(len(optimizer.state),0)


if __name__=='__main__':unittest.main()
