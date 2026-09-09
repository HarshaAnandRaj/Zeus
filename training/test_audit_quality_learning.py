"""Auditor qualification on nonregistered fixture worlds, never campaign seeds."""
import copy
import unittest
import torch
from core.quality_agent import QualityAgent, QualitySession, model_hash
from core.lifetime_world_v2 import QualityWorld, QualityInterface
from training import quality_learning_contract as K
from training.audit_quality_learning import audit_episode, decision, normalize
from training.run_quality_learning import adjudicate


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(77); cls.model = QualityAgent().eval().requires_grad_(False)
        cls.weights = copy.deepcopy(cls.model.state_dict())
        cls.fixtures = {}
        for arm in K.ARMS:
            world = QualityWorld(seed=991, changing=True); interface = QualityInterface(world)
            session = QualitySession(cls.model, fixed_weights=True); session.begin_episode()
            generator = torch.Generator().manual_seed(992)
            key = dict(trial=0,seed=991,changing=True,arm=arm)
            records = [dict(kind='start',**key,world=normalize(world.snapshot()),model_hash=model_hash(cls.model))]
            tick = 0; total = 0.; inspections = 0; unsafe = 0
            while interface.viable() and tick < K.HORIZON:
                if arm == 'reset_history': session.erase_history()
                action, output = session.act(interface.observation(),generator=generator)
                before = world.snapshot()
                unsafe += action == 3 and before['position'] in (0,4) and before['quality'][before['position']//4] == 0 and before['resources'][before['position']//4] > 0
                effect = interface.step(action); reward = K.reward(effect); total += reward; tick += 1
                inspections += action == 4
                session.record_outcome(effect.after,reward=reward,terminated=effect.terminated,truncated=tick==K.HORIZON and not effect.terminated)
                records.append(dict(kind='step',**key,tick=tick,action=action,reward=reward,
                    observation=list(effect.before.values()),next_observation=list(effect.after.values()),mask=list(effect.after.prediction_mask()),
                    state=output.state.tolist(),logits=output.logits.tolist(),value=output.value.tolist(),predictions=output.predictions.tolist(),terminated=effect.terminated))
            summary = dict(**key,survived=interface.viable(),ticks=tick,inspections=inspections,unsafe_harvests=unsafe,reward=total,final_world=normalize(world.snapshot()))
            records.append(dict(kind='end',episode=summary)); cls.fixtures[arm] = (records,summary)

    def test_all_three_arms_replay(self):
        for records,summary in self.fixtures.values():
            self.assertEqual(audit_episode(iter(records),summary,self.weights,992),summary['ticks'])

    def test_corrupted_trace_fields_are_rejected(self):
        records,summary = self.fixtures['intact']
        changes = {'action':lambda r:r.update(action=(r['action']+1)%6),
                   'reward':lambda r:r.update(reward=r['reward']+.1),
                   'state':lambda r:r['state'][0].__setitem__(0,999.),
                   'sensor':lambda r:r['next_observation'].__setitem__(0,.123),
                   'mask':lambda r:r['mask'].__setitem__(0,False),
                   'termination':lambda r:r.update(terminated=not r['terminated']),
                   'tick':lambda r:r.update(tick=2)}
        for name,change in changes.items():
            with self.subTest(name=name):
                altered = copy.deepcopy(records); change(altered[1])
                with self.assertRaises((ValueError,AssertionError)): audit_episode(iter(altered),summary,self.weights,992)

    def test_wrong_seed_weights_summary_and_missing_record_rejected(self):
        records,summary = self.fixtures['intact']
        altered = copy.deepcopy(records); altered[0]['world']['energy'] = .1
        with self.assertRaises(ValueError): audit_episode(iter(altered),summary,self.weights,992)
        wrong = copy.deepcopy(self.weights); wrong['actor.bias'][0] += .1
        with self.assertRaises(ValueError): audit_episode(iter(records),summary,wrong,992)
        wrong_summary = dict(summary,ticks=summary['ticks']+1)
        with self.assertRaises(ValueError): audit_episode(iter(records),wrong_summary,self.weights,992)
        with self.assertRaises(StopIteration): audit_episode(iter(records[:-1]),summary,self.weights,992)

    def test_independent_decisions_and_invalid_coverage(self):
        rows = [dict(trial=t,seed=s,changing=c,arm=a,survived=a=='intact')
                for t in range(4) for s in K.EVALUATION_SEEDS for c in (False,True) for a in K.ARMS]
        self.assertEqual(decision(rows),adjudicate(rows))
        for r in rows:
            r['survived'] = (r['seed']%7 != 0) if r['arm']=='intact' else r['seed']%3 == 0
        self.assertEqual(decision(rows),adjudicate(rows))
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError): decision(bad)
        rows[0]['survived']=1
        with self.assertRaises(ValueError): decision(rows)


if __name__=='__main__': unittest.main()
