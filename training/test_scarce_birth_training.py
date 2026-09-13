import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from training import scarce_birth_training as S,audit_lmb2 as E
from training.native_physical_replay import CheckedWorld


class ScarceBirthTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        cls.config=S.CONFIG|dict(training_base=992000,training_ecologies=4,body_horizon=16,updates=2,chunk=4)
        cls.data={arm:S.demonstrations(arm,cls.config) for arm in S.ARMS}

    def test_actual_profile_exposure_public_teacher_and_every_physical_balance(self):
        for arm,data in self.data.items():
            self.assertEqual(len(data),32);keys=[]
            for d in data:
                prep=d['preparation'];keys.append((d['profile_slot'],prep['index'],d['inherited']))
                actual=self.config['energies'][d['profile_slot']] if arm=='balanced' else .85
                self.assertEqual(prep['energy'],actual);self.assertEqual(prep['query'][0],actual)
                cfg=S.R.CONFIG|dict(base=self.config['training_base'],ecologies=4,horizon=16)
                self.assertEqual(S.A.preparation(prep,cfg),24)
                rows=[dict(tick=tick,**r) for tick,r in enumerate(d['records'])]
                world=CheckedWorld(S.A.world_for(prep['seed'],3,actual,cfg),rows)
                teacher=S.T.initial_teacher(prep['bodies'][0][2]['next_observation'] if d['inherited'] else None)
                for row in rows:
                    self.assertEqual(row['label'],S.T.action(world.observation(),teacher));effect=world.step(row['action'])
                    public=S.A.A.public_record(effect,row['tick'],16)
                    self.assertEqual(public,{k:row[k] for k in public});teacher=S.T.observe(effect,teacher)
                world.complete()
            self.assertEqual(len(set(keys)),32)
        cold=[d for d in self.data['balanced'] if d['preparation']['energy']==.12]
        self.assertTrue(all(not d['records'][-1]['terminated'] for d in cold if d['inherited']))
        empty=[d for d in cold if not d['inherited']]
        self.assertEqual(sum(d['records'][-1]['terminated'] for d in empty),2)

    def test_masked_deaths_and_independent_causal_encoding(self):
        body,_=S.initial(0);data=self.data['balanced'];encoded=S.C.encode(body,data,16)
        independent=E.independent_encoded(body.state_dict(),data,body.state_dict(),16)
        for key in ('inputs','z','label','active'):np.testing.assert_allclose(encoded[key],independent[key],atol=1e-5,rtol=0)
        for i,d in enumerate(data):
            actual=len(d['records']);self.assertEqual(int(encoded['active'][:,i].sum()),actual)
            self.assertFalse(encoded['active'][actual:,i].any())
        ids=S.indices(0,4,self.config);selected=[data[i] for i in ids.tolist()]
        self.assertEqual([(d['profile_slot'],d['inherited']) for d in selected],[(s,m) for s in range(4) for m in (True,False)])
        self.assertTrue(torch.equal(ids,S.indices(0,4,self.config)))

    def test_complete_optimizer_twins_actual_body_credit_and_frozen_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            checks={}
            for arm in S.ARMS:
                paths=[Path(directory)/(arm+'_'+twin) for twin in ('a','b')]
                for twin,path in zip(('a','b'),paths):S.fit(0,arm,twin,self.data[arm],self.config,path)
                a,ca=S.M.L.P.L3.checked_checkpoint(paths[0]);b,cb=S.M.L.P.L3.checked_checkpoint(paths[1])
                self.assertEqual(ca['logical_hash'],cb['logical_hash']);checks[arm]=ca['logical_hash']
                self.assertEqual(int(a['model']['revision']),a['initial_revision']+2)
                self.assertTrue(all(a['first_body_credit'][n]>0 for n in ('fast','reinstate','gate','actor')))
                self.assertEqual(a['first_body_credit']['store'],0);self.assertEqual(a['first_body_credit']['quality'],0)
                self.assertEqual(a['head_hash'],S.M.L.P.L3.tree_hash(a['heads']))
                initial,_=S.initial(0);self.assertEqual(a['frozen_hash'],S.M.L.frozen_hash(initial))
                self.assertEqual(a['logs'][0]['indices'],S.indices(0,4,self.config).tolist())
                for name in ('fast','reinstate','gate','actor'):
                    prefix=name+'.'
                    self.assertNotEqual(S.M.L.P.L3.tree_hash({k:v for k,v in a['model'].items() if k.startswith(prefix)}),
                                        S.M.L.P.L3.tree_hash({k:v for k,v in initial.state_dict().items() if k.startswith(prefix)}))
                encoded=S.C.encode(initial,self.data['balanced'],16)
                trained,_=S.initial(0);trained.load_state_dict(a['model'])
                probabilities=[]
                for model in (initial,trained):
                    h=torch.zeros(32,32)
                    for tick in range(3):logits,h=model.step_inputs(encoded['inputs'][tick],h,encoded['z'][tick])
                    probabilities.append(logits.softmax(-1).detach())
                self.assertGreater(float((probabilities[0]-probabilities[1]).abs().max()),1e-7)
            self.assertNotEqual(checks['balanced'],checks['original'])

    def test_mislabeled_exposure_cannot_create_fit_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'not_created'
            with self.assertRaises(AssertionError):S.fit(0,'original','a',self.data['balanced'],self.config,path)
            self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
