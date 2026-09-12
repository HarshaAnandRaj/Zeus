import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from training import run_lmb2 as R,lmb2_contract as K,learner_history_correction as C,audit_lmb2 as A


class LearnerHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config=K.CONFIG|dict(training_base=760000,training_ecologies=2,body_horizon=16,updates=2,refresh=1,
            collection_batch=2,batch=2,chunk=4,training_cue_base=761000)

    def test_frozen_rule_json_numpy_bool_roundtrip_and_exclusive_failure(self):
        old=R.L.P.L3.read(R.L.OUT/'endpoint.json');verdict=R.L.decide(old)
        self.assertTrue(any(isinstance(v,np.bool_) for v in verdict['gates'].values()))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'verdict.json';C.save(path,verdict)
            self.assertEqual(json.loads(path.read_text()),C.plain(verdict))
            with self.assertRaises(FileExistsError):C.save(path,verdict)
            bad=Path(directory)/'bad.json'
            with self.assertRaises(ValueError):C.save(bad,dict(metric=float('nan')))
            self.assertFalse(bad.exists())
            with self.assertRaises(TypeError):C.save(bad,dict(metric=object()))
            self.assertFalse(bad.exists())

    def test_executed_learner_histories_public_labels_and_tamper_rejection(self):
        data=R.L.demonstrations(self.config);model=C.candidate(0);collected=C.collect(model,data[:2],0,0,self.config)
        upstream=R.L.P.trained_model(0).base.state_dict();starts=A.native_starts(upstream,[d['preparation'] for d in collected])
        arrays={k:v.numpy() for k,v in model.state_dict().items() if isinstance(v,torch.Tensor)}
        for index,d in enumerate(collected):
            summary,count,pe,se=A.body_replay(iter(d['records']),d['preparation'],starts[index:index+1],arrays,
                self.config['collection_action_base']+index,self.config,True,d['inherited'])
            self.assertEqual(count,len(d['records']));self.assertLess(pe,2e-5);self.assertLess(se,1e-4)
        altered=copy.deepcopy(collected[0]);altered['records'][0]['label']=(altered['records'][0]['label']+1)%6
        with self.assertRaises(AssertionError):A.body_replay(iter(altered['records']),altered['preparation'],starts[:1],arrays,
            self.config['collection_action_base'],self.config,True,altered['inherited'])

    def test_padding_is_explicit_and_cannot_train_after_death(self):
        data=R.L.demonstrations(self.config);short=copy.deepcopy(data[0]);short['records']=short['records'][:1]
        with self.assertRaises(AssertionError):C.padded([short],16)
        short['records'][0]['body_done']=True;short['records'][0]['terminated']=True
        model=C.candidate(0);batch=C.encode(model,[short,data[1]],16)
        self.assertEqual(batch['active'].sum(0).tolist(),[1,16]);self.assertEqual(len(short['records']),1)
        baseline=C.loss(model,batch,self.config);altered=dict(batch,label=batch['label'].clone());altered['label'][~batch['active']]=5
        self.assertTrue(torch.equal(baseline,C.loss(model,altered,self.config)))
        independent=A.independent_encoded(model.state_dict(),[short,data[1]],R.L.P.trained_model(0).base.state_dict(),16)
        for key in ('inputs','z','label','active'):np.testing.assert_allclose(batch[key],independent[key],atol=1e-5,rtol=0)

    def test_correction_credit_reaches_all_motor_modules_not_frozen_store(self):
        data=R.L.demonstrations(self.config);model=C.candidate(0);batch=C.encode(model,data,16)
        loss=C.loss(model,batch,self.config);loss.backward()
        for name in ('fast','reinstate','gate','actor'):
            self.assertGreater(sum(float(p.grad.norm()) for p in getattr(model,name).parameters() if p.grad is not None),0)
        for name in ('store','quality'):self.assertTrue(all(p.grad is None and not p.requires_grad for p in getattr(model,name).parameters()))

    def test_exact_small_correction_twins_and_collection_sources(self):
        data=R.L.demonstrations(self.config)
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/t for t in ('a','b')]
            for twin,path in zip(('a','b'),paths):R.train_one(0,'learner_history',twin,data,self.config,path)
            a,ca=R.L.P.L3.checked_checkpoint(paths[0]);b,cb=R.L.P.L3.checked_checkpoint(paths[1]);self.assertEqual(ca['logical_hash'],cb['logical_hash'])
            self.assertEqual(int(a['model']['revision']),386)
            for refresh in range(2):self.assertEqual(R.L.P.L3.sha(paths[0]/f'collection_{refresh}.jsonl.gz'),R.L.P.L3.sha(paths[1]/f'collection_{refresh}.jsonl.gz'))

    def test_independent_joint_rules_and_failed_cells_roundtrip(self):
        old=R.L.P.L3.read(R.L.OUT/'endpoint.json');endpoints={arm:old for arm in K.CONFIG['arms']}
        primary=R.decide(endpoints);independent=A.independent_decide(endpoints,K.CONFIG)
        self.assertEqual(primary,independent);self.assertEqual(primary['verdict'],'FAIL');self.assertEqual(primary['learner_history_attribution'],'FAIL')
        self.assertIsNone(primary['selected_qualified_arm'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'joint.json';C.save(path,primary);self.assertEqual(json.loads(path.read_text()),primary)

    def test_fresh_roles_and_predeclared_balanced_collection(self):
        spans=[set(range(K.CONFIG[k],K.CONFIG[k]+K.CONFIG[s])) for k,s in
            (('training_base','training_ecologies'),('calibration_base','calibration_ecologies'),('evaluation_base','evaluation_ecologies'),('regression_base','regression_ecologies'))]
        self.assertTrue(all(not spans[i]&spans[j] for i in range(4) for j in range(i)))
        data=R.L.demonstrations(self.config|dict(body_horizon=8,training_ecologies=4))
        self.assertEqual(sorted((d['inherited'],d['preparation']['side'],d['preparation']['quality']) for d in data),
            sorted((m,s,q) for m in (False,True) for s in (0,1) for q in (0,1)))

if __name__=='__main__':unittest.main()
