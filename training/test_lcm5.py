import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from training import run_lcm5 as R,lcm5_contract as K,audit_lcm5 as A,audit_lcm4_compatibility as A4


class NativeReaderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.config=K.CONFIG|dict(training_base=720000,training_ecologies=8,updates=4,batch=8)

    def test_reader_only_gradient_and_no_parent_update(self):
        model=R.model_for(0,self.config);data=R.training_data(self.config);state,_=R.consolidate(model.base.store,data)
        before=R.L3.tree_hash(model.base.state_dict());side=torch.tensor([e['side'] for e in data]);quality=torch.tensor([e['quality'] for e in data],dtype=torch.float32)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model.reader(state['z']).gather(1,side[:,None]).squeeze(1),quality);loss.backward()
        self.assertGreater(float(model.reader.weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.base.parameters()));self.assertEqual(before,R.L3.tree_hash(model.base.state_dict()))

    def test_small_twins_and_independent_numpy_adam(self):
        data=R.training_data(self.config)
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/t for t in ('a','b')]
            for path in paths:R.train_one(0,'inherited',path,data,self.config)
            a,_=R.L3.checked_checkpoint(paths[0]);b,_=R.L3.checked_checkpoint(paths[1]);self.assertEqual(R.L3.tree_hash(a),R.L3.tree_hash(b))
            base=R.P.model_for(0);z=A4.reconstruct(base.state_dict(),data)['inherited']
            expected=A.numpy_training(a['initial_reader'],z,data,'inherited',self.config)
            for key in ('weight','bias'):np.testing.assert_allclose(expected[key],a['model']['reader.'+key],atol=2e-4,rtol=0)

    def test_no_write_does_not_learn_content_weights(self):
        data=R.training_data(self.config)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'control';R.train_one(0,'no_write',path,data,self.config);payload,_=R.L3.checked_checkpoint(path)
            self.assertTrue(torch.equal(payload['initial_reader']['weight'],payload['model']['reader.weight']))

    def test_reader_probability_and_parent_identity_validation(self):
        model=R.model_for(0,self.config);z=torch.randn(8,8);side=torch.arange(8)%2
        actual=model.reader(z).gather(1,side[:,None]).squeeze(1).sigmoid().detach().numpy()
        np.testing.assert_allclose(actual,A.reader_probability(model.state_dict(),z,side),atol=1e-6,rtol=0)
        state=copy.deepcopy(model.state_dict());state['_extra_state']['parent_hash']='wrong'
        with self.assertRaises(ValueError):model.load_state_dict(state)

    def test_disjoint_data_roles(self):
        roles=[set(range(K.CONFIG[key],K.CONFIG[key]+K.CONFIG[count])) for key,count in
            (('training_base','training_ecologies'),('development_base','development_ecologies'),('evaluation_base','evaluation_ecologies'))]
        self.assertTrue(all(not roles[i]&roles[j] for i in range(3) for j in range(i)))

    def test_joint_gate_quantity_failure_does_not_hide_in_cell_average(self):
        config=K.CONFIG|dict(evaluation_base=721000,evaluation_ecologies=64,trials=2,bootstrap_draws=100,synthetic_delays=(64,),synthetic_n=8,bin_min=1)
        data=R.P.native_episodes(K.native_config(config['evaluation_base'],config['evaluation_ecologies']))
        evaluation=dict(native=[],quality=[],synthetic=[])
        for trial in range(2):
            p=R.P.evaluate_one(R.P.model_for(trial),trial,data,K.native_config(config['evaluation_base'],config['evaluation_ecologies']))
            qrows=[]
            for row in p['rows']:
                action=row['target'] if row['control']=='full' else [1]*len(data) if row['control']=='reset' else [row['target'][i^2] for i in range(len(data))]
                row['action']=action;row['correct']=[a==b for a,b in zip(action,row['target'])]
                correct=[True]*len(data) if row['control']=='full' else [q==0 for q in row['quality']] if row['control']=='reset' else [False]*len(data)
                row['recall_correct']=correct;qrows.append(dict(control=row['control'],correct=correct))
            for control in ('original','initial','trained_no_write'):qrows.append(dict(control=control,correct=[e['quality']==0 for e in data]))
            evaluation['native'].append(p);evaluation['quality'].append(qrows)
            target=[1,2]*4;side=[0,0,1,1]*2;quality=[0,1]*4
            for control in ('full','reset','shuffle'):
                action=target if control=='full' else [1]*8 if control=='reset' else [target[i^1] for i in range(8)]
                evaluation['synthetic'].append(dict(trial=trial,delay=64,control=control,target=target,side=side,quality=quality,action=action,
                    correct=[a==b for a,b in zip(action,target)],recall_correct=[True]*8,storage_distance=0.))
        self.assertEqual(R.decide(evaluation,data,config),A.independent_decide(evaluation,data,config));self.assertEqual(R.decide(evaluation,data,config)['verdict'],'PASS')
        groups=[[i for i,e in enumerate(data) if e['side']==1 and e['quality']==0 and e['bodies'][0][2]['observation'][3]==coarse] for coarse in (.5,.75)]
        ids=min(groups,key=len)
        for i in ids[:int(len(ids)*.1)+1]:evaluation['native'][0]['rows'][0]['recall_correct'][i]=False
        verdict=R.decide(evaluation,data,config);self.assertTrue(verdict['gates']['recall_0_1_0'])
        self.assertEqual(verdict['verdict'],'FAIL');self.assertEqual(verdict,A.independent_decide(evaluation,data,config))

if __name__=='__main__':unittest.main()
