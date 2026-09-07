"""CYC5 mechanics on synthetic/nonregistered fixtures, without optimizer training."""
import tempfile
from pathlib import Path
import unittest
import torch
from tools import cyc5_learning_elimination_20260907 as N


class Mechanics(unittest.TestCase):
    def test_features_exclude_body(self):
        self.assertEqual(N.M.features([.1,.9,.5,.3,.25]),N.M.features([.8,.2,.1,.3,.25]))

    def test_masks_exclude_exposure_terminal_padding_and_reflexes(self):
        p=N.C.prepare(83,2);world=N.C.restore(p['boundary']);trace=[]
        for action in (1,1,3,3):trace.append(N.C.step_account(world,N.C.Action(action)))
        e=dict(trace=trace);row,valid=N.label_rows(p,e)
        self.assertEqual(valid,21);self.assertEqual(row['decision_mask'][:16],[0.]*16)
        self.assertEqual(row['decision_mask'][16],1.)
        self.assertEqual(row['decision_mask'][20:],[0.]*493)
        self.assertEqual(row['decision_mask'][18],0.) # current rich-cell food invokes HARVEST
        self.assertEqual(row['weight'][21:],[0.]*492)
        self.assertEqual(row['direction'][16],0)
        self.assertEqual(row['y'][16],[N.C.estimate(p['cache'],i,16) for i in range(9)])

    def test_decision_gradient_corrects_wrong_turn(self):
        x=torch.tensor([[[.01,0,0,0,0,1,0,0,0,0]]],dtype=torch.float32)
        pred=torch.full((1,1,9),.01);pred[0,0,2]=.1;pred[0,0,6]=.6;pred.requires_grad_()
        batch=dict(x=x,y=pred.detach().clone(),weight=torch.ones(1,1),direction=torch.zeros(1,1,dtype=torch.long),decision_mask=torch.ones(1,1))
        total,mse,ce=N.losses(pred,batch,True);total.backward()
        self.assertGreater(ce.item(),1.)
        self.assertLess(pred.grad[0,0,2].item(),0.)
        self.assertGreater(pred.grad[0,0,6].item(),0.)
        self.assertEqual(pred.grad[0,0,4].item(),0.)
        self.assertAlmostEqual(total.item(),.05*ce.item(),places=7)

    def test_current_cell_cannot_win_and_missing_side(self):
        x=torch.tensor([[[.01,1,0,0,0,0,0,0,0,0]]],dtype=torch.float32)
        pred=torch.zeros(1,1,9);pred[...,0]=1.
        logits=N.direction_logits(pred,x)
        self.assertEqual(logits[0,0,0].item(),-1e9)
        self.assertAlmostEqual(logits[0,0,1].item(),-.026/.05,places=6)

    def test_masked_decisions_have_zero_gradient_and_mse_ignores_ce(self):
        pred=torch.full((1,2,9),.5,requires_grad=True)
        x=torch.tensor([[[.01,1,0,0,0,0,0,0,0,0]]*2],dtype=torch.float32)
        batch=dict(x=x,y=pred.detach().clone(),weight=torch.tensor([[1.,0.]]),
                   direction=torch.zeros(1,2,dtype=torch.long),decision_mask=torch.zeros(1,2))
        total,mse,ce=N.losses(pred,batch,True);self.assertEqual(total.item(),0.);total.backward()
        self.assertEqual(pred.grad.abs().sum().item(),0.)
        batch['decision_mask'][:]=1.
        total,mse,ce=N.losses(pred,batch,False)
        self.assertEqual(total.item(),mse.item());self.assertGreater(ce.item(),1.)

    def test_gradient_reaches_history(self):
        torch.manual_seed(41);model=N.M.Memory()
        x=torch.randn(1,17,10,requires_grad=True);pred,_=model(x)
        pred[:,-1].sum().backward()
        self.assertGreater(x.grad[:,0].abs().sum().item(),0.)

    def test_local_labels_on_learner_paths(self):
        torch.manual_seed(43);model=N.M.Memory().eval();p=N.C.prepare(89,6)
        e=N.M.neural_rollout(model,p,p,'intact');row,valid=N.label_rows(p,e)
        world=N.C.restore(p['boundary']);cache=dict(p['cache'])
        for i,step in enumerate(e['trace'],16):
            N.C.observe(cache,world.observation(),i)
            self.assertEqual(row['y'][i],[N.C.estimate(cache,j,i) for j in range(9)])
            a,_=N.C.choose(world.observation(),i,cache)
            self.assertEqual(row['decision_mask'][i],float(int(a) in (1,2)))
            if int(a) in (1,2):self.assertEqual(row['direction'][i],int(a)-1)
            world.step(N.C.Action[step['action'].upper()])
        self.assertEqual(valid,world.body.age+1)

    def test_stream_identity_and_binary_multiple_comparison_gate(self):
        shared=[];arms={a:[] for a in N.ARMS}
        for i in range(128):
            os=[]
            for correct in (1,2):
                ep=dict(survived_256=False,survived_512=False,age=20,first_action='move_left')
                os.append(dict(correct_first_action=correct,teacher=dict(ep,survived_512=True),untrained=ep,
                    searches=[dict(first_action=a,status='EXHAUSTIVE_NO_SURVIVOR',expanded_transitions=1) for a in range(6) if a!=correct]))
            shared.append(dict(seed=N.CONFIG['eval_seed']+i,orientations=os))
            for arm in arms:
                arms[arm].append(dict(seed=N.CONFIG['eval_seed']+i,orientations=[dict(controls={
                    c:dict(ep,survived_256=c=='intact',survived_512=c=='intact') for c in ('intact','erased','swapped')}) for _ in (1,2)]))
        result=N.adjudicate(shared,arms,128)
        self.assertEqual(result['verdict'],'PASS');self.assertEqual(result['qualified_arms'],list(N.ARMS))
        self.assertLess(N.wilson(128,128)[0],N.C.wilson(128,128)[0])
        self.assertEqual(result['contrasts']['interaction']['point'],0.)
        for pairs in arms.values():
            for pair in pairs:
                for o in pair['orientations']:
                    o['controls']['intact'].update(survived_256=False,survived_512=False)
        self.assertEqual(N.adjudicate(shared,arms,128)['verdict'],'FAIL')
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.gz',Path(tmp)/'b.gz'
            N.write_rows(a,shared);N.write_rows(b,shared)
            self.assertEqual(N.stream_hash(a),N.stream_hash(b));self.assertEqual(list(N.rows(a)),shared)


if __name__=='__main__':unittest.main()
