import copy,gzip,io,json,tempfile,unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import torch
from core.quality_agent import QualityAgent,model_hash
from core.cdt_state_intervention import DriftController,projection,ARMS
from training import run_dri1 as R
from training.audit_dri1 import audit_episode,decision

class DriftTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);torch.manual_seed(79);self.model=QualityAgent().eval().requires_grad_(False);self.mean=torch.full((1,32),.1)
 def test_projection_is_nontrivial_and_complementary(self):
  w,row,null,rank=projection(self.model);self.assertEqual(rank,5)
  self.assertTrue(torch.allclose(row+null,torch.eye(32,dtype=torch.float64),atol=1e-12));self.assertLess(float((w@null).abs().max()),1e-12)
 def test_lag_amplitude_and_protected_vs_visible_effect(self):
  original=model_hash(self.model)
  for arm in ('null_repulsion','row_repulsion','null_noise'):
   controller=DriftController(self.model,arm,self.mean,80)
   for tick in range(75):
    obs=[.8-.003*tick,.9,.5,0.,0.,0.,0.,0.];out,info=controller.step(obs);controller.record_action(tick%6)
    if tick<8:self.assertEqual(info['impulse_rms'],0.)
    else:
     self.assertAlmostEqual(info['impulse_rms'],.05,places=6)
     if arm!='row_repulsion':self.assertLess(info['projection_error'],1e-6);self.assertLess(info['immediate_policy_tv'],1e-6)
   self.assertEqual(len(controller.history),64)
   if arm=='row_repulsion':self.assertGreater(info['projection_error'],.001)
  self.assertEqual(model_hash(self.model),original)
 def test_seeded_noise_and_frozen_boundary(self):
  with self.assertRaises(ValueError):DriftController(QualityAgent(),'intact',self.mean,80)
  with self.assertRaises(ValueError):DriftController(self.model,'unknown',self.mean,80)
  cs=[DriftController(self.model,'null_noise',self.mean,80) for _ in range(2)]
  for i in range(12):
   a=[c.step([.8,.9,.5,0.,0.,0.,0.,0.]) for c in cs];self.assertEqual(a[0][1],a[1][1])
   for c in cs:c.record_action(0)
 def test_fixture_exact_twins_and_independent_replay(self):
  parent=dict(weights=copy.deepcopy(self.model.state_dict()),mean=self.mean,projection_scale=torch.ones(6,dtype=torch.float64),rank=5,fit_steps=1,model_hash=model_hash(self.model))
  with tempfile.TemporaryDirectory() as tmp,patch.object(R,'OUT',Path(tmp)),patch.object(R,'SEEDS',(993,)),patch.object(R,'verify'):
   torch.save([parent],Path(tmp)/'preparation.pt')
   for twin in ('a','b'):
    with redirect_stdout(io.StringIO()):R.run({},twin)
   self.assertEqual(R.read(Path(tmp)/'a/completion.json'),R.read(Path(tmp)/'b/completion.json'))
   rows=R.read(Path(tmp)/'a/results.json')['episodes']
   with gzip.open(Path(tmp)/'a/trace.jsonl.gz','rt') as f:
    records=[json.loads(l) for l in f]
   stream=iter(records)
   for ep in rows:self.assertEqual(audit_episode(stream,ep,parent,0)['steps'],ep['ticks'])
   self.assertIsNone(next(stream,None))
   bad=copy.deepcopy(records);bad[1]['delta'][0][0]=99.
   with self.assertRaises(ValueError):audit_episode(iter(bad),rows[0],parent,0)
 def test_independent_binary_decision(self):
  rows=[dict(trial=t,seed=s,changing=c,arm=a,ticks=1024 if a=='null_repulsion' else 60,survived=a=='null_repulsion') for t in range(4) for s in R.SEEDS for c in (False,True) for a in ARMS]
  self.assertEqual(R.decide(rows),decision(rows));self.assertEqual(R.decide(rows)['directed_drift_benefit'],'PASS')
  for r in rows:
   if r['arm']=='null_noise':r.update(ticks=1024,survived=True)
  self.assertEqual(R.decide(rows)['directed_drift_benefit'],'FAIL')
  with self.assertRaises(ValueError):decision(rows[:-1])
if __name__=='__main__':unittest.main()
