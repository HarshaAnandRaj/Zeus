import copy,io,json,tempfile,unittest,gzip
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import torch
from core.lifetime_world_v2 import QualityWorld
from training import ql2_contract as K
from training import run_ql2 as R
from training.audit_ql2 import audit_training,independent_decision,audit_cues
from training.audit_quality_learning import audit_episode
from core.quality_agent import QualityAgent

class QL2Tests(unittest.TestCase):
 def test_start_intervention_changes_position_only(self):
  for seed in range(50,60):
   ordinary=QualityWorld(seed=seed,changing=False).snapshot()
   for phase in range(3):
    for arm in K.TRAIN_ARMS:
     s=K.start_world(seed,arm,phase).snapshot();pos=s['position'];s['position']=2
     self.assertEqual(s,ordinary)
     if arm=='curriculum' and phase<2:
      safe=ordinary['quality'].index(1)*4
      self.assertEqual(abs(pos-safe),phase)
     else:self.assertEqual(pos,2)

 def test_fixture_twins_fixed_step_budget_and_corruption_detection(self):
  torch.set_num_threads(1)
  with tempfile.TemporaryDirectory() as tmp,patch.object(K,'PHASE_STEPS',(70,5,5)),patch.object(K,'TRAIN_BASE',991):
   for arm in K.TRAIN_ARMS:
    paths=[Path(tmp)/f'{arm}_{t}' for t in ('a','b')]
    for path in paths:
     with redirect_stdout(io.StringIO()):R.train_one(path,77,arm)
    p,a=R.checked_checkpoint(paths[0]);_,b=R.checked_checkpoint(paths[1])
    self.assertEqual(a['logical_hash'],b['logical_hash']);self.assertEqual(a['trace_sha'],b['trace_sha'])
    self.assertEqual(audit_training(paths[0],p),80)
    with gzip.open(paths[0]/'trace.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
    rows[1]['reward']+=.5
    (paths[0]/'trace.jsonl.gz').rename(paths[0]/'original.gz')
    with R.trace_writer(paths[0]/'trace.jsonl.gz') as f:
     for row in rows:R.log(f,row)
    with self.assertRaises(AssertionError):audit_training(paths[0],p)

 def test_decisions_and_no_cue_rescue(self):
  rows=[dict(trial=t,seed=s,changing=c,arm=a,ticks=1024 if a=='curriculum' else 100,survived=a=='curriculum') for t in range(4) for s in K.EVALUATION_SEEDS for c in (False,True) for a in K.ARMS]
  contexts=[dict(trial=t,changing=False,harvest_probability_gap=.2) for t in range(4) for _ in range(32)]
  self.assertEqual(R.decide(rows,contexts),independent_decision(rows,contexts));self.assertEqual(R.decide(rows,contexts)['curriculum_transfer'],'PASS')
  self.assertEqual(R.decide(rows,[])['feeding_acquisition'],'FAIL')
  for row in rows:
   if row['arm']=='ordinary':row.update(ticks=1024,survived=True)
  self.assertEqual(R.decide(rows,contexts)['curriculum_transfer'],'FAIL')
  self.assertEqual(R.decide(rows,contexts),independent_decision(rows,contexts))
  with self.assertRaises(ValueError):independent_decision(rows[:-1],contexts)
  with self.assertRaises(ValueError):independent_decision(rows+[rows[0]],contexts)

 def test_seed_separation_and_256_boundary(self):
  self.assertFalse(set(range(K.TRAIN_BASE,K.TRAIN_BASE+sum(K.PHASE_STEPS)))&set(K.EVALUATION_SEEDS))
  self.assertEqual(sum(K.PHASE_STEPS),16384)
  rows=[dict(trial=t,seed=s,changing=c,arm=a,ticks=256,survived=False) for t in range(4) for s in K.EVALUATION_SEEDS for c in (False,True) for a in K.ARMS]
  self.assertEqual(R.decide(rows,[])['alive256_counts']['0']['curriculum'],0)

 def test_endpoint_and_cue_replay_on_fixture_seeds(self):
  torch.set_num_threads(1);torch.manual_seed(78);model=QualityAgent()
  p={(t,a):dict(initial=copy.deepcopy(model.state_dict()),final=copy.deepcopy(model.state_dict())) for t in range(4) for a in K.TRAIN_ARMS}
  with tempfile.TemporaryDirectory() as tmp,patch.object(R,'OUT',Path(tmp)),patch.object(K,'EVALUATION_SEEDS',(993,)),patch.object(R,'parents',return_value=p),patch.object(R,'verify'):
   (Path(tmp)/'training').mkdir();R.save(Path(tmp)/'training/completion.json',dict(passed=True))
   with redirect_stdout(io.StringIO()):R.evaluate({})
   rows=R.read(Path(tmp)/'evaluation/results.json')['episodes']
   with gzip.open(Path(tmp)/'evaluation/trace.jsonl.gz','rt') as f:
    stream=(json.loads(l) for l in f)
    for row in rows:
     self.assertEqual(audit_episode(stream,row,model.state_dict(),202696000+row['trial']*1000+int(row['changing'])),row['ticks'])
    self.assertIsNone(next(stream,None))
   self.assertEqual(audit_cues(p),R.read(Path(tmp)/'evaluation/contexts.json'))
if __name__=='__main__':unittest.main()
