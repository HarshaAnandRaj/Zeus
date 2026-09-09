"""Post-hoc shared-recurrence gradient probe on saved final-policy sequences only."""
import gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.quality_agent import QualityAgent,model_hash
from core.persistent_session import Transition
from training.persistent_learning import SequenceBatch,sequence_loss
from training import run_quality_learning as R

def main():
 torch.set_num_threads(1);assert R.read(R.OUT/'audit.json')['passed']
 models={};identities={};results=[]
 for t in range(4):
  p,_=R.checked_checkpoint(R.OUT/f'training/{t}_a');m=QualityAgent();m.load_state_dict(p['final']);models[t]=m;identities[t]=model_hash(m)
 def probe(start,steps):
  m=models[start['trial']];previous=-1;state=m.initial_state(1);rows=[]
  for r in steps[:64]:
   state_after=torch.tensor(r['state'],dtype=torch.float32)
   rows.append(Transition(torch.tensor(r['observation'],dtype=torch.float32),previous,previous==-1,state,r['action'],r['reward'],torch.tensor(r['next_observation'],dtype=torch.float32),r['terminated'],False,torch.tensor(r['logits'][0]),torch.tensor(r['value'][0]),torch.tensor(r['predictions'][0]),state_after,int(m.revision),True))
   state=state_after;previous=r['action']
  losses=sequence_loss(m,SequenceBatch.from_transitions(rows),R.K.SETTINGS)
  gradients={}
  for name,weight in [('actor',1.),('value',.5),('prediction',1.),('entropy',-.02)]:
   g=torch.autograd.grad(losses[name]*weight,tuple(m.recurrence.parameters()),retain_graph=True)
   gradients[name]=torch.cat([v.flatten() for v in g])
  a=gradients['actor'];p=gradients['prediction'];cos=float(torch.dot(a,p)/(a.norm()*p.norm()).clamp_min(1e-20))
  results.append(dict(trial=start['trial'],seed=start['seed'],steps=len(rows),norms={k:float(v.norm()) for k,v in gradients.items()},actor_prediction_cosine=cos))
 with gzip.open(R.OUT/'evaluation/trace.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line)
   if r['kind']=='start':start=r;steps=[]
   elif r['kind']=='step':
    if not start['changing'] and start['arm']=='intact' and len(steps)<64:steps.append(r)
   elif not start['changing'] and start['arm']=='intact':probe(start,steps)
 for t,m in models.items():assert model_hash(m)==identities[t]
 summary={str(t):dict(mean_norms={k:float(np.mean([r['norms'][k] for r in results if r['trial']==t])) for k in ('actor','value','prediction','entropy')},mean_actor_prediction_cosine=float(np.mean([r['actor_prediction_cosine'] for r in results if r['trial']==t]))) for t in range(4)}
 out=dict(grade='Post-hoc gradients at final checkpoints, not historical attribution of learning failure',episodes=len(results),summary=summary,rows=results,weights_unchanged=True)
 R.save(R.OUT/'gradient_diagnostics.json',out);print(json.dumps(summary))
if __name__=='__main__':main()
