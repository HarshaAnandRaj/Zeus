"""Read-only held-out-stream mean-state probe; no new world rollout or learning."""
import gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.quality_agent import QualityAgent,model_hash
from training import run_ql2 as R

def main():
 torch.set_num_threads(1);assert R.read(R.OUT/'audit.json')['passed'];R.verify(R.read(R.OUT/'manifest.json'))
 sums={t:np.zeros(32) for t in range(4)};counts={t:0 for t in range(4)}
 path=R.OUT/'evaluation/trace.jsonl.gz'
 with gzip.open(path,'rt') as f:
  for line in f:
   r=json.loads(line)
   if r['kind']=='start':start=r;incoming=None
   elif r['kind']=='step' and start['arm']=='curriculum' and not start['changing']:
    if incoming is not None and start['seed']%2==0:sums[start['trial']]+=incoming;counts[start['trial']]+=1
    incoming=np.asarray(r['state'][0])
 models={};means={};hashes={};values={t:[] for t in range(4)}
 for t in range(4):
  p,_=R.checked_checkpoint(R.OUT/f'training/{t}_curriculum_a');m=QualityAgent();m.load_state_dict(p['final']);m.eval().requires_grad_(False);models[t]=m;hashes[t]=model_hash(m);means[t]=torch.tensor([sums[t]/counts[t]],dtype=torch.float32)
 with gzip.open(path,'rt') as f:
  for line in f:
   r=json.loads(line)
   if r['kind']=='start':start=r;previous=-1
   elif r['kind']=='step' and start['arm']=='curriculum' and not start['changing']:
    if previous>=0 and start['seed']%2==1:
     t=start['trial'];m=models[t];obs=torch.tensor([r['observation']],dtype=torch.float32);prev=torch.tensor([previous]);flag=torch.tensor([False])
     with torch.no_grad():
      zero=m.step(obs,prev,m.initial_state(1),flag).logits.softmax(-1)[0].numpy()
      mean=m.step(obs,prev,means[t],flag).logits.softmax(-1)[0].numpy()
     logits=np.asarray(r['logits'][0]);p=np.exp(logits-logits.max());p/=p.sum()
     values[t].append((float(abs(p-zero).sum()/2),float(abs(p-mean).sum()/2)))
    previous=r['action']
 result={}
 for t,v in values.items():
  x=np.asarray(v);z,a=x.mean(0);result[str(t)]=dict(fit_steps=counts[t],test_steps=len(v),zero_state_mean_tv=float(z),fixed_mean_state_mean_tv=float(a),relative_tv_reduction=float(1-a/z),mean_state=means[t][0].tolist());assert model_hash(models[t])==hashes[t]
 out=dict(grade='Exploratory same-stream approximation, not a functional intervention trial',fit='even stable world seeds',test='odd stable world seeds; noninitial decisions',weights_unchanged=True,new_world_steps=0,per_model=result)
 R.save(R.OUT/'mean_state_diagnostics.json',out);print(json.dumps(result))
if __name__=='__main__':main()
