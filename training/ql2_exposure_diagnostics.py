"""Check whether QL2 actually delivered feeding consequences during training."""
import gzip,json,math,sys
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training import run_ql2 as R

def main():
 assert R.read(R.OUT/'audit.json')['passed'];R.verify(R.read(R.OUT/'manifest.json'));data=defaultdict(Counter)
 for trial in range(4):
  for arm in R.K.TRAIN_ARMS:
   with gzip.open(R.OUT/f'training/{trial}_{arm}_a/trace.jsonl.gz','rt') as f:
    for line in f:
     row=json.loads(line)
     if row['kind']=='start':phase=row['phase'];quality=row['world']['quality']
     elif row['kind']=='step':
      d=data[f'{trial}/{arm}/{phase}'];o=row['observation'];n=row['next_observation'];a=row['action'];p=int(o[2]*4)
      context='safe' if p in (0,4) and quality[p//4] else 'bad' if p in (0,4) else 'away'
      d['steps']+=1;d[context+'_steps']+=1;d['inspected_patch_steps']+=int(bool(o[4]) and p in (0,4))
      if a==3:
       d[context+'_harvests']+=1;d['energy_increasing_harvests']+=int(n[0]>o[0]);d['energy_gain_from_harvests']+=max(0.,n[0]-o[0])
      if a==5:d['valid_repairs' if p==2 else 'invalid_repairs']+=1
      d['reward_sum']+=row['reward']
 assert sum(d['steps'] for d in data.values())==131072
 aggregates={}
 for arm in R.K.TRAIN_ARMS:
  for phase in range(3):
   d=Counter()
   for t in range(4):d.update(data[f'{t}/{arm}/{phase}'])
   aggregates[f'{arm}/{phase}']=dict(d)
 report=dict(grade='post-hoc exposure accounting; counts are not learned competence',unique_steps=131072,by_trial_phase={k:dict(v) for k,v in data.items()},aggregates=aggregates)
 R.save(R.OUT/'exposure_diagnostics.json',report);print(json.dumps(aggregates),flush=True)
if __name__=='__main__':main()
