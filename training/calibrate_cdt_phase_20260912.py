"""Fixed diagnostic calibration: no adaptive threshold fitting or Zeus rollout."""
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CDT=ROOT.parent/'Configuration Drift Hypothesis'
sys.path.insert(0,str(CDT))
from gamma_probe import gamma_hat


def main():
    rows=[]
    for seed in range(202609120,202609128):
        rng=np.random.default_rng(seed)
        iid=rng.normal(size=(512,4))
        ar=np.zeros((1024,4))
        for t in range(1,len(ar)):ar[t]=.9*ar[t-1]+np.sqrt(1-.9**2)*rng.normal(size=4)
        ar=ar[512:];ar[:,1]+=.5*ar[:,0]
        t=np.arange(128);circle=np.column_stack([np.sin(2*np.pi*t/128),np.cos(2*np.pi*t/128)])
        periodic=np.tile(circle,(4,1))
        for name,x in [('iid_gaussian',iid),('stationary_ar09',ar),('periodic',periodic),('constant',np.ones((512,4)))]:
            r=gamma_hat(x,null='phase',M=512,K=12,nblocks=8,seed=seed)
            rows.append(dict(seed=seed,control=name,**r))
    summary={}
    for name in ('iid_gaussian','stationary_ar09','periodic','constant'):
        rs=[r for r in rows if r['control']==name]
        summary[name]=dict(n=len(rs),mean=float(np.mean([r['est'] for r in rs])),
            min=min(r['est'] for r in rs),max=max(r['est'] for r in rs),
            historical_threshold_hits=int(sum(r['est']>.4 and r['lo']>0 for r in rs)),
            ci_unavailable=sum('unavailable' in r['note'] for r in rs))
    result=dict(grade='Finite null calibration; no universal wall threshold licensed',
                source_sha=hashlib.sha256((CDT/'gamma_probe.py').read_bytes()).hexdigest(),
                configuration=dict(seeds=[202609120,202609127],length=512,M=512,K=12,nblocks=8),
                summary=summary,rows=rows)
    target=ROOT/'zeus_sandbox/universe/reports/cdt_phase_calibration_20260912.json'
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
