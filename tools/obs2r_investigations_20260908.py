"""OBS2R: preserve failed four-row assumption; use universal first three rows."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs2_investigations_20260908 as P
P.OUT=ROOT/'runs/obs2r_20260908'
original_sources=P.sources


def sources():
    m=original_sources()
    for name in ('docs/obs2r_measurement_correction_20260908.md','tools/obs2r_investigations_20260908.py',
                 'tools/test_obs2r_investigations_20260908.py','runs/obs2_20260908/invalid.json'):
        m['sources'][name]=P.O.sha(ROOT/name)
    m['execution']='OBS2R';m['prior_attempt']='OBS2_INVALID_FOUR_ROW_ASSUMPTION';return m


def stage1():
    out={};cases=[];checkpoints={};common={}
    for t in P.R.TRIALS:
        cp=P.R.N.load(P.R.OUT/t/'teacher_mse/a/final.pt');checkpoints[t]={k:P.params(cp[k]) for k in ('initial','final')}
        out[t]={};common[t]={c:[] for c in P.R.CONTROLS}
        for c in P.R.CONTROLS:
            h=np.load(P.O.OUT/f'{t}_{c}_samples.npz')['h'].reshape(256,16,32);out[t][c]=P.decompose(h)
        for i,pair in enumerate(P.R.N.rows(P.R.OUT/t/'evaluation_a.jsonl.gz')):
            for oi,o in enumerate(pair['orientations']):
                for c,e in o['controls'].items():
                    h,x,pred,body,age=P.O.arrays(e);assert len(h)>=3;common[t][c].append(h[:3])
                    if i in P.OFFSETS:cases.append(dict(trial=t,world=pair['seed'],orientation=oi,control=c,h=h,x=x,
                        initial_hidden=np.array(e['initial_hidden']).reshape(32)))
        for c in P.R.CONTROLS:out[t][c]['common_first_three']=P.decompose(np.array(common[t][c]))
        print('O1 corrected covariance complete',t,flush=True)
    assert len(cases)==256
    return dict(kind='OBS2_O1',execution='OBS2R',common_window=3,groups=out),cases,checkpoints


if __name__=='__main__':
    P.sources=sources;P.stage1=stage1;P.main()
