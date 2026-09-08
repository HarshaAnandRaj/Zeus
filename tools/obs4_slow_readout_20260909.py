"""OBS4 signed markers in local slow and readout-null directions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import subprocess
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs3_constant_input_20260908 as Q
P,O,R=Q.P,Q.O,Q.R
OUT=ROOT/'runs/obs4_20260909'
EPS=(1e-4,1e-5);NAMES=('slow','fast','null_slow');TIMES=(0,1,4,16,64,256,512)


def canonical(v):
    v=v/np.linalg.norm(v)
    return v if v[np.argmax(abs(v))]>=0 else -v


def directions(j,w):
    _,sw,vw=np.linalg.svd(w,full_matrices=True)
    assert np.sum(sw>sw[0]*max(w.shape)*np.finfo(float).eps)==9
    n=vw[9:].T;j64=np.linalg.matrix_power(j,64)
    _,ss,vs=np.linalg.svd(j64);_,sf,vf=np.linalg.svd(j);_,sn,vn=np.linalg.svd(j64@n,full_matrices=False)
    ds=np.array([canonical(vs[0]),canonical(vf[-1]),canonical(n@vn[0])])
    assert np.linalg.norm(w@ds[2])<1e-12
    return ds,dict(slow_singular_values=ss.tolist(),one_step_singular_values=sf.tolist(),null_singular_values=sn.tolist())


def response(plus,minus,eps):return (plus-minus)/(2*eps)


def agrees(a,b):return bool(np.linalg.norm(a-b)<=max(1e-6,.01*max(np.linalg.norm(a),np.linalg.norm(b))))


def qualified(h,y):
    return bool(all(np.linalg.norm(v)>=.01 for v in h) and all(np.linalg.norm(v)>=1e-4 for v in y)
                and agrees(h[0],h[1]) and agrees(y[0],y[1]))


def logits(h,p):return h@p['readout.weight'].T+p['readout.bias']


def sigmoid(x):return 1/(1+np.exp(-x))


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=False)
    prior=R.N.read(Q.OUT/'completion.json');audit=R.N.read(Q.OUT/'completion_audit.json')
    assert audit['passed'] and audit['completion_sha']==O.sha(Q.OUT/'completion.json')
    sources={str((Q.OUT/n).relative_to(ROOT)).replace('\\','/'):O.sha(Q.OUT/n) for n in ('completion.json','completion_audit.json')}
    for n in prior['artifacts']:
        if n.endswith('.npz') or n.endswith('.json'):
            assert O.sha(Q.OUT/n)==prior['artifacts'][n]
            sources[str((Q.OUT/n).relative_to(ROOT)).replace('\\','/')]=prior['artifacts'][n]
    cache_path=Q.PRIOR/'replay_inputs.pt';c2=R.N.read(Q.PRIOR/'completion.json')
    assert O.sha(cache_path)==c2['artifacts']['replay_inputs.pt']
    for path in (cache_path,ROOT/'docs/obs4_slow_readout_protocol_20260909.md',Path(__file__),
                 ROOT/'tools/test_obs4_slow_readout_20260909.py',ROOT/'tools/obs3_constant_input_20260908.py',
                 ROOT/'tools/obs2_investigations_20260908.py'):
        sources[str(path.relative_to(ROOT)).replace('\\','/')]=O.sha(path)
    manifest=dict(sources=sources,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  torch=torch.__version__,numpy=np.__version__,exploratory=True)
    R.C.save(OUT/'manifest.json',manifest)
    try:
        cache=R.N.load(cache_path);allrows=[]
        for trial in R.TRIALS:
            endpoints=[];caseids=[];xs=[]
            for origin in ('final','initial'):
                priorrows=R.N.read(Q.OUT/f'{trial}_{origin}.json')['cases'];a=np.load(Q.OUT/f'{trial}_{origin}.npz')
                endpoints.extend(a['state_16384'][::4]);xs.extend(a['x'][::4]);caseids.extend([{k:r[k] for k in ('trial','world','orientation','control')} for r in priorrows])
            endpoints=np.array(endpoints);xs=np.array(xs);assert len(endpoints)==32
            for kind in ('final','initial'):
                p=cache['checkpoints'][trial][kind];model=Q.native(p);ds=[];meta=[];js=[]
                for x,h in zip(xs,endpoints):
                    _,j=P.step(x,h,p,True);d,m=directions(j,p['readout.weight']);ds.append(d);meta.append(m);js.append(j)
                ds=np.array(ds);initial=[]
                for h,d in zip(endpoints,ds):
                    initial.append([h]+[h+sign*e*v for v in d for e in EPS for sign in (1,-1)])
                initial=np.array(initial);flat=initial.reshape(-1,32)
                y=Q.advance(model,np.repeat(xs,13,axis=0),flat,512).reshape(32,13,512,32)
                h=np.concatenate([initial[:,:,None],y],axis=2);l=logits(h,p);pred=sigmoid(l)
                rh=np.empty((32,3,2,513,32));rl=np.empty((32,3,2,513,9));ry=np.empty_like(rl)
                for di in range(3):
                    for ei,e in enumerate(EPS):
                        b=1+di*4+ei*2
                        for out,values in ((rh,h),(rl,l),(ry,pred)):out[:,di,ei]=response(values[:,b],values[:,b+1],e)
                gh=np.linalg.norm(rh,axis=-1);gl=np.linalg.norm(rl,axis=-1);gy=np.linalg.norm(ry,axis=-1)
                assert np.max(gl[:,2,:,0])<=1e-8
                saved=dict(directions=ds,jacobians=np.array(js),x=xs,checkpoints=h[:,:,TIMES],baseline=h[:,0],
                           hidden_gains=gh,logit_gains=gl,prediction_gains=gy,
                           final_hidden_responses=rh[:,:,:,-1],final_logit_responses=rl[:,:,:,-1],final_prediction_responses=ry[:,:,:,-1])
                np.savez_compressed(OUT/f'{trial}_{kind}.npz',**saved)
                rows=[]
                for ci,identity in enumerate(caseids):
                    row=dict(**identity,weights=kind,direction_metadata=meta[ci],directions={})
                    for di,name in enumerate(NAMES):
                        v=dict(epsilon_agreement_hidden=agrees(rh[ci,di,0,-1],rh[ci,di,1,-1]),
                               epsilon_agreement_prediction=agrees(ry[ci,di,0,-1],ry[ci,di,1,-1]),levels=[])
                        for ei,e in enumerate(EPS):
                            b=1+di*4+ei*2
                            v['levels'].append(dict(epsilon=e,hidden_at_checkpoints=gh[ci,di,ei,list(TIMES)].tolist(),
                                logit_at_checkpoints=gl[ci,di,ei,list(TIMES)].tolist(),prediction_at_checkpoints=gy[ci,di,ei,list(TIMES)].tolist(),
                                max_future_prediction=float(gy[ci,di,ei,1:].max()),first_max_step=1+int(np.argmax(gy[ci,di,ei,1:])),
                                final_midpoint_shift=float(np.linalg.norm((h[ci,b,-1]+h[ci,b+1,-1])/2-h[ci,0,-1])/e)))
                        row['directions'][name]=v
                    row['persistent_readable_marker']=qualified(rh[ci,2,:,-1],ry[ci,2,:,-1]);rows.append(row)
                allrows.extend(rows);R.C.save(OUT/f'{trial}_{kind}.json',dict(cases=rows));O.verify(manifest)
                print('completed',trial,kind,flush=True)
        assert len(allrows)==512
        R.C.save(OUT/'results.json',dict(cases=allrows,times=TIMES,eps=EPS,directions=NAMES))
        O.verify(manifest);artifacts={p.name:O.sha(p) for p in OUT.iterdir() if p.is_file()}
        R.C.save(OUT/'completion.json',dict(kind='OBS4_COMPLETE',manifest=manifest,artifacts=artifacts,functional_verdict_changed=False))
        print('OBS4 complete',flush=True)
    except Exception as exc:R.C.save(OUT/'invalid.json',dict(error=repr(exc)));raise


if __name__=='__main__':main()
