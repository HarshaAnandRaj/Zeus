"""OBS6 actual tangent decomposition and finite-size signed response ladder."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import subprocess
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs5_natural_history_20260909 as B
Q,P,O,R=B.Q,B.P,B.O,B.R
OUT=ROOT/'runs/obs6_20260909';LEVELS=(1.,.5,.125,.03125,.0078125,.001953125);NAMES=('full','row','null');TIMES=B.TIMES


def geometry(v):
    norms=np.linalg.norm(v,axis=1);den=norms[1]*norms[2]
    return dict(norms=norms.tolist(),cosine=None if min(norms[1:])<1e-6 else float(np.dot(v[1],v[2])/den),
                amplification=None if norms[0]<1e-6 else float(norms[1]/norms[0]),
                additivity=float(np.linalg.norm(v[0]-v[1]-v[2])/max(norms[0],norms[1]+norms[2],1e-6)))


def analyze(g,t):
    tangent=geometry(t);finite=[geometry(v) for v in g]
    agreement=[bool(np.all(np.linalg.norm(v-t,axis=1)<=.05*np.maximum(np.linalg.norm(t,axis=1),1e-6))) for v in g]
    small=all(agreement[-2:]);full=agreement[0]
    local=bool(min(tangent['norms'])>=1e-6 and tangent['cosine'] is not None and tangent['cosine']<=-.1 and tangent['amplification']>=1.10)
    amplified=bool(finite[0]['amplification'] is not None and finite[0]['amplification']>=1.10)
    if not amplified:label='NO_RESOLVED_FULL_SCALE_AMPLIFICATION'
    elif not small:label='LOCAL_COMPARISON_NOT_VERIFIED'
    elif local and full:label='LINEAR_EXPLAINS_FULL_SCALE'
    elif local:label='LINEAR_CORE_WITH_FINITE_NONLINEARITY'
    else:label='FINITE_SCALE_ONLY'
    return dict(tangent=tangent,finite=finite,agreement_by_level=agreement,small_scale_agreement=small,full_scale_agreement=full,
                local_cancellation=local,full_scale_amplification=amplified,label=label)


def tangent_path(p,x,baseline,directions):
    # Columns carry actual history differences; initial midpoint is baseline[0].
    d=directions.T.copy();log=[];pred=[]
    for time in range(len(baseline)):
        if time:_,j=P.step(x[time-1],baseline[time-1],p,True);d=j@d
        lg=p['readout.weight']@d;y=B.A.sigmoid(B.A.logits(baseline[time],p));pg=(y*(1-y))[:,None]*lg
        assert np.max(abs(d[:,0]-d[:,1]-d[:,2]))<=1e-10
        log.append(lg.T);pred.append(pg.T)
    return np.stack(log,axis=1),np.stack(pred,axis=1)


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);OUT.mkdir(exist_ok=False)
    prior=R.N.read(B.OUT/'completion.json');audit=R.N.read(B.OUT/'completion_audit.json')
    assert audit['passed'] and audit['completion_sha']==O.sha(B.OUT/'completion.json');O.verify(prior['manifest'])
    sources={}
    for name in prior['artifacts']:
        if name.endswith('.npz') or name.endswith('.json'):
            path=B.OUT/name;assert O.sha(path)==prior['artifacts'][name];sources[str(path.relative_to(ROOT)).replace('\\','/')]=prior['artifacts'][name]
    files=[B.OUT/'completion.json',B.OUT/'completion_audit.json',Q.PRIOR/'replay_inputs.pt',Path(__file__),
           ROOT/'tools/test_obs6_cancellation_20260909.py',ROOT/'docs/obs6_cancellation_protocol_20260909.md',
           ROOT/'tools/obs5_natural_history_20260909.py',ROOT/'tools/obs4_slow_readout_20260909.py',ROOT/'tools/obs3_constant_input_20260908.py',ROOT/'tools/obs2_investigations_20260908.py']
    for p in files:sources[str(p.relative_to(ROOT)).replace('\\','/')]=O.sha(p)
    manifest=dict(sources=sources,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),torch=torch.__version__,numpy=np.__version__,exploratory=True)
    R.C.save(OUT/'manifest.json',manifest)
    try:
        cache=R.N.load(Q.PRIOR/'replay_inputs.pt');allrows=[];prior_max=0.
        for trial in R.TRIALS:
            for kind in ('final','initial'):
                old=np.load(B.OUT/f'{trial}_{kind}.npz');oldrows=R.N.read(B.OUT/f'{trial}_{kind}.json')['cases'];p=cache['checkpoints'][trial][kind];model=Q.native(p)
                midpoint=old['midpoints'];directions=old['arm_differences'][:,[0,3,4]];drivers=old['drivers']
                initial=np.array([[m]+[m+sign*scale*v/2 for scale in LEVELS for v in ds for sign in (1,-1)] for m,ds in zip(midpoint,directions)])
                y=B.replay(model,np.repeat(drivers,37,axis=0),initial.reshape(-1,32)).reshape(16,37,496,32)
                h=np.concatenate([initial[:,:,None],y],axis=2);logits=B.A.logits(h,p);pred=B.A.sigmoid(logits);responses={}
                for name,values in (('logit',logits),('prediction',pred)):
                    g=np.empty((16,6,3,497,9))
                    for li,scale in enumerate(LEVELS):
                        for di in range(3):
                            b=1+li*6+di*2;g[:,li,di]=(values[:,b]-values[:,b+1])/scale
                    responses[name]=g
                for li in range(2):
                    for di,oldarm in enumerate((0,3,4)):
                        start=1+li*6+di*2;fresh=h[:,start:start+2,TIMES];expected=old['checkpoints'][:,li,oldarm]
                        prior_max=max(prior_max,float(np.max(abs(fresh-expected))));np.testing.assert_allclose(fresh,expected,rtol=1e-7,atol=1e-10)
                tl=[];tp=[]
                for ci in range(16):
                    lg,pg=tangent_path(p,drivers[ci],h[ci,0],directions[ci]);tl.append(lg);tp.append(pg)
                tangents={'logit':np.array(tl),'prediction':np.array(tp)}
                saved=dict(midpoints=midpoint,directions=directions,drivers=drivers,baseline=h[:,0],checkpoints=h[:,:,TIMES])
                for name,g in responses.items():saved[name+'_norms']=np.linalg.norm(g,axis=-1);saved['final_'+name+'_vectors']=g[:,:,:,-1];saved[name+'_tangents']=tangents[name]
                np.savez_compressed(OUT/f'{trial}_{kind}.npz',**saved)
                rows=[]
                for ci,oldrow in enumerate(oldrows):
                    row={k:oldrow[k] for k in ('trial','weights','world','orientation','driver')}
                    for name in responses:row[name]=analyze(responses[name][ci,:,:,-1],tangents[name][ci,:,-1])
                    rows.append(row)
                allrows.extend(rows);R.C.save(OUT/f'{trial}_{kind}.json',dict(cases=rows));O.verify(manifest);print('completed',trial,kind,flush=True)
        assert len(allrows)==256
        R.C.save(OUT/'results.json',dict(cases=allrows,levels=LEVELS,directions=NAMES,times=TIMES,max_prior_checkpoint_error=prior_max))
        O.verify(manifest);artifacts={p.name:O.sha(p) for p in OUT.iterdir() if p.is_file()}
        R.C.save(OUT/'completion.json',dict(kind='OBS6_COMPLETE',manifest=manifest,artifacts=artifacts,functional_verdict_changed=False));print('OBS6 complete',flush=True)
    except Exception as exc:R.C.save(OUT/'invalid.json',dict(error=repr(exc)));raise


if __name__=='__main__':main()
