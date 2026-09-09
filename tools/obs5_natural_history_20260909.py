"""OBS5 recorded-history component interventions, frozen offline GRUs."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import subprocess
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs4_slow_readout_20260909 as A
Q,P,O,R=A.Q,A.P,A.O,A.R
OUT=ROOT/'runs/obs5_20260909'
ARMS=('original','remove_slow_null','matched_null_shift','remove_all_null','only_null','erase_difference')
SCALES=(1.,.5);TIMES=(0,1,4,16,64,128,256,496)


def components(j,w,delta):
    ds,meta=A.directions(j,w);v=ds[2]
    _,_,vw=np.linalg.svd(w,full_matrices=True);n=vw[9:].T;dn=n@(n.T@delta);u=v*np.dot(v,delta)
    projection=lambda z:n@(n.T@z)-v*np.dot(v,z)
    control=projection(Q.DIRECTION)
    if np.linalg.norm(control)<=1e-12:
        control=next(projection(e) for e in np.eye(32) if np.linalg.norm(projection(e))>1e-12)
    control/=np.linalg.norm(control)
    if np.dot(control,delta)<0:control=-control
    shift=np.linalg.norm(u)*control
    arms=np.array([delta,delta-u,delta-shift,delta-dn,dn,np.zeros(32)])
    assert np.max(abs(w@np.stack([dn,u,shift],axis=1)))<1e-10
    return arms,dict(direction=v.tolist(),null_difference=dn.tolist(),slow_component=u.tolist(),control_shift=shift.tolist(),
                     delta_norm=float(np.linalg.norm(delta)),null_norm=float(np.linalg.norm(dn)),slow_norm=float(np.linalg.norm(u)),
                     null_singular_values=meta['null_singular_values'])


def decisions(hidden,pred,initial_null_norm):
    null_route=initial_null_norm>=1e-8;reductions=[];specific=True
    for si,scale in enumerate(SCALES):
        size=scale*initial_null_norm
        null_route=bool(null_route and size>0 and hidden[si,4]/size>=.01 and pred[si,4]/size>=1e-4)
        original=pred[si,0]
        reduction=None if original==0 else 1-pred[si,1]/original
        matched=None if original==0 else 1-pred[si,2]/original
        specific=bool(specific and original>=scale*1e-5 and reduction is not None and reduction>=.25 and reduction-matched>=.10)
        reductions.append(dict(scale=scale,slow_reduction=reduction,matched_reduction=matched))
    return dict(null_history_route=bool(null_route),slow_null_reduction=bool(specific),reductions=reductions)


def replay(model,x,h):
    with torch.no_grad():y,_=model(torch.tensor(x),torch.tensor(h)[None])
    return y.numpy()


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=False)
    previous=R.N.read(A.OUT/'completion.json');audit=R.N.read(A.OUT/'completion_audit.json')
    assert audit['passed'] and audit['completion_sha']==O.sha(A.OUT/'completion.json');O.verify(previous['manifest'])
    cachepath=Q.PRIOR/'replay_inputs.pt';c2=R.N.read(Q.PRIOR/'completion.json');assert O.sha(cachepath)==c2['artifacts']['replay_inputs.pt']
    files=[cachepath,Q.PRIOR/'completion.json',A.OUT/'completion.json',A.OUT/'completion_audit.json',Path(__file__),
           ROOT/'tools/test_obs5_natural_history_20260909.py',ROOT/'docs/obs5_natural_history_protocol_20260909.md',
           ROOT/'tools/obs4_slow_readout_20260909.py',ROOT/'tools/obs3_constant_input_20260908.py',ROOT/'tools/obs2_investigations_20260908.py']
    manifest=dict(sources={str(p.relative_to(ROOT)).replace('\\','/'):O.sha(p) for p in files},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),torch=torch.__version__,numpy=np.__version__,exploratory=True)
    R.C.save(OUT/'manifest.json',manifest)
    try:
        cache=R.N.load(cachepath);streams=cache['streams'];keys=list(streams);lookup={(c['trial'],c['world'],c['orientation'],c['control']):c for c in cache['cases']}
        prefix=np.array([streams[k][:16] for k in keys]);allrows=[];history_error=0.
        for trial in R.TRIALS:
            for kind in ('final','initial'):
                p=cache['checkpoints'][trial][kind];model=Q.native(p);hist=replay(model,prefix,np.zeros((8,32)))[:,-1]
                for i,(world,oi) in enumerate(keys):
                    saved=lookup[trial,world,oi,'intact' if kind=='final' else 'untrained']['initial_hidden']
                    error=float(np.max(abs(hist[i]-saved)));history_error=max(history_error,error);assert error<=1e-5
                    if kind=='final':
                        donor=hist[keys.index((world,1-oi))];err=float(np.max(abs(donor-lookup[trial,world,oi,'swapped']['initial_hidden'])))
                        history_error=max(history_error,err);assert err<=1e-5
                initial=[];drivers=[];metadata=[];jac=[];midpoints=[];deltas=[];components_saved=[]
                for i,(world,oi) in enumerate(keys):
                    ha=hist[i];hb=hist[keys.index((world,1-oi))];m=(ha+hb)/2;delta=ha-hb
                    _,j=P.step(streams[world,oi][16],m,p,True);arms,meta=components(j,p['readout.weight'],delta)
                    starts=np.array([m+sign*scale*arm/2 for scale in SCALES for arm in arms for sign in (1,-1)])
                    for driver,x in [('teacher',streams[world,oi][16:512]),('constant',np.tile(streams[world,oi][16],(496,1)))]:
                        initial.append(starts);drivers.append(x);metadata.append(dict(trial=trial,world=world,orientation=oi,weights=kind,driver=driver,geometry=meta))
                        jac.append(j);midpoints.append(m);deltas.append(delta);components_saved.append(arms)
                initial=np.array(initial);drivers=np.array(drivers)
                y=replay(model,np.repeat(drivers,24,axis=0),initial.reshape(-1,32)).reshape(16,24,496,32)
                states=np.concatenate([initial[:,:,None],y],axis=2).reshape(16,2,6,2,497,32)
                logits=A.logits(states,p);pred=A.sigmoid(logits)
                signed={name:value[:,:,:,0]-value[:,:,:,1] for name,value in [('hidden',states),('logit',logits),('prediction',pred)]}
                gains={name:np.linalg.norm(value,axis=-1) for name,value in signed.items()}
                assert np.max(abs(signed['logit'][:,:,4,0]))<=1e-10
                assert np.max(gains['hidden'][:,:,5])<=1e-10
                for arm in (1,2,3):assert np.max(abs(signed['logit'][:,:,arm,0]-signed['logit'][:,:,0,0]))<=1e-10
                saved=dict(prefix=prefix,histories=hist,drivers=drivers,jacobians=np.array(jac),midpoints=np.array(midpoints),
                           deltas=np.array(deltas),arm_differences=np.array(components_saved),checkpoints=states[:,:,:,:,TIMES])
                for name in signed:saved[name+'_gaps']=gains[name];saved['final_'+name+'_vectors']=signed[name][:,:,:,-1]
                np.savez_compressed(OUT/f'{trial}_{kind}.npz',**saved)
                rows=[]
                for ci,meta in enumerate(metadata):
                    row=dict(**meta,**decisions(gains['hidden'][ci,:,:,-1],gains['prediction'][ci,:,:,-1],meta['geometry']['null_norm']),arms={})
                    for ai,arm in enumerate(ARMS):
                        row['arms'][arm]=[{name:gains[name][ci,si,ai,list(TIMES)].tolist() for name in gains} for si in range(2)]
                    rows.append(row)
                allrows.extend(rows);R.C.save(OUT/f'{trial}_{kind}.json',dict(cases=rows));O.verify(manifest);print('completed',trial,kind,flush=True)
        assert len(allrows)==256
        R.C.save(OUT/'results.json',dict(cases=allrows,times=TIMES,scales=SCALES,arms=ARMS,max_history_reconstruction_error=history_error))
        O.verify(manifest);artifacts={p.name:O.sha(p) for p in OUT.iterdir() if p.is_file()}
        R.C.save(OUT/'completion.json',dict(kind='OBS5_COMPLETE',manifest=manifest,artifacts=artifacts,functional_verdict_changed=False));print('OBS5 complete',flush=True)
    except Exception as exc:R.C.save(OUT/'invalid.json',dict(error=repr(exc)));raise


if __name__=='__main__':main()
