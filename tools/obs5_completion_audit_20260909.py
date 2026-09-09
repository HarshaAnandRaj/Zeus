"""Independent full manual replay and intervention audit for OBS5."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs5_natural_history_20260909 as B


def close(a,b,atol=1e-10):np.testing.assert_allclose(a,b,rtol=1e-7,atol=atol)


def manual(p,x,h):
    states=[]
    for time in range(x.shape[1]):
        ih=x[:,time]@p['gru.weight_ih_l0'].T+p['gru.bias_ih_l0'];hh=h@p['gru.weight_hh_l0'].T+p['gru.bias_hh_l0']
        r=1/(1+np.exp(-(ih[:,:32]+hh[:,:32])));z=1/(1+np.exp(-(ih[:,32:64]+hh[:,32:64])))
        candidate=np.tanh(ih[:,64:]+r*hh[:,64:]);h=z*h+(1-z)*candidate;states.append(h.copy())
    return np.stack(states,axis=1)


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    complete=B.R.N.read(B.OUT/'completion.json');B.O.verify(complete['manifest'])
    for n,digest in complete['artifacts'].items():assert B.O.sha(B.OUT/n)==digest
    results=B.R.N.read(B.OUT/'results.json');allrows=results['cases'];assert len(allrows)==256
    cache=B.R.N.load(B.Q.PRIOR/'replay_inputs.pt');streams=cache['streams'];keys=list(streams);prefix=np.array([streams[k][:16] for k in keys])
    lookup={(c['trial'],c['world'],c['orientation'],c['control']):c for c in cache['cases']}
    count=0;histcount=0;jaccount=0;maxstate=0.;maxgap=0.;maxjac=0.;maxhistory=0.
    checks={'source_and_artifact_identity':True}
    for trial in B.R.TRIALS:
        for kind in ('final','initial'):
            p=cache['checkpoints'][trial][kind];d=np.load(B.OUT/f'{trial}_{kind}.npz');rows=B.R.N.read(B.OUT/f'{trial}_{kind}.json')['cases']
            np.testing.assert_array_equal(prefix,d['prefix']);hist=manual(p,prefix,np.zeros((8,32)))[:,-1];close(hist,d['histories']);histcount+=8
            for i,(world,oi) in enumerate(keys):
                e=float(np.max(abs(d['histories'][i]-lookup[trial,world,oi,'intact' if kind=='final' else 'untrained']['initial_hidden'])));maxhistory=max(maxhistory,e);assert e<=1e-5
                if kind=='final':
                    e=float(np.max(abs(d['histories'][keys.index((world,1-oi))]-lookup[trial,world,oi,'swapped']['initial_hidden'])));maxhistory=max(maxhistory,e);assert e<=1e-5
            initial=d['checkpoints'][:,:,:,:,0].reshape(16,24,32)
            replayed=manual(p,np.repeat(d['drivers'],24,axis=0),initial.reshape(-1,32)).reshape(16,24,496,32)
            h=np.concatenate([initial[:,:,None],replayed],axis=2).reshape(16,2,6,2,497,32)
            maxstate=max(maxstate,float(np.max(abs(h[:,:,:,:,B.TIMES]-d['checkpoints']))));close(h[:,:,:,:,B.TIMES],d['checkpoints'])
            logits=h@p['readout.weight'].T+p['readout.bias'];pred=1/(1+np.exp(-logits))
            for name,v in (('hidden',h),('logit',logits),('prediction',pred)):
                difference=v[:,:,:,0]-v[:,:,:,1];gap=np.linalg.norm(difference,axis=-1)
                maxgap=max(maxgap,float(np.max(abs(gap-d[name+'_gaps']))));close(gap,d[name+'_gaps'],1e-8)
                close(difference[:,:,:,-1],d['final_'+name+'_vectors'],1e-8)
            for ci,row in enumerate(rows):
                pi=ci//2;world,oi=keys[pi];driver=('teacher','constant')[ci%2]
                assert (row['trial'],row['world'],row['orientation'],row['weights'],row['driver'])==(trial,world,oi,kind,driver)
                assert row==next(r for r in allrows if all(r[k]==row[k] for k in ('trial','world','orientation','weights','driver')))
                xx=streams[world,oi][16:512] if driver=='teacher' else np.tile(streams[world,oi][16],(496,1));np.testing.assert_array_equal(xx,d['drivers'][ci])
                ha=d['histories'][pi];hb=d['histories'][keys.index((world,1-oi))];delta=ha-hb;m=(ha+hb)/2
                close(delta,d['deltas'][ci]);close(m,d['midpoints'][ci]);_,j=B.P.step(xx[0],m,p,True);close(j,d['jacobians'][ci])
                _,sv,vw=np.linalg.svd(p['readout.weight'],full_matrices=True);assert np.sum(sv>sv[0]*32*np.finfo(float).eps)==9
                n=vw[9:].T;dn=n@n.T@delta;geometry=row['geometry'];v=np.array(geometry['direction']);u=np.dot(v,delta)*v;c=np.array(geometry['control_shift'])
                close(np.linalg.norm(v),1.);assert np.linalg.norm(p['readout.weight']@v)<1e-12
                j64=np.linalg.matrix_power(j,64);singular=np.linalg.svd(j64@n,compute_uv=False);close(np.linalg.norm(j64@v),singular[0]);close(geometry['null_singular_values'],singular)
                close(geometry['null_difference'],dn);close(geometry['slow_component'],u);close(np.linalg.norm(c),np.linalg.norm(u));close(np.dot(c,v),0.)
                assert np.linalg.norm(p['readout.weight']@c)<1e-10
                projection=lambda z:n@(n.T@z)-v*np.dot(v,z)
                cv=projection(B.Q.DIRECTION)
                if np.linalg.norm(cv)<=1e-12:cv=next(projection(e) for e in np.eye(32) if np.linalg.norm(projection(e))>1e-12)
                cv/=np.linalg.norm(cv)
                if np.dot(cv,delta)<0:cv=-cv
                close(c,np.linalg.norm(u)*cv)
                for k,val in [('delta_norm',np.linalg.norm(delta)),('null_norm',np.linalg.norm(dn)),('slow_norm',np.linalg.norm(u))]:close(geometry[k],val)
                arms=np.array([delta,delta-u,delta-c,delta-dn,dn,np.zeros(32)]);close(arms,d['arm_differences'][ci])
                starts=np.array([m+sign*scale*arm/2 for scale in B.SCALES for arm in arms for sign in (1,-1)]);close(starts,initial[ci])
                actual=d['checkpoints'][ci];l=actual@p['readout.weight'].T+p['readout.bias'];dif=l[:,:,0]-l[:,:,1]
                assert np.max(abs(dif[:,4,0]))<=1e-10
                for arm in (1,2,3):assert np.max(abs(dif[:,arm,0]-dif[:,0,0]))<=1e-10
                assert np.max(d['hidden_gaps'][ci,:,5])<=1e-10
                nullnorm=np.linalg.norm(dn);route=nullnorm>=1e-8;specific=True
                for si,scale in enumerate(B.SCALES):
                    size=scale*nullnorm;hg=d['hidden_gaps'][ci,si,:,-1];yg=d['prediction_gaps'][ci,si,:,-1]
                    route=bool(route and size>0 and hg[4]/size>=.01 and yg[4]/size>=1e-4)
                    red=None if yg[0]==0 else 1-yg[1]/yg[0];ctrl=None if yg[0]==0 else 1-yg[2]/yg[0]
                    specific=bool(specific and yg[0]>=scale*1e-5 and red is not None and red>=.25 and red-ctrl>=.10)
                    actual=row['reductions'][si];assert actual['scale']==scale
                    if red is None:assert actual['slow_reduction'] is actual['matched_reduction'] is None
                    else:close(actual['slow_reduction'],red);close(actual['matched_reduction'],ctrl)
                    for ai,arm in enumerate(B.ARMS):
                        for name in ('hidden','logit','prediction'):close(row['arms'][arm][si][name],d[name+'_gaps'][ci,si,ai,list(B.TIMES)])
                assert row['null_history_route']==route and row['slow_null_reduction']==specific;count+=1
            ci=next(i for i,r in enumerate(rows) if r['world']==202678000 and r['orientation']==0 and r['driver']=='teacher')
            model=B.Q.native(p);state=torch.tensor(d['midpoints'][ci],requires_grad=True);xx=torch.tensor(d['drivers'][ci,0])[None,None]
            jauto=torch.autograd.functional.jacobian(lambda hh:model(xx,hh[None,None])[1].reshape(32),state).numpy()
            maxjac=max(maxjac,float(np.max(abs(jauto-d['jacobians'][ci]))));close(jauto,d['jacobians'][ci]);jaccount+=1
        print('audited',trial,flush=True)
    assert count==256 and histcount==128 and jaccount==16;close(maxhistory,results['max_history_reconstruction_error'])
    checks.update(all_128_history_reconstructions=True,all_6144_manual_branch_replays=True,all_saved_curves_and_checkpoint_states=True,
                  all_component_and_control_geometry=True,all_immediate_output_controls=True,all_decisions_and_summaries=True,
                  sixteen_native_autograd_jacobians=True)
    B.O.verify(complete['manifest']);checks['final_source_identity']=True
    result=dict(kind='OBS5_AUDIT',passed=all(checks.values()),checks=checks,combinations=count,branches=6144,
                max_state_error=maxstate,max_gap_error=maxgap,max_jacobian_error=maxjac,max_history_reconstruction_error=maxhistory,
                completion_sha=B.O.sha(B.OUT/'completion.json'),audit_sha=B.O.sha(Path(__file__)))
    B.R.C.save(B.OUT/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
