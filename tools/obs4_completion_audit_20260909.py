"""Audit every OBS4 branch by an independent batched manual GRU replay."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs4_slow_readout_20260909 as A


def close(a,b,atol=1e-10):np.testing.assert_allclose(a,b,atol=atol,rtol=1e-7)


def manual(p,x,h):
    ih=x@p['gru.weight_ih_l0'].T+p['gru.bias_ih_l0'];parts=[]
    for _ in range(512):
        hh=h@p['gru.weight_hh_l0'].T+p['gru.bias_hh_l0']
        r=1/(1+np.exp(-(ih[:,:32]+hh[:,:32])))
        z=1/(1+np.exp(-(ih[:,32:64]+hh[:,32:64])))
        n=np.tanh(ih[:,64:]+r*hh[:,64:]);h=z*h+(1-z)*n;parts.append(h.copy())
    return np.stack(parts,axis=1)


def agree(a,b):return np.linalg.norm(a-b)<=max(1e-6,.01*max(np.linalg.norm(a),np.linalg.norm(b)))


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    completion=A.R.N.read(A.OUT/'completion.json');A.O.verify(completion['manifest'])
    for name,digest in completion['artifacts'].items():assert A.O.sha(A.OUT/name)==digest
    cache=A.R.N.load(A.Q.PRIOR/'replay_inputs.pt');allrows=A.R.N.read(A.OUT/'results.json')['cases'];assert len(allrows)==512
    checks={'source_and_artifact_identity':True};combinations=0;jac_count=0;maxstate=0.;maxgain=0.;maxjac=0.
    for trial in A.R.TRIALS:
        identities=[];endpoints=[];xs=[]
        for origin in ('final','initial'):
            prior=A.R.N.read(A.Q.OUT/f'{trial}_{origin}.json')['cases'];old=np.load(A.Q.OUT/f'{trial}_{origin}.npz')
            identities.extend([{k:r[k] for k in ('trial','world','orientation','control')} for r in prior]);endpoints.extend(old['state_16384'][::4]);xs.extend(old['x'][::4])
        endpoints=np.array(endpoints);xs=np.array(xs)
        for kind in ('final','initial'):
            d=np.load(A.OUT/f'{trial}_{kind}.npz');rows=A.R.N.read(A.OUT/f'{trial}_{kind}.json')['cases'];p=cache['checkpoints'][trial][kind]
            np.testing.assert_array_equal(d['x'],xs)
            ds=d['directions'];initial=np.array([[h]+[h+sign*e*v for v in direction for e in A.EPS for sign in (1,-1)] for h,direction in zip(endpoints,ds)])
            np.testing.assert_array_equal(d['checkpoints'][:,:,0],initial)
            y=manual(p,np.repeat(xs,13,axis=0),initial.reshape(-1,32)).reshape(32,13,512,32)
            h=np.concatenate([initial[:,:,None],y],axis=2);maxstate=max(maxstate,float(np.max(abs(h[:,:,A.TIMES]-d['checkpoints']))))
            close(h[:,:,A.TIMES],d['checkpoints']);close(h[:,0],d['baseline'])
            logits=h@p['readout.weight'].T+p['readout.bias'];pred=1/(1+np.exp(-logits))
            responses={}
            for name,v in (('hidden',h),('logit',logits),('prediction',pred)):
                resp=np.empty((32,3,2,513,v.shape[-1]))
                for di in range(3):
                    for ei,eps in enumerate(A.EPS):
                        plus=1+4*di+2*ei;resp[:,di,ei]=(v[:,plus]-v[:,plus+1])/(2*eps)
                gain=np.sqrt((resp**2).sum(-1));maxgain=max(maxgain,float(np.max(abs(gain-d[f'{name}_gains']))))
                close(gain,d[f'{name}_gains'],1e-6);close(resp[:,:,:,-1],d[f'final_{name}_responses'],1e-6);responses[name]=resp
            for ci,(identity,row) in enumerate(zip(identities,rows)):
                assert all(identity[k]==row[k] for k in identity) and row['weights']==kind
                assert row==next(r for r in allrows if r['weights']==kind and all(r[k]==identity[k] for k in identity))
                close(np.linalg.norm(ds[ci],axis=1),np.ones(3));assert np.linalg.norm(p['readout.weight']@ds[ci,2])<1e-12
                _,j=A.P.step(xs[ci],endpoints[ci],p,True);close(j,d['jacobians'][ci]);j64=np.linalg.matrix_power(j,64)
                _,sv,vw=np.linalg.svd(p['readout.weight'],full_matrices=True);n=vw[9:].T
                close(np.linalg.norm(j64@ds[ci,0]),np.linalg.svd(j64,compute_uv=False)[0])
                close(np.linalg.norm(j@ds[ci,1]),np.linalg.svd(j,compute_uv=False)[-1])
                close(np.linalg.norm(j64@ds[ci,2]),np.linalg.svd(j64@n,compute_uv=False)[0])
                for key,matrix in [('slow_singular_values',j64),('one_step_singular_values',j),('null_singular_values',j64@n)]:close(row['direction_metadata'][key],np.linalg.svd(matrix,compute_uv=False))
                for di,name in enumerate(A.NAMES):
                    v=row['directions'][name]
                    for target in ('hidden','prediction'):
                        final=d[f'final_{target}_responses'][ci,di]
                        assert v[f'epsilon_agreement_{target}']==bool(agree(*final))
                    for ei,level in enumerate(v['levels']):
                        assert level['epsilon']==A.EPS[ei]
                        for target in ('hidden','logit','prediction'):close(level[f'{target}_at_checkpoints'],d[f'{target}_gains'][ci,di,ei,list(A.TIMES)])
                        gain=d['prediction_gains'][ci,di,ei];close(level['max_future_prediction'],max(gain[1:]));assert level['first_max_step']==1+int(np.argmax(gain[1:]))
                        plus=1+di*4+ei*2;nativefinal=d['checkpoints'][ci,:,-1]
                        close(level['final_midpoint_shift'],np.linalg.norm((nativefinal[plus]+nativefinal[plus+1])/2-nativefinal[0])/A.EPS[ei])
                rh=d['final_hidden_responses'][ci,2];ry=d['final_prediction_responses'][ci,2]
                decision=all(np.linalg.norm(v)>=.01 for v in rh) and all(np.linalg.norm(v)>=1e-4 for v in ry) and agree(*rh) and agree(*ry)
                assert row['persistent_readable_marker']==bool(decision);assert d['logit_gains'][ci,2,:,0].max()<=1e-8
                combinations+=1
            ci=next(i for i,r in enumerate(identities) if r['world']==202678000 and r['orientation']==0 and r['control']=='intact')
            model=A.Q.native(p);state=torch.tensor(endpoints[ci],requires_grad=True);xx=torch.tensor(xs[ci])[None,None]
            autograd=torch.autograd.functional.jacobian(lambda hh:model(xx,hh[None,None])[1].reshape(32),state).numpy()
            maxjac=max(maxjac,float(np.max(abs(autograd-d['jacobians'][ci]))));close(autograd,d['jacobians'][ci]);jac_count+=1
        print('audited',trial,flush=True)
    assert combinations==512 and jac_count==16
    checks.update(all_6656_manual_branch_replays=True,all_response_curves_and_checkpoints=True,
                  all_directions_and_singular_values=True,all_start_and_input_identities=True,
                  all_marker_decisions_and_summaries=True,sixteen_native_autograd_jacobians=True)
    A.O.verify(completion['manifest']);checks['final_source_identity']=True
    result=dict(kind='OBS4_AUDIT',passed=all(checks.values()),checks=checks,combinations=combinations,branches=6656,
                max_state_error=maxstate,max_response_gain_error=maxgain,max_autograd_jacobian_error=maxjac,
                audit_sha=A.O.sha(Path(__file__)),completion_sha=A.O.sha(A.OUT/'completion.json'))
    A.R.C.save(A.OUT/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
