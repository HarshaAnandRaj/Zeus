"""OBS6 full manual replay, signed-vector audit and end-to-end autograd."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs6_cancellation_20260909 as C
from tools.obs5_completion_audit_20260909 import manual


def close(a,b,atol=1e-10):np.testing.assert_allclose(a,b,rtol=1e-7,atol=atol)


def inspect_vectors(v):
    norms=np.sqrt((v*v).sum(1));cos=None if min(norms[1:])<1e-6 else float((v[1]*v[2]).sum()/(norms[1]*norms[2]))
    ratio=None if norms[0]<1e-6 else float(norms[1]/norms[0])
    residual=float(np.sqrt(np.sum((v[0]-v[1]-v[2])**2))/max(norms[0],norms[1]+norms[2],1e-6))
    return dict(norms=norms.tolist(),cosine=cos,amplification=ratio,additivity=residual)


def verify_analysis(actual,g,t):
    tangent=inspect_vectors(t);finite=[inspect_vectors(v) for v in g]
    for expected,recorded in [(tangent,actual['tangent'])]+list(zip(finite,actual['finite'])):
        for k in expected:
            if expected[k] is None:assert recorded[k] is None
            else:close(expected[k],recorded[k])
    agreement=[bool(all(np.linalg.norm(v[i]-t[i])<=.05*max(np.linalg.norm(t[i]),1e-6) for i in range(3))) for v in g]
    assert actual['agreement_by_level']==agreement
    small=all(agreement[-2:]);full=agreement[0]
    local=bool(min(tangent['norms'])>=1e-6 and tangent['cosine'] is not None and tangent['cosine']<=-.1 and tangent['amplification']>=1.1)
    amp=bool(finite[0]['amplification'] is not None and finite[0]['amplification']>=1.1)
    assert [actual[k] for k in ('small_scale_agreement','full_scale_agreement','local_cancellation','full_scale_amplification')]==[small,full,local,amp]
    label=('NO_RESOLVED_FULL_SCALE_AMPLIFICATION' if not amp else 'LOCAL_COMPARISON_NOT_VERIFIED' if not small
           else 'LINEAR_EXPLAINS_FULL_SCALE' if local and full else 'LINEAR_CORE_WITH_FINITE_NONLINEARITY' if local else 'FINITE_SCALE_ONLY')
    assert actual['label']==label


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    completion=C.R.N.read(C.OUT/'completion.json');C.O.verify(completion['manifest'])
    for name,digest in completion['artifacts'].items():assert C.O.sha(C.OUT/name)==digest
    cache=C.R.N.load(C.Q.PRIOR/'replay_inputs.pt');result=C.R.N.read(C.OUT/'results.json');allrows=result['cases'];assert len(allrows)==256
    checks={'source_and_artifact_identity':True};count=0;autocount=0;maxstate=0.;maxresponse=0.;maxtangent=0.;maxprior=0.
    for trial in C.R.TRIALS:
        for kind in ('final','initial'):
            d=np.load(C.OUT/f'{trial}_{kind}.npz');old=np.load(C.B.OUT/f'{trial}_{kind}.npz');rows=C.R.N.read(C.OUT/f'{trial}_{kind}.json')['cases'];p=cache['checkpoints'][trial][kind]
            priorrows=C.R.N.read(C.B.OUT/f'{trial}_{kind}.json')['cases']
            for key in ('midpoints','drivers'):np.testing.assert_array_equal(d[key],old[key])
            np.testing.assert_array_equal(d['directions'],old['arm_differences'][:,[0,3,4]])
            close(d['directions'][:,0],d['directions'][:,1]+d['directions'][:,2])
            initial=np.array([[m]+[m+sign*scale*v/2 for scale in C.LEVELS for v in ds for sign in (1,-1)] for m,ds in zip(d['midpoints'],d['directions'])])
            np.testing.assert_array_equal(initial,d['checkpoints'][:,:,0])
            replayed=manual(p,np.repeat(d['drivers'],37,axis=0),initial.reshape(-1,32)).reshape(16,37,496,32)
            h=np.concatenate([initial[:,:,None],replayed],axis=2);maxstate=max(maxstate,float(np.max(abs(h[:,:,C.TIMES]-d['checkpoints']))));close(h[:,:,C.TIMES],d['checkpoints']);close(h[:,0],d['baseline'])
            for li in range(2):
                for di,arm in enumerate((0,3,4)):
                    b=1+li*6+di*2;observed=d['checkpoints'][:,b:b+2];expected=old['checkpoints'][:,li,arm]
                    maxprior=max(maxprior,float(np.max(abs(observed-expected))));close(observed,expected)
            logits=h@p['readout.weight'].T+p['readout.bias'];pred=1/(1+np.exp(-logits))
            for name,values in (('logit',logits),('prediction',pred)):
                g=np.empty((16,6,3,497,9))
                for li,scale in enumerate(C.LEVELS):
                    for di in range(3):
                        b=1+li*6+di*2;g[:,li,di]=(values[:,b]-values[:,b+1])/scale
                norms=np.linalg.norm(g,axis=-1);maxresponse=max(maxresponse,float(np.max(abs(norms-d[name+'_norms']))));close(norms,d[name+'_norms'],1e-7);close(g[:,:,:,-1],d['final_'+name+'_vectors'],1e-7)
            for ci,row in enumerate(rows):
                assert all(row[k]==priorrows[ci][k] for k in ('trial','weights','world','orientation','driver'))
                assert row==next(r for r in allrows if all(row[k]==r[k] for k in ('trial','weights','world','orientation','driver')))
                direction=d['directions'][ci].T.copy()
                for time in range(497):
                    if time:
                        _,j=C.P.step(d['drivers'][ci,time-1],d['baseline'][ci,time-1],p,True);direction=j@direction
                    lg=p['readout.weight']@direction;y=1/(1+np.exp(-(d['baseline'][ci,time]@p['readout.weight'].T+p['readout.bias'])))
                    pg=(y*(1-y))[:,None]*lg
                    close(lg.T,d['logit_tangents'][ci,:,time],1e-8);close(pg.T,d['prediction_tangents'][ci,:,time],1e-8)
                    assert np.max(abs(direction[:,0]-direction[:,1]-direction[:,2]))<=1e-10
                for name in ('logit','prediction'):verify_analysis(row[name],d['final_'+name+'_vectors'][ci],d[name+'_tangents'][ci,:,-1])
                count+=1
            ci=next(i for i,r in enumerate(rows) if r['world']==202678000 and r['orientation']==0 and r['driver']=='teacher')
            model=C.Q.native(p);xx=torch.tensor(d['drivers'][ci])[None];m=torch.tensor(d['midpoints'][ci],requires_grad=True)
            w=torch.tensor(p['readout.weight']);bias=torch.tensor(p['readout.bias'])
            def endmap(hh):
                _,last=model(xx,hh[None,None]);lg=last.reshape(32)@w.T+bias
                return torch.cat([lg,torch.sigmoid(lg)])
            jac=torch.autograd.functional.jacobian(endmap,m).numpy();mapped=(jac@d['directions'][ci].T).T
            for name,v in [('logit',mapped[:,:9]),('prediction',mapped[:,9:])]:
                expected=d[name+'_tangents'][ci,:,-1];maxtangent=max(maxtangent,float(np.max(abs(v-expected))));close(v,expected,1e-8)
            autocount+=1
        print('audited',trial,flush=True)
    assert count==256 and autocount==16;close(maxprior,result['max_prior_checkpoint_error'])
    checks.update(all_9472_manual_branch_replays=True,all_saved_curves_and_checkpoints=True,all_decompositions_and_prior_agreement=True,
                  all_analytic_tangent_paths_and_additivity=True,all_vector_metrics_and_labels=True,sixteen_full_horizon_autograd_maps=True)
    C.O.verify(completion['manifest']);checks['final_source_identity']=True
    audit=dict(kind='OBS6_AUDIT',passed=all(checks.values()),checks=checks,combinations=count,branches=9472,
               max_state_error=maxstate,max_response_error=maxresponse,max_end_to_end_tangent_error=maxtangent,max_prior_checkpoint_error=maxprior,
               audit_sha=C.O.sha(Path(__file__)),manual_source_sha=C.O.sha(ROOT/'tools/obs5_completion_audit_20260909.py'),completion_sha=C.O.sha(C.OUT/'completion.json'))
    C.R.C.save(C.OUT/'completion_audit.json',audit);print(audit,flush=True)


if __name__=='__main__':main()
