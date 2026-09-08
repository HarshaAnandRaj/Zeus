"""Independent saved-array and manual/native replay audit for OBS3."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs3_constant_input_20260908 as Q


def close(a,b):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-7)


def stats(h):
    delta=h[1:]-h[:-1]
    return dict(max_step=float(np.sqrt((delta*delta).sum(1)).max()),rms=float(np.sqrt(np.var(h,axis=0).mean())))


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    completion=Q.R.N.read(Q.OUT/'completion.json');Q.O.verify(completion['manifest'])
    for name,digest in completion['artifacts'].items():assert Q.O.sha(Q.OUT/name)==digest
    cache=Q.R.N.load(Q.PRIOR/'replay_inputs.pt');rows=Q.R.N.read(Q.OUT/'results.json')['cases']
    assert len(rows)==256
    checks={'source_and_artifact_identity':True};branches=0;manual=0;autograd=0;manual_max=0.;jac_max=0.
    for t in Q.R.TRIALS:
        for kind in ('final','initial'):
            cases=[c for c in cache['cases'] if c['trial']==t and (c['control']=='untrained')==(kind=='initial')]
            records=Q.R.N.read(Q.OUT/f'{t}_{kind}.json')['cases'];assert len(cases)==len(records)
            d=np.load(Q.OUT/f'{t}_{kind}.npz');p=cache['checkpoints'][t][kind]
            expected_initial=np.array([[c['h'][-1],np.zeros(32),c['h'][-1]+1e-5*Q.DIRECTION,c['h'][-1]-1e-5*Q.DIRECTION] for c in cases]).reshape(-1,32)
            np.testing.assert_array_equal(d['initial'],expected_initial)
            np.testing.assert_array_equal(d['x'],np.repeat(np.array([c['x'][-1] for c in cases]),4,axis=0))
            for ci,(case,row) in enumerate(zip(cases,records)):
                assert all(case[k]==row[k] for k in ('trial','world','orientation','control'))
                assert row==next(v for v in rows if all(v[k]==row[k] for k in ('trial','world','orientation','control')))
                for si,start in enumerate(Q.STARTS):
                    j=ci*4+si;v=row['branches'][start];tail=d['tail'][j];s=stats(tail)
                    for key,val in s.items():close(v['tail'][key],val)
                    for time in Q.TIMES:
                        h=d[f'window_{time}'][j];assert h.shape==(193 if time==192 else 256,32)
                        np.testing.assert_array_equal(h[-1],d[f'state_{time}'][j])
                        for key,val in stats(h).items():close(v['windows'][str(time)][key],val)
                    np.testing.assert_array_equal(tail[-256:],d['window_16384'][j])
                    if s['max_step']<=1e-10:label='SETTLED'
                    elif all(v['windows']['16384'][k]<=.1*v['windows']['8192'][k] for k in ('rms','max_step')):label='DECAYING'
                    else:label='PERSISTENT_AT_HORIZON'
                    assert label==v['label']
                    ret=[np.sqrt(np.mean(np.sum((tail[k:]-tail[:-k])**2,axis=1))) for k in Q.LAGS];close(v['return_rms'],ret)
                    nxt,jac=Q.P.step(d['x'][j],tail[-1],p,True)
                    close(v['residual'],np.linalg.norm(nxt-tail[-1]));close(v['spectral_radius'],max(abs(np.linalg.eigvals(jac))))
                    branches+=1
                for time in (0,)+Q.TIMES:
                    states=d['initial'] if time==0 else d[f'state_{time}']
                    for si,start in enumerate(Q.STARTS[1:],1):close(row['separations'][str(time)][start],np.linalg.norm(states[ci*4+si]-states[ci*4]))
            # Fixed first world/orientation baseline; intact for trained weights.
            ci=next(i for i,c in enumerate(cases) if c['world']==202678000 and c['orientation']==0 and c['control']==('intact' if kind=='final' else 'untrained'))
            j=ci*4;x=d['x'][j];h=d['initial'][j].copy();tail=[]
            for time in range(1,16385):
                h,_=Q.P.step(x,h,p)
                if time in Q.TIMES:
                    manual_max=max(manual_max,float(np.max(abs(h-d[f'state_{time}'][j]))));close(h,d[f'state_{time}'][j])
                if time>15360:tail.append(h.copy())
            close(tail,d['tail'][j]);manual+=1
            m=Q.native(p);initial=torch.tensor(d['tail'][j,-1],requires_grad=True);xx=torch.tensor(x)[None,None]
            numeric=torch.autograd.functional.jacobian(lambda hh:m(xx,hh[None,None])[1].reshape(32),initial).numpy()
            _,analytic=Q.P.step(x,initial.detach().numpy(),p,True)
            jac_max=max(jac_max,float(np.max(abs(numeric-analytic))));close(numeric,analytic);autograd+=1
        print('audited',t,flush=True)
    assert branches==1024 and manual==autograd==16
    checks.update(all_1024_branch_statistics_and_labels=True,all_256_start_separation_curves=True,
                  all_saved_start_and_input_identities=True,all_final_residuals_and_jacobian_spectra=True,
                  sixteen_full_horizon_manual_replays=True,sixteen_native_autograd_jacobians=True)
    Q.O.verify(completion['manifest']);checks['final_source_identity']=True
    result=dict(kind='OBS3_AUDIT',passed=all(checks.values()),checks=checks,branches=branches,
                manual_replays=manual,autograd_jacobians=autograd,max_manual_checkpoint_error=manual_max,
                max_autograd_jacobian_error=jac_max,completion_sha=Q.O.sha(Q.OUT/'completion.json'),audit_sha=Q.O.sha(Path(__file__)))
    Q.R.C.save(Q.OUT/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
