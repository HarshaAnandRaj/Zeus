"""OBS3 fixed longer constant-input investigation; no training or world runs."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
from pathlib import Path
import sys
import subprocess
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import obs2_investigations_20260908 as P
O, R = P.O, P.R
OUT = ROOT / 'runs/obs3_20260908'
PRIOR = ROOT / 'runs/obs2r_20260908'
TIMES = (192, 512, 2048, 8192, 16384)
STARTS = ('baseline', 'zero', 'plus', 'minus')
DIRECTION = np.tile([1., -1.], 16) / np.sqrt(32)
LAGS = tuple(range(1, 65)) + (128, 256)


def window(h):
    return dict(max_step=float(np.linalg.norm(np.diff(h, axis=0), axis=1).max()),
                rms=float(np.sqrt(np.mean((h-h.mean(0))**2))))


def classify(tail, earlier, final):
    if tail['max_step'] <= 1e-10: return 'SETTLED'
    if final['max_step'] <= .1*earlier['max_step'] and final['rms'] <= .1*earlier['rms']: return 'DECAYING'
    return 'PERSISTENT_AT_HORIZON'


def native(p):
    model = torch.nn.GRU(10, 32, batch_first=True).double().eval()
    model.load_state_dict({k.removeprefix('gru.'): torch.tensor(v) for k,v in p.items() if k.startswith('gru.')})
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model


def advance(model, x, h, n):
    with torch.no_grad():
        y, _ = model(torch.tensor(x)[:, None].expand(-1,n,-1).contiguous(), torch.tensor(h)[None])
    return y.numpy()


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=False)
    completion = R.N.read(PRIOR/'completion.json'); audit = R.N.read(PRIOR/'completion_audit.json')
    assert audit['passed'] and audit['source_completion_sha'] == O.sha(PRIOR/'completion.json')
    for name in ('replay_inputs.pt', 'o3.json'): assert O.sha(PRIOR/name) == completion['artifacts'][name]
    sources = {str(p.relative_to(ROOT)).replace('\\','/'):O.sha(p) for p in
               [PRIOR/'completion.json', PRIOR/'completion_audit.json', PRIOR/'replay_inputs.pt', PRIOR/'o3.json',
                ROOT/'docs/obs3_constant_input_protocol_20260908.md', Path(__file__),
                ROOT/'tools/test_obs3_constant_input_20260908.py', ROOT/'tools/obs2_investigations_20260908.py']}
    manifest = dict(sources=sources, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    exploratory=True,torch=torch.__version__,numpy=np.__version__,horizon=16384)
    R.C.save(OUT/'manifest.json',manifest)
    try:
        cache=R.N.load(PRIOR/'replay_inputs.pt'); old=R.N.read(PRIOR/'o3.json')['cases']; results=[]
        for trial in R.TRIALS:
            for kind in ('final','initial'):
                cases=[c for c in cache['cases'] if c['trial']==trial and (c['control']=='untrained')==(kind=='initial')]
                p=cache['checkpoints'][trial][kind]; model=native(p)
                initial=np.array([[c['h'][-1],np.zeros(32),c['h'][-1]+1e-5*DIRECTION,c['h'][-1]-1e-5*DIRECTION] for c in cases]).reshape(-1,32)
                x=np.repeat(np.array([c['x'][-1] for c in cases]),4,axis=0)
                saved={'initial':initial,'x':x}; h=initial.copy(); buffer=h[:,None]; time=0
                for target in TIMES:
                    while time<target:
                        count=min(512,target-time); y=advance(model,x,h,count); h=y[:,-1]; time+=count
                        buffer=np.concatenate([buffer,y],axis=1)[:,-1024:]
                    saved[f'state_{target}']=h.copy(); saved[f'window_{target}']=buffer[:,-256:].copy()
                saved['tail']=buffer.copy(); np.savez_compressed(OUT/f'{trial}_{kind}.npz',**saved)
                for ci,c in enumerate(cases):
                    row={k:c[k] for k in ('trial','world','orientation','control')}; row['branches']={}
                    for si,start in enumerate(STARTS):
                        j=ci*4+si; stats={str(t):window(saved[f'window_{t}'][j]) for t in TIMES}; tail=window(buffer[j])
                        nxt,jac=P.step(x[j],h[j],p,True)
                        row['branches'][start]=dict(windows=stats,tail=tail,label=classify(tail,stats['8192'],stats['16384']),
                            residual=float(np.linalg.norm(nxt-h[j])),spectral_radius=float(np.max(np.abs(np.linalg.eigvals(jac)))),
                            return_rms=[float(np.sqrt(np.mean(np.sum((buffer[j,lag:]-buffer[j,:-lag])**2,axis=1)))) for lag in LAGS])
                    row['separations']={str(t):{s:float(np.linalg.norm((initial if t==0 else saved[f'state_{t}'])[ci*4+si]-(initial if t==0 else saved[f'state_{t}'])[ci*4]))
                                                for si,s in enumerate(STARTS) if si} for t in (0,)+TIMES}
                    prior=next(v for v in old if all(v[k]==row[k] for k in ('trial','world','orientation','control')))
                    w=saved['window_192'][ci*4]; difference=np.linalg.norm(w[-1]-w[-2])
                    assert abs(difference-prior['branches']['constant']['last_step_192'])<=1e-10
                    results.append(row)
                O.verify(manifest)
                R.C.save(OUT/f'{trial}_{kind}.json',dict(cases=results[-len(cases):])); print('completed',trial,kind,flush=True)
        assert len(results)==256
        R.C.save(OUT/'results.json',dict(cases=results,lags=LAGS,checkpoints=TIMES,starts=STARTS))
        O.verify(manifest)
        artifacts={p.name:O.sha(p) for p in OUT.iterdir() if p.is_file()}
        R.C.save(OUT/'completion.json',dict(kind='OBS3_COMPLETE',manifest=manifest,artifacts=artifacts,functional_verdict_changed=False))
        print('OBS3 complete',flush=True)
    except Exception as exc:
        R.C.save(OUT/'invalid.json',dict(error=repr(exc))); raise


if __name__=='__main__':main()
