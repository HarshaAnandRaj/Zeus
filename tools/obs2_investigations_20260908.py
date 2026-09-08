"""Six sequential frozen-checkpoint investigations, protocol 1d146f5."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs1_state_discovery_20260908 as O
R=O.R;OUT=ROOT/'runs/obs2_20260908';OFFSETS=(0,1,126,127)


def spectrum(c):
    e=np.maximum(np.linalg.eigvalsh(c),0)[::-1];total=float(e.sum())
    if total<1e-24:return dict(energy=total,dimension=None,d95=None,d99=None,rank=0)
    return dict(energy=total,dimension=float(total**2/(e@e)),d95=int(np.searchsorted(e.cumsum(),.95*total)+1),
        d99=int(np.searchsorted(e.cumsum(),.99*total)+1),rank=int(np.sum(e>e[0]*32*np.finfo(float).eps)))


def cov(h):
    h=h-h.mean(0);return h.T@h/len(h)


def decompose(h):
    means=h.mean(1);center=h-means[:,None,:]
    within=np.einsum('nti,ntj->ij',center,center)/(h.shape[0]*h.shape[1]);between=cov(means);total=cov(h.reshape(-1,32))
    error=float(np.max(np.abs(total-within-between)));assert error<1e-12
    return dict(within=spectrum(within),between=spectrum(between),pooled=spectrum(total),
        between_energy_fraction=float(np.trace(between)/np.trace(total)),identity_error=error)


def params(state):return {k:v.double().numpy() for k,v in state.items()}


def step(x,h,p,jacobian=False):
    gi=x@p['gru.weight_ih_l0'].T+p['gru.bias_ih_l0'];gh=h@p['gru.weight_hh_l0'].T+p['gru.bias_hh_l0']
    ir,iz,inn=np.split(gi,3);hr,hz,hn=np.split(gh,3)
    r=1/(1+np.exp(-(ir+hr)));z=1/(1+np.exp(-(iz+hz)));n=np.tanh(inn+r*hn)
    new=z*h+(1-z)*n
    if not jacobian:return new,(r,z,n)
    wr,wz,wn=np.split(p['gru.weight_hh_l0'],3)
    dr=(r*(1-r))[:,None]*wr;dz=(z*(1-z))[:,None]*wz
    dn=(1-n*n)[:,None]*(r[:,None]*wn+hn[:,None]*dr)
    j=np.diag(z)+(h-n)[:,None]*dz+(1-z)[:,None]*dn
    return new,j


def roll(xs,h,p):
    states=[];zs=[];ns=[]
    for x in xs:
        h,(_,z,n)=step(x,h,p);states.append(h);zs.append(z);ns.append(n)
    return np.array(states),np.array(zs),np.array(ns)


def predict(h,p):return 1/(1+np.exp(-(h@p['readout.weight'].T+p['readout.bias'])))


def jacobian_check(x,h,p):
    _,j=step(x,h,p,True);eps=1e-6;numeric=np.zeros((32,32))
    for i in range(32):
        d=np.zeros(32);d[i]=eps;numeric[:,i]=(step(x,h+d,p)[0]-step(x,h-d,p)[0])/(2*eps)
    error=float(np.max(np.abs(j-numeric)));assert error<1e-7;return error


def history_inputs(prep):
    obs=[R.C.restore(prep['initial']).observation()]+[r['after'] for r in prep['exposure'][:15]]
    return np.array([R.M.features(o) for o in obs],dtype=float)


def sources():
    c6=R.N.read(R.OUT/'verdict.json');obs=R.N.read(O.OUT/'catalogue.json')
    files=['docs/obs2_investigation_protocol_20260908.md','tools/obs2_investigations_20260908.py',
        'tools/test_obs2_investigations_20260908.py','tools/obs1_state_discovery_20260908.py',
        'tools/cyc4_learned_carryover_20260907.py','tools/cyc3_carryover_calibration_20260907.py','core/embodiment.py',
        'tools/cyc5_learning_elimination_20260907.py','tools/cyc6_initialization_robustness_20260907.py','tools/cycle_forensics_20260907.py',
        'runs/cyc6_20260907/verdict.json','runs/cyc6_20260907/completion_audit.json',
        'runs/obs1_20260908/catalogue.json','runs/obs1_20260908/completion_audit.json']
    hashes={p:O.sha(ROOT/p) for p in files}
    for name in ['calibration_a.jsonl.gz']+[f'{t}/{s}' for t in R.TRIALS for s in ('evaluation_a.jsonl.gz','teacher_mse/a/final.pt')]:
        assert O.sha(R.OUT/name)==c6['artifacts'][name];hashes[str((R.OUT/name).relative_to(ROOT)).replace('\\','/')]=c6['artifacts'][name]
    for t in R.TRIALS:
        for c in R.CONTROLS:
            name=f'{t}_{c}_samples.npz';assert O.sha(O.OUT/name)==obs['artifacts'][name]
            hashes[str((O.OUT/name).relative_to(ROOT)).replace('\\','/')]=obs['artifacts'][name]
    return dict(sources=hashes,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        exploratory=True,offline_dtype='float64',selected_world_offsets=OFFSETS,torch=torch.__version__,numpy=np.__version__)


def stage1():
    out={};cases=[];checkpoints={};common={}
    for t in R.TRIALS:
        cp=R.N.load(R.OUT/t/'teacher_mse/a/final.pt');checkpoints[t]={k:params(cp[k]) for k in ('initial','final')}
        out[t]={};common[t]={c:[] for c in R.CONTROLS}
        for c in R.CONTROLS:
            h=np.load(O.OUT/f'{t}_{c}_samples.npz')['h'].reshape(256,16,32)
            out[t][c]=decompose(h)
        for i,pair in enumerate(R.N.rows(R.OUT/t/'evaluation_a.jsonl.gz')):
            for oi,o in enumerate(pair['orientations']):
                for c,e in o['controls'].items():
                    h,x,pred,body,age=O.arrays(e);assert len(h)>=4;common[t][c].append(h[:4])
                    if i in OFFSETS:cases.append(dict(trial=t,world=pair['seed'],orientation=oi,control=c,h=h,x=x,
                        initial_hidden=np.array(e['initial_hidden']).reshape(32)))
        for c in R.CONTROLS:out[t][c]['common_first_four']=decompose(np.array(common[t][c]))
        print('O1 covariance complete',t,flush=True)
    assert len(cases)==256
    return dict(kind='OBS2_O1',groups=out),cases,checkpoints


def teacher_streams():
    streams={}
    for i,pair in enumerate(R.N.rows(R.OUT/'calibration_a.jsonl.gz')):
        if i in OFFSETS:
            for oi,o in enumerate(pair['orientations']):
                x=np.concatenate([history_inputs(o['preparation']),np.array([R.M.features(row['before']) for row in o['teacher']['trace']])])
                assert x.shape==(512,10);streams[(pair['seed'],oi)]=x
    assert len(streams)==8;return streams


def stage2(checkpoints,streams):
    out={};max_error=0.
    for t in R.TRIALS:
        out[t]={}
        for kind in ('final','initial'):
            p=checkpoints[t][kind];gru=torch.nn.GRU(10,32,batch_first=True).double().eval()
            gru.load_state_dict({k.removeprefix('gru.'):torch.tensor(v) for k,v in p.items() if k.startswith('gru.')})
            hs=[];zs=[];ns=[]
            for xs in streams.values():
                h,z,n=roll(xs,np.zeros(32),p)
                with torch.no_grad():ref,_=gru(torch.tensor(xs).unsqueeze(0))
                error=float(np.max(np.abs(h-ref[0].numpy())));assert error<1e-10;max_error=max(max_error,error)
                hs.append(h);zs.append(z);ns.append(n)
            h=np.concatenate(hs);z=np.concatenate(zs);n=np.concatenate(ns);sat=np.abs(h)>.95
            out[t][kind]=dict(streams=8,rows=len(h),hidden_saturation=float(sat.mean()),candidate_saturation=float((np.abs(n)>.95).mean()),
                mean_z=float(z.mean()),z_when_saturated=float(z[sat].mean()) if sat.any() else None,z_when_unsaturated=float(z[~sat].mean()) if (~sat).any() else None,
                hidden_rms=float(np.sqrt(np.mean(h*h))))
        print('O2 common-input gates complete',t,flush=True)
    return dict(kind='OBS2_O2',groups=out,max_manual_pytorch_error=max_error)


def stage3(cases,checkpoints):
    rows=[]
    for case in cases:
        p=checkpoints[case['trial']]['initial' if case['control']=='untrained' else 'final'];branches={}
        drivers={'constant':np.tile(case['x'][-1],(192,1))}
        if len(case['x'])>=6:drivers['six_input_motif']=np.tile(case['x'][-6:],(32,1))
        for name,xs in drivers.items():
            h,_,_=roll(xs,case['h'][-1].copy(),p);ratios,raw=O.displacements(h[-64:]);centered=float(np.sqrt(np.mean((h[-64:]-h[-64:].mean(0))**2)))
            branches[name]=dict(last_step_128=float(np.linalg.norm(h[127]-h[126])),last_step_192=float(np.linalg.norm(h[-1]-h[-2])),
                final64_centered_rms=centered,ratios=None if ratios is None else ratios.tolist(),raw_return_rms=raw.tolist(),
                numerical_stationary_at_192=bool(np.linalg.norm(h[-1]-h[-2])<1e-10),
                best_lag=None if ratios is None else int(O.LAGS[1+np.argmin(ratios[1:])]))
        rows.append(dict(**{k:case[k] for k in ('trial','world','orientation','control')},branches=branches))
    return dict(kind='OBS2_O3',cases=rows)


def stage4(cases,checkpoints):
    rows=[];max_error=0.
    for case in cases:
        p=checkpoints[case['trial']]['initial' if case['control']=='untrained' else 'final'];w=p['readout.weight']
        _,s,v=np.linalg.svd(w,full_matrices=True);rank=int(np.sum(s>s[0]*32*np.finfo(float).eps));assert rank==9
        bases=dict(row=v[:rank].T,null=v[rank:].T);index=len(case['h'])//2;h=case['h'][index].copy();prod=np.eye(32)
        gain=lambda mat:{k:float(np.linalg.norm(w@mat@b)/np.sqrt(b.shape[1])) for k,b in bases.items()}
        horizons={'0':gain(prod)};assert horizons['0']['null']<1e-12
        future=case['x'][index+1:index+17]
        if len(future):max_error=max(max_error,jacobian_check(future[0],h,p))
        for i,x in enumerate(future,1):
            h,j=step(x,h,p,True);prod=j@prod
            if i in (1,4,16):horizons[str(i)]=gain(prod)
        rows.append(dict(**{k:case[k] for k in ('trial','world','orientation','control')},horizons=horizons))
    return dict(kind='OBS2_O4',cases=rows,max_jacobian_finite_difference_error=max_error)


def stage5(cases,checkpoints,streams):
    lookup={(c['trial'],c['world'],c['orientation'],c['control']):c for c in cases};rows=[]
    for t in R.TRIALS:
        p=checkpoints[t]['final']
        for (world,oi),xs in streams.items():
            initials={c:lookup[t,world,oi,c]['initial_hidden'] for c in ('intact','erased','swapped')}
            drivers=dict(teacher=xs[16:144],constant=np.tile(xs[16],(128,1)))
            for name,driver in drivers.items():
                states={c:np.vstack([h,roll(driver,h.copy(),p)[0]]) for c,h in initials.items()}
                for c in ('erased','swapped'):
                    distance=np.linalg.norm(states['intact']-states[c],axis=1);d0=distance[0]
                    output=np.linalg.norm(predict(states['intact'],p)-predict(states[c],p),axis=1)
                    rows.append(dict(trial=t,world=world,orientation=oi,driver=name,control=c,distances=distance.tolist(),output_distances=output.tolist(),
                        final_initial_ratio=float(distance[-1]/d0) if d0>0 else None,max_initial_ratio=float(distance.max()/d0) if d0>0 else None))
    return dict(kind='OBS2_O5',cases=rows)


def weighted_fit(h,x,w,train):
    root=np.sqrt(w[train]);coef,_,rank,_=np.linalg.lstsq(x[train]*root[:,None],h[train]*root[:,None],rcond=None)
    average=np.average(h[train],axis=0,weights=w[train]);err=float(np.sum(w[~train,None]*(h[~train]-x[~train]@coef)**2))
    base=float(np.sum(w[~train,None]*(h[~train]-average)**2))
    return dict(r2=None if base<1e-24 else 1-err/base,rank=int(rank),train_rows=int(train.sum()),test_rows=int((~train).sum()),
        error=err,baseline=base,train_weight=float(w[train].sum()),test_weight=float(w[~train].sum()))


def stage6():
    out={}
    for t in R.TRIALS:
        data={c:{k:[] for k in ('h','current','polynomial','recent4','w','seed')} for c in R.CONTROLS}
        for pair,cal in zip(R.N.rows(R.OUT/t/'evaluation_a.jsonl.gz'),R.N.rows(R.OUT/'calibration_a.jsonl.gz')):
            assert pair['seed']==cal['seed']
            for oi,o in enumerate(pair['orientations']):
                for c,e in o['controls'].items():
                    h,x,_,_,_=O.arrays(e);n=min(16,len(h));idx=np.linspace(0,len(h)-1,n).astype(int);assert len(set(idx))==n
                    pre=np.zeros((16,10)) if c=='erased' else history_inputs(cal['orientations'][1-oi if c=='swapped' else oi]['preparation'])
                    full=np.vstack([pre,x]);recent=np.array([full[16+i-3:16+i+1].reshape(-1) for i in idx])
                    xx=x[idx];poly=np.concatenate([xx[:,1:]*xx[:,:1]**power for power in range(4)],axis=1)
                    values=dict(h=h[idx],current=np.column_stack([np.ones(n),xx]),polynomial=np.column_stack([np.ones(n),poly]),
                        recent4=np.column_stack([np.ones(n),recent]),w=np.full(n,1/n),seed=np.full(n,pair['seed']))
                    for k,v in values.items():data[c][k].append(v)
        out[t]={}
        for c,parts in data.items():
            d={k:np.concatenate(v) for k,v in parts.items()};train=d['seed']<202678064
            assert abs(d['w'][train].sum()-128)<1e-10 and abs(d['w'][~train].sum()-128)<1e-10
            out[t][c]={name:weighted_fit(d['h'],d[name],d['w'],train) for name in ('current','polynomial','recent4')}
            np.savez_compressed(OUT/f'o6_{t}_{c}_descriptions.npz',**d)
        print('O6 description complete',t,flush=True)
    return dict(kind='OBS2_O6',groups=out)


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=False);manifest=sources();R.C.save(OUT/'manifest.json',manifest)
    try:
        first,cases,checkpoints=stage1();R.C.save(OUT/'o1.json',first);print('O1 saved',flush=True)
        streams=teacher_streams();torch.save(dict(cases=cases,checkpoints=checkpoints,streams=streams),OUT/'replay_inputs.pt')
        functions=[lambda:stage2(checkpoints,streams),lambda:stage3(cases,checkpoints),lambda:stage4(cases,checkpoints),
            lambda:stage5(cases,checkpoints,streams),stage6]
        for index,fn in enumerate(functions,2):
            O.verify(manifest);result=fn();R.C.save(OUT/f'o{index}.json',result);print(f'O{index} saved',flush=True)
        O.verify(manifest)
        artifacts={p.name:O.sha(p) for p in OUT.iterdir() if p.is_file()}
        R.C.save(OUT/'completion.json',dict(kind='OBS2_COMPLETE',exploratory=True,manifest=manifest,artifacts=artifacts,
            functional_verdict_changed=False,automatic_followup=False));print('All six investigations complete',flush=True)
    except Exception as exc:
        R.C.save(OUT/'invalid.json',dict(status='INVALID_STOP',error=repr(exc)));raise


if __name__=='__main__':main()
