"""Independent covariance, native-GRU and regression checks for OBS2R."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs2r_investigations_20260908 as V
P=V.P;O=P.O;R=P.R;OUT=P.OUT


def close(a,b):
    if a is None or b is None:assert a is None and b is None
    else:np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-10)


def stats_from_rows(h,expected):
    s=np.linalg.svd(h-h.mean(0),compute_uv=False);e=s*s/len(h);total=float(e.sum())
    close(expected['energy'],total)
    if total<1e-24:assert expected['dimension'] is None;return
    close(expected['dimension'],total**2/np.sum(e*e))
    assert expected['d95']==int(np.searchsorted(e.cumsum(),.95*total)+1)
    assert expected['d99']==int(np.searchsorted(e.cumsum(),.99*total)+1)
    # Numerical rank near machine precision is algorithm-dependent; dimension
    # and variance coverage are independently checked without a rank claim.


def decomposition(h,d):
    mean=h.mean(1);within=(h-mean[:,None,:]).reshape(-1,32);pooled=h.reshape(-1,32)
    stats_from_rows(within,d['within']);stats_from_rows(mean,d['between']);stats_from_rows(pooled,d['pooled'])
    between=float(np.mean(np.sum((mean-mean.mean(0))**2,axis=1)))
    total=float(np.mean(np.sum((pooled-pooled.mean(0))**2,axis=1)))
    close(d['between_energy_fraction'],between/total);assert d['identity_error']<1e-12


def native(model,x,h):
    with torch.no_grad():states,_=model(torch.tensor(x)[None],torch.tensor(h)[None,None])
    return states[0].numpy()


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    completion=R.N.read(OUT/'completion.json');O.verify(completion['manifest'])
    assert all(O.sha(OUT/k)==v for k,v in completion['artifacts'].items())
    records={i:R.N.read(OUT/f'o{i}.json') for i in range(1,7)}
    cache=R.N.load(OUT/'replay_inputs.pt');cases=cache['cases'];cp=cache['checkpoints'];streams=cache['streams']
    lookup={(c['trial'],c['world'],c['orientation'],c['control']):c for c in cases};models={};checks={'source_and_artifact_hashes':True}
    for t in R.TRIALS:
        saved=R.N.load(R.OUT/t/'teacher_mse/a/final.pt')
        for kind in ('initial','final'):
            for key,value in saved[kind].items():np.testing.assert_array_equal(value.double().numpy(),cp[t][kind][key])
            m=torch.nn.GRU(10,32,batch_first=True).double().eval()
            m.load_state_dict({k.removeprefix('gru.'):torch.tensor(v) for k,v in cp[t][kind].items() if k.startswith('gru.')})
            for param in m.parameters():param.requires_grad_(False)
            models[t,kind]=m
        common={c:[] for c in R.CONTROLS}
        for pair in R.N.rows(R.OUT/t/'evaluation_a.jsonl.gz'):
            for oi,o in enumerate(pair['orientations']):
                for control,e in o['controls'].items():
                    h=np.array([row['hidden'][0][0] for row in e['trace']]);common[control].append(h[:3])
                    key=(t,pair['seed'],oi,control)
                    if key in lookup:
                        case=lookup[key];np.testing.assert_array_equal(h,case['h'])
                        np.testing.assert_array_equal(np.array([row['input'] for row in e['trace']]),case['x'])
                        np.testing.assert_array_equal(np.array(e['initial_hidden']).reshape(32),case['initial_hidden'])
        for control in R.CONTROLS:
            h=np.load(O.OUT/f'{t}_{control}_samples.npz')['h'].reshape(256,16,32)
            d=records[1]['groups'][t][control];decomposition(h,d);decomposition(np.array(common[control]),d['common_first_three'])
        print('audited O1 and replay inputs',t,flush=True)
    for i,pair in enumerate(R.N.rows(R.OUT/'calibration_a.jsonl.gz')):
        if i not in P.OFFSETS:continue
        for oi,o in enumerate(pair['orientations']):
            pre=[R.C.restore(o['preparation']['initial']).observation()]+[row['after'] for row in o['preparation']['exposure'][:15]]
            obs=pre+[row['before'] for row in o['teacher']['trace']]
            np.testing.assert_array_equal(np.array([R.M.features(x) for x in obs]),streams[pair['seed'],oi])
    checks.update(all_32_covariance_decompositions_independent_svd=True,all_32_common_three_state_views=True,all_256_case_and_eight_teacher_inputs_verified=True)
    for t in R.TRIALS:
        for kind in ('final','initial'):
            p=cp[t][kind];hh=[];zz=[];nn=[]
            for x in streams.values():
                h=native(models[t,kind],x,np.zeros(32));old=np.vstack([np.zeros(32),h[:-1]])
                gi=x@p['gru.weight_ih_l0'].T+p['gru.bias_ih_l0'];gh=old@p['gru.weight_hh_l0'].T+p['gru.bias_hh_l0']
                rr=1/(1+np.exp(-(gi[:,:32]+gh[:,:32])));z=1/(1+np.exp(-(gi[:,32:64]+gh[:,32:64])))
                n=np.tanh(gi[:,64:]+rr*gh[:,64:]);np.testing.assert_allclose(h,z*old+(1-z)*n,atol=1e-10,rtol=0)
                hh.append(h);zz.append(z);nn.append(n)
            h=np.concatenate(hh);z=np.concatenate(zz);n=np.concatenate(nn);sat=np.abs(h)>.95;v=records[2]['groups'][t][kind]
            for key,value in dict(hidden_saturation=sat.mean(),candidate_saturation=(np.abs(n)>.95).mean(),mean_z=z.mean(),
                z_when_saturated=z[sat].mean() if sat.any() else None,z_when_unsaturated=z[~sat].mean() if (~sat).any() else None,
                hidden_rms=np.sqrt(np.mean(h*h))).items():close(v[key],value)
    checks['all_128_gate_replays_native_pytorch']=True
    below_resolution=0;branch_count=0
    for row in records[3]['cases']:
        case=lookup[row['trial'],row['world'],row['orientation'],row['control']];model=models[row['trial'],'initial' if row['control']=='untrained' else 'final']
        for name,v in row['branches'].items():
            x=np.tile(case['x'][-1],(192,1)) if name=='constant' else np.tile(case['x'][-6:],(32,1));h=native(model,x,case['h'][-1]);branch_count+=1
            change=float(np.linalg.norm(h[-1]-h[-2]));rms=float(np.sqrt(np.mean((h[-64:]-h[-64:].mean(0))**2)))
            close(v['last_step_128'],np.linalg.norm(h[127]-h[126]));close(v['last_step_192'],change);close(v['final64_centered_rms'],rms)
            assert v['numerical_stationary_at_192']==(change<1e-10)
            ratios,raw=O.displacements(h[-64:]);close(v['raw_return_rms'],raw)
            if rms<1e-10:below_resolution+=1
            else:close(v['ratios'],ratios);assert v['best_lag']==int(O.LAGS[1+np.argmin(ratios[1:])])
    checks['all_constant_and_motif_replays_native_pytorch']=True
    # Above-resolved amplitudes get ratio checks. Ratios of ~zero variance are
    # numerically unstable; raw amplitudes/changes are still checked in all cases.
    checked=0
    for row in records[4]['cases']:
        if row['world']!=202678000 or row['orientation']!=0:continue
        key=(row['trial'],row['world'],row['orientation'],row['control']);case=lookup[key]
        kind='initial' if row['control']=='untrained' else 'final';model=models[row['trial'],kind];w=cp[row['trial']][kind]['readout.weight']
        _,_,v=np.linalg.svd(w,full_matrices=True);bs=dict(row=v[:9].T,null=v[9:].T);index=len(case['h'])//2
        for horizon,gains in row['horizons'].items():
            n=int(horizon)
            if n==0:j=np.eye(32)
            else:
                x=torch.tensor(case['x'][index+1:index+1+n])[None]
                def fn(h):return model(x,h[None,None])[1].reshape(32)
                j=torch.autograd.functional.jacobian(fn,torch.tensor(case['h'][index],requires_grad=True)).numpy()
            for name,b in bs.items():close(gains[name],np.linalg.norm(w@j@b)/np.sqrt(b.shape[1]))
        checked+=1
    assert checked==32;checks['null_propagation_32_cases_native_autograd']=True
    history_count=0
    for t in R.TRIALS:
        model=models[t,'final'];p=cp[t]['final']
        for (world,oi),teacher in streams.items():
            for driver in ('teacher','constant'):
                xs=teacher[16:144] if driver=='teacher' else np.tile(teacher[16],(128,1));hs={}
                for c in ('intact','erased','swapped'):
                    initial=lookup[t,world,oi,c]['initial_hidden'];hs[c]=np.vstack([initial,native(model,xs,initial)])
                for c in ('erased','swapped'):
                    row=next(v for v in records[5]['cases'] if (v['trial'],v['world'],v['orientation'],v['driver'],v['control'])==(t,world,oi,driver,c))
                    d=np.linalg.norm(hs['intact']-hs[c],axis=1);yp=lambda h:1/(1+np.exp(-(h@p['readout.weight'].T+p['readout.bias'])))
                    close(row['distances'],d);close(row['output_distances'],np.linalg.norm(yp(hs['intact'])-yp(hs[c]),axis=1))
                    close(row['final_initial_ratio'],d[-1]/d[0]);close(row['max_initial_ratio'],d.max()/d[0]);history_count+=1
    assert history_count==256;checks['all_history_distance_curves_native_pytorch']=True
    for t in R.TRIALS:
        for c in R.CONTROLS:
            d=np.load(OUT/f'o6_{t}_{c}_descriptions.npz');train=d['seed']<202678064;h=d['h'];w=d['w'];average=np.average(h[train],axis=0,weights=w[train])
            for name in ('current','polynomial','recent4'):
                x=d[name];a=x[train]*np.sqrt(w[train,None]);y=h[train]*np.sqrt(w[train,None]);u,s,v=np.linalg.svd(a,full_matrices=False)
                keep=s>s[0]*max(a.shape)*np.finfo(float).eps;coef=(v[keep].T/s[keep])@(u[:,keep].T@y)
                error=np.sum(w[~train,None]*(x[~train]@coef-h[~train])**2);base=np.sum(w[~train,None]*(h[~train]-average)**2)
                actual=records[6]['groups'][t][c][name];close(actual['error'],error);close(actual['baseline'],base);close(actual['r2'],1-error/base)
                assert actual['rank']==int(keep.sum()) and actual['train_rows']==int(train.sum()) and actual['test_rows']==int((~train).sum())
                close(actual['train_weight'],128.);close(actual['test_weight'],128.)
    checks['all_96_weighted_descriptions_independent_svd']=True
    O.verify(completion['manifest']);checks['post_audit_source_integrity']=True
    result=dict(kind='OBS2R_COMPLETION_AUDIT',passed=all(checks.values()),checks=checks,
        native_replay_branches=branch_count,below_resolution_return_profiles=below_resolution,
        below_resolution_rule='When tail RMS <1e-10, verify amplitudes and changes; do not compare unstable normalized ratios.',
        source_completion_sha=O.sha(OUT/'completion.json'),audit_source_sha=O.sha(Path(__file__)))
    R.C.save(OUT/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
