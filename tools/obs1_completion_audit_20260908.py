"""Independent raw-case and complete aggregate audit of OBS1 observations."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs1_state_discovery_20260908 as O


def close(a,b):
    if a is None or b is None:assert a is None and b is None
    else:np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-11)


def desc(vals,actual):
    v=np.array([x for x in vals if x is not None]);assert actual['n']==len(v)
    for key,q in [('median',50),('p10',10),('p90',90)]:close(actual[key],float(np.percentile(v,q)) if len(v) else None)


def geometry(h,w,g):
    centered=h-np.average(h,axis=0);energy=np.sum(centered*centered)/len(h)
    singular=np.linalg.svd(centered,compute_uv=False);ev=singular**2/len(h)
    close(g['hidden_rms'],np.linalg.norm(h)/np.sqrt(h.size));close(g['centered_rms'],np.sqrt(energy/32))
    close(g['saturation'],np.count_nonzero(np.abs(h)>.95)/h.size)
    close(g['participation_dimension'],ev.sum()**2/np.sum(ev**2) if energy>=1e-24 else None)
    close(g['top_two_fraction'],ev[:2].sum()/ev.sum() if energy>=1e-24 else None)
    projection=np.linalg.pinv(w)@w
    close(g['null_variation_fraction'],np.sum((centered-centered@projection)**2)/np.sum(centered**2) if energy*len(h)>=1e-24 else None)
    assert g['decisions']==len(h)


def recurrence(h,x,actions,index,r):
    if len(h)<64:assert r is None;return
    h=h[-64:];x=x[-64:];actions=actions[-64:];center=h-h.mean(0);energy=np.sum(center*center)/64
    if energy<1e-24:assert r['degenerate'];return
    ratios=[];raw=[]
    for k in O.LAGS:
        ms=sum(float(d@d) for d in h[k:]-h[:-k])/(64-k)
        ratios.append(ms/(2*energy));raw.append(np.sqrt(ms/32))
    close(r['ratios'],ratios);close(r['raw_rms'],raw);chosen=int(np.argmin(ratios[1:]))+1;lag=int(O.LAGS[chosen])
    assert r['best_lag']==lag;close(r['best_ratio'],ratios[chosen]);close(r['lag_one_ratio'],ratios[0])
    close(r['action_agreement'],sum(a==b for a,b in zip(actions[lag:],actions[:-lag]))/(64-lag))
    xe=np.mean(np.sum((x-x.mean(0))**2,axis=1))
    close(r['input_ratio_at_best'],np.mean(np.sum((x[lag:]-x[:-lag])**2,axis=1))/(2*xe) if xe>=1e-24 else None)
    rng=np.random.Generator(np.random.PCG64(20261201+index));mins=[]
    for _ in range(16):
        s=h[rng.permutation(64)]
        mins.append(min(np.mean(np.sum((s[k:]-s[:-k])**2,axis=1))/(2*energy) for k in O.LAGS[1:]))
    for key,v in dict(min=min(mins),max=max(mins),median=np.median(mins)).items():close(r['shuffle_minimum'][key],v)


def regression(h,x,train,r):
    u,s,v=np.linalg.svd(x[train],full_matrices=False);keep=s>np.finfo(float).eps*max(x[train].shape)*s[0]
    coef=(v[keep].T/s[keep])@(u[:,keep].T@h[train]);prediction=x[~train]@coef
    error=np.sum((prediction-h[~train])**2);baseline=np.sum((h[~train]-h[train].mean(0))**2)
    assert r['design_rank']==int(keep.sum()) and r['train_rows']==int(train.sum()) and r['test_rows']==int((~train).sum())
    close(r['test_error'],error);close(r['baseline_error'],baseline);close(r['test_r2'],1-error/baseline if baseline>=1e-24 else None)


def main():
    r=O.R.N.read(O.OUT/'catalogue.json');checks={};O.verify(r['manifest'])
    assert all(O.sha(O.OUT/p)==h for p,h in r['artifacts'].items());checks['source_and_artifact_hashes']=True
    raw_cases=prefixes=episodes=0
    for ti,t in enumerate(O.R.TRIALS):
        saved=O.R.N.read(O.OUT/f'{t}.json');cp=O.R.N.load(O.R.OUT/t/'teacher_mse/a/final.pt')
        weights={c:cp['initial' if c=='untrained' else 'final']['readout.weight'].double().numpy() for c in O.R.CONTROLS}
        sample={c:{k:[] for k in ('h','x','body','age','seed')} for c in O.R.CONTROLS}
        history_index=0;examples=np.load(O.OUT/f'{t}_examples.npz')
        for i,pair in enumerate(O.R.N.rows(O.R.OUT/t/'evaluation_a.jsonl.gz')):
            assert pair['seed']==202678000+i
            for oi,orientation in enumerate(pair['orientations']):
                parsed={}
                for ci,c in enumerate(O.R.CONTROLS):
                    e=orientation['controls'][c];rows=e['trace'];h=np.array([row['hidden'][0][0] for row in rows]);x=np.array([row['input'] for row in rows])
                    pred=np.array([row['prediction'] for row in rows]);body=np.array([row['before'][:3] for row in rows]);age=np.array([row['tick']-1 for row in rows])
                    parsed[c]=(h,x,pred);m=saved['episodes'][c][2*i+oi];episodes+=1
                    assert m['world_seed']==pair['seed'] and m['orientation']==oi and m['episode_index']==ti*1024+i*8+oi*4+ci
                    assert len(h)==m['geometry']['decisions']==e['age']-16
                    if i in (0,127):
                        geometry(h,weights[c],m['geometry']);recurrence(h,x,[row['action'] for row in rows],m['episode_index'],m['recurrence']);raw_cases+=1
                    idx=np.linspace(0,len(rows)-1,16).astype(int)
                    for k,v in dict(h=h[idx],x=x[idx],body=body[idx],age=age[idx],seed=np.full(16,pair['seed'])).items():sample[c][k].append(v)
                    if i==0 and oi==0:
                        for k,v in dict(h=h,x=x,age=age).items():np.testing.assert_array_equal(examples[c+'_'+k],v)
                for c in ('erased','swapped'):
                    p=saved['history_pairs'][history_index];history_index+=1;prefixes+=1
                    assert p['world_seed']==pair['seed'] and p['orientation']==oi and p['control']==c
                    ha,xa,pa=parsed['intact'];hb,xb,pb=parsed[c];n=0
                    while n<min(len(xa),len(xb)) and np.array_equal(xa[n],xb[n]):n+=1
                    assert n==p['equal_input_prefix'] and n>=1
                    for key,v in dict(initial_hidden_distance=np.linalg.norm(ha[0]-hb[0]),final_prefix_hidden_distance=np.linalg.norm(ha[n-1]-hb[n-1]),
                        initial_prediction_distance=np.linalg.norm(pa[0]-pb[0]),final_prefix_prediction_distance=np.linalg.norm(pa[n-1]-pb[n-1])).items():close(p[key],v)
                    diff=ha[0]-hb[0];proj=np.linalg.pinv(weights['intact'])@weights['intact']
                    close(p['initial_null_fraction'],np.sum((diff-diff@proj)**2)/np.sum(diff*diff) if np.sum(diff*diff)>=1e-24 else None)
        for c in O.R.CONTROLS:
            data=np.load(O.OUT/f'{t}_{c}_samples.npz')
            for k,parts in sample[c].items():np.testing.assert_array_equal(data[k],np.concatenate(parts))
            basic=np.column_stack([np.ones(len(data['h'])),data['x']]);extended=np.column_stack([basic,data['body'],data['age']/512]);train=data['seed']<202678064
            for label,design in [('current_input',basic),('input_body_age',extended)]:regression(data['h'],design,train,saved['linear'][c][label])
            group=saved['summaries'][c];es=saved['episodes'][c];assert len(es)==group['episodes']==256
            assert group['readout_rank']==np.linalg.matrix_rank(weights[c])
            for key in es[0]['geometry']:desc([e['geometry'][key] for e in es],group[key])
            tails=[e['recurrence'] for e in es if e['recurrence'] is not None and not e['recurrence']['degenerate']]
            assert group['tail_eligible']==sum(e['recurrence'] is not None for e in es) and group['tail_nondegenerate']==len(tails)
            for key in ('centered_rms','best_ratio','lag_one_ratio','input_ratio_at_best','action_agreement'):desc([a[key] for a in tails],group['tail_'+key])
            desc([a['shuffle_minimum']['median'] for a in tails],group['tail_shuffle_minimum'])
            assert group['tail_lag_counts']=={str(k):sum(a['best_lag']==k for a in tails) for k in O.LAGS[1:].tolist()}
            assert group==r['trials'][t]['summaries'][c] and saved['linear'][c]==r['trials'][t]['linear'][c]
        for c,group in r['trials'][t]['history_pairs'].items():
            for key,v in group.items():desc([p[key] for p in saved['history_pairs'] if p['control']==c],v)
        print('audited',t,'raw cases, all samples, prefixes and summaries',flush=True)
    assert episodes==8192 and raw_cases==128 and prefixes==4096
    checks.update(all_8192_episode_ids_and_lengths=True,raw_geometry_and_recurrence_128_cases=True,
        all_4096_history_prefixes_and_null_projections=True,all_sample_arrays_match_raw_trajectories=True,
        all_64_linear_descriptions_independent_svd=True,all_group_aggregates_and_eligibility_counts=True,fixed_example_arrays=True)
    O.verify(r['manifest']);checks['post_audit_sources']=True
    result=dict(kind='OBS1_COMPLETION_AUDIT',passed=all(checks.values()),checks=checks,raw_cases=raw_cases,prefixes=prefixes,episodes=episodes,
        catalogue_sha=O.sha(O.OUT/'catalogue.json'),audit_source_sha=O.sha(Path(__file__)))
    O.R.C.save(O.OUT/'completion_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
