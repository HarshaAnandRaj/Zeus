"""Separately arranged integer gates and paired contrasts for the prospective LMB6."""
import math
import numpy as np
from training.lmb6_contract import SCOPE


def decide(endpoints,preps,config):
    nt=config['trials'];n=config['evaluation_ecologies'];assert config['regression_ecologies']==n and n%4==0
    rng=np.random.default_rng(config['bootstrap_seed'])
    pt=rng.integers(0,nt,(config['bootstrap_draws'],nt));quartets=rng.integers(0,n//4,(config['bootstrap_draws'],n//4))
    ei=np.stack([quartets*4+j for j in range(4)],axis=2).reshape(config['bootstrap_draws'],n)
    def effect(values,minimum=None):
        a=np.asarray(values,dtype=int);assert a.shape==(nt,n)
        bounds=np.quantile(a[pt[:,:,None],ei[:,None,:]].mean(axis=(1,2)),(.025,.975)).tolist()
        mean=float(a.sum()/a.size);out=dict(mean=mean,bounds=bounds)
        if minimum is not None:out['passed']=bool(mean>=minimum and bounds[0]>0)
        return out
    maps={};arms={};reg_indices={e:[i for i,p in enumerate(preps) if p['energy']==e] for e in config['energies']}
    assert len(preps)==4*n and all(len(ii)==n for ii in reg_indices.values())
    for variant in (*config['arms'],'warm'):
        rows=endpoints[variant]['bodies'];keys=[(r['trial'],r['energy'],r['index'],r['mode']) for r in rows]
        expected={(t,e,i,'inherited') for t in range(nt) for e in config['energies'] for i in range(n)}
        if variant!='warm':expected.update((t,.85,i,'empty') for t in range(nt) for i in range(n))
        assert len(keys)==len(set(keys))==len(expected) and set(keys)==expected and all(r['variant']==variant for r in rows)
        maps[variant]=dict(zip(keys,rows))
        if variant=='warm':continue
        gates={};cells=[]
        for t in range(nt):
            for e in config['energies']:
                for mode in (('inherited','empty') if e==.85 else ('inherited',)):
                    for side in range(2):
                        for q in range(2):
                            rr=[r for r in rows if r['trial']==t and r['energy']==e and r['mode']==mode and r['side']==side and r['quality']==q]
                            count=len(rr);assert count==n//4
                            survivors=sum(bool(r['survived']) for r in rr);feeds=sum(r['feeding'] for r in rr);repairs=sum(r['repairs'] for r in rr);recalls=sum(bool(r['quality_correct']) for r in rr)
                            key=f'{t}_{e}_{mode}_{side}_{q}'
                            gates[key]=survivors>=math.ceil(config['survival_min']*count) and feeds>=config['feed_min']*count and repairs>=config['repair_min']*count and (mode=='empty' or recalls>=math.ceil(config['quality_min']*count))
                            gates['identity_'+key]=all(r['fast_reset'] is True and r['storage_distance']==0 and (r['initial_z']==r['written_z'] if mode=='inherited' else not any(r['initial_z'])) for r in rr)
                            cells.append(dict(trial=t,energy=e,mode=mode,side=side,quality=q,n=count,survival=survivors/count,
                                mean_feeding=feeds/count,mean_repairs=repairs/count,initial_quality_recall=recalls/count))
        queries=endpoints[variant]['readout'];assert len(queries)==nt
        controls=[]
        for t,query in enumerate(queries):
            assert query['trial']==t and query['storage_distance']==0 and query['written']==query['inherited']
            qq={r['control']:r for r in query['rows']};assert len(query['rows'])==3 and set(qq)=={'full','reset','opposite'};controls.append(qq)
            for e in config['energies']:
                for side in range(2):
                    for quality in range(2):
                        ii=[i for i,p in enumerate(preps) if (p['energy'],p['side'],p['quality'])==(e,side,quality)]
                        assert len(ii)==n//4;needed=math.ceil(config['quality_min']*len(ii));key=f'query_{t}_{e}_{side}_{quality}'
                        gates[key]=sum(qq['full']['action'][i]==preps[i]['target'] for i in ii)>=needed
                        gates['recall_'+key]=sum((qq['full']['recall_probability'][i]>=.5)==bool(quality) for i in ii)>=needed
                        gates['opposite_'+key]=sum(qq['opposite']['action'][i]==preps[i^1]['target'] for i in ii)>=needed
                        gates['opposite_recall_'+key]=sum((qq['opposite']['recall_probability'][i]>=.5)==bool(preps[i^1]['quality']) for i in ii)>=needed
        effects=[]
        for e in config['energies']:
            result=effect([[int(q['full']['action'][i]==preps[i]['target'])-int(q['reset']['action'][i]==preps[i]['target']) for i in reg_indices[e]] for q in controls],config['readout_effect_min'])
            effects.append(dict(energy=e,**result));gates[f'query_reset_effect_{e}']=result['passed']
        arms[variant]=dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,cells=cells,readout_effects=effects)
    contrasts=[];warm=[]
    for e in config['energies']:
        for control in ('recurrent','legacy'):
            difference=[[int(maps['current'][t,e,i,'inherited']['survived'])-int(maps[control][t,e,i,'inherited']['survived']) for i in range(n)] for t in range(nt)]
            contrasts.append(dict(control=control,energy=e,**effect(difference,config['attribution_min']),required=e in config['attribution_energies']))
        for variant in config['arms']:
            difference=[[int(maps[variant][t,e,i,'inherited']['survived'])-int(maps['warm'][t,e,i,'inherited']['survived']) for i in range(n)] for t in range(nt)]
            warm.append(dict(variant=variant,energy=e,**effect(difference)))
    selected=next((arm for arm in config['arms'] if arms[arm]['verdict']=='PASS'),None)
    required=[r for r in contrasts if r['required']]
    assert len(required)==2*len(config['attribution_energies'])
    attribution=arms['current']['verdict']=='PASS' and all(r['passed'] for r in required)
    return dict(verdict='PASS' if selected is not None else 'FAIL',selected_qualified_arm=selected,arms=arms,
        architecture_attribution='PASS' if attribution else 'FAIL',contrasts=contrasts,
        warm_contrasts=warm,pillar_promotion=False,scope=SCOPE)
