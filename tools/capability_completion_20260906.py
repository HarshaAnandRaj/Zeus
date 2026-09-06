"""Combine separately identified captures; preserve CAP1 DYN1 fidelity VOID."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.capability_components_20260906 import OUT,REPORTS,load,save,metrics,grid,dimension,channel,calibration,EmbodiedWorldV2,Action
from tools.cap1r_dimension_20260906 import OUT as DYNOUT,identities
from tools.qv1_postmortem_20260906 import sha


def main():
    import torch
    torch.set_num_threads(1)
    target=REPORTS/'capability_completion_20260906.json'
    if target.exists():
        raise RuntimeError('refuse report overwrite')
    manifest=identities()
    assert load(DYNOUT/'identity.json')['exact_twins']
    pol=load(OUT/'pol2.json.gz')
    logits=load(OUT/'pol2_logits_derived.json.gz')
    assert pol['original_replay_verified'] and logits['all_argmax_actions_match']
    assert logits['capture_sha256']==sha(OUT/'pol2.json.gz')
    dy=load(DYNOUT/'twin_a.json.gz')
    qv=load(REPORTS/'qv1_postmortem_20260906.json.gz')
    pm={k:[metrics(e) for e in eps] for k,eps in pol['conditions'].items()}
    qm={}
    for arm,cells in qv['D4'].items():
        qm[arm]={}
        for cell,data in cells.items():
            episodes=[]
            all_channels=[]
            for e in data['episodes']:
                w=EmbodiedWorldV2(seed=e['seed'])
                channels=[]
                for t in e['trace']:
                    assert list(w.observation())==t['before']
                    ch=channel(w)
                    a=int(Action[t['action'].upper()])
                    effect=w.step(a)
                    assert list(effect['after'])==t['after']==ch['outputs'][a]
                    assert effect['viable']==ch['viable'][a]
                    channels.append(ch)
                derived={'seed':e['seed'],'age':e['age'],'states':[t['state'] for t in e['trace']],
                         'observations':[t['before'] for t in e['trace']],
                         'actions':[int(Action[t['action'].upper()]) for t in e['trace']],'channels':channels}
                episodes.append(metrics(derived))
                all_channels.append({'seed':e['seed'],'channels':channels})
            qm[arm][cell]=episodes
            save(DYNOUT/(arm+'_'+cell+'_channels.json.gz'),{'source_manifest':manifest,'episodes':all_channels})
            print(f'CAP1 completion QV1 {arm} {cell}',flush=True)
    dm={a:[{'seed':e['seed'],**{b:dimension(e[b]['states']) for b in ['control','perturbed']}} for e in eps] for a,eps in dy['conditions'].items()}
    draws=np.random.Generator(np.random.PCG64(20260961)).integers(0,64,(10000,64))
    comparisons={}
    for control in ['fixed_rest','fixed_harvest']:
        values={'repertoire':[],'coverage_cells':[],'capacity_bits':[],'common_prefix_coverage_cells':[]}
        for i in range(64):
            a,b=pm['normal'][i],pm[control][i]
            assert a['seed']==b['seed']
            values['repertoire'].append(a['repertoire']['entropy_effective_actions']-b['repertoire']['entropy_effective_actions'])
            values['coverage_cells'].append(a['occupied_viable_cells']-b['occupied_viable_cells'])
            values['capacity_bits'].append(a['mean_viable_capacity_bits']-b['mean_viable_capacity_bits'])
            n=min(pol['conditions'][k][i]['age'] for k in ['normal','fixed_rest','fixed_harvest'])
            values['common_prefix_coverage_cells'].append(len({grid(o) for o in pol['conditions']['normal'][i]['observations'][:n]})-len({grid(o) for o in pol['conditions'][control][i]['observations'][:n]}))
        comparisons[control]={}
        for name,vs in values.items():
            x=np.array(vs,dtype=float)
            comparisons[control][name]={'mean_difference':float(x.mean()),'ci95':np.quantile(x[draws].mean(1),[.025,.975]).tolist()}
    separation=all(v['repertoire']['ci95'][0]>0 for v in comparisons.values())
    def overview(rows):
        return {'mean_age':float(np.mean([r['age'] for r in rows])),
                'mean_repertoire':float(np.mean([r['repertoire']['entropy_effective_actions'] for r in rows])),
                'mean_occupied_viable_cells':float(np.mean([r['occupied_viable_cells'] for r in rows])),
                'mean_participation_ratio':float(np.mean([r['dimension']['participation_ratio'] for r in rows])),
                'mean_capacity_bits':float(np.mean([r['mean_viable_capacity_bits'] for r in rows])),
                'mean_response_rank':float(np.mean([r['mean_local_response_rank'] for r in rows]))}
    result={'status':'MEASUREMENT COMPONENTS VALIDATED' if separation else 'NOT READY',
            'class':'O','continuation_authority':False,'manifest_sha256':identities(),'calibration':calibration(),
            'original_CAP1_DYN1_status':'VOID: first historical feature outside frozen tolerance; not rerun or repaired',
            'DYN1_measurement_scope':'Same frozen checkpoint; fresh registered seeds 20260970..20260985, exact CPU twins; not original DYN1 reproduction or re-adjudication.',
            'capture_fidelity':{'POL2_original_episodes':True,'POL2_derived_argmax':True,'DYN1_substrate_fresh_twins':True,'QV1_transitions':True},
            'POL2':pm,'QV1':qm,'DYN1_substrate_fresh':dm,'POL2_paired_comparisons':comparisons,
            'overviews':{'POL2':{k:overview(rows) for k,rows in pm.items()},'QV1':{a:{k:overview(rows) for k,rows in cells.items()} for a,cells in qm.items()}},
            'limits':['Coverage is a declared viable observation grid, not a reachable viability kernel.',
                      'Capacity is the one-step body/world channel, not learned policy access to it.',
                      'Response rank is one-step, not long-horizon controllability.',
                      'DYN1 has no body/action channel; viability and empowerment are not applicable.',
                      'No combined consciousness score, pillar pass, phase, or follow-up experiment is licensed.'],
            'capture_hashes':{str(p.relative_to(ROOT)):sha(p) for folder in [OUT,DYNOUT] for p in folder.glob('*.gz')}}
    save(target,result)
    print(result['status'],flush=True)
    print(result['overviews']['POL2'],flush=True)
    print(comparisons,flush=True)


if __name__=='__main__':
    main()
