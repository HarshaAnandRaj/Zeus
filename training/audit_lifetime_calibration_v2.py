"""V2 exact replay and independent quality/physical accounting."""
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.lifetime_world_v2 import QualityWorld, QualityConfig
from training import calibrate_lifetime_world_v2 as C
from training.audit_lifetime_calibration import read, close, normalize, independent_balance


def balance(before,action,c):
    result=independent_balance(before,action,c)
    if action==3 and before['position'] in (0,4) and before['quality'][before['position']//4]==0:
        amount=min(c['extraction_limit'],before['resources'][before['position']//4])
        energy=math.fsum((before['energy'],-c['metabolism'],-c['harvest_cost']))
        integrity=math.fsum((before['integrity'],-c['integrity_decay'],
            -c['starvation_damage']*max(0.,c['starvation_level']-energy),
            -c['contamination_damage']*amount/c['extraction_limit']))
        result['energy']=max(0.,min(1.,energy));result['integrity']=max(0.,min(1.,integrity))
    index=before['switch_index']
    if index<len(before['switches']) and result['tick']==before['switches'][index]:
        result['quality']=before['quality'][::-1];result['switch_index']=index+1
    return result


def main():
    complete=read(C.OUT/'completion.json');manifest=complete['manifest'];data=read(C.OUT/'results.json')
    assert all(C.sha(ROOT/p)==digest for p,digest in manifest['sources'].items())
    assert all(C.sha(C.OUT/p)==digest for p,digest in complete['artifacts'].items())
    assert manifest['config']==asdict(QualityConfig()) and manifest['new_neural_forward_passes']==0
    assert manifest['seeds']==list(C.SEEDS) and manifest['horizon']==1024 and manifest['policies']==list(C.POLICIES)
    episodes=data['episodes'];assert [(e['seed'],e['changing'],e['policy']) for e in episodes]==[(s,c,p) for s in C.SEEDS for c in (False,True) for p in C.POLICIES]
    count=0
    with gzip.open(C.OUT/'traces.jsonl.gz','rt',encoding='utf-8') as f:
        for ep in episodes:
            world=QualityWorld.restore(ep['initial'])
            assert normalize(world.snapshot())==normalize(QualityWorld(seed=ep['seed'],changing=ep['changing']).snapshot())
            state={};energy=0.;unsafe=0;counts={a.name.lower():0 for a in C.A}
            for tick in range(ep['ticks']):
                row=json.loads(next(f));assert (row['seed'],row['changing'],row['policy'])==(ep['seed'],ep['changing'],ep['policy'])
                before=C.physical(world.snapshot());assert before==row['before']
                reference={k:before[k] for k in ('quality','tool')} if ep['policy']=='informed' else None
                action=C.action_for(ep['policy'],world.observation(),tick,state,reference)
                assert int(action)==row['action'];close(balance(before,int(action),manifest['config']),row['after'])
                public=world.step(action);after=C.physical(world.snapshot());assert after==row['after']
                assert list(public.after.values())==row['observation'];assert list(public.after.prediction_mask())==row['prediction_mask']
                assert public.terminated==row['terminated']
                index=after['position']//4;on_patch=after['position'] in (0,4)
                resource=after['resources'][index] if on_patch else 0.
                quality=after['quality'][index] if on_patch else 0.
                obs=[after['energy'],after['integrity'],after['position']/4,
                     math.floor(resource*manifest['config']['resource_bins'])/manifest['config']['resource_bins'],
                     float(after['inspection']),resource if after['inspection'] else 0.,
                     after['tool'] if after['inspection'] else 0.,float(quality) if after['inspection'] else 0.]
                assert obs==row['observation'] and row['prediction_mask']==[True]*5+[after['inspection']]*3
                assert row['terminated']==(min(after['energy'],after['integrity'])<=manifest['config']['death_threshold'])
                bad=(int(action)==3 and before['position'] in (0,4) and before['quality'][before['position']//4]==0 and before['resources'][before['position']//4]>0)
                assert bad==row['unsafe_harvest'];unsafe+=bad;energy+=after['energy'];counts[action.name.lower()]+=1;count+=1
                if public.terminated:assert tick==ep['ticks']-1
            assert normalize(world.snapshot())==ep['final'];assert ep['survived']==world.viable()==ep['time_limited']
            assert ep['ticks']==1024 or not ep['survived']
            assert ep['actions']==counts and ep['unsafe_harvests']==unsafe;close(ep['mean_energy'],energy/ep['ticks'])
        assert not f.read().strip()
    for condition in ('stable','changing'):
        for policy in C.POLICIES:
            rows=[e for e in episodes if e['changing']==(condition=='changing') and e['policy']==policy]
            expected=dict(n=32,survivors=sum(e['survived'] for e in rows),mean_ticks=sum(e['ticks'] for e in rows)/32,
                mean_energy=sum(e['mean_energy'] for e in rows)/32,inspections=sum(e['actions']['inspect'] for e in rows),
                unsafe_harvests=sum(e['unsafe_harvests'] for e in rows))
            close(expected,data['aggregates'][condition][policy])
    a=data['aggregates'];s=a['stable'];c=a['changing'];m=c['public_memory']['survivors']
    bars=dict(informed_feasible=all(a[k]['informed']['survivors']>=29 for k in a),
        public_information_usable=all(a[k]['public_memory']['survivors']>=29 for k in a),
        revision_beats_frozen_map=s['frozen_map']['survivors']>=29 and m-c['frozen_map']['survivors']>=7,
        beats_reactive_and_fixed_route=m-max(c[p]['survivors'] for p in ('reactive','periodic'))>=7,
        constants_fail=all(a[k][p]['survivors']==0 for k in a for p in C.POLICIES if p.startswith('constant_')),
        map_saves_inspections=all(a[k]['public_memory']['survivors']>=a[k]['current_inspection']['survivors'] and
            a[k]['public_memory']['inspections']<=.5*a[k]['current_inspection']['inspections'] and a[k]['current_inspection']['inspections']>0 for k in a))
    assert bars==data['bars'];assert data['verdict']==('PASS' if all(bars.values()) else 'FAIL')
    assert data['memory_survival_advantage_over_current_inspection']==m-c['current_inspection']['survivors']
    assert not data['learned_adaptation_established'] and not data['automatic_followup']
    assert all(C.sha(ROOT/p)==digest for p,digest in manifest['sources'].items())
    audit=dict(passed=True,episodes=len(episodes),transitions=count,balance_absolute_tolerance=1e-12,
        checks=dict(source_artifact_identity=True,all_seeded_initializations=True,all_controller_choices=True,
                    all_deterministic_replays=True,independent_scalar_quality_and_event_balances=True,
                    public_sensors_masks_termination=True,all_episode_aggregates_and_decisions=True),
        completion_sha=C.sha(C.OUT/'completion.json'),audit_sha=C.sha(Path(__file__)),
        balance_helper_sha=C.sha(ROOT/'training/audit_lifetime_calibration.py'))
    C.save(C.OUT/'audit.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
