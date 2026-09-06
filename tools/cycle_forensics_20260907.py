"""Saved-action replay and prespecified cycle-survival analysis; no policy calls."""
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.embodiment import Action, EmbodiedWorldV2

REPORTS = ROOT / 'zeus_sandbox/universe/reports'
OUT = REPORTS / 'cycle_forensics_20260907.json.gz'
SOURCES = ['core/embodiment.py', 'tools/cycle_forensics_20260907.py',
           'docs/cycle_forensics_analysis_protocol_20260907.md',
           'zeus_sandbox/universe/reports/qv1_postmortem_20260906.json.gz',
           'runs/cap1_20260906/pol2.json.gz',
           'zeus_sandbox/universe/reports/pol2_endogenous_action_verdict_20260905.json',
           'zeus_sandbox/universe/reports/qv1_retention_lineage_verdict_20260906.json',
           'zeus_sandbox/universe/reports/cycle_ceiling_20260906.json',
           'training/cycle_ceiling_sim.py']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    with (gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)) as f:
        return json.load(f)

def cause(w):
    return '+'.join(k for k, bad in [('energy', w.body.energy <= .05),
        ('integrity', w.body.integrity <= .05),
        ('temperature', not .05 < w.body.temperature < .95)] if bad) or 'none'

def closures(positions, harvested):
    """harvested[t] belongs to action p[t] -> p[t+1]; inclusive path ends."""
    found, last = [], 0
    movement = np.array([a != b for a, b in zip(positions, positions[1:])])
    for end in range(1, len(positions)):
        for start in range(end - 1, last - 1, -1):
            if positions[start] != positions[end]:
                continue
            moves = int(movement[start:end].sum())
            if moves < 4:
                continue
            span = max(positions[start:end+1]) - min(positions[start:end+1])
            if span < 2 or not any(harvested[t] >= .03 and positions[t] != positions[start]
                                   for t in range(start, end)):
                continue
            found.append(dict(start=start, end=end, duration=end-start, span=span, movements=moves))
            last = end
            break
    durations = [x['duration'] for x in found]
    return dict(intervals=found, count=len(found), visited_cells=len(set(positions)),
                actual_movements=int(movement.sum()),
                path_cyclicity=sum(x['movements'] for x in found)/max(1, int(movement.sum())),
                duration_median=float(np.median(durations)) if durations else None,
                duration_cv=float(np.std(durations)/np.mean(durations)) if len(durations)>=3 else None)

def detector_checks():
    cases = [([1]*10, [0]*9, 0), ([0]*10, [0]*9, 0),
             ([0,1,2,3], [0]*3, 0), ([0,1,1,0,1,0], [0,.1,0,0,0], 0),
             ([0,1,2,2,1,0], [0,0,.1,0,0], 1),
             ([0,1,2,2,1,0], [0]*5, 0)]
    for p, h, expected in cases:
        assert closures(p,h)['count'] == expected
    p = [0,1,2,2,1,0,1,2,2,1,0]
    assert closures(p,[0,0,.1,0,0,0,0,.1,0,0])['count'] == 2
    return True

def step_account(w, action):
    b = w.body
    before, field, position = list(w.observation()), list(w.resources), b.position
    e, integ, temp = before[:3]
    e -= .018
    ambient = .5 + .42*math.sin((w.step_count+w.ambient_phase)/23)
    temp += .1*(ambient-temp)
    cost = {0:0, 1:.008, 2:.008, 3:0, 4:.012, 5:.006}[int(action)]
    amount = min(.13, field[position]) if action == Action.HARVEST else 0.
    penalty = .006 if action == Action.HARVEST and amount < .03 else 0.
    e += .8*amount-cost-penalty
    repair = 0.
    if action == Action.REST:
        repair = .018 if e > .25 and abs(temp-.5) < .25 else 0.
        temp += .05*(.5-temp)
    elif action == Action.REGULATE:
        temp += .45*(.5-temp)
        repair = .004 if abs(temp-.5)<.18 and e>.25 else 0.
    starvation, thermal = .10*max(0,.25-e), .12*max(0,abs(temp-.5)-.18)
    raw_i = integ+repair-.002-starvation-thermal
    clip_e, clip_i = w._clip(e)-e, w._clip(raw_i)-raw_i
    effect = w.step(action)
    assert abs(w.body.energy-(e+clip_e))<1e-12
    assert abs(w.body.integrity-(raw_i+clip_i))<1e-12
    assert abs(w.body.temperature-w._clip(temp))<1e-12
    return dict(tick=w.body.age, action=action.name.lower(), before=before,
                after=list(effect['after']), resources_before=field, resources_after=list(w.resources),
                position_before=position, position_after=w.body.position,
                actual_movement=int(position!=w.body.position), harvested_resource=amount,
                basal_cost=.018, action_cost=cost, harvested_energy=.8*amount,
                empty_harvest_penalty=penalty, energy_clipping=clip_e,
                integrity_repair=repair, integrity_basal=.002, starvation_damage=starvation,
                thermal_damage=thermal, integrity_clipping=clip_i, viable=effect['viable'])

def episode(saved, qv):
    w = EmbodiedWorldV2(seed=saved['seed'])
    initial = dict(observation=list(w.observation()), capacity=list(w.capacity),
                   resources=list(w.resources), ambient_phase=w.ambient_phase)
    traces = saved['trace'] if qv else saved['actions']
    result, positions, amounts = [], [w.body.position], []
    for j, item in enumerate(traces):
        assert w.viable(), 'saved trajectory continues after death'
        action = Action[item['action'].upper()] if qv else Action(item)
        expected = item['before'] if qv else saved['observations'][j]
        assert np.allclose(w.observation(), expected, atol=1e-12, rtol=0)
        row = step_account(w, action)
        if qv:
            assert np.allclose(row['after'], item['after'], atol=1e-12, rtol=0)
            assert row['viable'] == item['viable']
        result.append(row)
        positions.append(w.body.position)
        amounts.append(row['harvested_resource'])
    assert w.body.age == saved['age']
    survived = saved['survived'] if qv else saved['completed']
    assert survived == w.viable()
    if 'failure_cause' in saved:
        assert cause(w) == saved['failure_cause']
    tail = result[-20:] if not survived else []
    harvests = [r for r in tail if r['action']=='harvest']
    flags = dict(no_actual_movement=not any(r['actual_movement'] for r in tail),
                 regulation_majority=sum(r['action']=='regulate' for r in tail)>=len(tail)/2,
                 mostly_empty_harvest=bool(harvests) and sum(r['harvested_resource']<.03 for r in harvests)>=len(harvests)/2,
                 thermal_damage_present=any(r['thermal_damage']>0 for r in tail)) if tail else {}
    totals_keys = ['basal_cost','action_cost','harvested_energy','empty_harvest_penalty',
                   'energy_clipping','integrity_repair','integrity_basal','starvation_damage','thermal_damage','integrity_clipping']
    return dict(seed=saved['seed'], age=w.body.age, survived_256=survived, failure_cause=cause(w),
                initial=initial, final=list(w.observation()), cycles=closures(positions,amounts),
                tail=tail, tail_flags=flags, tail_actions=dict(Counter(r['action'] for r in tail)),
                tail_totals={k:sum(r[k] for r in tail) for k in totals_keys},
                lifetime_totals={k:sum(r[k] for r in result) for k in totals_keys},
                first_resource_below_0_03=next((r['tick'] for r in result if r['after'][3]<.03),None),
                terminal_action=result[-1]['action'])

def association(conditions, landmark):
    arrays, desc = {}, {}
    for name, episodes in conditions.items():
        # At the exact death tick the body is not viable; censoring is explicitly alive.
        rows = np.array([[e['age']>landmark or (e['age']==landmark and e['survived_256']),
                          any(c['end']<=landmark for c in e['cycles']['intervals']),
                          e['survived_256'], max(0,min(e['age'],256)-landmark)] for e in episodes], dtype=float)
        risk, cycle = rows[:,0].astype(bool), rows[:,1].astype(bool)
        groups = [risk & ~cycle, risk & cycle]
        n = [int(g.sum()) for g in groups]
        rates = [float(rows[g,2].mean()) if g.any() else None for g in groups]
        desc[name] = dict(noncycle_n=n[0], cycle_n=n[1], survival_rates=rates,
                          remaining_life_means=[float(rows[g,3].mean()) if g.any() else None for g in groups],
                          supported=min(n)>=8,
                          difference=rates[1]-rates[0] if min(n)>0 else None)
        arrays[name] = rows
    supported = [k for k,d in desc.items() if d['supported']]
    identified = 'POL2/normal' in supported and any(k.startswith('QV1/') for k in supported)
    point = float(np.mean([desc[k]['difference'] for k in supported])) if supported else None
    lower, upper, missing = [], [], 0
    if supported:
        rng = np.random.Generator(np.random.PCG64(20260987))
        for _ in range(10000):
            qidx, pidx = rng.integers(0,64,64), rng.integers(0,64,64)
            values = []
            for name in supported:
                rows = arrays[name][qidx if name.startswith('QV1/') else pidx]
                groups = [(rows[:,0]>0)&(rows[:,1]==v) for v in [0,1]]
                if not all(g.any() for g in groups):
                    break
                values.append(float(rows[groups[1],2].mean()-rows[groups[0],2].mean()))
            if len(values)!=len(supported):
                missing += 1
                lower.append(-1.)
                upper.append(1.)
            else:
                lower.append(float(np.mean(values)))
                upper.append(lower[-1])
    ci = [float(np.quantile(lower,.025)),float(np.quantile(upper,.975))] if lower else None
    negative = any(desc[k]['difference']<0 for k in supported)
    verdict = 'UNDECIDED'
    if identified and (point<=0 or negative):
        verdict = 'FAIL'
    elif identified and ci[0]>0 and not negative:
        verdict = 'PASS'
    return dict(landmark=landmark, conditions=desc, supported=supported, identified=identified,
                point=point, ci95=ci, bootstrap_missing_group_fraction=missing/10000 if supported else None,
                verdict=verdict)

def summarize(episodes):
    deaths = [e for e in episodes if not e['survived_256']]
    return dict(n=len(episodes), deaths=len(deaths), survivors_256=len(episodes)-len(deaths),
                mean_age=float(np.mean([e['age'] for e in episodes])),
                failure_causes=dict(Counter(e['failure_cause'] for e in deaths)),
                tail_flags={k:sum(e['tail_flags'][k] for e in deaths) for k in deaths[0]['tail_flags']} if deaths else {},
                tail_actions=dict(sum((Counter(e['tail_actions']) for e in deaths),Counter())),
                tail_total_means={k:float(np.mean([e['tail_totals'][k] for e in deaths])) for k in deaths[0]['tail_totals']} if deaths else {},
                lifetime_cyclers=sum(e['cycles']['count']>0 for e in episodes),
                lifetime_groups={str(c):dict(n=sum((e['cycles']['count']>0)==c for e in episodes),
                    survivors=sum(e['survived_256'] and (e['cycles']['count']>0)==c for e in episodes),
                    ages=[e['age'] for e in episodes if (e['cycles']['count']>0)==c]) for c in [False,True]},
                mean_actual_movements=float(np.mean([e['cycles']['actual_movements'] for e in episodes])))

def main():
    assert not OUT.exists(), 'refusing existing report'
    assert detector_checks()
    hashes = {s:sha(ROOT/s) for s in SOURCES}
    assert hashes['core/embodiment.py']=='984f0c2253204d57688ea972064aaccd684f4fc006bbcd378752b343c745b3b2'
    assert hashes['runs/cap1_20260906/pol2.json.gz']=='0579dd4c666a81ecfdee93a93fb00dc48c830e819143ecbfe2a15e8ef78fd718'
    q = read(ROOT/SOURCES[3])
    # Verify exact uncompressed archive identity against the prior published audit.
    with gzip.open(ROOT/SOURCES[3], 'rb') as f:
        assert hashlib.sha256(f.read()).hexdigest()=='089bd7e7bcdb986df2545e39fa5bf5cdcd3ad2dc83ceec995d962718ca3c3ab2'
    p = read(ROOT/SOURCES[4])
    conditions = {}
    for arm, splits in q['D4'].items():
        for split, cell in splits.items():
            name = f'QV1/{arm}/{split}'
            conditions[name] = [episode(e,True) for e in cell['episodes']]
            print('replayed',name,flush=True)
    for name, episodes in p['conditions'].items():
        conditions[f'POL2/{name}'] = [episode(e,False) for e in episodes]
        print('replayed POL2',name,flush=True)
    original = read(ROOT/SOURCES[5])['evaluations']['normal']['episodes']
    for old,new in zip(original,conditions['POL2/normal']):
        assert old['seed']==new['seed'] and old['age']==new['age']
        assert old['completed']==new['survived_256']
        if 'failure_cause' in old:
            assert old['failure_cause']==new['failure_cause']
    for old, new in zip(q['D5'],conditions['QV1/inherited_recurrent/heldout_greedy']):
        assert old['seed']==new['seed'] and old['age']==new['age']
        for key in ['basal_cost','action_cost','harvested_energy','empty_harvest_penalty']:
            assert abs(old[key]-new['lifetime_totals'][key])<1e-12
        assert abs(old['clipping_adjustment']-new['lifetime_totals']['energy_clipping'])<1e-12
    primary = {k:v for k,v in conditions.items() if 'heldout_' in k or k=='POL2/normal'}
    gate = association(primary,64)
    report = dict(kind='CYC-F', source_hashes=hashes, all_replays_and_balances_verified=True,
                  synthetic_detector_checks=True, D5_accounts_verified=True, grade='diagnostic_Class_O',
                  primary=gate, sensitivities={str(t):association(primary,t) for t in [32,128]},
                  summaries={k:summarize(v) for k,v in conditions.items()}, episodes=conditions,
                  verdict=gate['verdict'], cycle_training_proposal_eligible=gate['verdict']=='PASS',
                  cycle_training_launch_authorized=False)
    assert hashes=={s:sha(ROOT/s) for s in SOURCES}, 'sources changed during analysis'
    with gzip.open(OUT,'xt',encoding='utf-8') as f:
        json.dump(report,f,allow_nan=False,separators=(',',':'))
    print(json.dumps({k:report[k] for k in ['verdict','primary','summaries']},indent=2))

if __name__=='__main__':
    main()
