"""Complete audited scarce-birth anatomy; no neural edits or endpoint rescue."""
import gzip,json,math,subprocess,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training import run_known_birth_transfer2 as R,audit_known_birth_transfer2 as A,native_motor_teacher as T,audit_lmb1 as P
from training.native_physical_replay import CheckedWorld
from training.quality_learning_contract import reward

M=R.M
REPORT=ROOT/'zeus_sandbox/universe/reports/lbt2_failure_diagnosis_20260913.json'
AUDIT=ROOT/'zeus_sandbox/universe/reports/lbt2_audit_20260913.json'
SOURCES=('training/diagnose_lbt2_failures.py','training/test_lbt2_failure_diagnosis.py',
         'docs/lbt2_failure_diagnosis_protocol_20260913.md')


def prerequisites():
    manifest=R.verify();audit=M.L.P.L3.read(AUDIT)
    assert audit['status']=='PASS' and audit['verdict']=='FAIL' and audit['pillar_promotion'] is False
    assert audit['source']==manifest['source'] and audit['hardware_trajectory_exact'] is True
    assert audit['bodies']==2048 and audit['source_preparations']==512
    assert set(audit['evidence_sha'])=={'manifest.json','public_preparation.json','public_trace.jsonl.gz','endpoint.json','verdict.json'}
    for name,sha in audit['evidence_sha'].items():assert M.L.P.L3.sha(R.OUT/name)==sha
    assert audit['report_sha']==M.L.P.L3.sha(R.REPORT)
    assert M.L.P.L3.read(R.REPORT)==dict(**M.L.P.L3.read(R.OUT/'verdict.json'),manifest=manifest,independent_audit_required=True)
    assert M.L.P.L3.read(R.OUT/'verdict.json')['verdict']=='FAIL'
    assert 0<=audit['max_local_probability_error']<2e-5 and 0<=audit['max_local_state_error']<1e-4
    return manifest,audit


def group():
    return dict(bodies=0,steps=0,initial_quality_correct=0,initial_first_sample_correct=0,
        initial_first_argmax_correct=0,initial_target_probability_sum=0.,teacher_sample_agreement=0,
        teacher_argmax_agreement=0,actions=Counter(),interior_actions=Counter(),first8_interior_actions=Counter(),
        fatal_causes=Counter(),no_safe_harvest=0,no_positive_feeding=0,slow_state_changed=0,
        first_safe_harvest_steps=[],first_positive_feeding_steps=[],death_steps=[])


def decision(row,target):
    p=row['probability'];assert len(p)==6 and all(math.isfinite(x) and 0<=x<=1 for x in p)
    assert abs(sum(p)-1)<2e-5
    argmax=max(range(6),key=lambda j:p[j])
    return dict(tick=row['tick'],action=row['action'],argmax=argmax,teacher=target,
        observation=row['observation'],next_observation=row['next_observation'],reward=row['reward'],
        body_done=row['body_done'],terminated=row['terminated'],probability=p,h=row['h'],z=row['z'],
        audit_tool_before=row['audit_tool_before'],audit_tool_after=row['audit_tool_after'])


def anatomy(prep,natural,records,trial,config=R.CONFIG):
    assert len(records)==natural['ticks'] and records
    assert (natural['trial'],natural['energy'],natural['index'],natural['seed'],natural['side'],natural['quality'],natural['target'])==(
        trial,prep['energy'],prep['index'],prep['seed'],prep['side'],prep['quality'],prep['target'])
    assert natural['initial_z']==natural['written_z'] and natural['storage_distance']==0 and natural['fast_reset']
    teacher=T.initial_teacher(prep['bodies'][0][2]['next_observation']);safe=teacher['safe']
    world=CheckedWorld(A.world_for(prep['seed'],3,prep['energy'],config),records)
    assert list(world.observation().values())==prep['query']
    firsts={name:None for name in ('safe_patch','unsafe_patch','safe_harvest','positive_feeding','inspection','slow_state_change')}
    feeds=repairs=inspections=bad=0;last_feed=last_repair=None;rows=[];stats=group()
    for tick,row in enumerate(records):
        assert (row['trial'],row['energy'],row['index'],row['tick'])==(trial,prep['energy'],prep['index'],tick)
        assert row['observation']==list(world.observation().values())
        target=T.action(world.observation(),teacher);record=decision(row,target);effect=world.step(row['action'])
        public=P.public_record(effect,tick,config['horizon']);assert {name:row[name] for name in public}==public
        assert reward(effect)==row['reward']
        teacher=T.observe(effect,teacher);elapsed=tick+1
        positive=row['action']==3 and effect.after.energy>effect.before.energy
        restoration=row['audit_tool_after']>row['audit_tool_before']
        feeds+=positive;repairs+=restoration;inspections+=row['action']==4
        bad+=row['action']==3 and effect.after.integrity<effect.before.integrity-world.config.integrity_decay
        if positive:last_feed=elapsed
        if restoration:last_repair=elapsed
        events=dict(safe_patch=effect.after.position==safe,unsafe_patch=effect.after.position==1-safe,
            safe_harvest=row['action']==3 and effect.before.position==safe,
            positive_feeding=positive,inspection=row['action']==4,slow_state_change=row['z']!=natural['initial_z'])
        for name,happened in events.items():
            if happened and firsts[name] is None:firsts[name]=elapsed
        stats['steps']+=1;stats['actions'][row['action']]+=1
        stats['teacher_sample_agreement']+=row['action']==target;stats['teacher_argmax_agreement']+=record['argmax']==target
        if row['observation'][2] in (.25,.5,.75):
            stats['interior_actions'][row['action']]+=1
            if tick<8:stats['first8_interior_actions'][row['action']]+=1
        rows.append(record)
    world.complete();assert row['body_done'] and natural['survived']==(not effect.terminated)
    assert natural['ticks']==config['horizon'] or effect.terminated
    assert (feeds,repairs,inspections,bad)==(natural['feeding'],natural['repairs'],natural['inspections'],natural['bad_harvest'])
    assert (effect.after.energy,effect.after.integrity)==(natural['final_energy'],natural['final_integrity'])
    assert records[-1]['z']==natural['final_z']
    assert records[0]['action']==natural['first_action'] and natural['first_correct']==(records[0]['action']==prep['target'])
    assert natural['quality_correct']==((natural['quality_probability']>=.5)==bool(prep['quality']))
    energy=effect.after.energy<=world.config.death_threshold;integrity=effect.after.integrity<=world.config.death_threshold
    cause=None
    if not natural['survived']:
        assert energy or integrity;cause='both' if energy and integrity else 'energy_only' if energy else 'integrity_only'
        stats['fatal_causes'][cause]+=1;stats['death_steps'].append(natural['ticks'])
    stats['bodies']=1;stats['initial_quality_correct']=int(natural['quality_correct'])
    stats['initial_first_sample_correct']=int(natural['first_correct'])
    stats['initial_first_argmax_correct']=int(rows[0]['argmax']==prep['target'])
    stats['initial_target_probability_sum']=records[0]['probability'][prep['target']]
    stats['no_safe_harvest']=int(firsts['safe_harvest'] is None);stats['no_positive_feeding']=int(firsts['positive_feeding'] is None)
    stats['slow_state_changed']=int(firsts['slow_state_change'] is not None)
    if firsts['safe_harvest'] is not None:stats['first_safe_harvest_steps'].append(firsts['safe_harvest'])
    if firsts['positive_feeding'] is not None:stats['first_positive_feeding_steps'].append(firsts['positive_feeding'])
    summary=dict(trial=trial,index=prep['index'],energy=prep['energy'],side=prep['side'],quality=prep['quality'],
        survived=natural['survived'],ticks=natural['ticks'],cause=cause,initial_quality_correct=natural['quality_correct'],
        initial_quality_probability=natural['quality_probability'],first_sample_correct=natural['first_correct'],
        first_argmax_correct=rows[0]['argmax']==prep['target'],first_target_probability=stats['initial_target_probability_sum'],
        first_event_steps=firsts,terminal_energy=effect.after.energy,terminal_integrity=effect.after.integrity,
        terminal_position=effect.after.position,terminal_action=row['action'],
        steps_since_positive_feeding=None if last_feed is None else natural['ticks']-last_feed,
        steps_since_tool_restoration=None if last_repair is None else natural['ticks']-last_repair,
        final_slow_distance=math.sqrt(sum((a-b)**2 for a,b in zip(natural['final_z'],natural['initial_z']))),
        first8=rows[:8])
    fatal=dict(**summary,last16=rows[-16:]) if cause else None
    return summary,stats,fatal


def merge(target,source):
    assert set(target)==set(source)
    for name,value in source.items():
        if isinstance(value,Counter):target[name].update(value)
        elif isinstance(value,list):target[name].extend(value)
        else:target[name]+=value


def main():
    assert __debug__ and not REPORT.exists();manifest,audit=prerequisites()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in SOURCES:
        blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    preps=M.L.P.L3.read(R.OUT/'public_preparation.json');endpoint=M.L.P.L3.read(R.OUT/'endpoint.json')
    assert [(p['energy'],p['index']) for p in preps]==[(e,i) for e in R.CONFIG['energies'] for i in range(R.CONFIG['ecologies'])]
    source_steps=sum(A.preparation(p) for p in preps);groups={};summaries=[];fatal=[]
    for trial in range(R.CONFIG['trials']):
        for energy in R.CONFIG['energies']:
            for side in (0,1):
                for quality in (0,1):
                    for survived in (False,True):groups[trial,energy,side,quality,survived]=group()
    with gzip.open(R.OUT/'public_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        trace=(json.loads(line) for line in stream)
        for trial in range(R.CONFIG['trials']):
            for prep in preps:
                natural=endpoint[len(summaries)];records=[next(trace) for _ in range(natural['ticks'])]
                summary,stats,death=anatomy(prep,natural,records,trial)
                key=(trial,prep['energy'],prep['side'],prep['quality'],natural['survived']);merge(groups[key],stats)
                summaries.append(summary)
                if death:fatal.append(death)
            print('complete scarce-birth failure anatomy',trial,flush=True)
        assert next(trace,None) is None
    assert len(summaries)==len(endpoint)==audit['bodies']
    assert len(fatal)==sum(not r['survived'] for r in endpoint)
    steps=sum(g['steps'] for g in groups.values());assert steps==audit['decisions'] and steps+source_steps==audit['public_physical_steps_replayed']
    rows=[dict(trial=k[0],energy=k[1],side=k[2],quality=k[3],survived=k[4],**g) for k,g in groups.items()]
    result=dict(status='COMPLETE',verdict='FAIL',pillar_promotion=False,commit=commit,
        sources={name:M.L.P.L3.sha(ROOT/name) for name in SOURCES},transfer_audit_sha=M.L.P.L3.sha(AUDIT),
        source=manifest['source'],source_preparations=len(preps),source_steps_replayed=source_steps,bodies=len(summaries),
        steps_replayed=steps,fatal_bodies=len(fatal),groups=rows,body_summaries=summaries,fatal_traces=fatal,
        scope='Audited endpoint descriptive anatomy only; no neural intervention, optimality, recovered viability or memory-benefit claim')
    M.C.save(REPORT,result);print(json.dumps({k:v for k,v in result.items() if k not in ('sources','source','groups','body_summaries','fatal_traces')},indent=2))


if __name__=='__main__':main()
