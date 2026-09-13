"""Complete independently audited maintenance anatomy; no controller rescue."""
import gzip,json,math,subprocess,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training import run_long_neural_maintenance as R,audit_long_neural_maintenance as A,native_motor_teacher as T
from training.native_physical_replay import CheckedWorld
from training.quality_learning_contract import reward
M=R.R
AUDIT=ROOT/'zeus_sandbox/universe/reports/lmt1_audit_20260913.json'
REPORT=ROOT/'zeus_sandbox/universe/reports/lmt1_failure_diagnosis_20260913.json'
SOURCES=('training/diagnose_lmt1_failures.py','training/test_lmt1_failure_diagnosis.py','docs/lmt1_failure_diagnosis_protocol_20260913.md')


def prerequisites():
    manifest=R.verify();audit=M.L.P.L3.read(AUDIT)
    assert audit['status']=='PASS' and audit['verdict']=='FAIL' and audit['pillar_promotion'] is False
    assert audit['source']==manifest['source'] and audit['hardware_trajectory_exact'] is True and audit['bodies']==2048
    names={'manifest.json','public_preparation.json','public_trace.jsonl.gz','endpoint.json','verdict.json'}
    assert set(audit['evidence_sha'])==names
    for name,sha in audit['evidence_sha'].items():assert M.L.P.L3.sha(R.OUT/name)==sha
    assert audit['report_sha']==M.L.P.L3.sha(R.REPORT)
    assert M.L.P.L3.read(R.REPORT)==dict(**M.L.P.L3.read(R.OUT/'verdict.json'),manifest=manifest,independent_audit_required=True)
    assert 0<=audit['max_local_probability_error']<2e-5 and 0<=audit['max_local_state_error']<1e-4
    return manifest,audit


def group():
    return dict(bodies=0,steps=0,feeding=0,effective_repairs=0,ineffective_maintain=0,inspections=0,
        extra_harvest_integrity_loss=0,safe_patch_wait=0,unsafe_patch_harvest=0,teacher_sample_agreement=0,
        teacher_argmax_agreement=0,actions=Counter(),interior_actions=Counter(),fatal_causes=Counter(),
        no_feeding=0,no_effective_repairs=0,slow_state_changed=0,death_steps=[])


def anatomy(prep,natural,records,trial,mode,control,config=R.CONFIG):
    assert len(records)==natural['ticks'] and records
    assert (natural['trial'],natural['index'],natural['mode'],natural['control'],natural['seed'],natural['side'],natural['quality'])==(
        trial,prep['index'],mode,control,prep['seed'],prep['side'],prep['quality'])
    assert natural['fast_reset'] and natural['storage_distance']==0
    assert natural['initial_z']==(natural['written_z'] if mode=='inherited' else [0.]*8)
    world=CheckedWorld(A.world_for(prep['seed'],3,control=='enabled',config),records)
    assert list(world.observation().values())==prep['query']
    cue=prep['bodies'][0][2]['next_observation'];safe=int(cue[2]) if cue[7] else 1-int(cue[2])
    teacher=T.initial_teacher(cue if mode=='inherited' else None)
    firsts={name:None for name in ('safe_patch','positive_feeding','tool_restoration','inspection','slow_state_change')}
    last_feed=last_repair=None;rows=[];stats=group()
    for tick,row in enumerate(records):
        assert (row['trial'],row['index'],row['mode'],row['control'],row['tick'])==(trial,prep['index'],mode,control,tick)
        assert row['observation']==list(world.observation().values())
        target=T.action(world.observation(),teacher);p=row['probability'];assert len(p)==6 and all(math.isfinite(v) and 0<=v<=1 for v in p) and abs(sum(p)-1)<2e-5
        argmax=max(range(6),key=lambda j:p[j]);effect=world.step(row['action']);teacher=T.observe(effect,teacher)
        public=A.A.public_record(effect,tick,config['horizon']);assert {k:row[k] for k in public}==public and row['reward']==reward(effect)
        feeding=row['action']==3 and effect.after.energy>effect.before.energy
        restoration=row['audit_tool_after']>row['audit_tool_before'];elapsed=tick+1
        extra=row['action']==3 and effect.after.integrity<effect.before.integrity-world.config.integrity_decay
        stats['steps']+=1;stats['feeding']+=feeding;stats['effective_repairs']+=restoration
        stats['ineffective_maintain']+=row['action']==5 and not restoration;stats['inspections']+=row['action']==4
        stats['extra_harvest_integrity_loss']+=extra;stats['safe_patch_wait']+=row['action']==0 and effect.before.position==safe
        stats['unsafe_patch_harvest']+=row['action']==3 and effect.before.position==1-safe
        stats['teacher_sample_agreement']+=row['action']==target;stats['teacher_argmax_agreement']+=argmax==target
        stats['actions'][row['action']]+=1
        if effect.before.position in (.25,.5,.75):stats['interior_actions'][row['action']]+=1
        events=dict(safe_patch=effect.after.position==safe,positive_feeding=feeding,tool_restoration=restoration,
            inspection=row['action']==4,slow_state_change=row['z']!=natural['initial_z'])
        for name,happened in events.items():
            if happened and firsts[name] is None:firsts[name]=elapsed
        if feeding:last_feed=elapsed
        if restoration:last_repair=elapsed
        rows.append(dict(tick=tick,action=row['action'],argmax=argmax,teacher=target,observation=row['observation'],
            next_observation=row['next_observation'],reward=row['reward'],body_done=row['body_done'],terminated=row['terminated'],
            probability=p,h=row['h'],z=row['z'],audit_tool_before=row['audit_tool_before'],audit_tool_after=row['audit_tool_after']))
    world.complete();assert natural['survived']==(not effect.terminated) and rows[-1]['body_done']
    assert (stats['feeding'],stats['effective_repairs'],stats['inspections'],stats['extra_harvest_integrity_loss'])==(
        natural['feeding'],natural['repairs'],natural['inspections'],natural['bad_harvest'])
    assert (effect.after.energy,effect.after.integrity)==(natural['energy'],natural['integrity']) and row['z']==natural['final_z']
    assert natural['first_action']==records[0]['action'] and natural['first_correct']==(records[0]['action']==prep['target'])
    assert natural['quality_correct']==((natural['quality_probability']>=.5)==bool(prep['quality']))
    cause=None
    if not natural['survived']:
        low_e=effect.after.energy<=world.config.death_threshold;low_i=effect.after.integrity<=world.config.death_threshold;assert low_e or low_i
        cause='both' if low_e and low_i else 'energy_only' if low_e else 'integrity_only';stats['fatal_causes'][cause]+=1;stats['death_steps'].append(natural['ticks'])
    stats['bodies']=1;stats['no_feeding']=int(stats['feeding']==0);stats['no_effective_repairs']=int(stats['effective_repairs']==0)
    stats['slow_state_changed']=int(firsts['slow_state_change'] is not None)
    summary=dict(trial=trial,index=prep['index'],mode=mode,control=control,side=prep['side'],quality=prep['quality'],
        ticks=natural['ticks'],survived=natural['survived'],cause=cause,first_event_steps=firsts,first8=rows[:8],
        feeding=stats['feeding'],effective_repairs=stats['effective_repairs'],ineffective_maintain=stats['ineffective_maintain'],
        terminal_energy=effect.after.energy,terminal_integrity=effect.after.integrity,terminal_position=effect.after.position,
        terminal_action=row['action'],terminal_tool=row['audit_tool_after'],
        steps_since_feeding=None if last_feed is None else natural['ticks']-last_feed,
        steps_since_restoration=None if last_repair is None else natural['ticks']-last_repair)
    return summary,stats,dict(**summary,last16=rows[-16:]) if cause else None


def main():
    assert __debug__ and not REPORT.exists();manifest,audit=prerequisites()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in SOURCES:
        blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    preps=M.L.P.L3.read(R.OUT/'public_preparation.json');actual=M.L.P.L3.read(R.OUT/'endpoint.json');source_steps=sum(A.preparation(p) for p in preps)
    assert [p['index'] for p in preps]==list(range(R.CONFIG['ecologies']));groups={};summaries=[];fatal=[]
    for t in range(R.CONFIG['trials']):
        for mode in R.CONFIG['modes']:
            for control in R.CONFIG['controls']:
                for side in (0,1):
                    for q in (0,1):
                        for survived in (False,True):groups[t,mode,control,side,q,survived]=group()
    with gzip.open(R.OUT/'public_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        trace=(json.loads(line) for line in stream)
        for t in range(R.CONFIG['trials']):
            for prep in preps:
                for mode in R.CONFIG['modes']:
                    for control in R.CONFIG['controls']:
                        natural=actual[len(summaries)];records=[next(trace) for _ in range(natural['ticks'])]
                        summary,stats,death=anatomy(prep,natural,records,t,mode,control);summaries.append(summary)
                        target=groups[t,mode,control,prep['side'],prep['quality'],natural['survived']]
                        for name,value in stats.items():
                            if isinstance(value,Counter):target[name].update(value)
                            elif isinstance(value,list):target[name].extend(value)
                            else:target[name]+=value
                        if death:fatal.append(death)
            print('complete maintenance failure anatomy',t,flush=True)
        assert next(trace,None) is None
    steps=sum(g['steps'] for g in groups.values());assert len(summaries)==len(actual)==audit['bodies']
    assert len(fatal)==sum(not r['survived'] for r in actual) and steps==audit['decisions'] and steps+source_steps==audit['public_physical_steps']
    result=dict(status='COMPLETE',verdict='FAIL',pillar_promotion=False,commit=commit,sources={p:M.L.P.L3.sha(ROOT/p) for p in SOURCES},
        maintenance_audit_sha=M.L.P.L3.sha(AUDIT),source=manifest['source'],bodies=len(summaries),fatal_bodies=len(fatal),
        source_steps_replayed=source_steps,steps_replayed=steps,groups=[dict(trial=k[0],mode=k[1],control=k[2],side=k[3],quality=k[4],survived=k[5],**g) for k,g in groups.items()],
        body_summaries=summaries,fatal_traces=fatal,scope='Complete audited descriptive anatomy only; no intervention, optimum, recovered viability or newly fitted model claim')
    M.C.save(REPORT,result);print(json.dumps({k:v for k,v in result.items() if k not in ('sources','source','groups','body_summaries','fatal_traces')},indent=2))


if __name__=='__main__':main()
