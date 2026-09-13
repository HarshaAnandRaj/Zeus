"""Complete read-only anatomy of the independently verified LMB5 failure."""
import gzip, math, subprocess, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training import run_lmb5 as R, native_motor_teacher as T, audit_lmb1 as P
from training.native_physical_replay import CheckedWorld
from training.quality_learning_contract import reward
M=R.M
AUDIT=ROOT/'zeus_sandbox/universe/reports/lmb5_audit_20260913.json'
REPORT=ROOT/'zeus_sandbox/universe/reports/lmb5_failure_diagnosis2_20260913.json'
SOURCES=('training/diagnose_lmb5_failures.py','training/diagnose_lmb5_failures2.py','training/test_lmb5_failure_diagnosis2.py','docs/lmb5_failure_diagnosis2_protocol_20260913.md')

def prerequisites():
    manifest=R.verify();audit=M.L.P.L3.read(AUDIT)
    assert audit['status']=='PASS' and audit['verdict']=='FAIL' and audit['qualification'] is False
    assert audit['selected_qualified_arm'] is None and audit['pillar_promotion'] is False
    assert audit['hardware_trajectory_exact'] is True and audit['endpoint_bodies']==7168
    assert len(audit['fit_pairs'])==8 and audit['training_batches_verified']==768
    names={'manifest.json','calibration.json','calibration_audit.json','endpoint_public.json','regression_public.json','verdict.json','warm_endpoint.json'}
    names.update(f'{a}_{k}.json' for a in R.K.CONFIG['arms'] for k in ('calibration_public','training_a','training_b','endpoint'))
    names.update(f'{v}_trace.jsonl.gz' for v in (*R.K.CONFIG['arms'],'warm'))
    assert set(audit['evidence_sha'])==names
    for name,sha in audit['evidence_sha'].items():assert M.L.P.L3.sha(R.OUT/name)==sha
    assert audit['report_sha']==M.L.P.L3.sha(R.REPORT)
    verdict=M.L.P.L3.read(R.OUT/'verdict.json');assert verdict['verdict']=='FAIL'
    assert M.L.P.L3.read(R.REPORT)==dict(**verdict,manifest=manifest,independent_audit_required=True)
    assert 0<=audit['max_local_probability_error']<2e-5 and 0<=audit['max_local_state_error']<1e-4
    return manifest,audit

def group():
    return dict(bodies=0,steps=0,feeding=0,repairs=0,inspections=0,safe_patch_wait=0,
        slow_state_changed=0,no_feeding=0,actions=Counter(),fatal_causes=Counter(),
        teacher_sample_agreement=0,teacher_argmax_agreement=0)

def anatomy(prep,natural,records,trial,variant,mode,config=R.K.CONFIG):
    assert mode in ('inherited','empty') and variant in (*config['arms'],'warm')
    assert records and len(records)==natural['ticks']
    assert (natural['trial'],natural['variant'],natural['mode'],natural['index'],natural['energy'],natural['seed'],natural['side'],natural['quality'],natural['target'])==(
        trial,variant,mode,prep['index'],prep['energy'],prep['seed'],prep['side'],prep['quality'],prep['target'])
    assert natural['fast_reset'] is True and natural['storage_distance']==0
    assert natural['initial_z']==(natural['written_z'] if mode=='inherited' else [0.]*8)
    world=CheckedWorld(R.S.A.world_for(prep['seed'],3,prep['energy'],R.R.CONFIG|dict(horizon=config['body_horizon'])),records)
    assert list(world.observation().values())==prep['query']
    cue=prep['bodies'][0][2]['next_observation'];safe=int(cue[2]) if cue[7] else 1-int(cue[2])
    teacher=T.initial_teacher(cue if mode=='inherited' else None)
    firsts={k:None for k in ('safe_patch','positive_feeding','tool_restoration','inspection','slow_state_change')}
    stats=group();rows=[];bad=0;last_feed=last_repair=None
    for tick,row in enumerate(records):
        assert (row['trial'],row['variant'],row['mode'],row['energy'],row['index'],row['tick'])==(trial,variant,mode,prep['energy'],prep['index'],tick)
        assert row['observation']==list(world.observation().values())
        p=row['probability'];assert len(p)==6 and all(math.isfinite(v) and 0<=v<=1 for v in p) and abs(sum(p)-1)<2e-5
        target=T.action(world.observation(),teacher);argmax=max(range(6),key=lambda j:p[j])
        effect=world.step(row['action']);public=P.public_record(effect,tick,config['body_horizon'])
        assert public=={k:row[k] for k in public} and row['reward']==reward(effect)
        teacher=T.observe(effect,teacher);elapsed=tick+1
        feed=row['action']==3 and effect.after.energy>effect.before.energy
        repair=row['audit_tool_after']>row['audit_tool_before']
        stats['steps']+=1;stats['feeding']+=feed;stats['repairs']+=repair;stats['inspections']+=row['action']==4
        stats['safe_patch_wait']+=row['action']==0 and effect.before.position==safe
        stats['actions'][row['action']]+=1;stats['teacher_sample_agreement']+=row['action']==target;stats['teacher_argmax_agreement']+=argmax==target
        bad+=row['action']==3 and effect.after.integrity<effect.before.integrity-world.config.integrity_decay
        if feed:last_feed=elapsed
        if repair:last_repair=elapsed
        events=dict(safe_patch=effect.after.position==safe,positive_feeding=feed,tool_restoration=repair,
            inspection=row['action']==4,slow_state_change=row['z']!=natural['initial_z'])
        for name,happened in events.items():
            if happened and firsts[name] is None:firsts[name]=elapsed
        rows.append(dict(tick=tick,action=row['action'],argmax=argmax,teacher=target,**{k:row[k] for k in (
            'observation','next_observation','reward','body_done','terminated','probability','h','z','audit_tool_before','audit_tool_after')}))
    world.complete();assert row['body_done'] and natural['survived']==(not effect.terminated)
    assert natural['ticks']==config['body_horizon'] or effect.terminated
    assert (stats['feeding'],stats['repairs'],stats['inspections'],bad)==(natural['feeding'],natural['repairs'],natural['inspections'],natural['bad_harvest'])
    assert (effect.after.energy,effect.after.integrity,row['z'])==(natural['final_energy'],natural['final_integrity'],natural['final_z'])
    assert natural['quality_correct']==((natural['quality_probability']>=.5)==bool(prep['quality']))
    assert natural['first_action']==records[0]['action'] and natural['first_correct']==(records[0]['action']==prep['target'])
    e=effect.after.energy<=world.config.death_threshold;i=effect.after.integrity<=world.config.death_threshold;cause=None
    if not natural['survived']:
        assert e or i;cause='both' if e and i else 'energy_only' if e else 'integrity_only';stats['fatal_causes'][cause]+=1
    stats['bodies']=1;stats['slow_state_changed']=int(firsts['slow_state_change'] is not None);stats['no_feeding']=int(last_feed is None)
    summary=dict(trial=trial,variant=variant,mode=mode,index=prep['index'],energy=prep['energy'],side=prep['side'],quality=prep['quality'],
        survived=natural['survived'],ticks=natural['ticks'],cause=cause,first_event_steps=firsts,feeding=stats['feeding'],repairs=stats['repairs'],
        terminal_energy=effect.after.energy,terminal_integrity=effect.after.integrity,terminal_action=row['action'],
        steps_since_feeding=None if last_feed is None else natural['ticks']-last_feed,
        steps_since_restoration=None if last_repair is None else natural['ticks']-last_repair,first8=rows[:8])
    return summary,stats,dict(**summary,last16=rows[-16:]) if cause else None

def main():
    manifest,audit=prerequisites();assert not REPORT.exists(),'existing diagnosis preserved'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in SOURCES:
        assert subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT).replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    preps=M.L.P.L3.read(R.OUT/'endpoint_public.json');assert len(preps)==512
    preparation_config=R.R.CONFIG|dict(base=R.K.CONFIG['evaluation_base'],ecologies=R.K.CONFIG['evaluation_ecologies'],
        horizon=R.K.CONFIG['body_horizon'],energies=R.K.CONFIG['energies'])
    source_steps=sum(R.S.A.preparation(p,preparation_config) for p in preps)
    groups={};summaries=[];fatal=[]
    for variant in (*R.K.CONFIG['arms'],'warm'):
        endpoint=M.L.P.L3.read(R.OUT/f'{variant}_endpoint.json')['bodies'];cursor=0
        for trial in range(R.K.CONFIG['trials']):
            for prep,mode in R.cases(preps,variant):
                for survived in (False,True):groups.setdefault((variant,trial,prep['energy'],mode,prep['side'],prep['quality'],survived),group())
        with gzip.open(R.OUT/f'{variant}_trace.jsonl.gz','rt') as stream:
            import json
            trace=(json.loads(line) for line in stream)
            for trial in range(R.K.CONFIG['trials']):
                for prep,mode in R.cases(preps,variant):
                    natural=endpoint[cursor];cursor+=1;records=[next(trace) for _ in range(natural['ticks'])]
                    summary,stats,death=anatomy(prep,natural,records,trial,variant,mode)
                    aggregate=groups[(variant,trial,prep['energy'],mode,prep['side'],prep['quality'],natural['survived'])]
                    for k,v in stats.items():
                        if isinstance(v,Counter):aggregate[k].update(v)
                        else:aggregate[k]+=v
                    summaries.append(summary)
                    if death:fatal.append(death)
                print('complete LMB5 anatomy',variant,trial,flush=True)
            assert cursor==len(endpoint) and next(trace,None) is None
    steps=sum(g['steps'] for g in groups.values());query_decisions=len(R.K.CONFIG['arms'])*R.K.CONFIG['trials']*3*4*R.K.CONFIG['regression_ecologies']
    assert steps+query_decisions==audit['decisions'] and len(summaries)==7168
    assert len(fatal)==sum(not s['survived'] for s in summaries)
    result=dict(status='COMPLETE',verdict='FAIL',pillar_promotion=False,commit=commit,sources={n:M.L.P.L3.sha(ROOT/n) for n in SOURCES},
        maintenance_or_memory_claim=False,audit_sha=M.L.P.L3.sha(AUDIT),manifest_sha=M.L.P.L3.sha(R.OUT/'manifest.json'),
        bodies=len(summaries),fatal_bodies=len(fatal),source_steps_replayed=source_steps,steps_replayed=steps,readout_queries_in_audit=query_decisions,
        groups=[dict(variant=k[0],trial=k[1],energy=k[2],mode=k[3],side=k[4],quality=k[5],survived=k[6],**v) for k,v in groups.items()],
        body_summaries=summaries,fatal_traces=fatal,scope='Complete descriptive anatomy only; no intervention, optimum, recovered viability or new qualification')
    R.C.save(REPORT,result);print('LMB5 diagnosis COMPLETE',len(summaries),len(fatal),steps,flush=True)

if __name__=='__main__':main()

