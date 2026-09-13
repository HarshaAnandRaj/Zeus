"""LMB5 public teachers, complete fit twins, full neural/physical endpoint audit."""
import argparse,gzip,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from training import run_lmb5 as R,audit_lmb2 as E,audit_lmb1 as P,lmb5_judgment as J
from training.dual_body_replay import Replay
from training.native_physical_replay import CheckedWorld
from training.audit_lifetime_calibration import close
M=R.M;S=R.S


def teacher_audit(data,arm,base,n,config=R.K.CONFIG):
    assert len(data)==4*n*2
    assert [(d['profile_slot'],d['preparation']['index'],d['inherited']) for d in data]==[(s,i,m) for s in range(4) for i in range(n) for m in (True,False)]
    cfg=R.R.CONFIG|dict(base=base,ecologies=n,horizon=config['body_horizon']);steps=0
    for d in data:
        prep=d['preparation'];actual=config['energies'][d['profile_slot']] if arm=='balanced' else .85
        assert prep['energy']==actual and d['nominal_energy']==config['energies'][d['profile_slot']];steps+=S.A.preparation(prep,cfg)
        rows=[dict(tick=tick,**row) for tick,row in enumerate(d['records'])]
        world=CheckedWorld(S.A.world_for(prep['seed'],3,actual,cfg),rows)
        cue=prep['bodies'][0][2]['next_observation'];safe=(int(cue[2]) if cue[7] else 1-int(cue[2])) if d['inherited'] else None
        tool=cue[6] if d['inherited'] else None
        for tick,row in enumerate(rows):
            obs=world.observation();position=obs.position
            if safe is None:target=1 if position>0 else 4
            elif (tool is not None and tool<.75) or obs.integrity<.8:target=2 if position<.5 else 1 if position>.5 else 5
            else:target=2 if position<safe else 1 if position>safe else 3 if obs.energy<.5 else 0
            assert row['action']==row['label']==target,'public teacher provenance rejected'
            effect=world.step(target);public=P.public_record(effect,tick,config['body_horizon'])
            assert public=={k:row[k] for k in public}
            if target==3 and tool is not None:tool=max(0.,tool-world.config.harvest_wear)
            if target==5 and effect.after.position==.5 and tool is not None:tool=min(1.,tool+world.config.tool_repair)
            if target==4 and effect.after.position in (0,1):
                side=int(effect.after.position);safe=side if effect.after.resource_quality else 1-side;tool=effect.after.tool_condition
            assert row['audit_tool_before']==row['audit_physical_before']['tool'] and row['audit_tool_after']==row['audit_physical_after']['tool']
        world.complete();assert rows[-1]['body_done'];steps+=len(rows)
    return steps


def calibration_decide(data,config):
    gates={};cells=[]
    for arm in config['arms']:
        for slot,nominal in enumerate(config['energies']):
            for inherited in (True,False):
                rr=[d for d in data[arm] if d['profile_slot']==slot and d['inherited']==inherited]
                count=len(rr);assert count==config['calibration_ecologies'];actual=nominal if arm=='balanced' else .85
                survivors=sum(not d['records'][-1]['terminated'] for d in rr);needed=count//2 if actual==.12 and not inherited else count
                gates[f'{arm}_{slot}_{inherited}']=survivors==needed
                cells.append(dict(arm=arm,slot=slot,nominal_energy=nominal,actual_energy=actual,inherited=inherited,n=count,survivors=survivors,expected_survivors=needed))
    return dict(verdict='PASS' if all(gates.values()) else 'VOID',gates=gates,cells=cells,scope='Prospective public-teacher reachability only; no neural result')


def body_audit(weights,prep,records,trial,variant,mode,config=R.K.CONFIG):
    replay=Replay(weights);world=S.A.world_for(prep['seed'],3,prep['energy'],R.R.CONFIG|dict(horizon=config['body_horizon']))
    checked=CheckedWorld(world,records)
    body=replay.body(records,prep,config['action_base']+1000*trial+prep['index'],config['body_horizon'],mode=='inherited',world=checked);checked.complete()
    source=replay.native_starts([prep]);used=source if mode=='inherited' else torch.zeros_like(source)
    quality=float(F.linear(used,replay.w['quality.weight'],replay.w['quality.bias']).sigmoid()[0,prep['side']])
    nq=1/(1+np.exp(-(used.numpy()@replay.nw['quality.weight'].T+replay.nw['quality.bias'])))
    assert abs(quality-float(nq[0,prep['side']]))<1e-5
    result=dict(trial=trial,variant=variant,mode=mode,index=prep['index'],energy=prep['energy'],seed=prep['seed'],
        side=prep['side'],quality=prep['quality'],target=prep['target'],**{k:v for k,v in body.items() if k not in ('seed','energy','integrity')},
        final_energy=body['energy'],final_integrity=body['integrity'],first_action=records[0]['action'],
        first_correct=records[0]['action']==prep['target'],quality_probability=quality,quality_correct=(quality>=.5)==bool(prep['quality']),
        storage_distance=0.,fast_reset=True,written_z=source[0].tolist(),initial_z=used[0].tolist(),final_z=records[-1]['z'])
    return result,replay.receipt()


def query_audit(weights,item,preps,trial,config=R.K.CONFIG):
    replay=Replay(weights);source=replay.native_starts(preps);n=len(preps)
    assert item['trial']==trial and item['storage_distance']==0 and item['written']==item['inherited']==source.tolist()
    assert len(item['rows'])==3 and [r['control'] for r in item['rows']]==['full','reset','opposite']
    for ci,row in enumerate(item['rows']):
        used=source if ci==0 else torch.zeros_like(source) if ci==1 else source[torch.arange(n)^1]
        assert used.tolist()==row['used'];prob,_=replay.decide([p['query'] for p in preps],torch.zeros(n,32),used)
        assert torch.equal(prob,torch.tensor(row['probabilities'],dtype=torch.float32))
        actions=torch.multinomial(prob,1,generator=torch.Generator().manual_seed(config['regression_action_base']+1000*trial+ci)).squeeze(1).tolist()
        assert actions==row['action']
        side=torch.tensor([p['side'] for p in preps]);recall=F.linear(used,replay.w['quality.weight'],replay.w['quality.bias']).sigmoid()[torch.arange(n),side]
        assert recall.tolist()==row['recall_probability']
        nq=1/(1+np.exp(-(used.numpy()@replay.nw['quality.weight'].T+replay.nw['quality.bias'])))
        np.testing.assert_allclose(nq[np.arange(n),side],recall,atol=1e-5,rtol=0)
    return replay.receipt()


def audit_calibration():
    manifest=R.verify();cfg=R.K.CONFIG
    for name in R.SOURCES:
        blob=subprocess.check_output(['git','show',manifest['commit']+':'+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    data={arm:M.L.P.L3.read(R.OUT/f'{arm}_calibration_public.json') for arm in cfg['arms']}
    steps=sum(teacher_audit(rows,arm,cfg['calibration_base'],cfg['calibration_ecologies']) for arm,rows in data.items())
    result=calibration_decide(data,cfg);assert result==M.L.P.L3.read(R.OUT/'calibration.json')
    names=['manifest.json','calibration.json',*(arm+'_calibration_public.json' for arm in cfg['arms'])]
    receipt=dict(status='PASS',verdict=result['verdict'],teacher_bodies=1024,public_physical_steps_replayed=steps,
        evidence_sha={name:M.L.P.L3.sha(R.OUT/name) for name in names},scope='Independent public-teacher reachability/physical source ruler only')
    R.C.save(R.OUT/'calibration_audit.json',receipt);print('independent scarce-birth teacher calibration',receipt['verdict'],flush=True)


def main():
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);manifest=R.verify();cfg=R.K.CONFIG
    for name in R.SOURCES:
        blob=subprocess.check_output(['git','show',manifest['commit']+':'+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    cal_audit=R.calibration_qualified();physics=cal_audit['public_physical_steps_replayed']
    models={};heads={};fits=[];batches=0
    for arm in cfg['arms']:
        assert M.L.P.L3.sha(R.OUT/f'{arm}_training_a.json')==M.L.P.L3.sha(R.OUT/f'{arm}_training_b.json')
        data=M.L.P.L3.read(R.OUT/f'{arm}_training_a.json');physics+=teacher_audit(data,arm,cfg['training_base'],cfg['training_ecologies'])
        for trial in range(cfg['trials']):
            initial,ih=S.initial(trial);weights=initial.state_dict();head=ih.state_dict();directories=[R.OUT/f'{trial}_{arm}_{t}' for t in cfg['twins']]
            a,ca=M.L.P.L3.checked_checkpoint(directories[0]);b,cb=M.L.P.L3.checked_checkpoint(directories[1]);assert ca['logical_hash']==cb['logical_hash']
            assert a['config']==cfg and a['initial_hash']==M.L.P.L3.tree_hash(weights) and a['initial_revision']==int(initial.revision)
            assert a['frozen_hash']==M.L.frozen_hash(initial) and a['head_hash']==M.L.P.L3.tree_hash(head)==M.L.P.L3.tree_hash(a['heads'])
            assert a['input_hash']==M.L.P.L3.tree_hash(data) and len(a['logs'])==cfg['updates'] and int(a['model']['revision'])==int(initial.revision)+cfg['updates']
            for prefix in ('store.','quality.'):
                assert M.L.P.L3.tree_hash({k:v for k,v in a['model'].items() if k.startswith(prefix)})==M.L.P.L3.tree_hash({k:v for k,v in weights.items() if k.startswith(prefix)})
            for name in ('fast','reinstate','gate','actor'):
                prefix=name+'.';assert a['first_body_credit'][name]>0
                assert M.L.P.L3.tree_hash({k:v for k,v in a['model'].items() if k.startswith(prefix)})!=M.L.P.L3.tree_hash({k:v for k,v in weights.items() if k.startswith(prefix)})
            assert a['first_body_credit']['store']==a['first_body_credit']['quality']==0
            encoded=S.C.encode(initial,data,cfg['body_horizon']);independent=E.independent_encoded(weights,data,weights,cfg['body_horizon'])
            for key in ('inputs','z','label','active'):np.testing.assert_allclose(encoded[key],independent[key],atol=1e-5,rtol=0)
            for update,row in enumerate(a['logs']):
                rng=torch.Generator().manual_seed(cfg['batch_base']+update);eco=torch.randint(cfg['training_ecologies'],(8,),generator=rng)
                ids=torch.tensor([(slot*cfg['training_ecologies']+int(eco[2*slot+mode]))*2+mode for slot in range(4) for mode in range(2)])
                batch={k:encoded[k][:,ids] for k in ('inputs','z','label','active')}
                assert row['update']==update+1 and row['indices']==ids.tolist() and row['batch_hash']==M.L.P.L3.tree_hash(batch)
                assert row['live_training_steps']==int(batch['active'].sum()) and np.isfinite([row['loss'],row['body_loss'],row['cue_loss'],row['norm']]).all();batches+=1
            models[trial,arm]=a['model'];heads[trial,arm]=a['heads'];fits.append(dict(trial=trial,arm=arm,logical_hash=ca['logical_hash'],model=M.L.P.L3.tree_hash(a['model']),heads=a['head_hash']))
            print('scarce-birth fit audit',arm,trial,flush=True)
        del data
    preps=M.L.P.L3.read(R.OUT/'endpoint_public.json');reg=M.L.P.L3.read(R.OUT/'regression_public.json')
    for selected,base,n in ((preps,cfg['evaluation_base'],cfg['evaluation_ecologies']),(reg,cfg['regression_base'],cfg['regression_ecologies'])):
        assert [(p['energy'],p['index']) for p in selected]==[(e,i) for e in cfg['energies'] for i in range(n)]
        pcfg=R.R.CONFIG|dict(base=base,ecologies=n,horizon=cfg['body_horizon']);physics+=sum(S.A.preparation(p,pcfg) for p in selected)
    endpoints={v:M.L.P.L3.read(R.OUT/f'{v}_endpoint.json') for v in (*cfg['arms'],'warm')};stats=[];body_count=0
    for variant,endpoint in endpoints.items():
        with gzip.open(R.OUT/f'{variant}_trace.jsonl.gz','rt',encoding='utf-8') as stream:
            trace=(json.loads(line) for line in stream);cursor=0
            for trial in range(cfg['trials']):
                if variant=='warm':body,hd=S.initial(trial);weights=body.state_dict();head=hd.state_dict()
                else:weights=models[trial,variant];head=heads[trial,variant]
                assert endpoint['hashes'][trial]==dict(trial=trial,model=M.L.P.L3.tree_hash(weights),heads=M.L.P.L3.tree_hash(head))
                ordered=[(p,'inherited') for p in preps]
                if variant!='warm':ordered.extend((p,'empty') for p in preps if p['energy']==.85)
                for prep,mode in ordered:
                    primary=endpoint['bodies'][cursor];cursor+=1
                    assert (primary['trial'],primary['variant'],primary['mode'],primary['energy'],primary['index'])==(trial,variant,mode,prep['energy'],prep['index'])
                    records=[]
                    for tick in range(primary['ticks']):
                        row=next(trace);assert (row['trial'],row['variant'],row['mode'],row['energy'],row['index'],row['tick'])==(trial,variant,mode,prep['energy'],prep['index'],tick);records.append(row)
                    result,stat=body_audit(weights,prep,records,trial,variant,mode);assert result==primary
                    physics+=result['ticks'];body_count+=1;stats.append(stat)
                if variant!='warm':stats.append(query_audit(weights,endpoint['readout'][trial],reg,trial))
                print('scarce-birth endpoint audit',variant,trial,flush=True)
            assert cursor==len(endpoint['bodies']) and next(trace,None) is None and len(endpoint['hashes'])==cfg['trials']
            assert len(endpoint['readout'])==(0 if variant=='warm' else cfg['trials'])
    result=J.decide(endpoints,reg,cfg);close(result,M.L.P.L3.read(R.OUT/'verdict.json'))
    assert M.L.P.L3.read(R.REPORT)==dict(**M.L.P.L3.read(R.OUT/'verdict.json'),manifest=manifest,independent_audit_required=True)
    assert len(fits)==8 and batches==768 and body_count==7168
    names=['manifest.json','calibration.json','calibration_audit.json','endpoint_public.json','regression_public.json','verdict.json']
    names.extend(f'{arm}_{kind}.json' for arm in cfg['arms'] for kind in ('calibration_public','training_a','training_b','endpoint'))
    names.extend(['warm_endpoint.json',*(v+'_trace.jsonl.gz' for v in (*cfg['arms'],'warm'))])
    receipt=dict(status='PASS',verdict=result['verdict'],qualification=result['verdict']=='PASS',selected_qualified_arm=result['selected_qualified_arm'],
        exposure_attribution=result['exposure_attribution'],fit_pairs=fits,training_batches_verified=batches,endpoint_bodies=body_count,
        public_physical_steps_replayed=physics,decisions=sum(s['decisions'] for s in stats),
        max_local_probability_error=max(s['max_local_probability_error'] for s in stats),max_local_state_error=max(s['max_local_state_error'] for s in stats),
        evidence_sha={p:M.L.P.L3.sha(R.OUT/p) for p in names},report_sha=M.L.P.L3.sha(R.REPORT),hardware_trajectory_exact=True,pillar_promotion=False,
        scope='Motor/native-query prerequisite and separate exposure attribution; exact fit twins, no independent Adam, synthetic regression or engine portability claim')
    R.C.save(ROOT/'zeus_sandbox/universe/reports/lmb5_audit_20260913.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['calibration','full'])
    {'calibration':audit_calibration,'full':main}[parser.parse_args().command]()
