"""Prospective current-observation correction with matched architecture controls."""
import argparse,gzip,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training import scarce_birth_training as S, anchored_body_training as B, lmb6_contract as K
from training import probe_lmb5_interior_state as P, lmb4_qualified_source as Q, run_lmb5 as OLD
R=S.R;M=R.M;C=S.C
OUT=ROOT/'runs/lmb6_20260914';REPORT=ROOT/'zeus_sandbox/universe/reports/lmb6_20260914.json'
SOURCES=tuple(dict.fromkeys((*OLD.SOURCES,*P.D.SOURCES,*P.SOURCES,
    'training/lmb4_qualified_source.py','core/anchored_body_agent.py',
    'training/anchored_body_training.py','training/anchored_body_replay.py',
    'training/test_anchored_body_agent.py','training/test_anchored_body_training.py','training/test_anchored_body_replay.py',
    'training/lmb6_contract.py','training/lmb6_fit_guard.py','training/run_lmb6.py',
    'training/lmb6_judgment.py','training/audit_lmb6.py','training/test_lmb6.py',
    'docs/lmb6_protocol_20260914.md')))


def prerequisites():
    source=Q.qualified_source()
    manifest,audit=P.D.prerequisites()
    diagnosis=M.L.P.L3.read(P.D.REPORT);probe=M.L.P.L3.read(P.REPORT)
    assert source==manifest['prerequisites']['source']
    assert diagnosis['status']=='COMPLETE' and diagnosis['verdict']=='FAIL' and diagnosis['pillar_promotion'] is False
    assert diagnosis['bodies']==7168 and diagnosis['fatal_bodies']==1454
    assert diagnosis['audit_sha']==M.L.P.L3.sha(P.D.AUDIT) and diagnosis['manifest_sha']==M.L.P.L3.sha(OLD.OUT/'manifest.json')
    assert diagnosis['steps_replayed']+diagnosis['readout_queries_in_audit']==audit['decisions']
    assert probe['status']=='COMPLETE' and probe['pillar_promotion'] is False
    assert probe['diagnosis_sha']==M.L.P.L3.sha(P.D.REPORT) and probe['audit_sha']==M.L.P.L3.sha(P.D.AUDIT)
    assert probe['body_decisions_consumed']==629550 and probe['selected_records']==36602
    assert 0<=probe['max_intact_probability_error']<2e-5
    for result,names in ((diagnosis,P.D.SOURCES),(probe,P.SOURCES)):
        assert result['sources']=={name:M.L.P.L3.sha(ROOT/name) for name in names}
        for name in names:
            blob=subprocess.check_output(['git','show',result['commit']+':'+name],cwd=ROOT)
            assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    initials={};parents={}
    for trial in range(4):
        raw,head=S.initial(trial)
        assert M.L.P.L3.tree_hash(raw.state_dict())==source['models'][str(trial)]
        assert M.L.P.L3.tree_hash(head.state_dict())==source['heads'][str(trial)]
        parents[str(trial)]=raw.parent_hash
        for arm in K.CONFIG['arms']:
            body,heads=B.initial(trial,arm)
            # Remove only the declared zero correction and restore source metadata.
            restored={k:v for k,v in body.state_dict().items() if not k.startswith('anchor.')}
            restored['_extra_state']=raw.state_dict()['_extra_state']
            assert M.L.P.L3.tree_hash(restored)==source['models'][str(trial)]
            assert M.L.P.L3.tree_hash(heads.state_dict())==source['heads'][str(trial)]
            if arm!='legacy':
                assert body.mode==arm and not bool(body.anchor.weight.any()) and not bool(body.anchor.bias.any())
            initials[f'{trial}_{arm}']=M.L.P.L3.tree_hash(body.state_dict())
    return dict(source=source,diagnosis_sha=M.L.P.L3.sha(P.D.REPORT),probe_sha=M.L.P.L3.sha(P.REPORT),
        initial_hashes=initials,parent_identities=parents)


def prepare():
    K.validate();source=prerequisites();assert not OUT.exists(),'existing campaign preserved'
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    OUT.mkdir();C.save(OUT/'manifest.json',dict(prerequisites=source,config=K.CONFIG,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={p:M.L.P.L3.sha(ROOT/p) for p in SOURCES},
        torch=torch.__version__,numpy=np.__version__))


def verify():
    K.validate();source=prerequisites();m=M.L.P.L3.read(OUT/'manifest.json')
    assert m['prerequisites']==source and m['config']==K.CONFIG and m['sources']=={p:M.L.P.L3.sha(ROOT/p) for p in SOURCES}
    assert m['torch']==torch.__version__ and m['numpy']==np.__version__
    for name in SOURCES:
        blob=subprocess.check_output(['git','show',m['commit']+':'+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    return m


def calibration_decide(data,config=K.CONFIG):
    # All neural arms receive this one public teacher family, so count it once.
    return OLD.calibration_decide({'balanced':data},config|dict(arms=['balanced']))


def calibrate():
    verify()
    data=S.demonstrations('balanced',K.CONFIG,base=K.CONFIG['calibration_base'],n=K.CONFIG['calibration_ecologies'])
    C.save(OUT/'calibration_public.json',data)
    result=calibration_decide(data,K.CONFIG);C.save(OUT/'calibration.json',result)
    print('LMB6 teacher calibration',result['verdict'],flush=True)


def calibration_qualified():
    audit=M.L.P.L3.read(OUT/'calibration_audit.json')
    assert audit['status']==audit['verdict']=='PASS','unqualified/VOID teacher route cannot train'
    names={'manifest.json','calibration.json','calibration_public.json'}
    assert set(audit['evidence_sha'])==names and audit['teacher_bodies']==8*K.CONFIG['calibration_ecologies']
    for name,sha in audit['evidence_sha'].items():assert M.L.P.L3.sha(OUT/name)==sha
    assert M.L.P.L3.read(OUT/'calibration.json')['verdict']=='PASS'
    return audit


def train():
    manifest=verify();calibration_qualified()
    data=[]
    for twin in K.CONFIG['twins']:
        actual=S.demonstrations('balanced',K.CONFIG);C.save(OUT/f'training_{twin}.json',actual);data.append(actual)
    assert M.L.P.L3.sha(OUT/'training_a.json')==M.L.P.L3.sha(OUT/'training_b.json')
    from training.audit_lmb5 import teacher_audit
    teacher_audit(data[0],'balanced',K.CONFIG['training_base'],K.CONFIG['training_ecologies'],K.CONFIG)
    for arm in K.CONFIG['arms']:
        for trial in range(K.CONFIG['trials']):
            for twin,actual in zip(K.CONFIG['twins'],data):
                payload=B.fit(trial,arm,twin,actual,K.CONFIG,OUT/f'{trial}_{arm}_{twin}')
                assert payload['initial_hash']==manifest['prerequisites']['initial_hashes'][f'{trial}_{arm}']


def complete_fits(manifest):
    from training.lmb6_fit_guard import verify_fits
    return verify_fits(OUT,manifest,K.CONFIG)


def trained(trial,variant):
    body,heads=S.initial(trial) if variant=='warm' else B.initial(trial,variant)
    if variant!='warm':
        payload,_=M.L.P.L3.checked_checkpoint(OUT/f'{trial}_{variant}_a');body.load_state_dict(payload['model']);heads.load_state_dict(payload['heads'])
    body.requires_grad_(False);heads.requires_grad_(False);return body,heads


def preparations(base,n,config=K.CONFIG):
    return R.episodes(R.CONFIG|dict(base=base,ecologies=n,horizon=config['body_horizon'],energies=config['energies']))


def cases(preps,variant):
    result=[(p,'inherited') for p in preps]
    if variant!='warm':result.extend((p,'empty') for p in preps if p['energy']==.85)
    return result


@torch.no_grad()
def operate(body,trial,variant,prep,mode,config=K.CONFIG,emit=None):
    assert mode in ('inherited','empty')
    world=S.A.world_for(prep['seed'],3,prep['energy'],R.CONFIG|dict(horizon=config['body_horizon']))
    result=R.O.operate(body,prep,world,action_seed=config['action_base']+1000*trial+prep['index'],
        horizon=config['body_horizon'],inherited=mode=='inherited',emit=emit)
    result['final_energy']=result.pop('energy');result['final_integrity']=result.pop('integrity')
    return dict(trial=trial,variant=variant,mode=mode,index=prep['index'],energy=prep['energy'],seed=prep['seed'],
        side=prep['side'],quality=prep['quality'],target=prep['target'],**result)


@torch.no_grad()
def readout(body,trial,preps,config=K.CONFIG):
    full,written=S.consolidate(body.store,preps);z=full['z'];n=len(preps)
    assert torch.equal(z,written);donors=torch.arange(n)^1
    assert all(preps[int(i)]['energy']==p['energy'] and preps[int(i)]['side']==p['side'] and preps[int(i)]['quality']==1-p['quality'] for i,p in zip(donors,preps))
    query=torch.tensor([p['query'] for p in preps],dtype=torch.float32);side=torch.tensor([p['side'] for p in preps]);rows=[]
    for ci,control in enumerate(('full','reset','opposite')):
        used=z if control=='full' else torch.zeros_like(z) if control=='reset' else z[donors]
        action,_,prob=body.act(query,body.initial(n,used),torch.Generator().manual_seed(config['regression_action_base']+1000*trial+ci))
        recall=body.quality(used).sigmoid()[torch.arange(n),side]
        rows.append(dict(control=control,used=used.tolist(),action=action.tolist(),probabilities=prob.tolist(),recall_probability=recall.tolist()))
    return dict(trial=trial,storage_distance=float((z-written).abs().max()),written=written.tolist(),inherited=z.tolist(),rows=rows)


@torch.no_grad()
def evaluate():
    manifest=verify();calibration_qualified();complete_fits(manifest)
    preps=preparations(K.CONFIG['evaluation_base'],K.CONFIG['evaluation_ecologies'],K.CONFIG);C.save(OUT/'endpoint_public.json',preps)
    regression=preparations(K.CONFIG['regression_base'],K.CONFIG['regression_ecologies'],K.CONFIG);C.save(OUT/'regression_public.json',regression)
    for variant in (*K.CONFIG['arms'],'warm'):
        bodies=[];queries=[];hashes=[]
        with M.L.trace_writer(OUT/f'{variant}_trace.jsonl.gz') as stream:
            for trial in range(K.CONFIG['trials']):
                body,heads=trained(trial,variant);before=M.L.P.L3.tree_hash(body.state_dict());head=M.L.P.L3.tree_hash(heads.state_dict())
                for prep,mode in cases(preps,variant):
                    tags=dict(trial=trial,variant=variant,mode=mode,index=prep['index'],energy=prep['energy'])
                    bodies.append(operate(body,trial,variant,prep,mode,K.CONFIG,emit=lambda row:M.L.log(stream,dict(**tags,**row))))
                if variant!='warm':queries.append(readout(body,trial,regression,K.CONFIG))
                assert before==M.L.P.L3.tree_hash(body.state_dict()) and head==M.L.P.L3.tree_hash(heads.state_dict())
                hashes.append(dict(trial=trial,model=before,heads=head));print('LMB6 endpoint',variant,trial,flush=True)
        C.save(OUT/f'{variant}_endpoint.json',dict(bodies=bodies,readout=queries,hashes=hashes))


def decide(endpoints,preps,config=K.CONFIG):
    judgments={};nt=config['trials'];n=config['evaluation_ecologies'];regn=config['regression_ecologies'];assert regn==n
    rng=np.random.default_rng(config['bootstrap_seed']);parents=rng.integers(nt,size=(config['bootstrap_draws'],nt))
    blocks=rng.integers(n//4,size=(config['bootstrap_draws'],n//4));ids=(4*blocks[:,:,None]+np.arange(4)).reshape(config['bootstrap_draws'],n)
    lookups={}
    for variant in (*config['arms'],'warm'):
        rows=endpoints[variant]['bodies'];lookup={(r['trial'],r['energy'],r['index'],r['mode']):r for r in rows}
        expected={(t,e,i,'inherited') for t in range(nt) for e in config['energies'] for i in range(n)}
        if variant!='warm':expected|={(t,.85,i,'empty') for t in range(nt) for i in range(n)}
        assert len(rows)==len(lookup)==len(expected) and set(lookup)==expected and all(r['variant']==variant for r in rows);lookups[variant]=lookup
        if variant=='warm':continue
        armg={};cells=[]
        for t in range(nt):
            for e in config['energies']:
                for mode in ('inherited','empty') if e==.85 else ('inherited',):
                    for side in (0,1):
                        for q in (0,1):
                            rs=[r for r in rows if (r['trial'],r['energy'],r['mode'],r['side'],r['quality'])==(t,e,mode,side,q)];count=len(rs);assert count==n//4
                            p=sum(r['survived'] for r in rs)/count;feed=sum(r['feeding'] for r in rs)/count;repair=sum(r['repairs'] for r in rs)/count;recall=sum(r['quality_correct'] for r in rs)/count
                            key=f'{t}_{e}_{mode}_{side}_{q}';armg[key]=p>=config['survival_min'] and feed>=config['feed_min'] and repair>=config['repair_min'] and (mode=='empty' or recall>=config['quality_min'])
                            armg['identity_'+key]=all(r['storage_distance']==0 and r['fast_reset'] and (r['initial_z']==r['written_z'] if mode=='inherited' else all(v==0 for v in r['initial_z'])) for r in rs)
                            cells.append(dict(trial=t,energy=e,mode=mode,side=side,quality=q,n=count,survival=p,mean_feeding=feed,mean_repairs=repair,initial_quality_recall=recall))
        assert len(endpoints[variant]['readout'])==nt
        assert len(preps)==4*regn
        for t,item in enumerate(endpoints[variant]['readout']):
            assert item['trial']==t and len(item['rows'])==3 and item['storage_distance']==0 and item['written']==item['inherited']
            by={r['control']:r for r in item['rows']};assert set(by)=={'full','reset','opposite'}
            for e in config['energies']:
                for side in (0,1):
                    for q in (0,1):
                        ii=[i for i,p in enumerate(preps) if (p['energy'],p['side'],p['quality'])==(e,side,q)];assert len(ii)==regn//4
                        key=f'query_{t}_{e}_{side}_{q}'
                        armg[key]=sum(by['full']['action'][i]==preps[i]['target'] for i in ii)/len(ii)>=config['quality_min']
                        armg['recall_'+key]=sum((by['full']['recall_probability'][i]>=.5)==bool(q) for i in ii)/len(ii)>=config['quality_min']
                        armg['opposite_'+key]=sum(by['opposite']['action'][i]==preps[i^1]['target'] for i in ii)/len(ii)>=config['quality_min']
                        armg['opposite_recall_'+key]=sum((by['opposite']['recall_probability'][i]>=.5)==bool(preps[i^1]['quality']) for i in ii)/len(ii)>=config['quality_min']
        query_effects=[]
        for e in config['energies']:
            ii=[i for i,p in enumerate(preps) if p['energy']==e];assert len(ii)==regn
            delta=[]
            for item in endpoints[variant]['readout']:
                by={r['control']:r for r in item['rows']}
                delta.append([int(by['full']['action'][i]==preps[i]['target'])-int(by['reset']['action'][i]==preps[i]['target']) for i in ii])
            delta=np.array(delta);mean=float(delta.mean());bounds=np.quantile(delta[parents[:,:,None],ids[:,None,:]].mean((1,2)),[.025,.975]).tolist()
            passed=bool(mean>=config['readout_effect_min'] and bounds[0]>0);armg[f'query_reset_effect_{e}']=passed
            query_effects.append(dict(energy=e,mean=mean,bounds=bounds,passed=passed))
        judgments[variant]=dict(verdict='PASS' if all(armg.values()) else 'FAIL',gates=armg,cells=cells,readout_effects=query_effects)
    contrasts=[];warm_contrasts=[]
    for e in config['energies']:
        for control in ('recurrent','legacy'):
            delta=np.array([[int(lookups['current'][t,e,i,'inherited']['survived'])-int(lookups[control][t,e,i,'inherited']['survived']) for i in range(n)] for t in range(nt)])
            mean=float(delta.mean());bounds=np.quantile(delta[parents[:,:,None],ids[:,None,:]].mean((1,2)),[.025,.975]).tolist()
            contrasts.append(dict(control=control,energy=e,mean=mean,bounds=bounds,
                required=e in config['attribution_energies'],passed=bool(mean>=config['attribution_min'] and bounds[0]>0)))
        for variant in config['arms']:
            delta=np.array([[int(lookups[variant][t,e,i,'inherited']['survived'])-int(lookups['warm'][t,e,i,'inherited']['survived']) for i in range(n)] for t in range(nt)])
            warm_contrasts.append(dict(variant=variant,energy=e,mean=float(delta.mean()),bounds=np.quantile(delta[parents[:,:,None],ids[:,None,:]].mean((1,2)),[.025,.975]).tolist()))
    selected=next((arm for arm in config['arms'] if judgments[arm]['verdict']=='PASS'),None)
    attribution=judgments['current']['verdict']=='PASS' and all(r['passed'] for r in contrasts if r['required'])
    return dict(verdict='PASS' if selected else 'FAIL',selected_qualified_arm=selected,arms=judgments,
        architecture_attribution='PASS' if attribution else 'FAIL',contrasts=contrasts,
        warm_contrasts=warm_contrasts,pillar_promotion=False,scope=K.SCOPE)


def finalize():
    manifest=verify();calibration_qualified();complete_fits(manifest);endpoints={variant:M.L.P.L3.read(OUT/f'{variant}_endpoint.json') for variant in (*K.CONFIG['arms'],'warm')}
    result=decide(endpoints,M.L.P.L3.read(OUT/'regression_public.json'),K.CONFIG);C.save(OUT/'verdict.json',result)
    C.save(REPORT,dict(**result,manifest=manifest,independent_audit_required=True));print('LMB6 raw',result['verdict'],result['architecture_attribution'],flush=True)


if __name__=='__main__':
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','calibrate','train','evaluate','finalize'])
    globals()[parser.parse_args().command]()
