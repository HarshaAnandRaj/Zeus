"""LMB6 complete public teacher, fit, anchored neural and physical audit."""
import argparse,gzip,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from training import run_lmb6 as R,audit_lmb2 as E,audit_lmb1 as P,lmb6_judgment as J
from training.dual_body_replay import Replay as LegacyReplay
from training.anchored_body_replay import Replay as AnchorReplay
from training.audit_lmb5 import teacher_audit as old_teacher_audit, calibration_decide as old_calibration_decide
from training.native_physical_replay import CheckedWorld
from training.audit_lifetime_calibration import close
M=R.M;S=R.S


def teacher_audit(data,base,n,config=R.K.CONFIG):
    return old_teacher_audit(data,'balanced',base,n,config)


def calibration_decide(data,config):
    return old_calibration_decide({'balanced':data},config|dict(arms=['balanced']))


def replayer(weights,variant,parent_identity):
    assert variant in (*R.K.CONFIG['arms'],'warm') and parent_identity
    assert weights['_extra_state']['parent_hash']==parent_identity
    if variant in ('current','recurrent'):
        return AnchorReplay(weights,expected_mode=variant,expected_parent_hash=parent_identity)
    assert not any(k.startswith('anchor.') for k in weights)
    assert weights['_extra_state']['version']=='native-body-agent-v1-20260913'
    return LegacyReplay(weights)


def body_audit(weights,prep,records,trial,variant,mode,parent_identity,config=R.K.CONFIG):
    replay=replayer(weights,variant,parent_identity);world=S.A.world_for(prep['seed'],3,prep['energy'],R.R.CONFIG|dict(horizon=config['body_horizon']))
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


def query_audit(weights,item,preps,trial,variant,parent_identity,config=R.K.CONFIG):
    replay=replayer(weights,variant,parent_identity);source=replay.native_starts(preps);n=len(preps)
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
    data=M.L.P.L3.read(R.OUT/'calibration_public.json')
    steps=teacher_audit(data,cfg['calibration_base'],cfg['calibration_ecologies'],cfg)
    result=calibration_decide(data,cfg);assert result==M.L.P.L3.read(R.OUT/'calibration.json')
    names=['manifest.json','calibration.json','calibration_public.json']
    receipt=dict(status='PASS',verdict=result['verdict'],teacher_bodies=8*cfg['calibration_ecologies'],public_physical_steps_replayed=steps,
        evidence_sha={name:M.L.P.L3.sha(R.OUT/name) for name in names},scope='Independent public-teacher reachability/physical source ruler only')
    R.C.save(R.OUT/'calibration_audit.json',receipt);print('independent LMB6 teacher calibration',receipt['verdict'],flush=True)


def main():
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);manifest=R.verify();cfg=R.K.CONFIG
    for name in R.SOURCES:
        blob=subprocess.check_output(['git','show',manifest['commit']+':'+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    cal_audit=R.calibration_qualified();physics=cal_audit['public_physical_steps_replayed']
    fit_receipt=R.complete_fits(manifest);fits=fit_receipt['fit_pairs'];batches=fit_receipt['training_batches_verified']
    data=M.L.P.L3.read(R.OUT/'training_a.json')
    physics+=teacher_audit(data,cfg['training_base'],cfg['training_ecologies'],cfg);del data
    models={};heads={}
    for arm in cfg['arms']:
        for trial in range(cfg['trials']):
            payload,_=M.L.P.L3.checked_checkpoint(R.OUT/f'{trial}_{arm}_a')
            models[trial,arm]=payload['model'];heads[trial,arm]=payload['heads']
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
                    result,stat=body_audit(weights,prep,records,trial,variant,mode,manifest['prerequisites']['parent_identities'][str(trial)],cfg);assert result==primary
                    physics+=result['ticks'];body_count+=1;stats.append(stat)
                if variant!='warm':stats.append(query_audit(weights,endpoint['readout'][trial],reg,trial,variant,manifest['prerequisites']['parent_identities'][str(trial)],cfg))
                print('LMB6 endpoint audit',variant,trial,flush=True)
            assert cursor==len(endpoint['bodies']) and next(trace,None) is None and len(endpoint['hashes'])==cfg['trials']
            assert len(endpoint['readout'])==(0 if variant=='warm' else cfg['trials'])
    result=J.decide(endpoints,reg,cfg);close(result,M.L.P.L3.read(R.OUT/'verdict.json'))
    assert M.L.P.L3.read(R.REPORT)==dict(**M.L.P.L3.read(R.OUT/'verdict.json'),manifest=manifest,independent_audit_required=True)
    assert len(fits)==len(cfg['arms'])*cfg['trials'] and batches==len(fits)*cfg['updates']
    assert body_count==cfg['trials']*cfg['evaluation_ecologies']*(5*len(cfg['arms'])+4)
    body_decisions=sum(sum(b['ticks'] for b in item['bodies']) for item in endpoints.values())
    readout_queries=3*cfg['trials']*len(cfg['energies'])*cfg['regression_ecologies']*len(cfg['arms'])
    assert sum(s['decisions'] for s in stats)==body_decisions+readout_queries
    names=['manifest.json','calibration.json','calibration_audit.json','calibration_public.json','training_a.json','training_b.json','endpoint_public.json','regression_public.json','verdict.json']
    names.extend(f'{arm}_endpoint.json' for arm in cfg['arms'])
    names.extend(['warm_endpoint.json',*(v+'_trace.jsonl.gz' for v in (*cfg['arms'],'warm'))])
    receipt=dict(status='PASS',verdict=result['verdict'],qualification=result['verdict']=='PASS',selected_qualified_arm=result['selected_qualified_arm'],
        architecture_attribution=result['architecture_attribution'],fit_pairs=fits,training_batches_verified=batches,endpoint_bodies=body_count,
        public_physical_steps_replayed=physics,decisions=sum(s['decisions'] for s in stats),
        body_decisions=body_decisions,readout_queries=readout_queries,
        max_local_probability_error=max(s['max_local_probability_error'] for s in stats),max_local_state_error=max(s['max_local_state_error'] for s in stats),
        evidence_sha={p:M.L.P.L3.sha(R.OUT/p) for p in names},report_sha=M.L.P.L3.sha(R.REPORT),hardware_trajectory_exact=True,pillar_promotion=False,
        scope='Motor/native-query prerequisite and separate architecture attribution; exact fit twins, no independent Adam, own acquisition or engine portability claim')
    R.C.save(ROOT/'zeus_sandbox/universe/reports/lmb6_audit_20260914.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['calibration','full'])
    {'calibration':audit_calibration,'full':main}[parser.parse_args().command]()
