"""LCM5 frozen-parent native quality-reader repair; no actor/store optimization."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from core.native_quality_readout import NativeQualityReadout
from core.native_memory_adapter import consolidate
from training import run_lcm4_compatibility as P,lcm5_contract as K,run_lcm3 as L3,lcm2_contract as DATA

OUT=ROOT/'runs/lcm5_20260913';REPORT=ROOT/'zeus_sandbox/universe/reports/lcm5_20260913.json'
SOURCES=tuple(dict.fromkeys((*P.SOURCES,'core/native_quality_readout.py','training/lcm5_contract.py',
    'training/run_lcm5.py','training/audit_lcm5.py','training/test_lcm5.py','docs/lcm5_protocol_20260913.md')))


def model_for(trial,config=K.CONFIG):
    base=P.model_for(trial);identity=L3.tree_hash(base.state_dict())
    torch.manual_seed(config['initialization_base']+trial);return NativeQualityReadout(base,identity)


def training_data(config=K.CONFIG):
    return P.native_episodes(K.native_config(config['training_base'],config['training_ecologies']))


def train_one(trial,arm,path,episodes,config=K.CONFIG):
    path.mkdir();model=model_for(trial,config);fixed=L3.tree_hash(model.base.state_dict())
    initial={k:v.detach().clone() for k,v in model.reader.state_dict().items()}
    state,_=consolidate(model.base.store,episodes);z=state['z'].detach()
    if arm=='no_write':z=torch.zeros_like(z)
    side=torch.tensor([e['side'] for e in episodes]);quality=torch.tensor([e['quality'] for e in episodes],dtype=torch.float32)
    opt=torch.optim.Adam(model.reader.parameters(),lr=config['lr'],foreach=False,fused=False);logs=[]
    for update in range(config['updates']):
        ids=K.batch_indices(update,len(episodes),config)
        logits=model.quality_logits(z[ids]).gather(1,side[ids,None]).squeeze(1)
        loss=F.binary_cross_entropy_with_logits(logits,quality[ids])
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.reader.parameters(),config['clip'],error_if_nonfinite=True)
        opt.step();model.revision.add_(1)
        logs.append(dict(update=update+1,loss=float(loss.detach()),norm=float(norm),index_hash=L3.tree_hash(ids)))
    assert L3.tree_hash(model.base.state_dict())==fixed
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),initial_reader=initial,
        frozen_parent_hash=fixed,logs=logs,input_hash=L3.tree_hash(episodes))
    torch.save(payload,path/'checkpoint.pt');L3.save(path/'completion.json',dict(logical_hash=L3.tree_hash(payload),checkpoint_sha=L3.sha(path/'checkpoint.pt')))


def trained_model(trial,arm='inherited'):
    payload,_=L3.checked_checkpoint(OUT/f'{trial}_{arm}_a');model=model_for(trial)
    model.load_state_dict(payload['model']);model.requires_grad_(False);return model


@torch.no_grad()
def evaluate_native(trial,episodes,config=K.CONFIG):
    model=trained_model(trial);no_write=trained_model(trial,'no_write');initial=model_for(trial,config)
    native=K.native_config(config['evaluation_base'],config['evaluation_ecologies'])
    native['bootstrap_draws']=config['bootstrap_draws'];p=P.evaluate_one(model.base,trial,episodes,native)
    quality_rows=[]
    for row in p['rows']:
        used=torch.tensor(row['used'],dtype=torch.float32);side=torch.tensor(row['side']);quality=torch.tensor(row['quality'])
        reader=model.quality_logits(used).gather(1,side[:,None]).squeeze(1).sigmoid()
        quality_rows.append(dict(control=row['control'],probability=reader.tolist(),correct=((reader>=.5)==quality.bool()).tolist()))
        row['recall_probability']=reader.tolist();row['recall_correct']=quality_rows[-1]['correct']
    z=torch.tensor(p['inherited'],dtype=torch.float32);side=torch.tensor([e['side'] for e in episodes]);quality=torch.tensor([e['quality'] for e in episodes])
    for control,head,state in (('original',model.base.store.quality,z),('initial',initial.reader,z),('trained_no_write',no_write.reader,torch.zeros_like(z))):
        prob=head(state).gather(1,side[:,None]).squeeze(1).sigmoid()
        quality_rows.append(dict(control=control,probability=prob.tolist(),correct=((prob>=.5)==quality.bool()).tolist()))
    return p,quality_rows


@torch.no_grad()
def evaluate_synthetic(trial,delay,config=K.CONFIG):
    model=trained_model(trial)
    settings=L3.K.CONFIG|dict(evaluation_base=config['synthetic_base'],evaluation_n=config['synthetic_n'],evaluation_action_base=config['synthetic_action_base'])
    rows=[];full,z=L3.endpoint(model.base,trial,delay,'full',settings)
    for control in ('full','reset','shuffle'):
        row=full if control=='full' else L3.endpoint(model.base,trial,delay,control,settings,donor=z[torch.arange(len(z))^1] if control=='shuffle' else None)[0]
        used=z if control=='full' else torch.zeros_like(z) if control=='reset' else z[torch.arange(len(z))^1]
        side=torch.tensor(row['side']);quality=torch.tensor(row['quality'])
        prob=model.quality_logits(used).gather(1,side[:,None]).squeeze(1).sigmoid()
        row['recall_probability']=prob.tolist();row['recall_correct']=((prob>=.5)==quality.bool()).tolist();rows.append(row)
    return rows


def decide(evaluation,episodes,config=K.CONFIG):
    native=K.native_config(config['evaluation_base'],config['evaluation_ecologies'])|dict(trials=config['trials'],bootstrap_draws=config['bootstrap_draws'])
    verdict=P.decide(evaluation['native'],native);gates=verdict['gates'].copy();bins=[]
    for trial,p in enumerate(evaluation['native']):
        full=next(r for r in p['rows'] if r['control']=='full')
        for side in (0,1):
            for q in (0,1):
                for coarse in (.5,.75):
                    ids=[i for i,e in enumerate(episodes) if e['side']==side and e['quality']==q and e['bodies'][0][2]['observation'][3]==coarse]
                    assert len(ids)>=config['bin_min'],'VOID: required physical quantity-bin coverage missing'
                    recall=float(np.mean([full['recall_correct'][i] for i in ids]));passed=recall>=config['recall_min']
                    gates[f'bin_{trial}_{side}_{q}_{coarse}']=passed;bins.append(dict(trial=trial,side=side,quality=q,before_coarse=coarse,n=len(ids),recall=recall))
    n=2*config['evaluation_ecologies'];nt=config['trials'];rng=np.random.default_rng(config['bootstrap_seed'])
    ti=rng.integers(nt,size=(config['bootstrap_draws'],nt));cluster=rng.integers(n//4,size=(config['bootstrap_draws'],n//4))
    wi=(cluster[:,:,None]*4+np.arange(4)).reshape(config['bootstrap_draws'],n);quality_effects=[]
    indexed={(t,r['control']):r for t,rows in enumerate(evaluation['quality']) for r in rows}
    for control in ('reset','opposite','trained_no_write'):
        matrix=np.array([np.array(indexed[t,'full']['correct'],float)-np.array(indexed[t,control]['correct'],float) for t in range(nt)])
        bounds=np.quantile(matrix[ti[:,:,None],wi[:,None,:]].mean((1,2)),[.025,.975]).tolist()
        margin=config['quality_opposite_margin'] if control=='opposite' else config['quality_reset_margin']
        passed=bool(matrix.mean()>=margin and bounds[0]>0);gates[f'quality_{control}']=passed
        quality_effects.append(dict(control=control,mean=float(matrix.mean()),bounds=bounds,passed=passed))
    synthetic=[]
    for row in evaluation['synthetic']:
        if row['control']!='full':continue
        opposite=next(r for r in evaluation['synthetic'] if r['trial']==row['trial'] and r['delay']==row['delay'] and r['control']=='shuffle')
        gates[f'synthetic_identity_{row["trial"]}_{row["delay"]}']=row['storage_distance']==0
        for side in (0,1):
            for q in (0,1):
                ids=[i for i in range(len(row['side'])) if row['side'][i]==side and row['quality'][i]==q]
                action=float(np.mean([row['correct'][i] for i in ids]));recall=float(np.mean([row['recall_correct'][i] for i in ids]))
                donor=float(np.mean([opposite['action'][i]==row['target'][i^1] for i in ids]))
                passed=action>=config['accuracy_min'] and recall>=config['recall_min'] and donor>=.8
                gates[f'synthetic_{row["trial"]}_{row["delay"]}_{side}_{q}']=passed
                synthetic.append(dict(trial=row['trial'],delay=row['delay'],side=side,quality=q,accuracy=action,recall=recall,donor_follow=donor))
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,native_cells=verdict['cells'],bins=bins,
        action_effects=verdict['effects'],quality_effects=quality_effects,synthetic_cells=synthetic,pillar_promotion=False)


def verify():
    P.verify();m=L3.read(OUT/'manifest.json');assert m['sources']=={p:L3.sha(ROOT/p) for p in SOURCES}
    assert m['config']==json.loads(json.dumps(K.CONFIG));assert m['parents']=={str(t):P.parent(t)[1] for t in range(K.CONFIG['trials'])}
    assert m['torch']==torch.__version__ and m['numpy']==np.__version__;return m


def prepare():
    P.verify();assert L3.read(ROOT/'zeus_sandbox/universe/reports/lcm4_compatibility_audit_20260912.json')['status']=='PASS'
    if OUT.exists():verify();return
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config=json.loads(json.dumps(K.CONFIG)),
        sources={p:L3.sha(ROOT/p) for p in SOURCES},parents={str(t):P.parent(t)[1] for t in range(K.CONFIG['trials'])},
        torch=torch.__version__,numpy=np.__version__,parent_audit_sha=L3.sha(ROOT/'zeus_sandbox/universe/reports/lcm4_compatibility_audit_20260912.json'))
    OUT.mkdir();L3.save(OUT/'manifest.json',manifest)


def train():
    verify()
    for twin in K.CONFIG['twins']:
        inputs=OUT/f'training_{twin}.json'
        if not inputs.exists():L3.save(inputs,training_data())
        episodes=L3.read(inputs)
        for trial in range(K.CONFIG['trials']):
            for arm in K.CONFIG['arms']:
                path=OUT/f'{trial}_{arm}_{twin}'
                if not (path/'completion.json').exists():
                    assert not path.exists(),'partial training preserved without automatic retry';train_one(trial,arm,path,episodes)
                print('quality reader trained',trial,arm,twin,flush=True)
    assert L3.sha(OUT/'training_a.json')==L3.sha(OUT/'training_b.json')
    for trial in range(K.CONFIG['trials']):
        for arm in K.CONFIG['arms']:
            a,ca=L3.checked_checkpoint(OUT/f'{trial}_{arm}_a');b,cb=L3.checked_checkpoint(OUT/f'{trial}_{arm}_b');assert ca['logical_hash']==cb['logical_hash']


def evaluate():
    verify();path=OUT/'public_evaluation.json';assert not path.exists(),'endpoint already exposed'
    episodes=P.native_episodes(K.native_config(K.CONFIG['evaluation_base'],K.CONFIG['evaluation_ecologies']));L3.save(path,episodes)
    evaluation=dict(native=[],quality=[],synthetic=[])
    for trial in range(K.CONFIG['trials']):
        p,q=evaluate_native(trial,episodes);evaluation['native'].append(p);evaluation['quality'].append(q)
        for delay in K.CONFIG['synthetic_delays']:evaluation['synthetic']+=evaluate_synthetic(trial,delay)
        print('quality reader evaluated',trial,flush=True)
    L3.save(OUT/'evaluation.json',evaluation)


def finalize():
    manifest=verify();evaluation=L3.read(OUT/'evaluation.json');episodes=L3.read(OUT/'public_evaluation.json');verdict=decide(evaluation,episodes)
    L3.save(OUT/'verdict.json',verdict);L3.save(REPORT,dict(**verdict,manifest=manifest,independent_audit_required=True,
        quality_summary=[dict(trial=t,control=r['control'],accuracy=float(np.mean(r['correct']))) for t,rows in enumerate(evaluation['quality']) for r in rows],
        scope='Native public quality-reader and unchanged action compatibility, no motor viability or pillar'))
    print('LCM5',verdict['verdict'],'failed gates',[k for k,v in verdict['gates'].items() if not v],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','train','evaluate','finalize','all'));phase=parser.parse_args().phase
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if phase in ('prepare','all'):prepare()
    if phase in ('train','all'):train()
    if phase in ('evaluate','all'):evaluate()
    if phase in ('finalize','all'):finalize()

if __name__=='__main__':main()
