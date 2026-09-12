"""LCM4-C: committed native-public adapter qualification, exact twins, binary gates."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.lineage_ecology import LineageEcology,LineageConfig
from core.native_memory_adapter import public_body_records,consolidate
from training import run_lcm3 as P,lcm4_compatibility_contract as K

OUT=ROOT/'runs/lcm4_compatibility_20260912'
REPORT=ROOT/'zeus_sandbox/universe/reports/lcm4_compatibility_20260912.json'
SOURCES=tuple(dict.fromkeys((*P.SOURCES,'core/native_memory_adapter.py',
    'training/lcm4_compatibility_contract.py','training/run_lcm4_compatibility.py',
    'training/audit_lcm4_compatibility.py','training/test_lcm4_compatibility.py',
    'docs/lcm4_native_compatibility_protocol_20260912.md')))


def parent(trial,config=K.CONFIG):
    payload,complete=P.checked_checkpoint(P.OUT/f"{trial}_{config['parent_arm']}_a")
    return payload,dict(model_hash=P.tree_hash(payload['model']),**complete)


def model_for(trial,config=K.CONFIG):
    payload,_=parent(trial,config);model=P.model_for(trial,config['parent_arm'])
    model.load_state_dict(payload['model']);model.requires_grad_(False);return model


def native_episodes(config=K.CONFIG):
    rows=[]
    for index in range(config['ecologies']):
        seed=config['ecology_base']+index
        for side in (0,1):
            ecology=LineageEcology(seed=seed,config=LineageConfig(4,config['body_horizon']))
            bodies=[public_body_records(ecology.body(c),side,config['body_horizon'],inspect=c==0) for c in range(3)]
            cue=bodies[0][2]['next_observation'];quality=int(cue[7])
            # Labels are derived only from the public inspection. Never passed to consolidate/logits.
            safe=side if quality else 1-side
            rows.append(dict(seed=seed,side=side,quality=quality,target=1 if safe==0 else 2,bodies=bodies,
                query=list(ecology.body(3).observation().values())))
    return rows


@torch.no_grad()
def evaluate_one(model,trial,episodes,config=K.CONFIG):
    before=P.tree_hash(model.state_dict());state,written=consolidate(model.store,episodes)
    z=state['z'];query=torch.tensor([e['query'] for e in episodes],dtype=torch.float32)
    side=torch.tensor([e['side'] for e in episodes]);target=torch.tensor([e['target'] for e in episodes])
    quality=torch.tensor([e['quality'] for e in episodes]);rows=[]
    for control in config['controls']:
        used=z if control=='full' else torch.zeros_like(z) if control=='reset' else z[K.donor_indices(len(z))]
        probability=model.logits(query,used).softmax(-1)
        rng=torch.Generator().manual_seed(config['action_base']+trial)
        action=torch.multinomial(probability,1,generator=rng).squeeze(1)
        recall=model.store.quality(used).gather(1,side[:,None]).squeeze(1).sigmoid()
        rows.append(dict(trial=trial,control=control,action=action.tolist(),target=target.tolist(),
            side=side.tolist(),quality=quality.tolist(),correct=(action==target).tolist(),
            recall_correct=((recall>=.5)==quality.bool()).tolist(),probabilities=probability.tolist(),
            recall_probability=recall.tolist(),used=used.tolist()))
    assert P.tree_hash(model.state_dict())==before
    return dict(rows=rows,written=written.tolist(),inherited=z.tolist(),storage_distance=float((z-written).abs().max()),
        fast_reset=all(torch.count_nonzero(state[k])==0 for k in ('h','previous_reward')) and bool((state['previous']==-1).all()),
        model_hash=before,input_hash=P.tree_hash(episodes))


def decide(payloads,config=K.CONFIG):
    rows=[r for p in payloads for r in p['rows']];lookup={(r['trial'],r['control']):r for r in rows}
    gates={};summaries=[];nt=config['trials'];n=2*config['ecologies']
    donor=K.donor_indices(n)
    for trial,p in enumerate(payloads):
        gates[f'identity_{trial}']=p['storage_distance']==0 and p['fast_reset']
        full=lookup[trial,'full'];opposite=lookup[trial,'opposite']
        for side in (0,1):
            for quality in (0,1):
                indices=[i for i in range(n) if full['side'][i]==side and full['quality'][i]==quality]
                accuracy=float(np.mean([full['correct'][i] for i in indices]))
                recall=float(np.mean([full['recall_correct'][i] for i in indices]))
                follows=float(np.mean([opposite['action'][i]==full['target'][donor[i]] for i in indices]))
                gates[f'accuracy_{trial}_{side}_{quality}']=accuracy>=config['accuracy_min']
                gates[f'recall_{trial}_{side}_{quality}']=recall>=config['recall_min']
                gates[f'donor_{trial}_{side}_{quality}']=follows>=config['opposite_direction_min']
                summaries.append(dict(trial=trial,side=side,quality=quality,n=len(indices),accuracy=accuracy,recall=recall,donor_follow=follows))
    rng=np.random.default_rng(config['bootstrap_seed']);ti=rng.integers(nt,size=(config['bootstrap_draws'],nt))
    clusters=rng.integers(n//4,size=(config['bootstrap_draws'],n//4))
    wi=(clusters[:,:,None]*4+np.arange(4)).reshape(config['bootstrap_draws'],n)
    effects=[]
    for control in ('reset','opposite'):
        matrix=np.array([np.array(lookup[t,'full']['correct'],float)-np.array(lookup[t,control]['correct'],float) for t in range(nt)])
        samples=matrix[ti[:,:,None],wi[:,None,:]].mean((1,2));bounds=np.quantile(samples,[.025,.975]).tolist()
        margin=config['reset_effect_min'] if control=='reset' else config['opposite_effect_min']
        passed=bool(matrix.mean()>=margin and bounds[0]>0);gates[control]=passed
        effects.append(dict(control=control,mean=float(matrix.mean()),bounds=bounds,passed=passed))
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,cells=summaries,effects=effects,pillar_promotion=False)


def verify():
    P.verify();m=P.read(OUT/'manifest.json')
    assert m['sources']=={p:P.sha(ROOT/p) for p in SOURCES}
    assert m['config']==json.loads(json.dumps(K.CONFIG))
    assert m['parents']=={str(t):parent(t)[1] for t in range(K.CONFIG['trials'])}
    assert m['torch']==torch.__version__ and m['numpy']==np.__version__;return m


def prepare():
    P.verify();assert P.read(ROOT/'zeus_sandbox/universe/reports/lcm3_audit_20260912.json')['status']=='PASS'
    if OUT.exists():verify();return
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources={p:P.sha(ROOT/p) for p in SOURCES},config=json.loads(json.dumps(K.CONFIG)),
        parents={str(t):parent(t)[1] for t in range(K.CONFIG['trials'])},torch=torch.__version__,numpy=np.__version__,
        parent_audit_sha=P.sha(ROOT/'zeus_sandbox/universe/reports/lcm3_audit_20260912.json'))
    OUT.mkdir();P.save(OUT/'manifest.json',manifest)


def evaluate():
    verify()
    for twin in K.CONFIG['twins']:
        path=OUT/twin
        if (path/'completion.json').exists():continue
        assert not path.exists(),'partial artifacts preserved without automatic retry';path.mkdir()
        episodes=native_episodes();P.save(path/'public_episodes.json',episodes);payloads=[]
        for trial in range(K.CONFIG['trials']):
            payloads.append(evaluate_one(model_for(trial),trial,episodes));print('native compatibility',twin,trial,flush=True)
        P.save(path/'evaluation.json',payloads)
        P.save(path/'completion.json',dict(logical_hash=P.tree_hash(payloads),input_hash=P.tree_hash(episodes),
            evaluation_sha=P.sha(path/'evaluation.json'),input_sha=P.sha(path/'public_episodes.json')))


def finalize():
    manifest=verify();a=P.read(OUT/'a/completion.json');b=P.read(OUT/'b/completion.json');assert a==b
    payloads=P.read(OUT/'a/evaluation.json');assert P.tree_hash(payloads)==a['logical_hash']
    verdict=decide(payloads);P.save(OUT/'verdict.json',verdict)
    P.save(REPORT,dict(**verdict,manifest=manifest,twins=a,independent_audit_required=True,
        scope='Real public observation adapter development qualification, no motor learning or native viability'))
    print(json.dumps(verdict,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','evaluate','finalize','all'));phase=parser.parse_args().phase
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if phase in ('prepare','all'):prepare()
    if phase in ('evaluate','all'):evaluate()
    if phase in ('finalize','all'):finalize()

if __name__=='__main__':main()
