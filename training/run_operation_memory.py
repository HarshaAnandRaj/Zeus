"""OM1 fixed standalone memory-credit assay. Never modifies Zeus checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from core.operation_memory import OperationMemory,events
from training.run_quality_learning import tree_hash,sha,save,read

OUT=ROOT/'runs/om1_20260912'
ARMS=('full','no_writer_credit','shuffled_writer_credit','no_reader_credit')
SEEDS=tuple(range(2026091200,2026091204))
SOURCES=('core/operation_memory.py','training/run_operation_memory.py','training/test_operation_memory.py',
         'docs/om1_protocol_20260912.md')
CONFIG=dict(updates=800,batch=256,lr=.03,entropy_weight=.001,delay=128,evaluation_n=4096,
            seeds=SEEDS,arms=ARMS,twins=('a','b'))


def learn(seed,arm):
    torch.manual_seed(seed);model=OperationMemory()
    # Small declared initialization variation; paired exactly across arms.
    with torch.no_grad():
        for p in model.parameters():p.add_(torch.randn_like(p)*.02)
    initial=tree_hash(model.state_dict())
    opt=torch.optim.Adam(model.parameters(),lr=.03,foreach=False)
    data_rng=torch.Generator().manual_seed(seed+100)
    action_rng=torch.Generator().manual_seed(seed+200)
    shuffle_rng=torch.Generator().manual_seed(seed+300)
    logs=[]
    for update in range(800):
        e=events(256,data_rng)
        result=model(e.keys,e.values,e.priority,e.query,action_rng)
        reward=(result['answer']==e.target).float()
        advantage=reward-(reward.sum()-reward)/(len(reward)-1)
        perm=torch.randperm(len(reward),generator=shuffle_rng)
        writer_adv=advantage[perm] if arm=='shuffled_writer_credit' else advantage
        writer_term=0 if arm=='no_writer_credit' else writer_adv.detach()*result['writer_logp']
        reader_term=0 if arm=='no_reader_credit' else advantage.detach()*result['reader_logp']
        loss=-(writer_term+reader_term+advantage.detach()*result['answer_logp']).mean()
        loss=loss-.001*result['entropy'].mean()
        loss.backward()
        # Explicitly freeze the disabled operation, including optimizer state.
        if arm=='no_writer_credit':model.writer.grad=None
        if arm=='no_reader_credit':model.reader.grad=None
        opt.step();opt.zero_grad(set_to_none=True)
        if update%100==99:logs.append(dict(update=update+1,reward=float(reward.mean()),loss=float(loss.detach())))
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),initial_hash=initial,logs=logs,
                data_rng=data_rng.get_state(),action_rng=action_rng.get_state(),shuffle_rng=shuffle_rng.get_state())


@torch.no_grad()
def evaluate(payload,seed,control='intact'):
    model=OperationMemory();model.load_state_dict(payload['model']);model.requires_grad_(False)
    e=events(4096,torch.Generator().manual_seed(seed+10000))
    r=model(e.keys,e.values,e.priority,e.query,torch.Generator().manual_seed(seed+20000),content_control=control)
    correct=(r['answer']==e.target)
    return dict(correct=correct.tolist(),accuracy=float(correct.float().mean()),
                retrieval_match=float(r['matched'].mean()),read_rate=float(r['read'].mean()))


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=True)
    manifest=dict(config=json.loads(json.dumps(CONFIG)),sources={p:sha(ROOT/p) for p in SOURCES},torch=torch.__version__)
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest
    else:save(OUT/'manifest.json',manifest)
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            hashes=[];payload=None
            for twin in ('a','b'):
                path=OUT/f'{seed}_{arm}_{twin}.pt'
                if path.exists():payload=torch.load(path,weights_only=False)
                else:payload=learn(seed,arm);torch.save(payload,path)
                hashes.append(tree_hash(payload))
            assert hashes[0]==hashes[1]
            result=evaluate(payload,seed)
            controls={c:evaluate(payload,seed,c)['accuracy'] for c in ('zero','flip')} if arm=='full' else {}
            rows.append(dict(seed=seed,arm=arm,logical_hash=hashes[0],controls=controls,**result))
            print(seed,arm,'exact twins complete',flush=True)
    keyed={(r['seed'],r['arm']):r for r in rows}
    # Explicit fixed gates. Paired event differences; every initialization must pass.
    gates={}
    gates['accuracy']=all(keyed[s,'full']['accuracy']>=.80 for s in SEEDS)
    gates['writer_credit']=all(keyed[s,'full']['accuracy']-keyed[s,'no_writer_credit']['accuracy']>=.15 for s in SEEDS)
    gates['targeted_writer_credit']=all(keyed[s,'full']['accuracy']-keyed[s,'shuffled_writer_credit']['accuracy']>=.15 for s in SEEDS)
    gates['reader_credit']=all(keyed[s,'full']['accuracy']-keyed[s,'no_reader_credit']['accuracy']>=.08 for s in SEEDS)
    gates['content_necessary']=all(keyed[s,'full']['accuracy']-keyed[s,'full']['controls']['zero']>=.20 for s in SEEDS)
    import numpy as np
    paired=[]
    for s in SEEDS:
        for arm in ARMS[1:]:
            delta=np.array(keyed[s,'full']['correct'],dtype=float)-np.array(keyed[s,arm]['correct'],dtype=float)
            half=1.96*delta.std(ddof=1)/np.sqrt(len(delta))
            paired.append(dict(seed=s,control=arm,mean=float(delta.mean()),
                               bounds=[float(delta.mean()-half),float(delta.mean()+half)]))
    gates['positive_paired_lower_bounds']=all(r['bounds'][0]>0 for r in paired)
    result=dict(grade='Standalone engineered memory-operation assay; no Zeus pillar promotion',
                gates=gates,verdict='PASS' if all(gates.values()) else 'FAIL',rows=rows,paired_intervals=paired,manifest=manifest)
    save(ROOT/'zeus_sandbox/universe/reports/om1_20260912.json',result)
    print(json.dumps(dict(gates=gates,verdict=result['verdict'])),flush=True)


if __name__=='__main__':main()
