"""Independent NumPy policy reconstruction for OM1 held-out decisions."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.operation_memory import OperationMemory
from training.run_quality_learning import tree_hash,sha,read,save


def reconstruct(state,seed,control):
    # Recreate public episode inputs independently of core.operation_memory.events.
    n=4096;data=torch.Generator().manual_seed(seed+10000)
    keys=torch.rand(n,6,generator=data).numpy().argsort(axis=1)
    values=torch.randint(0,2,(n,6),generator=data).numpy().astype(np.float32)*2-1
    marked=torch.randint(0,6,(n,),generator=data).numpy()
    offset=torch.randint(1,6,(n,),generator=data).numpy()
    query_position=np.where(torch.rand(n,generator=data).numpy()<.8,marked,(marked+offset)%6)
    index=np.arange(n);query=keys[index,query_position];target=values[index,query_position]>0
    priority=(np.arange(6)[None]==marked[:,None]).astype(np.float32)
    rng=torch.Generator().manual_seed(seed+20000)
    sigmoid=lambda x:1/(1+np.exp(-x))
    writer=np.float32(state['writer']);reader=state['reader'].numpy();answer=state['answer'].numpy()
    weight=np.exp(writer*priority);total=weight[:,0].copy();slot=np.zeros(n,dtype=int)
    for t in range(1,6):
        probability=sigmoid(np.log(weight[:,t])-np.log(total))
        slot=np.where(torch.rand(n,generator=rng).numpy()<probability,t,slot)
        total+=weight[:,t]
    match=(keys[index,slot]==query).astype(np.float32)
    take=torch.rand(n,generator=rng).numpy()<sigmoid(reader[0]+reader[1]*match)
    value=values[index,slot]
    if control=='zero':value=np.zeros_like(value)
    if control=='flip':value=-value
    selected=torch.rand(n,generator=rng).numpy()<sigmoid(answer[0]*(value*take)+answer[1])
    return selected==target


def main():
    torch.set_num_threads(1)
    report=read(ROOT/'zeus_sandbox/universe/reports/om1_20260912.json')
    manifest=report['manifest']
    assert all(sha(ROOT/p)==h for p,h in manifest['sources'].items())
    verified=0;checks=[]
    for row in report['rows']:
        seed,arm=row['seed'],row['arm']
        paths=[ROOT/f'runs/om1_20260912/{seed}_{arm}_{t}.pt' for t in ('a','b')]
        a,b=[torch.load(p,weights_only=False) for p in paths]
        assert tree_hash(a)==tree_hash(b)==row['logical_hash']
        torch.manual_seed(seed);initial=OperationMemory()
        with torch.no_grad():
            for p in initial.parameters():p.add_(torch.randn_like(p)*.02)
        assert tree_hash(initial.state_dict())==a['initial_hash']
        if arm=='no_writer_credit':assert torch.equal(initial.writer,a['model']['writer'])
        if arm=='no_reader_credit':assert torch.equal(initial.reader,a['model']['reader'])
        correct=reconstruct(a['model'],seed,'intact')
        assert np.array_equal(correct,np.array(row['correct']))
        assert abs(float(correct.mean())-row['accuracy'])<1e-8
        verified+=len(correct)
        for control,accuracy in row['controls'].items():
            c=reconstruct(a['model'],seed,control)
            assert abs(float(c.mean())-accuracy)<1e-8;verified+=len(c)
        checks.append(dict(seed=seed,arm=arm,checkpoint_hashes=[sha(p) for p in paths]))
    # Recompute the fixed decision without using runner.decide or training helpers.
    keyed={(r['seed'],r['arm']):r for r in report['rows']}
    gates=dict(accuracy=True,writer_credit=True,targeted_writer_credit=True,reader_credit=True,
               content_necessary=True,positive_paired_lower_bounds=True)
    for seed in manifest['config']['seeds']:
        full=keyed[seed,'full']
        gates['accuracy'] &= full['accuracy']>=.80
        gates['content_necessary'] &= full['accuracy']-full['controls']['zero']>=.20
        for arm,gate,margin in [('no_writer_credit','writer_credit',.15),
                               ('shuffled_writer_credit','targeted_writer_credit',.15),
                               ('no_reader_credit','reader_credit',.08)]:
            other=keyed[seed,arm]
            diff=np.array(full['correct'],float)-np.array(other['correct'],float)
            gates[gate] &= float(diff.mean())>=margin
            gates['positive_paired_lower_bounds'] &= bool(diff.mean()-1.96*diff.std(ddof=1)/np.sqrt(len(diff))>0)
    assert gates==report['gates']
    assert report['verdict']==('PASS' if all(gates.values()) else 'FAIL')
    save(ROOT/'zeus_sandbox/universe/reports/om1_audit_20260912.json',
         dict(passed=True,independent_evaluation_decisions=verified,twins=16,
              report_sha=sha(ROOT/'zeus_sandbox/universe/reports/om1_20260912.json'),checks=checks,
              limitation='Independent evaluation and identities; training gradients checked mechanically, not independently reimplemented.'))
    print(f'PASS: {verified} independent evaluation decisions, 16 exact twin pairs',flush=True)


if __name__=='__main__':main()
