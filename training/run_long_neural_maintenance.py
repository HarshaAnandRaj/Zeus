"""LMT1: guarded zero-update continuous repair-dependent neural operation."""
import argparse,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from core.lineage_ecology import LineageConfig
from core.lineage_repair_ecology import RepairDependentEcology
from core.native_memory_adapter import public_body_records
from training import run_lmb4 as R,lmb4_qualified_source as S,native_body_operation as O,calibrate_repair_dependent_world as W

OUT=ROOT/'runs/lmt1_20260913';REPORT=ROOT/'zeus_sandbox/universe/reports/lmt1_20260913.json'
CONFIG=dict(base=217113000,ecologies=128,trials=4,horizon=4096,action_base=217413000,survival_min=.90,quality_min=.90,
    modes=['inherited','empty'],controls=['enabled','disabled'],disabled_max_ticks=598)
SOURCES=tuple(dict.fromkeys((*R.SOURCES,*W.SOURCES,'training/lmb4_qualified_source.py','training/test_lmb4_qualified_source.py',
    'training/native_body_operation.py','training/test_native_body_operation.py','training/native_physical_replay.py',
    'training/test_native_physical_replay.py','training/run_long_neural_maintenance.py','training/audit_long_neural_maintenance.py',
    'training/test_long_neural_maintenance.py','docs/lmt1_protocol_20260913.md')))

def prerequisites():
    source=S.qualified_source();W.verify();receipt=W.read(W.OUT/'audit.json')
    assert receipt['status']==receipt['verdict']=='PASS'
    assert set(receipt['evidence_sha'])=={'manifest.json','trace.jsonl.gz','rows.json','verdict.json'}
    for name,digest in receipt['evidence_sha'].items():assert W.sha(W.OUT/name)==digest
    return source,W.sha(W.OUT/'audit.json')

def episodes(config=CONFIG):
    result=[]
    for index in range(config['ecologies']):
        side=(index//2)%2;ecology=RepairDependentEcology(seed=config['base']+index,config=LineageConfig(4,config['horizon']))
        bodies=[public_body_records(ecology.body(c),side,8,inspect=c==0) for c in range(3)]
        assert all(len(b)==8 and not b[-1]['terminated'] for b in bodies)
        cue=bodies[0][2]['next_observation'];q=int(cue[7]);safe=side if q else 1-side
        result.append(dict(seed=ecology.seed,index=index,side=side,quality=q,target=1+safe,bodies=bodies,query=list(ecology.body(3).observation().values())))
    return result

def verify():
    source,world_sha=prerequisites();m=R.L.P.L3.read(OUT/'manifest.json')
    assert m['source']==source and m['world_audit_sha']==world_sha and m['config']==CONFIG
    assert m['sources']=={p:R.L.P.L3.sha(ROOT/p) for p in SOURCES}
    assert m['torch']==torch.__version__ and m['numpy']==R.np.__version__
    return m

def prepare():
    source,world_sha=prerequisites();assert not OUT.exists(),'existing campaign preserved'
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    OUT.mkdir();R.C.save(OUT/'manifest.json',dict(source=source,world_audit_sha=world_sha,config=CONFIG,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={p:R.L.P.L3.sha(ROOT/p) for p in SOURCES},
        torch=torch.__version__,numpy=R.np.__version__))

def decide(rows,config=CONFIG):
    expected={(t,i,m,c) for t in range(config['trials']) for i in range(config['ecologies']) for m in config['modes'] for c in config['controls']}
    lookup={(r['trial'],r['index'],r['mode'],r['control']):r for r in rows};assert len(rows)==len(lookup)==len(expected) and set(lookup)==expected
    cells=[];gates={}
    for trial in range(config['trials']):
        for mode in config['modes']:
            for side in (0,1):
                for q in (0,1):
                    rs=[r for r in rows if (r['trial'],r['mode'],r['control'],r['side'],r['quality'])==(trial,mode,'enabled',side,q)]
                    n=len(rs);assert n==config['ecologies']//4
                    p=sum(r['survived'] for r in rs)/n;recall=sum(r['quality_correct'] for r in rs)/n;k=f'{trial}_{mode}_{side}_{q}'
                    gates['survival_'+k]=p>=config['survival_min']
                    gates['identity_'+k]=all(r['storage_distance']==0 and r['fast_reset'] and (r['initial_z']==r['written_z'] if mode=='inherited' else r['initial_z']==[0.]*8) for r in rs)
                    gates['effective_repair_'+k]=all(not r['survived'] or r['repairs']>0 for r in rs)
                    if mode=='inherited':gates['recall_'+k]=recall>=config['quality_min']
                    critical=1.959963984540054;den=1+critical**2/n;center=(p+critical**2/(2*n))/den
                    radius=critical*math.sqrt(p*(1-p)/n+critical**2/(4*n*n))/den
                    cells.append(dict(trial=trial,mode=mode,side=side,quality=q,n=n,survival=p,survival_wilson95=[center-radius,center+radius],
                        mean_feeding=sum(r['feeding'] for r in rs)/n,mean_effective_repairs=sum(r['repairs'] for r in rs)/n,initial_quality_recall=recall))
    disabled=[r for r in rows if r['control']=='disabled']
    gates['disabled_restoration_fails']=all(not r['survived'] and r['repairs']==0 and r['ticks']<=config['disabled_max_ticks'] for r in disabled)
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,cells=cells,pillar_promotion=False,
        scope='Bounded continuous repair-dependent neural maintenance only, not memory benefit or six-pillar promotion')

@torch.no_grad()
def evaluate():
    manifest=verify();data=episodes();R.C.save(OUT/'public_preparation.json',data);rows=[]
    with R.L.trace_writer(OUT/'public_trace.jsonl.gz') as stream:
        for trial in range(CONFIG['trials']):
            model,_=R.trained(trial,manifest['source']['arm']);before=R.L.P.L3.tree_hash(model.state_dict())
            assert before==manifest['source']['models'][str(trial)]
            for prep in data:
                for mode in CONFIG['modes']:
                    for control in CONFIG['controls']:
                        tags=dict(trial=trial,index=prep['index'],mode=mode,control=control)
                        world=RepairDependentEcology(seed=prep['seed'],config=LineageConfig(4,CONFIG['horizon']),tool_repair_enabled=control=='enabled').body(3)
                        result=O.operate(model,prep,world,action_seed=CONFIG['action_base']+1000*trial+prep['index'],horizon=CONFIG['horizon'],
                            inherited=mode=='inherited',emit=lambda row:R.L.log(stream,dict(**tags,**row)))
                        rows.append(dict(**tags,seed=prep['seed'],side=prep['side'],quality=prep['quality'],target=prep['target'],**result))
            assert before==R.L.P.L3.tree_hash(model.state_dict());print('long maintenance endpoint',trial,flush=True)
    R.C.save(OUT/'endpoint.json',rows)

def finalize():
    m=verify();result=decide(R.L.P.L3.read(OUT/'endpoint.json'));R.C.save(OUT/'verdict.json',result)
    R.C.save(REPORT,dict(**result,manifest=m,independent_audit_required=True));print('LMT1 raw',result['verdict'],flush=True)

if __name__=='__main__':
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','evaluate','finalize']);globals()[parser.parse_args().command]()
