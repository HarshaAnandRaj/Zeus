"""LBT2: fresh known-memory actuation, bound to independently qualified LMB4."""
import argparse,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from core.lineage_ecology import LineageConfig
from core.lineage_energy_ecology import BirthResourceEcology,BirthResources
from core.native_memory_adapter import public_body_records
from training import run_lmb4 as M,lmb4_qualified_source as S,native_body_operation as O,calibrate_birth_resource_memory as B

OUT=ROOT/'runs/lbt2_20260913';REPORT=ROOT/'zeus_sandbox/universe/reports/lbt2_20260913.json'
CONFIG=dict(base=218113000,ecologies=128,trials=4,energies=[.12,.20,.35,.85],horizon=256,action_base=218413000,
    survival_min=.90,feed_min=16.,repair_min=.5,quality_min=.90)
SOURCES=tuple(dict.fromkeys((*M.SOURCES,*B.SOURCES,'training/lmb4_qualified_source.py','training/test_lmb4_qualified_source.py',
    'training/native_body_operation.py','training/test_native_body_operation.py','training/native_physical_replay.py',
    'training/test_native_physical_replay.py','training/run_known_birth_transfer2.py','training/audit_known_birth_transfer2.py',
    'training/test_known_birth_transfer2.py','docs/lbt2_protocol_20260913.md')))

def prerequisites():
    source=S.qualified_source();path=ROOT/'zeus_sandbox/universe/reports/birth_resource_memory_calibration_audit_20260913.json'
    audit=M.L.P.L3.read(path);assert audit['status']==audit['verdict']=='PASS'
    manifest=M.L.P.L3.read(B.OUT/'manifest.json');report=M.L.P.L3.read(ROOT/'zeus_sandbox/universe/reports/birth_resource_memory_calibration_20260913.json')
    assert audit['manifest_sha']==M.L.P.L3.sha(B.OUT/'manifest.json') and audit['rows_sha']==M.L.P.L3.sha(B.OUT/'rows.json')
    assert audit['trace_sha']==M.L.P.L3.sha(B.OUT/'public_trace.jsonl.gz')==report['trace_sha']
    assert report==dict(**M.L.P.L3.read(B.OUT/'verdict.json'),manifest=manifest,trace_sha=audit['trace_sha'])
    assert manifest['config']==M.C.plain(B.CONFIG) and manifest['sources']=={p:M.L.P.L3.sha(ROOT/p) for p in B.SOURCES}
    for name in B.SOURCES:
        blob=subprocess.check_output(['git','show',manifest['commit']+':'+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    return source,M.L.P.L3.sha(path)

def episodes(config=CONFIG):
    result=[]
    for energy in config['energies']:
        for index in range(config['ecologies']):
            side=(index//2)%2;ecology=BirthResourceEcology(seed=config['base']+index,resources=BirthResources(energy),config=LineageConfig(4,config['horizon']))
            bodies=[public_body_records(ecology.body(c),side,8,inspect=c==0) for c in range(3)]
            assert all(len(b)==8 and not b[-1]['terminated'] for b in bodies),'source preparation died'
            cue=bodies[0][2]['next_observation'];q=int(cue[7]);safe=side if q else 1-side
            result.append(dict(seed=ecology.seed,index=index,energy=energy,side=side,quality=q,target=1+safe,bodies=bodies,query=list(ecology.body(3).observation().values())))
    return result

def verify():
    source,resource_sha=prerequisites();m=M.L.P.L3.read(OUT/'manifest.json')
    assert m['source']==source and m['resource_audit_sha']==resource_sha and m['config']==CONFIG
    assert m['sources']=={p:M.L.P.L3.sha(ROOT/p) for p in SOURCES}
    assert m['torch']==torch.__version__ and m['numpy']==M.np.__version__
    return m

def prepare():
    source,resource_sha=prerequisites();assert not OUT.exists(),'existing transfer preserved'
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    OUT.mkdir();M.C.save(OUT/'manifest.json',dict(source=source,resource_audit_sha=resource_sha,config=CONFIG,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={p:M.L.P.L3.sha(ROOT/p) for p in SOURCES},
        torch=torch.__version__,numpy=M.np.__version__))

def decide(rows,config=CONFIG):
    expected={(t,e,i) for t in range(config['trials']) for e in config['energies'] for i in range(config['ecologies'])}
    keys={(r['trial'],r['energy'],r['index']) for r in rows};assert len(rows)==len(keys)==len(expected) and keys==expected
    gates={};cells=[]
    for trial in range(config['trials']):
        for energy in config['energies']:
            for side in (0,1):
                for q in (0,1):
                    rs=[r for r in rows if (r['trial'],r['energy'],r['side'],r['quality'])==(trial,energy,side,q)]
                    n=len(rs);assert n==config['ecologies']//4
                    p=sum(r['survived'] for r in rs)/n;feed=sum(r['feeding'] for r in rs)/n;repair=sum(r['repairs'] for r in rs)/n;recall=sum(r['quality_correct'] for r in rs)/n
                    key=f'{trial}_{energy}_{side}_{q}';gates[key]=p>=config['survival_min'] and feed>=config['feed_min'] and repair>=config['repair_min'] and recall>=config['quality_min']
                    gates['identity_'+key]=all(r['storage_distance']==0 and r['fast_reset'] and r['initial_z']==r['written_z'] for r in rs)
                    z=1.959963984540054;den=1+z*z/n;center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
                    cells.append(dict(trial=trial,energy=energy,side=side,quality=q,n=n,survival=p,survival_wilson95=[center-radius,center+radius],
                        mean_feeding=feed,mean_repairs=repair,initial_quality_recall=recall,first_direction=sum(r['first_correct'] for r in rs)/n))
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,cells=cells,pillar_promotion=False,
        scope='Zero-update known-memory scarce-birth actuation only, not native inheritance benefit or a pillar')

@torch.no_grad()
def run_body(model,trial,prep,config=CONFIG,emit=None):
    world=BirthResourceEcology(seed=prep['seed'],resources=BirthResources(prep['energy']),config=LineageConfig(4,config['horizon'])).body(3)
    result=O.operate(model,prep,world,action_seed=config['action_base']+1000*trial+prep['index'],horizon=config['horizon'],emit=emit)
    result['final_energy']=result.pop('energy');result['final_integrity']=result.pop('integrity')
    return dict(trial=trial,index=prep['index'],energy=prep['energy'],seed=prep['seed'],side=prep['side'],quality=prep['quality'],target=prep['target'],**result)

@torch.no_grad()
def evaluate():
    manifest=verify();data=episodes();M.C.save(OUT/'public_preparation.json',data);rows=[]
    with M.L.trace_writer(OUT/'public_trace.jsonl.gz') as stream:
        for trial in range(CONFIG['trials']):
            model,_=M.trained(trial,manifest['source']['arm']);before=M.L.P.L3.tree_hash(model.state_dict());assert before==manifest['source']['models'][str(trial)]
            for prep in data:
                tags=dict(trial=trial,index=prep['index'],energy=prep['energy'])
                rows.append(run_body(model,trial,prep,emit=lambda row:M.L.log(stream,dict(**tags,**row))))
            assert before==M.L.P.L3.tree_hash(model.state_dict());print('known memory birth endpoint',trial,flush=True)
    M.C.save(OUT/'endpoint.json',rows)

def finalize():
    m=verify();result=decide(M.L.P.L3.read(OUT/'endpoint.json'));M.C.save(OUT/'verdict.json',result)
    M.C.save(REPORT,dict(**result,manifest=m,independent_audit_required=True));print('LBT2 raw',result['verdict'],flush=True)

if __name__=='__main__':
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','evaluate','finalize']);globals()[parser.parse_args().command]()
