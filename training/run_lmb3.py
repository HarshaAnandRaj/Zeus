"""Fresh zero-update body qualification of four declared, unqualified candidates."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training import run_lmb2 as N,learner_history_correction as C,validate_dual_body_replay as V
L=N.L
CONFIG=N.K.CONFIG|dict(updates=0,arms=('fixed_demonstration_candidates',),evaluation_base=213113000,
    regression_base=213123000,synthetic_regression_base=213223000,action_base=213413000,
    regression_action_base=213423000,bootstrap_seed=213513000)
OUT=ROOT/'runs/lmb3_20260913';REPORT=ROOT/'zeus_sandbox/universe/reports/lmb3_20260913.json'
SOURCES=tuple(dict.fromkeys((*N.SOURCES,*V.SOURCES,'training/run_lmb3.py','training/audit_lmb3.py',
    'training/test_lmb3.py','docs/lmb3_protocol_20260913.md')))


def candidate(trial):return N.trained_model(trial,'demonstration')


def prerequisites():
    N.verify();receipt=L.P.L3.read(V.REPORT)
    assert receipt['status']=='PASS' and receipt['qualification'] is False and receipt['hardware_trajectory_exact'] is True
    assert receipt['bodies']==256 and len(receipt['source_phases'])==32
    assert receipt['original_manifest_sha']==L.P.L3.sha(N.OUT/'manifest.json') and receipt['original_report_sha']==L.P.L3.sha(N.REPORT)
    assert receipt['sources']=={name:L.P.L3.sha(ROOT/name) for name in V.SOURCES}
    assert receipt['original_campaign_status']=='VOID'
    assert receipt['original_rejection_sha']==L.P.L3.sha(ROOT/'zeus_sandbox/universe/reports/lmb2_replay_rejection_20260913.json')
    for trial in range(4):
        a,ca=L.P.L3.checked_checkpoint(N.OUT/f'{trial}_demonstration_a');b,cb=L.P.L3.checked_checkpoint(N.OUT/f'{trial}_demonstration_b')
        assert ca['logical_hash']==cb['logical_hash']==receipt['exact_demonstration_fit_pairs'][trial]
    return receipt


def identities():
    return dict(candidates={str(t):L.P.L3.tree_hash(candidate(t).state_dict()) for t in range(4)},
        references={f'{t}_{kind}':L.P.L3.tree_hash(model.state_dict()) for t in range(4)
            for kind,model in (('initial',L.model_for(t)),('warm',L.trained_model(t)))})


def prepare():
    prerequisites();assert not OUT.exists(),'existing campaign preserved'
    assert not subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT,text=True).strip()
    manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        config=C.plain(CONFIG),sources={name:L.P.L3.sha(ROOT/name) for name in SOURCES},
        instrument_receipt_sha=L.P.L3.sha(V.REPORT),torch=torch.__version__,numpy=np.__version__,**identities())
    OUT.mkdir();C.save(OUT/'manifest.json',manifest)


def verify():
    prerequisites();manifest=L.P.L3.read(OUT/'manifest.json')
    assert manifest['config']==C.plain(CONFIG) and manifest['sources']=={name:L.P.L3.sha(ROOT/name) for name in SOURCES}
    assert manifest['instrument_receipt_sha']==L.P.L3.sha(V.REPORT)
    assert manifest['torch']==torch.__version__ and manifest['numpy']==np.__version__
    for key,value in identities().items():assert manifest[key]==value
    return manifest


@torch.no_grad()
def evaluate():
    verify();prep=L.P.P.native_episodes(L.P.K.native_config(CONFIG['evaluation_base'],CONFIG['evaluation_ecologies']))
    selected=[prep[2*i+(i//2)%2] for i in range(CONFIG['evaluation_ecologies'])]
    regression=L.P.P.native_episodes(L.P.K.native_config(CONFIG['regression_base'],CONFIG['regression_ecologies']))
    C.save(OUT/'endpoint_public.json',selected);C.save(OUT/'regression_public.json',regression);reference=[]
    with L.trace_writer(OUT/'reference_trace.jsonl.gz') as stream:
        for trial in range(4):
            models=(('initial',L.model_for(trial)),('warm',L.trained_model(trial)))
            for index,p in enumerate(selected):
                for mode in ('inherited','empty'):
                    for kind,model in models:reference.append(L.evaluate_body(model,trial,index,p,mode,CONFIG,stream,condition=kind+'_'+mode))
            print('fresh qualification references',trial,flush=True)
    C.save(OUT/'reference.json',reference);result=dict(bodies=list(reference),memory=[],synthetic=[])
    with L.trace_writer(OUT/'candidate_trace.jsonl.gz') as stream:
        for trial in range(4):
            model=candidate(trial)
            for index,p in enumerate(selected):
                for mode in ('inherited','empty'):result['bodies'].append(L.evaluate_body(model,trial,index,p,mode,CONFIG,stream))
            result['memory'].append(L.memory_regression(model,trial,regression,CONFIG))
            for delay in CONFIG['synthetic_delays']:result['synthetic']+=L.synthetic_regression(model,trial,delay,CONFIG)
            assert L.P.L3.tree_hash(model.state_dict())==L.P.L3.read(OUT/'manifest.json')['candidates'][str(trial)]
            print('fresh qualification candidate',trial,flush=True)
    C.save(OUT/'endpoint.json',result)


def finalize():
    manifest=verify();endpoint=L.P.L3.read(OUT/'endpoint.json');result=C.plain(L.decide(endpoint,CONFIG))
    C.save(OUT/'verdict.json',result);C.save(REPORT,dict(**result,manifest=manifest,independent_audit_required=True,
        scope='Fresh fixed-candidate body qualification; original LMB2 remains VOID'))
    print(json.dumps({k:v for k,v in result.items() if k!='gates'},indent=2))


if __name__=='__main__':
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','evaluate','finalize'))
    globals()[parser.parse_args().command]()
