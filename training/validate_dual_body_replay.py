"""Exposed source-collection instrument validation; no endpoint qualification."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training import run_lmb2 as R,lmb2_contract as K,learner_history_correction as C,audit_lmb1 as A
from training.dual_body_replay import Replay,VERSION

SOURCES=('training/dual_body_replay.py','training/validate_dual_body_replay.py','training/test_dual_body_replay.py',
    'docs/dual_body_replay_validation_protocol_20260913.md')
REPORT=ROOT/'zeus_sandbox/universe/reports/dual_body_replay_validation_20260913.json'


def committed_sources():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    hashes={}
    for name in SOURCES:
        blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n'), 'uncommitted instrument source'
        hashes[name]=R.L.P.L3.sha(ROOT/name)
    return commit,hashes


@torch.no_grad()
def main():
    assert __debug__;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert not REPORT.exists(),'existing validation preserved'
    commit,sources=committed_sources();manifest=R.verify();data=R.L.P.L3.read(R.OUT/'training_a.json')
    assert R.L.P.L3.sha(R.OUT/'training_a.json')==R.L.P.L3.sha(R.OUT/'training_b.json')
    physics=A.teacher_replay(data,K.CONFIG['training_base'],K.CONFIG['training_ecologies'],K.CONFIG['body_horizon'])
    calibration=R.L.P.L3.read(R.OUT/'calibration_public.json')
    physics+=A.teacher_replay(calibration,K.CONFIG['calibration_base'],K.CONFIG['calibration_ecologies'],K.CONFIG['body_horizon'])
    assert R.L.P.L3.read(R.OUT/'calibration.json')==dict(passed=True,input_hash=R.L.P.L3.tree_hash(calibration))
    cases=[];phase_hashes=[];pairs=[];bodies=steps=decisions=writes=0;pe=se=0.
    for trial in range(K.CONFIG['trials']):
        dirs=[R.OUT/f'{trial}_demonstration_{t}' for t in K.CONFIG['twins']]
        a,ca=R.L.P.L3.checked_checkpoint(dirs[0]);b,cb=R.L.P.L3.checked_checkpoint(dirs[1])
        assert ca['logical_hash']==cb['logical_hash'];pairs.append(ca['logical_hash'])
        for refresh in range(K.CONFIG['updates']//K.CONFIG['refresh']):
            source=[torch.load(d/f'source_{refresh}.pt',weights_only=False) for d in dirs]
            assert R.L.P.L3.tree_hash(source[0])==R.L.P.L3.tree_hash(source[1])
            paths=[d/f'collection_{refresh}.jsonl.gz' for d in dirs]
            assert R.L.P.L3.sha(paths[0])==R.L.P.L3.sha(paths[1]);collected=R.load_collection(paths[0])
            selected=data[refresh*K.CONFIG['collection_batch']:(refresh+1)*K.CONFIG['collection_batch']]
            assert len(collected)==len(selected)==8
            replay=Replay(source[0]['model'])
            for index,(row,expected) in enumerate(zip(collected,selected)):
                assert row['preparation']==expected['preparation'] and row['inherited']==expected['inherited']
                prep=row['preparation'];physics+=A.preparation(prep,prep['seed'],prep['side'])
                summary=replay.body(row['records'],prep,K.CONFIG['collection_action_base']+trial*100000+refresh*1000+index,
                    K.CONFIG['body_horizon'],row['inherited'],True)
                bodies+=1;steps+=summary['ticks']
                cases.append(dict(trial=trial,refresh=refresh,index=index,ticks=summary['ticks'],survived=summary['survived']))
            r=replay.receipt();decisions+=r['decisions'];writes+=r['eligible_writes']
            pe=max(pe,r['max_local_probability_error']);se=max(se,r['max_local_state_error'])
            phase_hashes.append(dict(trial=trial,refresh=refresh,source_hash=R.L.P.L3.tree_hash(source[0]),collection_sha=R.L.P.L3.sha(paths[0])))
            print('dual exposed-source validation',trial,refresh,bodies,flush=True)
    assert bodies==256 and decisions==steps and len(phase_hashes)==32 and len(pairs)==4
    assert any((r['trial'],r['refresh'],r['index'])==(0,4,0) and r['ticks']>=151 for r in cases)
    result=dict(status='PASS',qualification=False,instrument_version=VERSION,commit=commit,sources=sources,
        original_campaign_status='VOID',original_manifest_sha=R.L.P.L3.sha(R.OUT/'manifest.json'),
        original_rejection_sha=R.L.P.L3.sha(ROOT/'zeus_sandbox/universe/reports/lmb2_replay_rejection_20260913.json'),
        original_report_sha=R.L.P.L3.sha(R.REPORT),original_source_manifest=manifest,
        exact_demonstration_fit_pairs=pairs,source_phases=phase_hashes,cases=cases,bodies=bodies,
        public_physical_steps_replayed=physics+steps,decisions=decisions,eligible_writes=writes,
        max_local_probability_error=pe,max_local_state_error=se,hardware_trajectory_exact=True,
        scope='Exposed source instrument validation only; shared hardware operators plus independent local mathematics and public physics',
        earned_followup='Prepare fresh fixed-candidate body qualification; no transfer or pillar earned')
    C.save(REPORT,result);print(json.dumps({k:v for k,v in result.items() if k not in ('sources','original_source_manifest','cases','source_phases','exact_demonstration_fit_pairs')},indent=2))

if __name__=='__main__':main()
