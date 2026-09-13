"""Read-only completion/provenance guard before the frozen LMB5 endpoint."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from training import run_lmb5 as R
M=R.M


def pair(trial,arm,manifest,input_hash=None):
    paths=[R.OUT/f'{trial}_{arm}_{t}' for t in R.K.CONFIG['twins']]
    a,ca=M.L.P.L3.checked_checkpoint(paths[0]);b,cb=M.L.P.L3.checked_checkpoint(paths[1])
    assert ca['logical_hash']==cb['logical_hash']
    source=manifest['prerequisites']['source'];key=str(trial)
    assert a['initial_hash']==source['models'][key] and a['head_hash']==source['heads'][key]
    assert a['config']==R.K.CONFIG and len(a['logs'])==R.K.CONFIG['updates']
    assert int(a['model']['revision'])==a['initial_revision']+R.K.CONFIG['updates']
    if input_hash is not None:assert a['input_hash']==input_hash
    initial,heads=R.S.initial(trial);initial_weights=initial.state_dict()
    assert M.L.P.L3.tree_hash(initial_weights)==a['initial_hash'] and M.L.frozen_hash(initial)==a['frozen_hash']
    assert M.L.P.L3.tree_hash(a['heads'])==M.L.P.L3.tree_hash(heads.state_dict())==a['head_hash']
    for name in ('fast','reinstate','gate','actor'):
        assert a['first_body_credit'][name]>0;prefix=name+'.'
        assert M.L.P.L3.tree_hash({k:v for k,v in a['model'].items() if k.startswith(prefix)})!=M.L.P.L3.tree_hash({k:v for k,v in initial_weights.items() if k.startswith(prefix)})
    initial.load_state_dict(a['model']);assert M.L.frozen_hash(initial)==a['frozen_hash']
    assert a['first_body_credit']['store']==a['first_body_credit']['quality']==0
    for update,row in enumerate(a['logs']):
        assert row['update']==update+1 and row['indices']==R.S.indices(update,R.K.CONFIG['training_ecologies'],R.K.CONFIG).tolist()
        assert row['live_training_steps']>0 and np.isfinite([row['loss'],row['body_loss'],row['cue_loss'],row['norm']]).all()
    return dict(trial=trial,arm=arm,logical_hash=ca['logical_hash'],model=M.L.P.L3.tree_hash(a['model']),
        heads=a['head_hash'],input_hash=a['input_hash'],checkpoint_sha=[ca['checkpoint_sha'],cb['checkpoint_sha']])


def main():
    assert __debug__;manifest=R.verify();R.calibration_qualified()
    assert not (R.OUT/'endpoint_public.json').exists(),'pre-exposure guard cannot be backdated'
    assert not (R.OUT/'fit_completion_guard.json').exists(),'previous guard preserved'
    for arm in R.K.CONFIG['arms']:
        for trial in range(R.K.CONFIG['trials']):
            for twin in R.K.CONFIG['twins']:assert (R.OUT/f'{trial}_{arm}_{twin}'/'completion.json').exists(),'incomplete fit blocks endpoint'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();name='training/guard_lmb5_completed_fits.py'
    blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
    assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    fits=[];inputs={}
    for arm in R.K.CONFIG['arms']:
        files=[R.OUT/f'{arm}_training_{t}.json' for t in R.K.CONFIG['twins']]
        assert M.L.P.L3.sha(files[0])==M.L.P.L3.sha(files[1])
        tree=M.L.P.L3.tree_hash(M.L.P.L3.read(files[0]));inputs[arm]=dict(file_sha=M.L.P.L3.sha(files[0]),logical_hash=tree)
        for trial in range(R.K.CONFIG['trials']):fits.append(pair(trial,arm,manifest,tree))
    assert len(fits)==8
    result=dict(status='PASS',functional_qualification=False,fit_pairs=fits,training_inputs=inputs,
        manifest_sha=M.L.P.L3.sha(R.OUT/'manifest.json'),calibration_audit_sha=M.L.P.L3.sha(R.OUT/'calibration_audit.json'),
        commit=commit,source_sha=M.L.P.L3.sha(ROOT/name),endpoint_unexposed=True,
        scope='Pre-exposure whole-fit completion/provenance only; final independent trajectory audit and function remain owed')
    R.C.save(R.OUT/'fit_completion_guard.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
