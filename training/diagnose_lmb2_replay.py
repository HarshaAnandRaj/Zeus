"""Post-hoc locate frozen numerical rejection; cannot qualify or edit LMB2."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training import run_lmb2 as R,audit_lmb2 as A,learner_history_correction as C


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);m=R.verify();assert __debug__
    data=R.L.P.L3.read(R.OUT/'training_a.json');trial=0;arm='demonstration';path=R.OUT/f'{trial}_{arm}_a';successes=0
    upstream=R.L.P.trained_model(trial).base.state_dict()
    for refresh in range(R.K.CONFIG['updates']//R.K.CONFIG['refresh']):
        source=torch.load(path/f'source_{refresh}.pt',weights_only=False);records=R.load_collection(path/f'collection_{refresh}.jsonl.gz')
        selected=data[refresh*R.K.CONFIG['collection_batch']:(refresh+1)*R.K.CONFIG['collection_batch']]
        starts=A.native_starts(upstream,[d['preparation'] for d in selected]);arrays={k:v.numpy() for k,v in source['model'].items() if isinstance(v,torch.Tensor)}
        for index,d in enumerate(records):
            try:
                A.body_replay(iter(d['records']),d['preparation'],starts[index:index+1],arrays,
                    R.K.CONFIG['collection_action_base']+trial*100000+refresh*1000+index,R.K.CONFIG,True,d['inherited'])
            except AssertionError as error:
                tb=error.__traceback__;frame=None;line=None
                while tb is not None:
                    if Path(tb.tb_frame.f_code.co_filename).resolve()==ROOT/'training/audit_lmb2.py' and tb.tb_frame.f_code.co_name=='body_replay':frame=tb.tb_frame;line=tb.tb_lineno
                    tb=tb.tb_next
                assert frame is not None and line==42,'not the observed frozen probability rejection'
                variables=frame.f_locals;tick=variables['tick'];row=variables['row'];model=C.candidate(trial);model.load_state_dict(source['model']);model.requires_grad_(False)
                rerun=C.collect(model,selected,trial,refresh,R.K.CONFIG)
                primary_exact=R.L.P.L3.tree_hash(rerun)==R.L.P.L3.tree_hash(records);assert primary_exact,'primary collection does not reproduce its source'
                if tick:
                    previous=d['records'][tick-1];h=np.asarray([previous['h']],np.float32);z=np.asarray([previous['z']],np.float32);action=previous['action'];reward=previous['reward'];done=previous['body_done']
                else:
                    inherited,_=R.L.consolidate(model.store,[d['preparation']]);h=np.zeros((1,32),np.float32);z=inherited['z'].numpy() if d['inherited'] else np.zeros((1,8),np.float32);action=-1;reward=0.;done=True
                probability,hidden=A.A.numpy_logits(arrays,[row['observation']],h,z,action,reward,done)
                local_probability_error=float(np.max(np.abs(probability[0]-row['probability'])));local_hidden_error=float(np.max(np.abs(hidden[0]-row['h'])))
                receipt=dict(campaign_status='VOID',qualification=False,frozen_rule_functional_verdict='PASS',failed_audit='Accumulating NumPy probability reconstruction exceeds frozen2e-5',
                    case=dict(trial=trial,arm=arm,refresh=refresh,index=index,tick=tick,seed=d['preparation']['seed'],inherited=d['inherited']),
                    full_numpy_probability_error=variables['maximum'],local_numpy_probability_error=local_probability_error,local_numpy_hidden_error=local_hidden_error,
                    primary_entire_collection_exact=primary_exact,prior_cases_without_rejection=successes,
                    recorded_probability=row['probability'],full_numpy_probability=variables['probability'][0].tolist(),local_numpy_probability=probability[0].tolist(),
                    manifest_sha=R.L.P.L3.sha(R.OUT/'manifest.json'),source_checkpoint_sha=R.L.P.L3.sha(path/f'source_{refresh}.pt'),
                    collection_sha=R.L.P.L3.sha(path/f'collection_{refresh}.jsonl.gz'),audit_log_sha=R.L.P.L3.sha(ROOT/'runs/lmb2_audit_20260913.log'),
                    scope='Exposed-data diagnosis only; original frozen audit remains rejected and no tolerance changes')
                C.save(ROOT/'zeus_sandbox/universe/reports/lmb2_replay_rejection_20260913.json',receipt);print(json.dumps(receipt,indent=2));return
            successes+=1
    raise AssertionError('observed frozen rejection did not reproduce')

if __name__=='__main__':main()
