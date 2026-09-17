"""Full fit receipt required before LMB6 endpoint exposure, also rechecked by audit."""
import math
from pathlib import Path
import numpy as np
import torch
from training import anchored_body_training as B,audit_lmb2 as E
from training.audit_lmb5 import teacher_audit

M=B.M


def finite(value):
    if isinstance(value,torch.Tensor):
        assert bool(torch.isfinite(value).all()),'nonfinite checkpoint tensor'
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)
    elif isinstance(value,(float,np.floating)):
        assert math.isfinite(value),'nonfinite checkpoint scalar'


def verify_fits(directory,manifest,config):
    root=Path(directory);bound=manifest['prerequisites'];pairs=[];batches=0;matching={}
    assert M.L.P.L3.sha(root/'training_a.json')==M.L.P.L3.sha(root/'training_b.json')
    data=M.L.P.L3.read(root/'training_a.json');B.validate_data(data,config)
    teacher_steps=teacher_audit(data,'balanced',config['training_base'],config['training_ecologies'],config)
    data_hash=M.L.P.L3.tree_hash(data)
    for arm in config['arms']:
        for trial in range(config['trials']):
            initial,heads=B.initial(trial,arm);weights=initial.state_dict()
            assert M.L.P.L3.tree_hash(weights)==bound['initial_hashes'][f'{trial}_{arm}']
            assert initial.parent_hash==bound['parent_identities'][str(trial)]
            assert M.L.P.L3.tree_hash(heads.state_dict())==bound['source']['heads'][str(trial)]
            payload,ca=M.L.P.L3.checked_checkpoint(root/f'{trial}_{arm}_a')
            twin,cb=M.L.P.L3.checked_checkpoint(root/f'{trial}_{arm}_b')
            finite(payload);finite(twin)
            assert ca['logical_hash']==cb['logical_hash'],'complete fit twins differ'
            assert payload['arm']==arm and payload['trial']==trial and payload['config']==config
            assert payload['initial_hash']==M.L.P.L3.tree_hash(weights) and payload['initial_revision']==int(initial.revision)
            assert payload['head_hash']==M.L.P.L3.tree_hash(heads.state_dict())==M.L.P.L3.tree_hash(payload['heads'])
            assert payload['frozen_hash']==M.L.frozen_hash(initial) and payload['input_hash']==data_hash
            assert set(payload['model'])==set(weights)
            assert payload['model']['_extra_state']==weights['_extra_state']
            assert int(payload['model']['revision'])==int(initial.revision)+config['updates']
            for name in ('store','quality'):
                select=lambda w:{k:v for k,v in w.items() if k.startswith(name+'.')}
                assert M.L.P.L3.tree_hash(select(payload['model']))==M.L.P.L3.tree_hash(select(weights))
                assert payload['first_body_credit'][name]==0
            for name in B.active_modules(arm):
                select=lambda w:{k:v for k,v in w.items() if k.startswith(name+'.')}
                assert payload['first_body_credit'][name]>0
                assert M.L.P.L3.tree_hash(select(payload['model']))!=M.L.P.L3.tree_hash(select(weights)),name+' did not update'
            parameters=list(initial.active_parameters())
            optimizer=payload['optimizer'];groups=optimizer['param_groups']
            expected=torch.optim.Adam(parameters,lr=config['lr'],foreach=False,fused=False).state_dict()['param_groups']
            assert groups==expected and set(optimizer['state'])==set(groups[0]['params'])
            for identifier,parameter in zip(groups[0]['params'],parameters):
                state=optimizer['state'][identifier]
                assert set(state)=={'step','exp_avg','exp_avg_sq'} and float(state['step'])==config['updates']
                for key in ('exp_avg','exp_avg_sq'):
                    assert state[key].shape==parameter.shape and state[key].dtype==parameter.dtype
                assert bool((state['exp_avg_sq']>=0).all())
            # Independently reconstruct public observations, writes, labels and padding.
            encoded=B.C.encode(initial,data,config['body_horizon'])
            independent=E.independent_encoded(weights,data,weights,config['body_horizon'])
            for key in ('inputs','z','label','active'):
                np.testing.assert_allclose(encoded[key],independent[key],atol=1e-5,rtol=0)
            assert len(payload['logs'])==config['updates']
            for update,row in enumerate(payload['logs']):
                rng=torch.Generator().manual_seed(config['batch_base']+update)
                eco=torch.randint(config['training_ecologies'],(8,),generator=rng)
                ids=torch.tensor([(s*config['training_ecologies']+int(eco[2*s+m]))*2+m for s in range(4) for m in range(2)])
                batch={key:encoded[key][:,ids] for key in ('inputs','z','label','active')}
                actual=(ids.tolist(),M.L.P.L3.tree_hash(batch),int(batch['active'].sum()))
                assert row['update']==update+1 and (row['indices'],row['batch_hash'],row['live_training_steps'])==actual
                assert matching.setdefault((trial,update),actual)==actual,'arm exposure differs'
                assert row['norm']>=0;batches+=1
            # Strict loading checks all keys, shapes and architecture metadata.
            for name,value in payload['model'].items():
                if isinstance(value,torch.Tensor):
                    assert value.dtype==weights[name].dtype and value.shape==weights[name].shape
            initial.load_state_dict(payload['model'],strict=True);heads.load_state_dict(payload['heads'],strict=True)
            pairs.append(dict(trial=trial,arm=arm,logical_hash=ca['logical_hash'],
                model=M.L.P.L3.tree_hash(payload['model']),heads=payload['head_hash']))
    assert len(pairs)==len(config['arms'])*config['trials']
    return dict(fit_pairs=pairs,training_batches_verified=batches,input_hash=data_hash,teacher_physical_steps=teacher_steps)
