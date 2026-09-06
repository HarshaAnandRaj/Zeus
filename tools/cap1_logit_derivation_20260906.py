"""Derive the omitted POL2 telemetry from saved states without replay."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.capability_components_20260906 import OUT,load,save,check
from tools.qv1_postmortem_20260906 import read,sha
from training.evaluate_homeostatic_policy_v2 import _load_artifact


@torch.no_grad()
def main():
    target=OUT/'pol2_logits_derived.json.gz'
    if target.exists():
        raise RuntimeError('refuse overwrite')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest=check()
    report=read('pol2_endogenous_action_verdict_20260905.json')
    artifact_path=Path(report['artifacts']['a']['path'])
    artifact=_load_artifact(artifact_path)
    weights=artifact['action_head']
    # core/model.py declares Linear -> Tanh -> Linear; reject other layouts.
    assert set(weights)=={'0.weight','0.bias','2.weight','2.bias'}
    capture_path=OUT/'pol2.json.gz'
    capture=load(capture_path)
    rows=[]
    for episode in capture['conditions']['normal']:
        logits=[]
        for state,action in zip(episode['states'],episode['actions']):
            features=torch.cat([torch.tensor(state,dtype=torch.float32),torch.zeros(5)])
            hidden=torch.tanh(torch.nn.functional.linear(features,weights['0.weight'],weights['0.bias']))
            output=torch.nn.functional.linear(hidden,weights['2.weight'],weights['2.bias'])
            assert torch.isfinite(output).all() and int(output.argmax())==action
            logits.append(output.tolist())
        assert len(logits)==episode['age']
        rows.append({'seed':episode['seed'],'logits':logits})
    save(target,{'kind':'derived telemetry, not original captured logits',
                 'manifest_sha256':manifest,'policy_sha256':sha(artifact_path),
                 'capture_sha256':sha(capture_path),'derivation_sha256':sha(__file__),
                 'all_argmax_actions_match':True,'episodes':rows})
    print('Derived and verified logits for all '+str(len(rows))+' POL2 episodes.',flush=True)


if __name__=='__main__':
    main()
