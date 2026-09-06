"""Fresh-seed diagnostic capture, preserving the original fidelity failure."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.capability_components_20260906 import Observer,save,load,check,calibration,dyn,ZeusCore,ZeusConfig
from tools.qv1_postmortem_20260906 import sha,read

OUT=ROOT/'runs/cap1r_20260906'
MANIFEST=ROOT/'docs/registrations/cap1r_manifest.json'


def identities():
    check()
    m=load(MANIFEST)
    for path,h in m.items():
        assert sha(ROOT/path)==h, path
    return sha(MANIFEST)


@torch.no_grad()
def campaign():
    original=read('dyn1_resilience_20260905.json')
    checkpoint=Path(original['artifact'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    result={}
    for arm in ['trained','random_init']:
        torch.manual_seed(20260969)
        trained=ZeusCore.load(checkpoint,device='cpu') if arm=='trained' else None
        records=[]
        for index,seed in enumerate(range(20260970,20260986)):
            if arm=='random_init':
                torch.manual_seed(seed)
                model=ZeusCore(ZeusConfig(**payload['config']),tokenizer_path=payload.get('tokenizer')).eval()
            else:
                model=trained
            observer=Observer(model)
            row=dyn.evaluate_pair(observer,seed)
            assert len(observer.states)==640 and row['pre_kick_state_equal']
            assert row['control']['finite'] and row['perturbed']['finite']
            row['control']['states']=torch.stack(observer.states[320:384]).tolist()
            row['perturbed']['states']=torch.stack(observer.states[576:640]).tolist()
            records.append(row)
            print(f'CAP1R {arm}: {index+1}/16 seeds',flush=True)
        result[arm]=records
    return {'manifest':identities(),'conditions':result}


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    phase=sys.argv[1]
    if phase=='register':
        paths=[Path(__file__),ROOT/'docs/cap1r_dimension_protocol_20260906.md',ROOT/'tools/capability_completion_20260906.py',ROOT/'docs/registrations/cap1_manifest.json',ROOT/'tools/cap1_logit_derivation_20260906.py',ROOT/'docs/cap1_logit_derivation_20260906.md']
        save(MANIFEST,{str(p.relative_to(ROOT)):sha(p) for p in paths})
        return
    identities()
    calibration()
    OUT.mkdir(exist_ok=True)
    target=OUT/(phase+'.json.gz')
    if target.exists():
        raise RuntimeError('refuse existing campaign')
    save(target,campaign())
    if phase=='twin_b':
        a,b=load(OUT/'twin_a.json.gz'),load(target)
        exact=a==b
        save(OUT/'identity.json',{'exact_twins':exact,'manifest':identities(),'a_sha256':sha(OUT/'twin_a.json.gz'),'b_sha256':sha(target)})
        if not exact:
            raise RuntimeError('CAP1R VOID: twin mismatch')
        print('CAP1R full saved-state and feature twins match exactly.',flush=True)


if __name__=='__main__':
    main()
