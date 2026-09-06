"""Independent read-only endpoint/accounting audit; no training or policy rollout."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.embodiment import EmbodiedWorldV2
from training.clone_orbit_policy import exact

def read(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:
        return json.load(f)

def main():
    checks={}
    folder=ROOT/'runs/cyc1_20260907'
    c=read(folder/'verdict.json')
    f=read(ROOT/'zeus_sandbox/universe/reports/cycle_forensics_20260907.json.gz')
    for name,manifest in [('CYC1',c['source_manifest']['sources']),('CYC-F',f['source_hashes'])]:
        checks[name+'_sources_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    checks['CYC1_saved_artifact_hashes']=all(hashlib.sha256((folder/p).read_bytes()).hexdigest()==h for p,h in c['artifacts'].items())
    a=torch.load(folder/'twin_a.pt',map_location='cpu',weights_only=False)
    b=torch.load(folder/'twin_b.pt',map_location='cpu',weights_only=False)
    checks['all_reloaded_training_tensors_traces_optimizer_exact']=exact(a,b)
    checks['dataset_count_shape']=(a['features'].shape==(32768,48) and a['labels'].shape==(32768,))
    checks['dataset_finite']=bool(torch.isfinite(a['features']).all())
    ev=read(folder/'twin_a_evaluation.json.gz')
    checks['all_reloaded_evaluation_traces_exact']=ev==read(folder/'twin_b_evaluation.json.gz')
    calibration=read(folder/'calibration.json.gz')
    checks['teacher_control_64_survivors']=sum(e['survived_512'] for e in calibration['teacher'])==64
    checks['fixed_controls_no_survivors']=all(not e['survived_256'] for k in ['fixed_rest','fixed_harvest'] for e in calibration[k])
    replayed=0
    for es in list(ev.values())+list(calibration.values())+[a['teacher']]:
        for e in es:
            w=EmbodiedWorldV2(seed=e['seed'])
            for row in e['trace']:
                assert w.viable()
                before=w.observation()
                effect=w.step(row['action'])
                assert np.allclose(before,row['effect']['before'],atol=1e-12,rtol=0)
                assert np.allclose(effect['after'],row['effect']['after'],atol=1e-12,rtol=0)
                assert effect['viable']==row['effect']['viable']
            assert w.body.age==e['age']
            replayed+=1
    checks['CYC1_saved_action_replays']=replayed==512
    episodes=[e for es in f['episodes'].values() for e in es]
    for e in episodes:
        t=e['lifetime_totals']
        energy=e['initial']['observation'][0]+t['harvested_energy']-t['basal_cost']-t['action_cost']-t['empty_harvest_penalty']+t['energy_clipping']
        integrity=e['initial']['observation'][1]+t['integrity_repair']-t['integrity_basal']-t['starvation_damage']-t['thermal_damage']+t['integrity_clipping']
        assert abs(energy-e['final'][0])<1e-12 and abs(integrity-e['final'][1])<1e-12
    checks['960_forensic_lifetime_energy_integrity_balances']=len(episodes)==960
    checks['955_death_tails']=sum(len(e['tail'])==min(20,e['age']) for e in episodes if not e['survived_256'])==955
    checks['cycle_gate_correctly_unidentified']=f['verdict']=='UNDECIDED' and not f['primary']['identified'] and f['primary']['conditions']['POL2/normal']['noncycle_n']==0
    draws=np.random.Generator(np.random.PCG64(20260953)).integers(0,64,(10000,64))
    es=ev['normal']
    for key in ['executed_moves','zero_flips']:
        numer=np.array([e[key] for e in es]); denom=np.array([e['decisions'] for e in es])
        ci=np.quantile(numer[draws].sum(1)/denom[draws].sum(1),[.025,.975]).tolist()
        checks[key+'_bootstrap_recomputed']=ci==c['bars'][key]['ci95']
    checks['zero_survival_both_horizons']=all(not e['survived_256'] and not e['survived_512'] for e in es)
    checks['functional_failure_unrescued']=c['verdict']=='FAIL' and c['bars']['survival_256']['ci95'][1]<.9 and c['bars']['survival_512']['ci95'][1]<.8
    checks['no_followup_launch_authority']=not c['followup_authorized'] and not f['cycle_training_launch_authorized']
    assert all(checks.values()),checks
    result=dict(all_passed=True,checks=checks,checks_count=len(checks),CYC1_replayed_episodes=replayed,
                forensic_episodes=len(episodes),forensic_deaths=955)
    out=ROOT/'zeus_sandbox/universe/reports/cycle_completion_audit_20260907.json'
    with out.open('x') as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
