"""Saved-report capability availability audit; not a capability gate."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.qv1_postmortem_20260906 import REPORTS, read, repertoire, sha


def main():
    output = REPORTS / 'capability_pilot_20260906.json'
    if output.exists():
        raise RuntimeError('refuse overwrite')
    pol = read('pol2_endogenous_action_verdict_20260905.json')
    qv = read('qv1_retention_lineage_verdict_20260906.json')
    dyn = read('dyn1_resilience_20260905.json')
    cal = read('embodiment_v2_calibration_20260905.json')
    assert cal['seeds'] == pol['conditions']['eval_seeds']
    assert cal['horizon'] == pol['conditions']['eval_horizon']
    rows = {}
    for name, condition in pol['evaluations'].items():
        counts = sum((Counter(e['selected_actions']) for e in condition['episodes']),Counter())
        rows['POL2/'+name] = repertoire(counts)
    for name, condition in qv['conditions'].items():
        rows['QV1/'+name] = repertoire(condition['selected_actions'])
    for name, condition in cal['results'].items():
        rows['V2/'+name] = repertoire(condition['actions'])
    pol_value = rows['POL2/normal']['entropy_effective_actions']
    separation = all(pol_value > rows['V2/'+n]['entropy_effective_actions'] for n in ['fixed_rest','fixed_harvest'])
    ranks = {name: [{'seed':r['seed'], 'control_rank90':r['control']['features']['effective_rank_90'],
                     'perturbed_rank90':r['perturbed']['features']['effective_rank_90']} for r in dyn[name]['records']]
             for name in ['trained','random_init']}
    result = {'grade':'diagnostic only', 'status':'NOT READY', 'continuation_authority':False,
              'source_hashes':{n:sha(REPORTS/n) for n in ['pol2_endogenous_action_verdict_20260905.json','qv1_retention_lineage_verdict_20260906.json','dyn1_resilience_20260905.json','embodiment_v2_calibration_20260905.json']},
              'instrument_sha256':sha(__file__), 'repertoire':rows, 'DYΝ1_original_rank90':ranks,
              'POL2_separates_saved_fixed_reflexes':separation,
              'viable_set_coverage':None, 'covariance_participation_ratio':None, 'empowerment':None,
              'availability':{
                  'QV1':'Saved formal report contains episode counts and outcomes, not per-tick states or causal action channels.',
                  'POL2':'Saved formal report and calibration contain action counts and outcomes, not per-tick states or counterfactual action channels.',
                  'DYN1':'Saved report has rank90 feature summaries; no full covariance spectrum, action channel, or embodied viable observations.',
              },
              'interpretation':'Repertoire distinguishes a multi-action policy from constant action but does not prove useful control. Full instrument components are unidentifiable from these saved reports. No template activation; no further data collection is authorized by this pilot.'}
    with output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'separation':separation,'POL2_effective_actions':pol_value,'path':str(output)}))


if __name__=='__main__':
    main()
