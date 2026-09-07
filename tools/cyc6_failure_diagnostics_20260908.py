"""Post-verdict saved-trace analysis only; no new world or model evaluation."""
import copy
from collections import Counter
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import cyc6_initialization_robustness_20260907 as R


def main():
    report=R.N.read(R.OUT/'verdict.json');result={}
    for trial in R.TRIALS:
        if report['trials'][trial]['verdict']!='FAIL':continue
        episodes=[]
        for pair,cal in zip(R.N.rows(R.OUT/trial/'evaluation_a.jsonl.gz'),R.N.rows(R.OUT/'calibration_a.jsonl.gz')):
            assert pair['seed']==cal['seed']
            for oi,o in enumerate(pair['orientations']):
                episodes.append((pair['seed'],oi,o['controls']['intact'],cal['orientations'][oi]['preparation']))
        assert len(episodes)==256
        episodes.sort(key=lambda item:(item[2]['age'],item[0],item[1]))
        seed,oi,e,p=episodes[128];ages=[a[2]['age'] for a in episodes]
        cache=copy.deepcopy(p['cache']);obs=p['observation'];details=[]
        for row in e['trace']:
            age=row['tick']-1;R.C.observe(cache,obs,age)
            action,_=R.C.choose(obs,age,cache)
            details.append(dict(age=age,position=round(8*obs[4]),action=row['action'],
                explicit_encounter_cache_action=action.name.lower(),prediction=row['prediction'],
                explicit_encounter_cache_estimates=[R.C.estimate(cache,j,age) for j in range(9)]))
            obs=row['after']
        correct=[a[2]['age'] for a in episodes if a[2]['first_action']==R.C.Action(a[1]+1).name.lower()]
        result[trial]=dict(initialization=report['trials'][trial]['initialization_seed'],
            intact=report['trials'][trial]['summaries']['intact'],death_causes=dict(Counter(a[2]['failure_cause'] for a in episodes)),
            ages=dict(min=min(ages),median=float(np.median(ages)),max=max(ages)),
            correct_first_action_ages=dict(n=len(correct),min=min(correct) if correct else None,median=float(np.median(correct)) if correct else None,max=max(correct) if correct else None),
            upper_median_example=dict(sorted_index=128,seed=seed,rich_target=(2 if oi==0 else 6),age=e['age'],
                final_physical=e['final_physical'],tail_actions=dict(Counter(x['action'] for x in details[-20:])),
                tail_disagreements=sum(x['action']!=x['explicit_encounter_cache_action'] for x in details[-20:]),tail=details[-20:]))
    output=dict(kind='CYC6_POST_VERDICT_FAILURE_DIAGNOSTICS',diagnostic_only=True,selection='all failed trials; upper median intact episode ordered by age,world seed,orientation',
        source_verdict_sha=R.C.sha(R.OUT/'verdict.json'),source_sha=R.C.sha(Path(__file__)),trials=result)
    R.C.save(R.OUT/'failure_diagnostics_20260908.json',output)
    for trial,v in result.items():
        print(trial,v['ages'],'correct-first-action',v['correct_first_action_ages'],'median-tail disagreements',v['upper_median_example']['tail_disagreements'],flush=True)


if __name__=='__main__':main()
