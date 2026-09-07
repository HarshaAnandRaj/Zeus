"""Post-verdict saved-trajectory diagnostics only; no training or new rollouts."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import cyc4_learned_carryover_20260907 as M
from tools import cyc3_completion_audit_20260907 as A


def main():
    run=M.OUT; audit=A.read(run/'completion_audit.json'); assert audit['passed']
    data=torch.load(run/'training_data.pt',map_location='cpu',weights_only=False)
    model=torch.load(run/'twin_a.pt',map_location='cpu',weights_only=False)
    error=(model['teacher_predictions']-data['y']).square()
    fit=dict(unweighted_all_observations_mse=float(error.mean()),
             exposure_mse=float(error[:,:17].mean()),continuation_mse=float(error[:,17:].mean()))
    del data,model,error
    pairs=A.read(run/'campaign_a.json.gz'); summaries={}
    for condition in M.CONTROLS:
        es=[o['controls'][condition] for p in pairs for o in p['orientations']]
        final20=[r for e in es for r in e['trace'][-20:]]
        summaries[condition]=dict(min_age=min(e['age'] for e in es),max_age=max(e['age'] for e in es),
            median_age=float(np.median([e['age'] for e in es])),
            mean_cycles=float(np.mean([e['cycles']['count'] for e in es])),
            worlds_reaching_outer_cells=sum(any(r['position_after'] in (0,1,7,8) for r in e['trace']) for e in es),
            mean_resources_remaining_at_death=float(np.mean([sum(e['final_physical']['resources']) for e in es])),
            last_up_to_20_ticks=dict(rows=len(final20),harvested_energy=sum(r['harvested_energy'] for r in final20),
                basal_cost=sum(r['basal_cost'] for r in final20),action_cost=sum(r['action_cost'] for r in final20),
                empty_harvest_penalty=sum(r['empty_harvest_penalty'] for r in final20)))
    orientations=sorted([o for p in pairs for o in p['orientations']],
                         key=lambda o:(o['controls']['intact']['age'],o['preparation']['seed'],o['preparation']['rich_target']))
    o=orientations[len(orientations)//2]; e=o['controls']['intact']; cache=copy.deepcopy(o['preparation']['cache']); tail=[]
    for row in e['trace']:
        obs=row['before']; tick=row['tick']-1; cache[str(row['position_before'])]=dict(resource=obs[3],tick=tick)
        oracle=A.independent_choice(obs,tick,cache)
        tail.append(dict(tick=tick,position=row['position_before'],energy=obs[0],local=obs[3],
            action=row['action'],explicit_action=M.C.Action(oracle).name.lower(),harvest=row['harvested_resource'],
            energy_after=row['after'][0],decision=row['decision']))
    result=dict(kind='CYC4_POSTHOC_SAVED_FAILURE_DIAGNOSTICS',gate_authority=False,training_fit=fit,summaries=summaries,
                median_intact_example=dict(seed=o['preparation']['seed'],target=o['preparation']['rich_target'],age=e['age'],
                    final_resources=e['final_physical']['resources'],last_12=tail[-12:]),
                source_verdict_sha=A.digest(run/'verdict.json'),source_audit_sha=A.digest(run/'completion_audit.json'),
                diagnostic_source_sha=A.digest(Path(__file__)))
    M.C.save(run/'failure_diagnostics.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
