"""Read-only exposed-development diagnostics. Splices are nonphysical algebraic probes."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training import run_lcm4_compatibility as R,lcm4_compatibility_contract as K,lcm2_contract as DATA,lcm3_contract as L3


def stats(values):
    x=np.asarray(values,float)
    return dict(min=float(x.min()),median=float(np.median(x)),max=float(x.max()),
        quantiles=np.quantile(x,[.05,.25,.5,.75,.95]).tolist())


@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);R.verify()
    episodes=R.P.read(R.OUT/'a/public_episodes.json');payloads=R.P.read(R.OUT/'a/evaluation.json')
    records=[e['bodies'][0][2] for e in episodes]
    before=torch.tensor([r['observation'] for r in records],dtype=torch.float32)
    cue=torch.tensor([r['next_observation'] for r in records],dtype=torch.float32)
    reward=torch.tensor([r['reward'] for r in records],dtype=torch.float32)
    side=torch.tensor([e['side'] for e in episodes]);quality=torch.tensor([e['quality'] for e in episodes])
    query=torch.tensor([e['query'] for e in episodes],dtype=torch.float32);target=torch.tensor([e['target'] for e in episodes])
    names=('energy','integrity','position','coarse_resource','inspection_valid','precise_resource','tool_condition','resource_quality')
    source=DATA.episodes(L3.CONFIG['normalization_seed'],L3.CONFIG['normalization_n'],1)
    ranges=[]
    for i,name in enumerate(names):
        source_values=source['cue'][:,i];native=cue[:,i]
        ranges.append(dict(sensor=name,native=stats(native),source_training_role=stats(source_values),
            outside_finite_source_sample=float(((native<source_values.min())|(native>source_values.max())).float().mean())))
    cells=[];splices=[];coarse_groups=[]
    for trial,p in enumerate(payloads):
        model=R.model_for(trial);original=R.P.tree_hash(model.state_dict());full=next(r for r in p['rows'] if r['control']=='full')
        probability=np.array(full['probabilities']);recall=np.array(full['recall_probability'])
        for s in (0,1):
            for q in (0,1):
                ids=np.flatnonzero((side.numpy()==s)&(quality.numpy()==q))
                cells.append(dict(trial=trial,side=s,quality=q,n=len(ids),
                    recall_probability=stats(recall[ids]),quality_accuracy=float(np.mean((recall[ids]>=.5)==q)),
                    correct_action_probability=stats(probability[ids,target.numpy()[ids]])))
                for pair in sorted({(float(before[i,3]),float(cue[i,3])) for i in ids}):
                    group=[i for i in ids if (float(before[i,3]),float(cue[i,3]))==pair]
                    coarse_groups.append(dict(trial=trial,side=s,quality=q,before_coarse=pair[0],after_coarse=pair[1],
                        n=len(group),recall_accuracy=float(np.mean((recall[group]>=.5)==q)),recall_probability=stats(recall[group])))
        for treatment in ('native','energy_anchor','integrity_anchor','coarse_anchor','precise_anchor','tool_anchor',
                          'zero_reward','synchronize_before','all_source_anchor'):
            obs=before.clone();nxt=cue.clone();rr=reward.clone()
            for name,index,value in (('energy_anchor',0,.85),('integrity_anchor',1,.9),('coarse_anchor',3,.5)):
                if treatment in (name,'all_source_anchor'):obs[:,index]=value;nxt[:,index]=value
            for name,index,value in (('precise_anchor',5,.75),('tool_anchor',6,.75)):
                if treatment in (name,'all_source_anchor'):nxt[:,index]=value
            if treatment in ('zero_reward','all_source_anchor'):rr.zero_()
            if treatment in ('synchronize_before','all_source_anchor'):obs[:,:4]=nxt[:,:4]
            state=model.store.initial(len(episodes))
            state=model.store.observe(obs,torch.full((len(episodes),),4),rr,nxt,
                torch.zeros(len(episodes),dtype=torch.bool),state,torch.ones(len(episodes),dtype=torch.bool))
            z=state['z'];r=model.store.quality(z).gather(1,side[:,None]).squeeze(1).sigmoid()
            probability=model.logits(query,z).softmax(-1)
            if treatment=='native':np.testing.assert_allclose(z,p['written'],atol=0,rtol=0)
            for s in (0,1):
                for q in (0,1):
                    mask=(side==s)&(quality==q)
                    splices.append(dict(trial=trial,treatment=treatment,side=s,quality=q,
                        recall_accuracy=float(((r[mask]>=.5)==quality[mask].bool()).float().mean()),
                        recall_probability=stats(r[mask]),
                        correct_action_probability=stats(probability[mask].gather(1,target[mask,None]).squeeze(1))))
        assert R.P.tree_hash(model.state_dict())==original
    report=dict(diagnostic_only=True,verdict=R.P.read(R.OUT/'verdict.json')['verdict'],
        source_sha=R.P.sha(Path(__file__)),evaluation_sha=R.P.sha(R.OUT/'a/evaluation.json'),
        input_sha=R.P.sha(R.OUT/'a/public_episodes.json'),native_reward=stats(reward),sensor_ranges=ranges,cells=cells,
        coarse_groups=coarse_groups,splices=splices,
        limitation='Exposed development data only, no training. Sensor anchors/splices are algebraic counterfactual writes, not physical interventions or new qualification results.')
    R.P.save(ROOT/'zeus_sandbox/universe/reports/lcm4_compatibility_diagnosis_20260912.json',report)
    print('Native sensor ranges',json.dumps(ranges,indent=2))
    for row in cells:
        if row['trial']==2 and row['side']==1 and row['quality']==0:print('Failed native cell',row)
    for row in splices:
        if row['trial']==2 and row['side']==1 and row['quality']==0:print('Algebraic probe',row)

if __name__=='__main__':main()
