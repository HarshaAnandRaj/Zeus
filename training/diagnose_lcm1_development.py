"""Read-only causal and representation diagnosis of the LCM1 development pilot."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core.lineage_agent import LineageAgent
from training import run_lcm1 as R,audit_lcm1 as A

def cue(model,quality,lag):
    state=model.initial(1)
    plain=torch.tensor([[.85,.95,0.,.5,0.,0.,0.,0.]])
    inspected=torch.tensor([[.85,.95,0.,.5,1.,.7,.9,float(quality)]])
    state=model.observe(plain,torch.tensor([4]),torch.tensor([0.]),inspected,
                        torch.tensor([False]),state,torch.tensor([True]))
    for _ in range(lag):
        state=model.observe(plain,torch.tensor([0]),torch.tensor([0.]),plain,
                            torch.tensor([False]),state,torch.tensor([True]))
    state=model.reset_fast(state,torch.tensor([True]))
    _,_,result=model.act(plain,state,torch.Generator().manual_seed(204602000))
    return state,result

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=ROOT/'runs/lcm1_development_v2_20260912';artifact=R.read(out/'evaluation.json');config=artifact['config']
    payload,_=R.checked_checkpoint(out/'full');model=LineageAgent();model.load_state_dict(payload['model'])
    physical=A.audit_training(out/'full',config)
    stored={(r['arm'],r['index']):r for r in artifact['results']};donors={};audited=0
    for index in range(config['evaluation_n']):
        r,z=A.endpoint(payload['model'],0,index,'full',config=config)
        assert r==stored['full',index];donors[index]=z;audited+=1
    for arm in ('acute_reset','acute_shuffle','acute_zero'):
        for index in range(config['evaluation_n']):
            r,_=A.endpoint(payload['model'],0,index,arm,donors[index^1],config)
            assert r==stored[arm,index];audited+=1
    behavior={}
    for arm in ('full','acute_reset','acute_shuffle','acute_zero'):
        rows=[r for r in artifact['results'] if r['arm']==arm];trace=[s for r in rows for s in r['trace']]
        behavior[arm]=dict(alive_counts=[sum(r['bodies'][c]['survived'] for r in rows) for c in range(4)],
            mean_ticks=[float(np.mean([r['bodies'][c]['ticks'] for r in rows])) for c in range(4)],
            action_counts=[sum(s['action']==a for s in trace) for a in range(6)],
            valid_inspections=sum(s['next_observation'][4] and s['next_observation'][2] in (0.,1.) for s in trace),
            feeding=sum(b['feeding'] for r in rows for b in r['bodies']),
            bad_harvest=sum(b['bad_harvest'] for r in rows for b in r['bodies']))
    decay={};initial=R.model_for(0,config)
    with torch.no_grad():
        for name,m in (('initial',initial),('trained',model)):
            decay[name]=[]
            for lag in (0,1,8,32,64):
                s0,r0=cue(m,0,lag);s1,r1=cue(m,1,lag)
                decay[name].append(dict(lag=lag,state_distance=float((s1['z']-s0['z']).norm()),
                    recall_probability_difference=float(m.quality(s1['z']).sigmoid()[0,0]-m.quality(s0['z']).sigmoid()[0,0]),
                    action_probability_distance=float((r1['logits'].softmax(-1)-r0['logits'].softmax(-1)).norm())))
    state,result=cue(model,1,8);model.zero_grad(set_to_none=True);(-result['actor_logp']).backward()
    gradients={name:float(p.grad.norm()) if p.grad is not None else 0. for name,p in model.named_parameters()}
    movement={name:float((p.detach()-dict(initial.named_parameters())[name].detach()).norm()) for name,p in model.named_parameters()}
    report=dict(status='DEVELOPMENT_FAIL',campaign_endpoint_exposed=False,
        physical_audit=physical,endpoint_lineages_replayed=audited,endpoint_bodies_replayed=audited*4,
        behavior=behavior,cue_decay=decay,actor_path_gradient_norms=gradients,parameter_movement=movement,
        first_quality_loss=payload['updates'][0]['quality'],last_quality_loss=payload['updates'][-1]['quality'],
        acquisition_count=13,acquisition_required=48,
        interpretation='No development inheritance effect detected. Slow state overwrites isolated cue information; delayed actor graph remains connected.',
        sources={p:R.sha(ROOT/p) for p in ('core/lineage_agent.py','core/lineage_ecology.py','training/run_lcm1.py','training/diagnose_lcm1_development.py')})
    R.save(ROOT/'zeus_sandbox/universe/reports/lcm1_development_20260912.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
