"""Independent policy/factory reconstruction, scalar balances and energy bound."""
import copy,gzip,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.lineage_ecology import LineageEcology,LineageConfig
from core.lifetime_world_v2 import QualityWorld
from training import calibrate_repair_dependent_world as R
from training.audit_lifetime_calibration_v2 import balance
from training.audit_lifetime_calibration import close

def main():
    assert __debug__;manifest=R.verify();cfg=R.CONFIG;rows=[];steps=0;envelope_checks=0
    # Arithmetic-series bound independently arranged from the draft's term sum.
    c=manifest['physics'];terms=math.ceil(c['initial_tool']/c['harvest_wear'])
    bonus=c['extraction_limit']*c['efficiency_gain']*(terms*c['initial_tool']-c['harvest_wear']*terms*(terms-1)/2)
    decay=c['metabolism']-(c['extraction_limit']*c['efficiency_floor']-c['harvest_cost'])
    assert decay>0 and c['initial_energy']-decay*598+bonus<c['death_threshold']
    with gzip.open(R.OUT/'trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for seed in range(cfg['base'],cfg['base']+cfg['ecologies']):
            for control in cfg['controls']:
                original=LineageEcology(seed=seed,config=LineageConfig(2,cfg['horizon'])).body(0).snapshot()
                modified=copy.deepcopy(original);modified['config']['efficiency_floor']=.05
                if control=='tool_repair_disabled':modified['config']['tool_repair']=0.
                candidate=R.RepairDependentEcology(seed=seed,tool_repair_enabled=control!='tool_repair_disabled').body(0).snapshot()
                assert candidate==modified,'factory altered undeclared physical/RNG state'
                world=QualityWorld.restore(modified);safe=tool=None;counts=[0]*6;increases=0
                for tick in range(cfg['horizon']):
                    row=json.loads(next(stream));before=R.N.V.physical(world.snapshot());obs=list(world.observation().values());position=obs[2]
                    if obs[4] and position in (0.,1.):safe=int(position) if obs[7] else 1-int(position);tool=obs[6]
                    if control=='passive':chosen=0
                    elif safe is None:chosen=1 if position>0 else 4
                    elif control in ('repair','tool_repair_disabled') and ((tool is not None and tool<.75) or obs[1]<.8):
                        chosen=2 if position<.5 else 1 if position>.5 else 5
                    elif position<safe:chosen=2
                    elif position>safe:chosen=1
                    else:chosen=3 if obs[0]<( .5 if control in ('repair','tool_repair_disabled') else .65) else 0
                    assert (row['seed'],row['control'],row['tick'],row['action'])==(seed,control,tick,chosen)
                    assert row['before']==before and row['observation']==obs
                    close(balance(before,chosen,modified['config']),row['after'])
                    effect=world.step(chosen);after=R.N.V.physical(world.snapshot());assert row['after']==after
                    assert row['next_observation']==list(effect.after.values()) and row['terminated']==effect.terminated
                    if control in ('repair','tool_repair_disabled'):
                        if chosen==3 and tool is not None:tool=max(0.,tool-.008)
                        # Deliberately preserve the same supplied reference's expectation in the disabled arm.
                        if chosen==5 and after['position']==2 and tool is not None:tool=min(1.,tool+.40)
                    counts[chosen]+=1;increases+=after['tool']>before['tool'];steps+=1
                    if control!='repair':
                        assert after['energy']<=c['initial_energy']-decay*(tick+1)+bonus+1e-10
                        assert increases==0;envelope_checks+=1
                    if effect.terminated:break
                rows.append(dict(seed=seed,control=control,ticks=tick+1,survived=not effect.terminated,actions=counts,
                    tool_increases=increases,final_energy=after['energy'],final_integrity=after['integrity']))
            if (seed-cfg['base']+1)%8==0:print('repair independent replay ecologies',seed-cfg['base']+1,flush=True)
        assert next(stream,None) is None
    assert rows==R.read(R.OUT/'rows.json')
    # Reconstruct qualification without the primary decision function.
    groups={control:[r for r in rows if r['control']==control] for control in cfg['controls']};assert all(len(rs)==32 for rs in groups.values())
    summary=[dict(control=control,n=32,survivors=sum(r['survived'] for r in rs),min_ticks=min(r['ticks'] for r in rs),max_ticks=max(r['ticks'] for r in rs),
        min_tool_increases=min(r['tool_increases'] for r in rs),max_tool_increases=max(r['tool_increases'] for r in rs)) for control,rs in groups.items()]
    gates=dict(public_repair_feasible=sum(r['survived'] for r in groups['repair'])>=math.ceil(.90*32),
        actual_tool_restoration=all(r['tool_increases']>0 for r in groups['repair']),
        no_repair_fails=all(not r['survived'] and r['tool_increases']==0 and r['ticks']<=598 for r in groups['no_repair']),
        disabled_restoration_fails=all(not r['survived'] and r['tool_increases']==0 and r['ticks']<=598 for r in groups['tool_repair_disabled']),
        passive_fails=all(not r['survived'] for r in groups['passive']))
    result=dict(verdict='PASS' if all(gates.values()) else 'FAIL',gates=gates,summary=summary,
        scope='Engineered repair-dependent reference feasibility only; no neural maintenance or pillar promotion')
    assert result==R.read(R.OUT/'verdict.json')
    receipt=dict(status='PASS',verdict=result['verdict'],bodies=len(rows),physical_steps=steps,energy_envelope_checks=envelope_checks,
        maximum_initial_tool_bonus=bonus,no_restoring_action_energy_upper_at598=c['initial_energy']-decay*598+bonus,
        evidence_sha={p:R.sha(R.OUT/p) for p in ('manifest.json','trace.jsonl.gz','rows.json','verdict.json')},
        scope='Exact public/controller/physical replay plus independent scalar/bound arithmetic; shared world primitives, no neural qualification')
    R.save(R.OUT/'audit.json',receipt);print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
