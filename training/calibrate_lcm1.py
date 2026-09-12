"""Development world range; privileged oracle is never used for agent training."""
import random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.lineage_ecology import LineageEcology
from training import run_lcm1 as R

def main():
    result={}
    for policy in ('random','privileged_oracle'):
        survived=ticks=feeding=0
        for seed in range(204002000,204002512):
            ecology=LineageEcology(seed=seed);rng=random.Random(seed+204602000)
            safe=ecology.audit_snapshot()['safe_patch']*4
            for cycle in range(4):
                world=ecology.body(cycle)
                for tick in range(64):
                    obs=world.observation()
                    if policy=='random':action=rng.randrange(6)
                    elif obs.position*4<safe:action=2
                    elif obs.position*4>safe:action=1
                    elif obs.energy<.5:action=3
                    else:action=0
                    effect=world.step(action);feeding+=action==3 and effect.after.energy>effect.before.energy
                    if effect.terminated:break
                ticks+=tick+1;survived+=not effect.terminated
        result[policy]=dict(bodies=2048,survived=survived,survival_fraction=survived/2048,
                            mean_ticks=ticks/2048,feeding=feeding)
    report=dict(development_only=True,ecology_base=204002000,ecologies=512,action_rng_offset=204602000,
        policies=result,sources={p:R.sha(ROOT/p) for p in ('core/lineage_ecology.py','core/lifetime_world.py',
        'core/lifetime_world_v2.py','training/calibrate_lcm1.py')})
    R.save(ROOT/'zeus_sandbox/universe/reports/lcm1_world_calibration_20260912.json',report)
    print(result)

if __name__=='__main__':main()
