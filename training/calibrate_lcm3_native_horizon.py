"""Development horizon checks only; no agent training or native held-out exposure."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.lineage_ecology import LineageEcology,LineageConfig
from training import run_lcm3 as R

def main():
    rows=[]
    for horizon in (64,128,256):
        for policy in ('wait','public_inspection_map'):
            survivors=ticks=feeding=inspections=0
            for seed in range(204002000,204002512):
                ecology=LineageEcology(seed=seed,config=LineageConfig(4,horizon))
                for cycle in range(4):
                    world=ecology.body(cycle);known_safe=None
                    for tick in range(horizon):
                        obs=world.observation()
                        if policy=='wait':action=0
                        elif known_safe is None:action=1 if obs.position>0 else 4
                        elif obs.position*4<known_safe:action=2
                        elif obs.position*4>known_safe:action=1
                        elif obs.energy<.5:action=3
                        else:action=0
                        effect=world.step(action);feeding+=action==3 and effect.after.energy>effect.before.energy
                        inspections+=action==4
                        if action==4 and effect.after.inspection_valid:
                            known_safe=0 if effect.after.resource_quality==1 else 4
                        if effect.terminated:break
                    ticks+=tick+1;survivors+=not effect.terminated
            rows.append(dict(horizon=horizon,policy=policy,bodies=2048,survived=survivors,
                survival_fraction=survivors/2048,mean_ticks=ticks/2048,feeding=feeding,inspections=inspections))
    report=dict(development_only=True,ecology_base=204002000,ecologies=512,cycles=4,rows=rows,
        limitation='Scripted public reference and passive control only; no learned motor or inheritance qualification.',
        source_sha=R.sha(Path(__file__)),
        world_sources={p:R.sha(R.ROOT/p) for p in ('core/lineage_ecology.py','core/lifetime_world.py','core/lifetime_world_v2.py')})
    R.save(ROOT/'zeus_sandbox/universe/reports/lcm3_native_horizon_calibration_20260912.json',report)
    for r in rows:print(r)

if __name__=='__main__':main()
