"""Development calibration only; never uses LCM1 campaign seeds."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from core.lineage_agent import LineageAgent
from training import run_lcm1 as R

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    config=R.CONFIG|dict(trials=1,train_base=204022000,evaluation_base=204032000,
        initialization_base=204222000,train_action_base=204322000,evaluation_action_base=204422000)
    out=ROOT/'runs/lcm1_development_v2_20260912';out.mkdir()
    R.save(out/'config.json',config)
    R.train_one(0,'full',out/'full',config)
    payload,_=R.checked_checkpoint(out/'full');model=LineageAgent();model.load_state_dict(payload['model'])
    results=[];donors={}
    for index in range(config['evaluation_n']):
        row,z=R.evaluate_lineage(model,0,index,'full',config=config);results.append(row);donors[index]=z
    for arm in ('acute_reset','acute_shuffle','acute_zero'):
        for index in range(config['evaluation_n']):
            pair=index^1
            row,_=R.evaluate_lineage(model,0,index,arm,donors=donors[pair],config=config);results.append(row)
    summary={arm:[sum(r['bodies'][cycle]['survived'] for r in results if r['arm']==arm)
                  for cycle in range(config['cycles'])] for arm in ('full','acute_reset','acute_shuffle','acute_zero')}
    R.save(out/'evaluation.json',dict(config=config,development_only=True,results=results,alive_counts=summary))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
