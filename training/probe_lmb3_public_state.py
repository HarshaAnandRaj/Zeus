"""Offline public-state linear decodability; no functional endpoint or model edits."""
import gzip,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training import run_lmb3 as R,learner_history_correction as C
from training.dual_body_replay import Replay

RIDGE=.001
SOURCES=('training/probe_lmb3_public_state.py','docs/lmb3_public_state_probe_protocol_20260913.md')
REPORT=ROOT/'zeus_sandbox/universe/reports/lmb3_public_state_probe_20260913.json'


def fit(inputs,target):
    inputs=np.asarray(inputs,np.float64);target=np.asarray(target,np.float64)
    mean=inputs.mean(0);scale=inputs.std(0);scale[scale<1e-12]=1.
    design=np.column_stack(((inputs-mean)/scale,np.ones(len(inputs))))
    penalty=np.eye(design.shape[1])*RIDGE;penalty[-1,-1]=0.
    coefficient=np.linalg.solve(design.T@design/len(design)+penalty,design.T@target/len(design))
    return dict(mean=mean,scale=scale,coefficient=coefficient)


def metrics(prediction,target):
    error=prediction-target;n=len(target)
    levels=np.array([0.,.25,.5,.75,1.]);position=levels[np.abs(prediction[:,2,None]-levels).argmin(1)]
    return dict(n=n,rmse=np.sqrt((error**2).mean(0)).tolist(),mae=np.abs(error).mean(0).tolist(),
        position_accuracy=float((position==target[:,2]).mean()))


def main():
    assert __debug__ and not REPORT.exists();torch.set_num_threads(1);R.verify()
    audit=R.L.P.L3.read(ROOT/'zeus_sandbox/universe/reports/lmb3_audit_20260913.json')
    assert audit['status']=='PASS' and audit['verdict']=='FAIL'
    for name,sha in audit['evidence_sha'].items():assert R.L.P.L3.sha(R.OUT/name)==sha
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in SOURCES:assert subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT).replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
    endpoint=R.L.P.L3.read(R.OUT/'endpoint.json');preps=R.L.P.L3.read(R.OUT/'endpoint_public.json')
    lookup={(r['trial'],r['index'],r['mode']):r for r in endpoint['bodies'] if r['mode'] in ('inherited','empty')}
    groups={};records=0;bodies=0;output_error=0.
    with gzip.open(R.OUT/'candidate_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);trial=row['trial'];index=row['index'];mode=row['mode'];key=(trial,mode)
            if row['tick']==0:
                model=R.candidate(trial);weights={k:v.numpy() for k,v in model.state_dict().items() if isinstance(v,torch.Tensor)}
                z=Replay(model.state_dict()).native_starts([preps[index]]).numpy()[0];z=z if mode=='inherited' else np.zeros(8,np.float32);bodies+=1
            h=np.asarray(row['h'],np.float32);gate=1/(1+np.exp(-(np.concatenate((h,z))@weights['gate.weight'].T+weights['gate.bias'])))
            mouth=h+gate*(z@weights['reinstate.weight'].T);logits=mouth@weights['actor.weight'].T+weights['actor.bias']
            exp=np.exp(logits-logits.max());prob=exp/exp.sum();error=float(np.max(np.abs(prob-row['probability'])))
            assert error<2e-5;output_error=max(output_error,error)
            data=groups.setdefault(key,dict(h=[],mouth=[],z=[],target=[],index=[],fatal=[],interior_wait=[]))
            for name,value in (('h',h),('mouth',mouth),('z',z.copy()),('target',row['observation'][:3])):data[name].append(value)
            data['index'].append(index);data['fatal'].append(not lookup[trial,index,mode]['survived'])
            data['interior_wait'].append(row['observation'][2] in (.25,.5,.75) and int(np.argmax(row['probability']))==0)
            z=np.asarray(row['z'],np.float32);records+=1
    assert bodies==1024 and len(groups)==8
    results=[];parameters=[]
    for (trial,mode),data in groups.items():
        target=np.asarray(data['target'],np.float64);index=np.asarray(data['index']);train=index<64;test=~train
        assert set(index[train])==set(range(64)) and set(index[test])==set(range(64,128))
        for name in ('h','mouth','z'):
            x=np.asarray(data[name]);p=fit(x[train],target[train]);prediction=np.column_stack(((x-p['mean'])/p['scale'],np.ones(len(x))))@p['coefficient']
            parameters.append(dict(trial=trial,mode=mode,representation=name,**{k:v.tolist() for k,v in p.items()}))
            selections={'all':test,'fatal':test&np.asarray(data['fatal']),'survived':test&~np.asarray(data['fatal']),
                'interior_wait':test&np.asarray(data['interior_wait'])}
            for selection,mask in selections.items():
                results.append(dict(trial=trial,mode=mode,representation=name,selection=selection,
                    metrics=metrics(prediction[mask],target[mask]) if mask.any() else dict(n=0)))
        print('offline public-state probes',trial,mode,flush=True)
    result=dict(status='COMPLETE',qualification=False,commit=commit,sources={name:R.L.P.L3.sha(ROOT/name) for name in SOURCES},
        audit_sha=R.L.P.L3.sha(ROOT/'zeus_sandbox/universe/reports/lmb3_audit_20260913.json'),ridge=RIDGE,
        bodies=bodies,records=records,max_recorded_output_reconstruction_error=output_error,parameters=parameters,results=results,
        scope='Exposed-data offline linear decodability only; public-state availability is not causal usage, absent decodability is not absent information')
    C.save(REPORT,result);print(json.dumps({k:v for k,v in result.items() if k not in ('sources','parameters','results')},indent=2))

if __name__=='__main__':main()
