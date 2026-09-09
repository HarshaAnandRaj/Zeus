"""OBS7 factual resource prediction under frozen hidden-component edits."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import subprocess
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs6_cancellation_20260909 as C
B,Q,P,O,R=C.B,C.Q,C.P,C.O,C.R
OUT=ROOT/'runs/obs7_20260909';ARMS=B.ARMS+('matched_full_null_shift',);SCALES=B.SCALES;TIMES=B.TIMES


def validate_targets(orientation,driver):
    trace=orientation['teacher']['trace'];assert len(trace)==496
    target=np.array([r['resources_before'] for r in trace]);after=np.array([r['resources_after'] for r in trace]);positions=np.array([r['position_before'] for r in trace],dtype=int)
    assert target.shape==(496,9) and np.all((target>=0)&(target<=1))
    np.testing.assert_array_equal(np.array([R.M.features(r['before']) for r in trace]),driver)
    np.testing.assert_array_equal([r['tick'] for r in trace],np.arange(17,513))
    np.testing.assert_array_equal(target[np.arange(496),positions],driver[:,0])
    np.testing.assert_array_equal(driver[:,1:].argmax(1),positions)
    np.testing.assert_array_equal(target[1:],after[:-1]);np.testing.assert_array_equal(target[0],orientation['preparation']['boundary']['resources'])
    np.testing.assert_array_equal(after[-1],orientation['teacher']['final_physical']['resources'])
    mask=np.ones((496,9),dtype=bool);mask[np.arange(496),positions]=False
    return target,mask


def full_null_control(w,dn):
    size=np.linalg.norm(dn)
    if size<=1e-12:return np.zeros(32)
    _,_,vw=np.linalg.svd(w,full_matrices=True);n=vw[9:].T;v=dn/size
    project=lambda z:n@(n.T@z)-v*np.dot(v,z)
    control=project(Q.DIRECTION)
    if np.linalg.norm(control)<=1e-12:control=next(project(e) for e in np.eye(32) if np.linalg.norm(project(e))>1e-12)
    return size*B.A.canonical(control)


def loss_curves(pred,target,mask):
    squared=(pred-target[:,None,None,None])**2;unobserved=mask[:,None,None,None]
    return dict(unobserved=(squared*unobserved).sum(-1)/8,current=(squared*(~unobserved)).sum(-1),all_cells=squared.mean(-1))


def decision(mean,lower):return bool(mean>=1e-6 and lower>0)


def bootstrap(benefit,specificity,mi,wi):
    result={}
    for name,matrix in [('benefit',benefit),('specificity',specificity)]:
        samples=matrix[mi[:,:,None],wi[:,None,:]].mean((1,2));bounds=np.quantile(samples,[.025,.975])
        result[name]=dict(matrix=matrix.tolist(),mean=float(matrix.mean()),bounds=bounds.tolist(),model_means=matrix.mean(1).tolist(),world_means=matrix.mean(0).tolist())
    benefit_pass=decision(result['benefit']['mean'],result['benefit']['bounds'][0])
    result['factual_benefit']=benefit_pass
    result['specific_benefit']=bool(benefit_pass and decision(result['specificity']['mean'],result['specificity']['bounds'][0]))
    return result


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);OUT.mkdir(exist_ok=False)
    prior5=R.N.read(B.OUT/'completion.json');prior6=R.N.read(C.OUT/'completion.json')
    for directory,prior in [(B.OUT,prior5),(C.OUT,prior6)]:
        audit=R.N.read(directory/'completion_audit.json');assert audit['passed'] and audit['completion_sha']==O.sha(directory/'completion.json');O.verify(prior['manifest'])
    verdict=R.N.read(R.OUT/'verdict.json');calibration=R.OUT/'calibration_a.jsonl.gz';assert O.sha(calibration)==verdict['artifacts']['calibration_a.jsonl.gz']
    files=[calibration,R.OUT/'verdict.json',B.OUT/'completion.json',B.OUT/'completion_audit.json',C.OUT/'completion.json',C.OUT/'completion_audit.json',
           C.OUT/'results.json',Q.PRIOR/'replay_inputs.pt',Path(__file__),ROOT/'tools/test_obs7_factual_accuracy_20260909.py',
           ROOT/'docs/obs7_factual_accuracy_protocol_20260909.md',ROOT/'tools/obs5_natural_history_20260909.py',ROOT/'tools/obs4_slow_readout_20260909.py',
           ROOT/'tools/obs3_constant_input_20260908.py',ROOT/'tools/obs2_investigations_20260908.py']
    sources={str(p.relative_to(ROOT)).replace('\\','/'):O.sha(p) for p in files}
    for name in prior5['artifacts']:
        if name.endswith('.npz') or name.endswith('.json'):
            path=B.OUT/name;assert O.sha(path)==prior5['artifacts'][name];sources[str(path.relative_to(ROOT)).replace('\\','/')]=prior5['artifacts'][name]
    manifest=dict(sources=sources,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),numpy=np.__version__,torch=torch.__version__,exploratory=True)
    R.C.save(OUT/'manifest.json',manifest)
    try:
        cache=R.N.load(Q.PRIOR/'replay_inputs.pt');keys=list(cache['streams']);selected={}
        for pair in R.N.rows(calibration):
            if (pair['seed'],0) in keys:
                for oi,o in enumerate(pair['orientations']):selected[pair['seed'],oi]=o
        targets=[];masks=[]
        for key in keys:
            t,m=validate_targets(selected[key],cache['streams'][key][16:512]);targets.append(t);masks.append(m)
        targets=np.array(targets);masks=np.array(masks);np.savez_compressed(OUT/'targets.npz',targets=targets,masks=masks,keys=np.array(keys))
        allrows=[];maxprior=0.;prior6rows=R.N.read(C.OUT/'results.json')['cases']
        for trial in R.TRIALS:
            for kind in ('final','initial'):
                with np.load(B.OUT/f'{trial}_{kind}.npz') as a:old={k:a[k] for k in a.files}
                oldrows=R.N.read(B.OUT/f'{trial}_{kind}.json')['cases'][::2];p=cache['checkpoints'][trial][kind];model=Q.native(p)
                midpoint=old['midpoints'][::2];delta=old['deltas'][::2];drivers=old['drivers'][::2];priorarms=old['arm_differences'][::2]
                controls=np.array([full_null_control(p['readout.weight'],dn) for dn in priorarms[:,4]])
                arms=np.concatenate([priorarms,(delta-controls)[:,None]],axis=1)
                starts=np.array([[[m+sign*scale*d/2 for sign in (1,-1)] for scale in SCALES for d in diffs] for m,diffs in zip(midpoint,arms)]).reshape(8,28,32)
                y=B.replay(model,np.repeat(drivers,28,axis=0),starts.reshape(-1,32)).reshape(8,28,496,32)
                states=np.concatenate([starts[:,:,None],y],axis=2).reshape(8,2,7,2,497,32)
                expected=old['checkpoints'][::2];observed=states[:,:,:6,:,TIMES];maxprior=max(maxprior,float(np.max(abs(observed-expected))));np.testing.assert_allclose(observed,expected,rtol=1e-7,atol=1e-10)
                initial_logits=B.A.logits(states[:,:,:,:,0],p)
                np.testing.assert_allclose(initial_logits[:,:,6],initial_logits[:,:,0],rtol=0,atol=1e-10)
                pred=B.A.sigmoid(B.A.logits(states[:,:,:,:,1:],p));loss=loss_curves(pred,targets,masks)
                saved=dict(midpoints=midpoint,deltas=delta,controls=controls,arm_differences=arms,drivers=drivers,
                           checkpoints=states[:,:,:,:,TIMES],late_states=states[:,:,:,:,369:],predictions=pred)
                for name,value in loss.items():saved[name+'_loss']=value
                np.savez_compressed(OUT/f'{trial}_{kind}.npz',**saved)
                rows=[]
                for ci,oldrow in enumerate(oldrows):
                    assert (oldrow['world'],oldrow['orientation'])==keys[ci]
                    row={k:oldrow[k] for k in ('trial','weights','world','orientation','driver')};row['arms']={}
                    for ai,arm in enumerate(ARMS):
                        row['arms'][arm]=[[{name:dict(late=float(value[ci,si,ai,sign,368:].mean()),whole=float(value[ci,si,ai,sign].mean()),final=float(value[ci,si,ai,sign,-1])) for name,value in loss.items()} for sign in range(2)] for si in range(2)]
                    row['obs6_amplified']=next(r['prediction']['full_scale_amplification'] for r in prior6rows if all(r[k]==row[k] for k in ('trial','weights','world','orientation','driver')))
                    original=row['arms']['original'][0][0]['unobserved']['late'];removed=row['arms']['remove_all_null'][0][0]['unobserved']['late'];control=row['arms']['matched_full_null_shift'][0][0]['unobserved']['late']
                    row.update(benefit=removed-original,specificity=removed-control);rows.append(row)
                allrows.extend(rows);R.C.save(OUT/f'{trial}_{kind}.json',dict(cases=rows));O.verify(manifest);print('completed',trial,kind,flush=True)
        assert len(allrows)==128
        rng=np.random.default_rng(20260997);mi=rng.integers(0,8,size=(10000,8));wi=rng.integers(0,4,size=(10000,4));np.savez_compressed(OUT/'bootstrap_indices.npz',models=mi,worlds=wi)
        worlds=sorted(set(k[0] for k in keys));summaries={}
        for kind in ('final','initial'):
            matrices={name:np.array([[np.mean([r[name] for r in allrows if r['trial']==trial and r['weights']==kind and r['world']==world]) for world in worlds] for trial in R.TRIALS]) for name in ('benefit','specificity')}
            summaries[kind]=bootstrap(matrices['benefit'],matrices['specificity'],mi,wi)
        R.C.save(OUT/'results.json',dict(cases=allrows,summaries=summaries,worlds=worlds,max_prior_checkpoint_error=maxprior,primary='full-scale plus; final128; unobserved cells'))
        O.verify(manifest);artifacts={p.name:O.sha(p) for p in OUT.iterdir() if p.is_file()}
        R.C.save(OUT/'completion.json',dict(kind='OBS7_COMPLETE',manifest=manifest,artifacts=artifacts,prior_functional_verdict_changed=False));print('OBS7 complete',flush=True)
    except Exception as exc:R.C.save(OUT/'invalid.json',dict(error=repr(exc)));raise


if __name__=='__main__':main()
