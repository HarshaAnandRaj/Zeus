"""Independent raw-target, full manual replay, loss and bootstrap audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs7_factual_accuracy_20260909 as D
from tools.obs5_completion_audit_20260909 import manual


def close(a,b,atol=1e-10):np.testing.assert_allclose(a,b,rtol=1e-7,atol=atol)


def arrays(path):
    with np.load(path) as a:return {k:a[k] for k in a.files}


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    complete=D.R.N.read(D.OUT/'completion.json');D.O.verify(complete['manifest'])
    for n,digest in complete['artifacts'].items():assert D.O.sha(D.OUT/n)==digest
    data=D.R.N.read(D.OUT/'results.json');allrows=data['cases'];assert len(allrows)==128
    cache=D.R.N.load(D.Q.PRIOR/'replay_inputs.pt');truth=arrays(D.OUT/'targets.npz');keys=[tuple(map(int,k)) for k in truth['keys']]
    targets=[];masks=[];selected={}
    for pair in D.R.N.rows(D.R.OUT/'calibration_a.jsonl.gz'):
        if (pair['seed'],0) in keys:
            for oi,o in enumerate(pair['orientations']):selected[pair['seed'],oi]=o
    for key in keys:
        o=selected[key];trace=o['teacher']['trace'];assert len(trace)==496;previous=np.array(o['preparation']['boundary']['resources']);target=[];mask=[]
        for i,row in enumerate(trace):
            field=np.array(row['resources_before']);np.testing.assert_array_equal(field,previous);previous=np.array(row['resources_after'])
            assert row['tick']==17+i;position=row['position_before'];assert position==round(row['before'][4]*8)
            assert field[position]==row['before'][3];np.testing.assert_array_equal(D.R.M.features(row['before']),cache['streams'][key][16+i])
            target.append(field);mask.append(np.arange(9)!=position)
        np.testing.assert_array_equal(previous,o['teacher']['final_physical']['resources']);targets.append(target);masks.append(mask)
    targets=np.array(targets);masks=np.array(masks);np.testing.assert_array_equal(targets,truth['targets']);np.testing.assert_array_equal(masks,truth['masks'])
    checks={'source_and_artifact_identity':True,'all_eight_raw_target_streams_and_timing':True};count=0;maxstate=0.;maxloss=0.;maxprior=0.
    prior6=D.R.N.read(D.C.OUT/'results.json')['cases']
    for trial in D.R.TRIALS:
        for kind in ('final','initial'):
            d=arrays(D.OUT/f'{trial}_{kind}.npz');old=arrays(D.B.OUT/f'{trial}_{kind}.npz');rows=D.R.N.read(D.OUT/f'{trial}_{kind}.json')['cases'];p=cache['checkpoints'][trial][kind]
            for name in ('midpoints','deltas','drivers'):np.testing.assert_array_equal(d[name],old[name][::2])
            np.testing.assert_array_equal(d['arm_differences'][:,:6],old['arm_differences'][::2])
            starts=np.array([[m+sign*scale*arm/2 for scale in D.SCALES for arm in arms for sign in (1,-1)] for m,arms in zip(d['midpoints'],d['arm_differences'])])
            np.testing.assert_array_equal(starts,d['checkpoints'][:,:,:,:,0].reshape(8,28,32))
            replayed=manual(p,np.repeat(d['drivers'],28,axis=0),starts.reshape(-1,32)).reshape(8,28,496,32)
            h=np.concatenate([starts[:,:,None],replayed],axis=2).reshape(8,2,7,2,497,32)
            maxstate=max(maxstate,float(np.max(abs(h[:,:,:,:,D.TIMES]-d['checkpoints']))));close(h[:,:,:,:,D.TIMES],d['checkpoints']);close(h[:,:,:,:,369:],d['late_states'])
            maxprior=max(maxprior,float(np.max(abs(d['checkpoints'][:,:,:6]-old['checkpoints'][::2]))));close(d['checkpoints'][:,:,:6],old['checkpoints'][::2])
            pred=1/(1+np.exp(-(h[:,:,:,:,1:]@p['readout.weight'].T+p['readout.bias'])));close(pred,d['predictions'])
            error=pred-targets[:,None,None,None];square=error*error;unobserved=masks[:,None,None,None]
            losses=dict(unobserved=np.sum(np.where(unobserved,square,0),axis=-1)/8,
                        current=np.sum(np.where(unobserved,0,square),axis=-1),all_cells=np.sum(square,axis=-1)/9)
            for name,v in losses.items():maxloss=max(maxloss,float(np.max(abs(v-d[name+'_loss']))));close(v,d[name+'_loss'])
            _,_,vw=np.linalg.svd(p['readout.weight'],full_matrices=True);n=vw[9:].T
            for ci,row in enumerate(rows):
                assert (row['world'],row['orientation'])==keys[ci] and row['trial']==trial and row['weights']==kind and row['driver']=='teacher'
                assert row==next(r for r in allrows if all(r[k]==row[k] for k in ('trial','weights','world','orientation','driver')))
                dn=d['arm_differences'][ci,4];control=d['controls'][ci];size=np.linalg.norm(dn)
                close(np.linalg.norm(control),size);assert np.linalg.norm(p['readout.weight']@control)<1e-10;close(np.dot(control,dn),0.)
                if size>1e-12:
                    v=dn/size;project=lambda z:n@n.T@z-v*np.dot(v,z);raw=project(D.Q.DIRECTION)
                    if np.linalg.norm(raw)<=1e-12:raw=next(project(e) for e in np.eye(32) if np.linalg.norm(project(e))>1e-12)
                    raw/=np.linalg.norm(raw)
                    if raw[np.argmax(abs(raw))]<0:raw=-raw
                    close(control,size*raw)
                close(d['arm_differences'][ci,6],d['deltas'][ci]-control)
                immediate=h[ci,:,:,:,0]@p['readout.weight'].T+p['readout.bias'];close(immediate[:,6],immediate[:,0])
                for ai,arm in enumerate(D.ARMS):
                    for si in range(2):
                        for sign in range(2):
                            for name in losses:
                                v=d[name+'_loss'][ci,si,ai,sign];stored=row['arms'][arm][si][sign][name]
                                close(stored['late'],sum(v[368:])/128);close(stored['whole'],sum(v)/496);close(stored['final'],v[-1])
                primary=d['unobserved_loss'][ci,0,:,0,368:].mean(-1)
                close(row['benefit'],primary[3]-primary[0]);close(row['specificity'],primary[3]-primary[6])
                assert row['obs6_amplified']==next(r['prediction']['full_scale_amplification'] for r in prior6 if all(r[k]==row[k] for k in ('trial','weights','world','orientation','driver')))
                count+=1
        print('audited',trial,flush=True)
    assert count==128;close(maxprior,data['max_prior_checkpoint_error'])
    indices=arrays(D.OUT/'bootstrap_indices.npz');rng=np.random.default_rng(20260997);mi=rng.integers(0,8,size=(10000,8));wi=rng.integers(0,4,size=(10000,4))
    np.testing.assert_array_equal(mi,indices['models']);np.testing.assert_array_equal(wi,indices['worlds'])
    for kind in ('final','initial'):
        summary=data['summaries'][kind]
        for name in ('benefit','specificity'):
            matrix=np.array([[np.mean([r[name] for r in allrows if r['weights']==kind and r['trial']==trial and r['world']==world]) for world in data['worlds']] for trial in D.R.TRIALS])
            actual=summary[name];close(matrix,actual['matrix']);close(matrix.mean(),actual['mean']);close(matrix.mean(1),actual['model_means']);close(matrix.mean(0),actual['world_means'])
            # Explicit independent resampling loop, same fixed indices.
            boot=np.array([np.mean(matrix[np.ix_(models,worlds)]) for models,worlds in zip(mi,wi)])
            close(np.percentile(boot,[2.5,97.5]),actual['bounds'])
        benefit=summary['benefit'];specific=summary['specificity'];passed=bool(benefit['mean']>=1e-6 and benefit['bounds'][0]>0)
        assert summary['factual_benefit']==passed
        assert summary['specific_benefit']==bool(passed and specific['mean']>=1e-6 and specific['bounds'][0]>0)
    checks.update(all_3584_manual_branches_and_prior_agreement=True,all_prediction_loss_curves_and_summaries=True,
                  full_null_control_geometry_and_output_invariance=True,all_case_effects_and_strata=True,
                  fixed_two_axis_bootstrap_and_decisions=True)
    D.O.verify(complete['manifest']);checks['final_source_identity']=True
    result=dict(kind='OBS7_AUDIT',passed=all(checks.values()),checks=checks,combinations=count,branches=3584,
                max_state_error=maxstate,max_loss_error=maxloss,max_prior_checkpoint_error=maxprior,
                completion_sha=D.O.sha(D.OUT/'completion.json'),audit_sha=D.O.sha(Path(__file__)),manual_source_sha=D.O.sha(ROOT/'tools/obs5_completion_audit_20260909.py'))
    D.R.C.save(D.OUT/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
