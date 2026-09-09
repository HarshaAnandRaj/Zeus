"""Independent raw-observation estimator reconstruction and score audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs8_target_mismatch_20260909 as E
D,R,O=E.D,E.R,E.O


def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)


def arrays(path):
    with np.load(path) as a:return {k:a[k] for k in a.files}


def main():
    complete=R.N.read(E.OUT/'completion.json');O.verify(complete['manifest'])
    assert complete['kind']=='OBS8_COMPLETE' and not complete['obs7_factual_verdict_changed']
    assert complete['manifest']['new_model_forward_passes']==0
    for name,digest in complete['artifacts'].items():assert O.sha(E.OUT/name)==digest
    result=R.N.read(E.OUT/'results.json');prior=R.N.read(D.OUT/'results.json');assert len(result['cases'])==128
    assert not result['obs7_factual_verdict_changed'];assert result['worlds']==prior['worlds']
    saved=arrays(E.OUT/'proxy_targets.npz');data=arrays(D.OUT/'targets.npz');keys=[tuple(map(int,k)) for k in data['keys']]
    np.testing.assert_array_equal(saved['keys'],data['keys']);raw={}
    for pair in R.N.rows(R.OUT/'calibration_a.jsonl.gz'):
        if (pair['seed'],0) in keys:
            for oi,o in enumerate(pair['orientations']):raw[pair['seed'],oi]=o
    reconstructed=[]
    for key in keys:
        o=raw[key];p=o['preparation'];trace=o['teacher']['trace']
        observations=[list(R.C.restore(p['initial']).observation())]+[r['after'] for r in p['exposure'][:15]]+[r['before'] for r in trace]
        assert len(observations)==512
        last_tick=[None]*9;last_value=[None]*9;targets=[]
        for tick,obs in enumerate(observations):
            cell=round(obs[4]*8);last_tick[cell]=tick;last_value[cell]=obs[3]
            targets.append([.40 if last_tick[j] is None else max(0.,min(1.,.575+(last_value[j]-.575)*.992**(tick-last_tick[j]))) for j in range(9)])
        x,y,_,valid=R.M.data_rows(p,o['teacher']);assert valid==513
        close(targets,np.array(y)[:512]);close(targets[16:],saved['targets'][len(reconstructed)])
        reconstructed.append(targets[16:])
    proxy=np.array(reconstructed);truth=data['targets'];mask=data['masks'];m=mask[:,None,None,None];t=proxy[:,None,None,None];w=truth[:,None,None,None]
    checked=[];maxloss=0.;maxidentity=0.;maxgap=0.
    checks=dict(source_and_artifact_identity=True,all_raw_estimator_targets_and_helper_agreement=True)
    for trial in R.TRIALS:
        for kind in ('final','initial'):
            old=arrays(D.OUT/f'{trial}_{kind}.npz');saved_group=arrays(E.OUT/f'{trial}_{kind}.npz');rows=R.N.read(E.OUT/f'{trial}_{kind}.json')['cases'];pred=old['predictions']
            for ci,key in enumerate(keys):
                np.testing.assert_array_equal(old['drivers'][ci],np.array([R.M.features(r['before']) for r in raw[key]['teacher']['trace']]))
            average=lambda a:np.where(m,a,0.).sum(-1)/8
            values=dict(proxy=average((pred-t)**2),target_bias=np.where(mask,(proxy-truth)**2,0.).sum(-1)/8,
                        cross=average(2*(pred-t)*(t-w)),factual=average((pred-w)**2))
            for name,v in values.items():
                close(v,saved_group[name]);maxloss=max(maxloss,float(np.max(abs(v-saved_group[name]))))
            close(values['factual'],old['unobserved_loss'])
            close(np.where(m,0.,(pred-t)**2).sum(-1),saved_group['proxy_current']);close(((pred-t)**2).mean(-1),saved_group['proxy_all_cells'])
            identity=values['factual']-values['proxy']-values['target_bias'][:,None,None,None]-values['cross'];close(identity,0.)
            maxidentity=max(maxidentity,float(np.max(abs(identity))))
            for ci,row in enumerate(rows):
                assert (row['world'],row['orientation'])==keys[ci] and row['trial']==trial and row['weights']==kind and row['driver']=='teacher'
                assert row==next(r for r in result['cases'] if all(r[k]==row[k] for k in ('trial','weights','world','orientation','driver')))
                for ai,arm in enumerate(D.ARMS):
                    for si in range(2):
                        for sign in range(2):
                            for window,sl in [('late',slice(368,None)),('whole',slice(None))]:
                                for name,v in values.items():close(row['arms'][arm][si][sign][window][name],np.mean(v[ci,sl] if name=='target_bias' else v[ci,si,ai,sign,sl]))
                primary=values['proxy'][ci,0,:,0,368:].mean(-1);factual=values['factual'][ci,0,:,0,368:].mean(-1)
                for name,ai in [('benefit',0),('specificity',6)]:
                    effect=float(primary[3]-primary[ai]);fact=float(factual[3]-factual[ai]);close(row[name],effect)
                    term=float(np.where(mask[ci,368:],2*(pred[ci,0,3,0,368:]-pred[ci,0,ai,0,368:])*(truth[ci,368:]-proxy[ci,368:]),0.).sum(-1).mean()/8)
                    stored=row['target_gap_terms'][name];close(stored['proxy_effect'],effect);close(stored['factual_effect'],fact);close(stored['target_disagreement_term'],term)
                    close(effect-fact,term);maxgap=max(maxgap,abs(effect-fact-term))
                checked.append(row)
        print('audited',trial,flush=True)
    assert len(checked)==128;indices=arrays(D.OUT/'bootstrap_indices.npz')
    for kind in ('final','initial'):
        summary=result['summaries'][kind]
        for name in ('benefit','specificity'):
            matrix=np.array([[np.mean([r[name] for r in checked if r['weights']==kind and r['trial']==trial and r['world']==world]) for world in result['worlds']] for trial in R.TRIALS])
            s=summary[name];close(matrix,s['matrix']);close(matrix.mean(),s['mean']);close(matrix.mean(0),s['world_means']);close(matrix.mean(1),s['model_means'])
            samples=[matrix[np.ix_(mi,wi)].mean() for mi,wi in zip(indices['models'],indices['worlds'])];close(np.percentile(samples,[2.5,97.5]),s['bounds'])
        b=summary['benefit'];s=summary['specificity'];f=prior['summaries'][kind]['benefit']
        passed=bool(b['mean']>=1e-6 and b['bounds'][0]>0)
        assert summary['proxy_benefit']==passed
        assert summary['proxy_specific_benefit']==bool(passed and s['mean']>=1e-6 and s['bounds'][0]>0)
        assert summary['target_tradeoff']==bool(passed and f['mean']<=-1e-6 and f['bounds'][1]<0)
    close(maxidentity,result['max_loss_identity_error']);close(maxgap,result['max_target_gap_identity_error'])
    checks.update(all_3584_cached_branch_scores_and_decompositions=True,all_case_summaries_and_signed_target_gap_identities=True,
                  independent_fixed_bootstrap_and_decisions=True,unchanged_obs7_factual_verdict=True)
    O.verify(complete['manifest']);checks['final_source_identity']=True
    audit=dict(kind='OBS8_AUDIT',passed=all(checks.values()),checks=checks,combinations=len(checked),branches=3584,
               max_loss_error=maxloss,max_loss_identity_error=maxidentity,max_target_gap_identity_error=maxgap,
               completion_sha=O.sha(E.OUT/'completion.json'),audit_sha=O.sha(Path(__file__)))
    R.C.save(E.OUT/'completion_audit.json',audit);print(audit,flush=True)


if __name__=='__main__':main()
