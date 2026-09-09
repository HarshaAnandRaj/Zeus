"""OBS8 rescores frozen predictions against intended teaching targets."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import obs7_factual_accuracy_20260909 as D
R,O,Q,B=D.R,D.O,D.Q,D.B
OUT=ROOT/'runs/obs8_20260909'


def arrays(path):
    with np.load(path) as a:return {k:a[k] for k in a.files}


def teacher_values(preparation,teacher,driver):
    x,y,_,valid=R.M.data_rows(preparation,teacher);assert valid==513
    np.testing.assert_array_equal(np.array(x)[16:512],driver)
    return np.array(y,dtype=np.float64)[16:512]


def score(square,mask):return (square*mask).sum(-1)/8


def decompose(pred,proxy,truth,mask):
    t=proxy[:,None,None,None];w=truth[:,None,None,None];m=mask[:,None,None,None]
    fidelity=score((pred-t)**2,m);bias=score((proxy-truth)**2,mask);cross=score(2*(pred-t)*(t-w),m);factual=score((pred-w)**2,m)
    np.testing.assert_allclose(factual,fidelity+bias[:,None,None,None]+cross,rtol=0,atol=1e-12)
    return dict(proxy=fidelity,target_bias=bias,cross=cross,factual=factual)


def summarize(benefit,specificity,mi,wi,factual):
    result=D.bootstrap(benefit,specificity,mi,wi)
    result['proxy_benefit']=result.pop('factual_benefit');result['proxy_specific_benefit']=result.pop('specific_benefit')
    result['target_tradeoff']=bool(result['proxy_benefit'] and factual['mean']<=-1e-6 and factual['bounds'][1]<0)
    return result


def main():
    OUT.mkdir(exist_ok=False);complete=D.R.N.read(D.OUT/'completion.json');audit=D.R.N.read(D.OUT/'completion_audit.json')
    assert audit['passed'] and audit['completion_sha']==O.sha(D.OUT/'completion.json');O.verify(complete['manifest'])
    sources={}
    for name,digest in complete['artifacts'].items():
        path=D.OUT/name;assert O.sha(path)==digest;sources[str(path.relative_to(ROOT)).replace('\\','/')]=digest
    files=[D.OUT/'completion.json',D.OUT/'completion_audit.json',R.OUT/'calibration_a.jsonl.gz',Path(__file__),
           ROOT/'docs/obs8_target_mismatch_protocol_20260909.md',ROOT/'tools/test_obs8_target_mismatch_20260909.py',
           ROOT/'tools/cyc4_learned_carryover_20260907.py',ROOT/'tools/cyc3_carryover_calibration_20260907.py',ROOT/'tools/obs7_factual_accuracy_20260909.py']
    for p in files:sources[str(p.relative_to(ROOT)).replace('\\','/')]=O.sha(p)
    manifest=dict(sources=sources,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),numpy=np.__version__,exploratory=True,new_model_forward_passes=0)
    R.C.save(OUT/'manifest.json',manifest)
    try:
        oldresults=R.N.read(D.OUT/'results.json');data=arrays(D.OUT/'targets.npz');truth=data['targets'];mask=data['masks'];keys=[tuple(map(int,k)) for k in data['keys']]
        with np.load(D.OUT/'seed_0_final.npz') as a:drivers=a['drivers']
        raw={}
        for pair in R.N.rows(R.OUT/'calibration_a.jsonl.gz'):
            if (pair['seed'],0) in keys:
                for oi,o in enumerate(pair['orientations']):raw[pair['seed'],oi]=o
        proxy=np.array([teacher_values(raw[k]['preparation'],raw[k]['teacher'],drivers[i]) for i,k in enumerate(keys)])
        np.savez_compressed(OUT/'proxy_targets.npz',targets=proxy,keys=np.array(keys))
        allrows=[];maxidentity=0.;maxgapidentity=0.
        for trial in R.TRIALS:
            for kind in ('final','initial'):
                old=arrays(D.OUT/f'{trial}_{kind}.npz');oldrows=R.N.read(D.OUT/f'{trial}_{kind}.json')['cases'];pred=old['predictions']
                np.testing.assert_array_equal(old['drivers'],drivers)
                parts=decompose(pred,proxy,truth,mask);np.testing.assert_allclose(parts['factual'],old['unobserved_loss'],atol=1e-12,rtol=0)
                maxidentity=max(maxidentity,float(np.max(abs(parts['factual']-parts['proxy']-parts['target_bias'][:,None,None,None]-parts['cross']))))
                scores=D.loss_curves(pred,proxy,mask)
                np.savez_compressed(OUT/f'{trial}_{kind}.npz',**parts,proxy_current=scores['current'],proxy_all_cells=scores['all_cells'])
                rows=[]
                for ci,prior in enumerate(oldrows):
                    row={k:prior[k] for k in ('trial','weights','world','orientation','driver')};row['arms']={}
                    for ai,arm in enumerate(D.ARMS):
                        row['arms'][arm]=[[{window:{name:float((v[ci,selection] if name=='target_bias' else v[ci,si,ai,sign,selection]).mean()) for name,v in parts.items()}
                            for window,selection in [('late',slice(368,None)),('whole',slice(None))]} for sign in range(2)] for si in range(2)]
                    losses=parts['proxy'][ci,0,:,0,368:].mean(-1);benefit=float(losses[3]-losses[0]);specificity=float(losses[3]-losses[6])
                    shifts={}
                    for name,comparison,reference in [('benefit',0,prior['benefit']),('specificity',6,prior['specificity'])]:
                        error_change=pred[ci,0,3,0,368:]-pred[ci,0,comparison,0,368:]
                        target_term=float(np.mean(score(2*error_change*(truth[ci,368:]-proxy[ci,368:]),mask[ci,368:])))
                        proxy_effect=benefit if name=='benefit' else specificity;error=abs(proxy_effect-reference-target_term);assert error<=1e-12;maxgapidentity=max(maxgapidentity,error)
                        shifts[name]=dict(proxy_effect=proxy_effect,factual_effect=reference,target_disagreement_term=target_term)
                    row.update(benefit=benefit,specificity=specificity,target_gap_terms=shifts);rows.append(row)
                allrows.extend(rows);R.C.save(OUT/f'{trial}_{kind}.json',dict(cases=rows));O.verify(manifest);print('scored',trial,kind,flush=True)
        assert len(allrows)==128;indices=arrays(D.OUT/'bootstrap_indices.npz');worlds=oldresults['worlds'];summaries={}
        for kind in ('final','initial'):
            matrices={name:np.array([[np.mean([r[name] for r in allrows if r['weights']==kind and r['trial']==trial and r['world']==world]) for world in worlds] for trial in R.TRIALS]) for name in ('benefit','specificity')}
            summaries[kind]=summarize(matrices['benefit'],matrices['specificity'],indices['models'],indices['worlds'],oldresults['summaries'][kind]['benefit'])
        R.C.save(OUT/'results.json',dict(cases=allrows,summaries=summaries,worlds=worlds,max_loss_identity_error=maxidentity,max_target_gap_identity_error=maxgapidentity,obs7_factual_verdict_changed=False))
        O.verify(manifest);artifacts={p.name:O.sha(p) for p in OUT.iterdir() if p.is_file()}
        R.C.save(OUT/'completion.json',dict(kind='OBS8_COMPLETE',manifest=manifest,artifacts=artifacts,obs7_factual_verdict_changed=False));print('OBS8 complete',flush=True)
    except Exception as exc:R.C.save(OUT/'invalid.json',dict(error=repr(exc)));raise


if __name__=='__main__':main()
