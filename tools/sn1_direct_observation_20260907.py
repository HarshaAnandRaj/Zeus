"""Direct fixed-history readout observations; no training, sampling or deployment."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.model import ZeusCore
from tools import cyc4_learned_carryover_20260907 as M
C=M.C
OUT=ROOT/'runs/sn1_20260907'
PROMPTS=['hello','tell me about the past','once upon a time','what happened','the river flowed quietly']
MODES=('coupled','zero_s','zero_h','zero_both','self_source','reverse_tokens')


def js(a,b):
    p,q=F.softmax(a.double(),-1),F.softmax(b.double(),-1);mid=(p+q)/2
    return float(.5*(p*(p.clamp_min(1e-300).log()-mid.clamp_min(1e-300).log())).sum()+
                 .5*(q*(q.clamp_min(1e-300).log()-mid.clamp_min(1e-300).log())).sum())


@torch.no_grad()
def observe(model,snapshot,mode):
    untouched=copy.deepcopy(snapshot)
    model.restore_runtime(snapshot);model.deploy_self_source=False
    if mode in ('zero_s','zero_both'):model.S.zero_()
    if mode in ('zero_h','zero_both'):model.H.zero_()
    if mode=='self_source':model.deploy_self_source=True
    if mode=='reverse_tokens':
        model.E_hist.copy_(model.E_hist.flip(0).clone());model.last_e=model.E_hist[-1].detach().clone()
    assert mode in MODES
    if mode!='reverse_tokens':
        assert torch.equal(model.E_hist,snapshot['E_hist']) and torch.equal(model.last_e,snapshot['last_e'])
    if mode not in ('zero_s','zero_both'):assert torch.equal(model.S,snapshot['S'])
    if mode not in ('zero_h','zero_both'):assert torch.equal(model.H,snapshot['H'])
    logits=model.observe().detach().cpu().clone()
    assert M.exact(snapshot,untouched),'probe mutated its source snapshot'
    model.restore_runtime(snapshot);assert M.exact(model.snapshot_runtime(),snapshot)
    return logits


def main():
    M.configure();OUT.mkdir(exist_ok=False);config_path=ROOT/'zeus_sandbox/config.json';config=json.loads(config_path.read_text())
    checkpoint=ROOT/config['ckpt']
    paths=[config_path,checkpoint,ROOT/'core/model.py',Path(__file__),ROOT/'docs/sn1_direct_observation_protocol_20260907.md']
    manifest=dict(sources={str(p.relative_to(ROOT)).replace('\\','/'):C.sha(p) for p in paths},
        config=dict(checkpoint=config['ckpt'],voice_self_source=config.get('voice_self_source',False),skip_pad_window=config.get('skip_pad_window',False)),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),torch=torch.__version__)
    C.save(OUT/'manifest.json',manifest)
    try:
        model=ZeusCore.load(str(checkpoint),'cpu').eval();rows=[];artifacts=[];pairs=[]
        for prompt in PROMPTS:
            group=[]
            for seed in (7,8):
                model.deploy_self_source=False;model.reset_state(.12,torch.Generator('cpu').manual_seed(seed));ids=model.encode(prompt)
                if not config.get('skip_pad_window',False) and ids:model.pad_window(ids[0])
                model.ingest(ids);snapshot=model.snapshot_runtime()
                logits={mode:observe(model,snapshot,mode) for mode in MODES}
                assert torch.equal(logits['coupled'],observe(model,snapshot,'coupled'))
                rows.append(dict(prompt=prompt,seed=seed,js={mode:js(logits['coupled'],logits[mode]) for mode in MODES if mode!='coupled'},
                    argmax={mode:int(value.argmax()) for mode,value in logits.items()}))
                saved=dict(prompt=prompt,seed=seed,snapshot=snapshot,logits=logits);artifacts.append(saved);group.append(saved)
            a,b=group
            assert torch.equal(a['snapshot']['E_hist'],b['snapshot']['E_hist']) and torch.equal(a['snapshot']['last_e'],b['snapshot']['last_e'])
            same_source=torch.equal(a['logits']['self_source'],b['logits']['self_source'])
            pairs.append(dict(prompt=prompt,initialization_js=js(a['logits']['coupled'],b['logits']['coupled']),
                initialization_argmax_differs=int(a['logits']['coupled'].argmax())!=int(b['logits']['coupled'].argmax()),
                self_source_exact=same_source,state_distance=float((a['snapshot']['S']-b['snapshot']['S']).norm())))
            assert same_source,'self-source depends on supposedly excluded initialization'
        full=torch.stack([a['logits']['coupled'] for a in artifacts]).double().flatten()
        zero=torch.stack([a['logits']['zero_both'] for a in artifacts]).double().flatten();residual=full-zero
        vf=float(full.var(unbiased=False));vz=float(zero.var(unbiased=False));vr=float(residual.var(unbiased=False))
        cov=float(((zero-zero.mean())*(residual-residual.mean())).mean());identity=vf-(vz+vr+2*cov)
        assert abs(identity)<1e-10*max(1.,vf)
        torch.save(artifacts,OUT/'observations.pt');C.verify(manifest)
        report=dict(kind='SN1_DIRECT_OBSERVATION',diagnostic_only=True,manifest=manifest,
            checkpoint_step=torch.load(checkpoint,map_location='cpu',weights_only=True).get('step'),
            configured_gains=dict(s_scale=model.readout.s_scale,gate_gain=model.readout.gate_gain,ctx_gain=model.readout.ctx_gain),
            rows=rows,initialization_pairs=pairs,summary=dict(mean_js={mode:sum(r['js'][mode] for r in rows)/len(rows) for mode in MODES if mode!='coupled'},
                mean_initialization_js=sum(p['initialization_js'] for p in pairs)/len(pairs),
                initialization_argmax_disagreements=sum(p['initialization_argmax_differs'] for p in pairs),pairs=len(pairs),
                all_self_source_exact=all(p['self_source_exact'] for p in pairs)),
            variance=dict(full=vf,zero_both=vz,residual=vr,covariance=cov,residual_to_full=vr/vf,zero_to_full=vz/vf,
                covariance_term_to_full=2*cov/vf,identity_residual=identity),
            observation_sha=C.sha(OUT/'observations.pt'))
        C.save(OUT/'report.json',report);print(json.dumps(report),flush=True)
    except Exception as exc:
        C.save(OUT/'invalid.json',dict(status='INVALID_STOP',error=repr(exc)));raise


if __name__=='__main__':main()
