"""SN2: matched spectral buffers and token windows; diagnostic only."""
import json
from pathlib import Path
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import sn1_direct_observation_20260907 as P
OUT=ROOT/'runs/sn2_20260907'


def prepare(model,prompt,seed,skip_pad,initial_buffers):
    with torch.no_grad():
        for name,buffer in model.named_buffers():
            buffer.copy_(initial_buffers[name])
            assert torch.equal(buffer,initial_buffers[name])
    model.deploy_self_source=False;model.reset_state(.12,torch.Generator('cpu').manual_seed(seed))
    model.E_hist.zero_()
    ids=model.encode(prompt)
    if not skip_pad and ids:model.pad_window(ids[0])
    model.ingest(ids)
    return model.snapshot_runtime()


def main():
    P.M.configure();OUT.mkdir(exist_ok=False);config_path=ROOT/'zeus_sandbox/config.json';config=json.loads(config_path.read_text())
    checkpoint=ROOT/config['ckpt']
    paths=[config_path,checkpoint,ROOT/'core/model.py',Path(__file__),ROOT/'tools/sn1_direct_observation_20260907.py',
        ROOT/'docs/sn1_direct_observation_protocol_20260907.md',ROOT/'docs/sn1r_direct_observation_protocol_20260907.md',ROOT/'runs/sn1_20260907/invalid.json', ROOT/'runs/sn1r_20260907/report.json',
        ROOT/'tools/sn1r_direct_observation_20260907.py',ROOT/'docs/sn2_direct_observation_protocol_20260907.md']
    manifest=dict(sources={str(p.relative_to(ROOT)).replace('\\','/'):P.C.sha(p) for p in paths},
        config=dict(checkpoint=config['ckpt'],voice_self_source=config.get('voice_self_source',False),skip_pad_window=config.get('skip_pad_window',False)),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),torch=torch.__version__,construction_seed=20260907)
    P.C.save(OUT/'manifest.json',manifest)
    try:
        torch.manual_seed(20260907)
        model=P.ZeusCore.load(str(checkpoint),'cpu').eval();rows=[];artifacts=[];pairs=[]
        initial_buffers={name:buffer.detach().clone() for name,buffer in model.named_buffers()}
        torch.save(initial_buffers,OUT/'initial_buffers.pt')
        for prompt in P.PROMPTS:
            group=[]
            for seed in (7,8):
                snapshot=prepare(model,prompt,seed,config.get('skip_pad_window',False),initial_buffers)
                logits={mode:P.observe(model,snapshot,mode) for mode in P.MODES}
                assert torch.equal(logits['coupled'],P.observe(model,snapshot,'coupled'))
                rows.append(dict(prompt=prompt,seed=seed,js={mode:P.js(logits['coupled'],logits[mode]) for mode in P.MODES if mode!='coupled'},
                    argmax={mode:int(value.argmax()) for mode,value in logits.items()}))
                saved=dict(prompt=prompt,seed=seed,snapshot=snapshot,logits=logits);artifacts.append(saved);group.append(saved)
            a,b=group
            assert all(P.M.exact(a['snapshot'][k],b['snapshot'][k]) for k in ('E_hist','last_e','anchor_vec'))
            same_source=torch.equal(a['logits']['self_source'],b['logits']['self_source'])
            pairs.append(dict(prompt=prompt,initialization_js=P.js(a['logits']['coupled'],b['logits']['coupled']),
                initialization_argmax_differs=int(a['logits']['coupled'].argmax())!=int(b['logits']['coupled'].argmax()),
                self_source_exact=same_source,state_distance=float((a['snapshot']['S']-b['snapshot']['S']).norm())))
            assert same_source
        full=torch.stack([a['logits']['coupled'] for a in artifacts]).double().flatten()
        zero=torch.stack([a['logits']['zero_both'] for a in artifacts]).double().flatten();residual=full-zero
        vf=float(full.var(unbiased=False));vz=float(zero.var(unbiased=False));vr=float(residual.var(unbiased=False))
        cov=float(((zero-zero.mean())*(residual-residual.mean())).mean());identity=vf-(vz+vr+2*cov)
        assert abs(identity)<1e-10*max(1.,vf)
        torch.save(artifacts,OUT/'observations.pt');P.C.verify(manifest)
        report=dict(kind='SN2_DIRECT_OBSERVATION',diagnostic_only=True,prior_attempt='SN1R_UNMATCHED_SPECTRAL_BUFFERS',manifest=manifest,
            checkpoint_step=torch.load(checkpoint,map_location='cpu',weights_only=True).get('step'),
            configured_gains=dict(s_scale=model.readout.s_scale,gate_gain=model.readout.gate_gain,ctx_gain=model.readout.ctx_gain),
            rows=rows,initialization_pairs=pairs,summary=dict(mean_js={mode:sum(r['js'][mode] for r in rows)/len(rows) for mode in P.MODES if mode!='coupled'},
                mean_initialization_js=sum(p['initialization_js'] for p in pairs)/len(pairs),
                initialization_argmax_disagreements=sum(p['initialization_argmax_differs'] for p in pairs),pairs=len(pairs),
                all_self_source_exact=all(p['self_source_exact'] for p in pairs)),
            variance=dict(full=vf,zero_both=vz,residual=vr,covariance=cov,residual_to_full=vr/vf,zero_to_full=vz/vf,
                covariance_term_to_full=2*cov/vf,identity_residual=identity),initial_buffers_sha=P.C.sha(OUT/'initial_buffers.pt'),observation_sha=P.C.sha(OUT/'observations.pt'))
        P.C.save(OUT/'report.json',report);print(json.dumps(report),flush=True)
    except Exception as exc:
        P.C.save(OUT/'invalid.json',dict(status='INVALID_STOP',error=repr(exc)));raise


if __name__=='__main__':main()
