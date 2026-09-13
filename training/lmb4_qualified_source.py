"""Bind the complete audited motor source before any fresh follow-up preparation."""
import torch
from training import run_lmb4 as R

AUDIT=R.ROOT/'zeus_sandbox/universe/reports/lmb4_audit_20260913.json'

def qualified_source():
    manifest=R.verify();receipt=R.L.P.L3.read(AUDIT);report=R.L.P.L3.read(R.REPORT)
    assert receipt['status']=='PASS' and receipt['verdict']=='PASS' and receipt['qualification'] is True,'no independently qualified LMB4 source'
    assert report['verdict']=='PASS' and receipt['report_sha']==R.L.P.L3.sha(R.REPORT)
    verdict=R.L.P.L3.read(R.OUT/'verdict.json')
    assert report==dict(**verdict,manifest=manifest,independent_audit_required=True,scope='Public grounding motor prerequisite only')
    assert receipt['grounding_attribution']==verdict['grounding_attribution']
    arm=verdict['selected_qualified_arm'];assert arm==next(a for a in R.K.CONFIG['arms'] if verdict['arms'][a]['verdict']=='PASS')
    required={'manifest.json','training_a.json','training_b.json','own_public_a.json','own_public_b.json',
        'calibration_public.json','calibration.json','endpoint_public.json','regression_public.json','reference.json',
        'reference_trace.jsonl.gz','grounded_endpoint.json','detached_endpoint.json','grounded_trace.jsonl.gz','detached_trace.jsonl.gz','verdict.json'}
    assert set(receipt['evidence_sha'])==required
    for name,digest in receipt['evidence_sha'].items():assert R.L.P.L3.sha(R.OUT/name)==digest,'audited endpoint artifact changed: '+name
    pairs={(p['trial'],p['arm']):p['logical_hash'] for p in receipt['fit_pairs']}
    assert len(receipt['fit_pairs'])==len(pairs)==8 and set(pairs)=={(t,a) for t in range(4) for a in R.K.CONFIG['arms']}
    models={};heads={}
    for t,a in sorted(pairs):
        payload,complete=R.L.P.L3.checked_checkpoint(R.OUT/f'{t}_{a}_a')
        other,twin=R.L.P.L3.checked_checkpoint(R.OUT/f'{t}_{a}_b')
        assert complete['logical_hash']==twin['logical_hash']==pairs[t,a]
        assert payload['initial_hash']==manifest['parents'][str(t)] and payload['initial_head_hash']==manifest['heads'][str(t)]
        assert R.L.P.L3.tree_hash(payload['model'])==receipt['candidate_hashes'][a][str(t)]
        assert R.L.P.L3.tree_hash(payload['heads'])==receipt['head_hashes'][a][str(t)]
        if a==arm:models[str(t)]=receipt['candidate_hashes'][a][str(t)];heads[str(t)]=receipt['head_hashes'][a][str(t)]
    phases={(p['trial'],p['arm'],p['refresh']):p for p in receipt['collection_phases']}
    assert len(receipt['collection_phases'])==len(phases)==64
    assert set(phases)=={(t,a,r) for t in range(4) for a in R.K.CONFIG['arms'] for r in range(8)}
    for (t,a,r),phase in sorted(phases.items()):
        dirs=[R.OUT/f'{t}_{a}_{v}' for v in R.K.CONFIG['twins']]
        sources=[d/f'source_{r}.pt' for d in dirs];collections=[d/f'collection_{r}.jsonl.gz' for d in dirs]
        assert R.L.P.L3.sha(sources[0])==phase['source_sha']
        assert R.L.P.L3.tree_hash(torch.load(sources[0],weights_only=False))==R.L.P.L3.tree_hash(torch.load(sources[1],weights_only=False))
        assert all(R.L.P.L3.sha(p)==phase['collection_sha'] for p in collections)
    assert receipt['training_batches_verified']==768 and receipt['source_bodies']==512 and receipt['endpoint_bodies']==4096
    assert receipt['hardware_trajectory_exact'] is True and receipt['pillar_promotion'] is False
    assert 0<=receipt['max_local_probability_error']<2e-5 and 0<=receipt['max_local_state_error']<1e-4
    assert 0<=receipt['max_auxiliary_head_error']<1e-5
    return dict(arm=arm,models=models,heads=heads,audit_sha=R.L.P.L3.sha(AUDIT),report_sha=receipt['report_sha'],manifest_sha=receipt['evidence_sha']['manifest.json'])
