"""Post-run evidence audit; no experiment, selection change, or model rollout."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.qualify_tagged_recall import source_checks
from training.utility_memory import canonical_hash,select_entries
from tools.capability_components_20260906 import check as cap1_check,calibration,dimension
from tools.cap1r_dimension_20260906 import identities as cap1r_check

REPORTS=ROOT/'zeus_sandbox/universe/reports'


def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    if isinstance(value,dict):
        return all(finite(v) for v in value.values())
    if isinstance(value,list):
        return all(finite(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)


def main():
    target=REPORTS/'retention_completion_audit_20260907.json'
    if target.exists():
        raise RuntimeError('refuse audit overwrite')
    torch.set_num_threads(1)
    checks={}
    manifest=read(ROOT/'docs/registrations/sel1_manifest.json')
    checks['SEL1_all_14_registered_identities_and_committed_files']=len(source_checks(manifest))==14
    base=ROOT/'runs/sel1_20260906'
    gens=[read(base/t/'generation.json') for t in ['twin_a','twin_b']]
    evs=[read(base/t/'evaluation.json') for t in ['twin_a','twin_b']]
    payloads=[torch.load(base/t/'inheritance.pt',map_location='cpu',weights_only=False) for t in ['twin_a','twin_b']]
    report=read(REPORTS/'sel1_utility_inheritance_20260906.json')
    checks['SEL1_full_generation_and_evaluation_twins']=gens[0]==gens[1] and evs[0]==evs[1]
    checks['SEL1_canonical_payload_twins']=canonical_hash(payloads[0])==canonical_hash(payloads[1])==gens[0]['payload_hash']==evs[0]['payload_hash']
    checks['SEL1_report_matches_twins']=report['generation']==gens[0] and report['evaluation']==evs[0] and report['exact_twins']
    selected,estimates=select_entries(gens[0]['selection_rows'],gens[0]['candidate_count'])
    checks['SEL1_selection_recomputed_exactly']=selected==gens[0]['selected_ids'] and estimates==gens[0]['estimates']
    checks['SEL1_registered_selection_failure']=report['verdict']=='FAIL' and len(selected)==2 and gens[0]['candidate_count']==123 and len(selected)<8
    checks['SEL1_no_calibration_or_endpoint_exposure']=all(e['endpoint_examples_exposed']==0 and e['stage']=='selection' and e['reload_verified'] for e in evs) and all(g['endpoint_examples_exposed']==0 for g in gens) and all(not (base/t/'calibration.json').exists() for t in ['twin_a','twin_b'])
    checks['SEL1_full_draw_rows']=len(gens[0]['proposals'])==128 and len(gens[0]['selection_rows'])==256*32
    checks['SEL1_finite_report']=finite(report)
    tokenizer=Tokenizer.from_file(str(ROOT/'corpus/data/tokenizer/bpe_8192.json'))
    entries=[{'memory_id':p['memory_id'],'context':tokenizer.decode(p['context']),
              'target':tokenizer.decode([p['target']]),'provenance':p['provenance'],
              'action_origin':p['action_origin'],'selection_estimate':estimates[p['memory_id']]}
             for p in gens[0]['proposals'] if p['memory_id'] in selected]
    checks['SEL1_read_all_retained_entries']=len(entries)==2 and all(e['provenance']=='observed_token' and not e['action_origin'] for e in entries)
    qpath=REPORTS/'qv1_postmortem_20260906.json.gz'
    q=read(qpath)
    checks['QV1_archive_equals_raw']=gzip.decompress(qpath.read_bytes())==(REPORTS/'qv1_postmortem_20260906.json').read_bytes()
    checks['QV1_twins_and_original_greedy_replay']=q['exact_campaign_twins'] and q['greedy_original_replay_verified']
    checks['QV1_all_768_traces_complete']=sum(len(c['episodes']) for a in q['D4'].values() for c in a.values())==768 and all(len(e['trace'])==e['age'] and sum(e['selected_actions'].values())==e['age'] for a in q['D4'].values() for c in a.values() for e in c['episodes'])
    checks['QV1_all_KM_death_events']=len(q['D2'])==7 and all(sum(r['deaths'] for r in rows['table'])==64 and sum(r['censored'] for r in rows['table'])==0 for rows in q['D2'].values())
    checks['QV1_all_energy_ledgers']=all(abs(e['initial_energy']+e['harvested_energy']-e['basal_cost']-e['action_cost']-e['empty_harvest_penalty']+e['clipping_adjustment']-e['final_energy'])<1e-12 for e in q['D5'])
    checks['QV1_full_initial_join']=len(q['D3']['training_initials'])==1280 and len(q['D3']['heldout_initials'])==64 and q['D3']['seed_overlap']==0 and q['D3']['full_initial_overlap']==0
    checks['CAP1_manifest_current']=bool(cap1_check()) and bool(cap1r_check())
    checks['CAP1_calibration']=all(calibration().values())
    cap=read(REPORTS/'capability_completion_20260906.json')
    checks['CAP1_all_capture_hashes']=all(sha(ROOT/p)==h for p,h in cap['capture_hashes'].items())
    da,db=[read(ROOT/'runs/cap1r_20260906'/f'{t}.json.gz') for t in ['twin_a','twin_b']]
    checks['CAP1R_exact_full_state_twins']=da==db and read(ROOT/'runs/cap1r_20260906/identity.json')['exact_twins']
    checks['CAP1R_all_64_windows_and_dimensions']=sum(len(rows)*2 for rows in da['conditions'].values())==64 and all(len(e[b]['states'])==64 and dimension(e[b]['states'])==cap['DYN1_substrate_fresh'][arm][i][b] for arm,rows in da['conditions'].items() for i,e in enumerate(rows) for b in ['control','perturbed'])
    checks['CAP1_finite_components_and_fidelity']=finite(cap) and all(cap['capture_fidelity'].values())
    checks['CAP1_fixed_reflex_separation']=all(c['repertoire']['ci95'][0]>0 for c in cap['POL2_paired_comparisons'].values())
    checks['CAP1_preserves_historical_VOID_and_no_authority']=cap['original_CAP1_DYN1_status'].startswith('VOID') and not cap['continuation_authority'] and cap['class']=='O'
    tag=read(REPORTS/'tag1_qualification_20260906.json')
    checks['TAG1_VOID_not_functional_FAIL']=tag['verdict']=='VOID' and tag['training_steps']==0 and tag['endpoint_examples_exposed']==0 and set(tag['failed_checks'])=={'source_identifiable','source_permutation_changes_values','gradient_connected'}
    result={'all_evidence_checks_pass':all(checks.values()),'checks':checks,'retained_entries_direct_read':entries,
            'SEL1_bars':{'1_identity_and_integrity':'PASS for executed phases; later interventions not reached',
                         '2_consolidation_and_coverage':'FAIL: 2 entries, minimum 8; endpoint coverage not evaluated',
                         '3_independent_dense_advantage':'NOT EVALUATED: registered early stop',
                         '4_independent_positive_utility_fraction':'NOT EVALUATED: registered early stop'},
            'registered_outcome':'SEL1 FAIL at selection; no retention success',
            'limits':['This audit verifies execution evidence, not the six-pillar ultimate goal.',
                      'QV1 original protocol has a disclosed 63-character digest defect; affected source matches its registration commit after newline normalization.',
                      'No calibration or endpoint outcome is inferred for SEL1 from selection utility.'],
            'reports_sha256':{p.name:sha(p) for p in [REPORTS/'sel1_utility_inheritance_20260906.json',qpath,REPORTS/'capability_completion_20260906.json',REPORTS/'tag1_qualification_20260906.json']}}
    with target.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'all_evidence_checks_pass':result['all_evidence_checks_pass'],'checks':checks}),flush=True)
    if not result['all_evidence_checks_pass']:
        raise RuntimeError('Completion evidence audit failed')


if __name__=='__main__':
    main()
