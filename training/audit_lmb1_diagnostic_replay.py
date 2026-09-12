"""Run the frozen auditor unchanged; record its expected final-file rejection.

This cannot issue a qualifying receipt. Source/endpoint/partial verdict stay frozen.
The caught frame records checks completed before the corrupt-file comparison.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training import audit_lmb1 as A,diagnose_lmb1 as D


def main():
    D.verify();assert __debug__,'audit assertions must be enabled'
    try:
        A.main()
    except json.JSONDecodeError as error:
        frame=None;traceback=error.__traceback__
        while traceback is not None:
            if Path(traceback.tb_frame.f_code.co_filename).resolve()==ROOT/'training/audit_lmb1.py' and traceback.tb_frame.f_code.co_name=='main':
                assert traceback.tb_lineno==202,'unexpected audit rejection';frame=traceback.tb_frame
            traceback=traceback.tb_next
        assert frame is not None,'not the frozen final-file comparison'
        data=frame.f_locals;verdict=D.plain(data['verdict']);endpoint=D.R.P.L3.read(D.R.OUT/'endpoint.json')
        assert verdict==D.plain(D.R.decide(endpoint));D.verify()
        receipt=dict(campaign_status='VOID',physical_neural_replay_status='PASS',qualification=False,
            frozen_rule_functional_verdict=verdict['verdict'],expected_rejection=str(error),rejection_line=202,
            exact_twin_pairs=4,frozen_store_reader_parents=4,training_batches_verified=data['batches'],
            public_physical_steps_replayed=data['steps'],numpy_endpoint_decisions=data['decisions'],
            max_probability_error=data['error'],max_state_error=data['state_error'],
            auditor_sha=D.AUDITOR_SHA,endpoint_sha=D.ENDPOINT_SHA,partial_verdict_sha=D.PARTIAL_SHA,
            trace_sha=D.R.P.L3.sha(D.R.OUT/'endpoint_trace.jsonl.gz'),
            observer_source_sha=D.R.P.L3.sha(Path(__file__)),scope='Post-hoc replay evidence; frozen campaign VOID and functional failures preserved')
        D.R.P.L3.save(ROOT/'zeus_sandbox/universe/reports/lmb1_diagnostic_replay_20260913.json',receipt)
        print(json.dumps(receipt,indent=2));return
    raise AssertionError('expected corrupt frozen final verdict to be rejected')

if __name__=='__main__':main()
