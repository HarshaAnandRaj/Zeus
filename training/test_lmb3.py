import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from training import run_lmb3 as R,audit_lmb3 as A,learner_history_correction as C,audit_lmb1 as A1
from training.dual_body_replay import Replay


class FixedCandidateTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_instrument_gate_rejects_incomplete_or_changed_receipt(self):
        original=R.L.P.L3.read;receipt=original(R.V.REPORT)
        for field,value in (('status','FAIL'),('bodies',255),('original_report_sha','changed')):
            changed=dict(receipt,**{field:value})
            with patch.object(R.L.P.L3,'read',side_effect=lambda p:changed if Path(p)==R.V.REPORT else original(p)):
                with self.assertRaises(AssertionError):R.prerequisites()

    def test_all_declared_candidates_unchanged_and_no_new_optimizer(self):
        R.prerequisites();self.assertEqual(R.CONFIG['updates'],0)
        self.assertEqual(len(R.identities()['candidates']),4)
        for trial in range(4):
            model=R.candidate(trial);before=R.L.P.L3.tree_hash(model.state_dict())
            model.act(torch.tensor([[.85,.95,.5,0.,0.,0.,0.,0.]]),model.initial(1),torch.Generator().manual_seed(780000))
            self.assertEqual(before,R.L.P.L3.tree_hash(model.state_dict()));self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_native_and_synthetic_query_checks_reject_changed_controls(self):
        cfg=R.CONFIG|dict(regression_base=781000,regression_ecologies=2,synthetic_regression_base=782000,synthetic_n=8)
        model=R.candidate(0);data=R.L.P.P.native_episodes(R.L.P.K.native_config(cfg['regression_base'],cfg['regression_ecologies']))
        native=R.L.memory_regression(model,0,data,cfg);replay=Replay(model.state_dict());z=replay.native_starts(data)
        for row in native['rows']:
            used=z if row['control']=='full' else torch.zeros_like(z) if row['control']=='reset' else z[torch.arange(len(z))^2]
            A.query_check(replay,row,used,[d['query'] for d in data],cfg['regression_action_base'],[d['side'] for d in data],[d['quality'] for d in data],[d['target'] for d in data])
        synthetic=A1.A2.public_inputs(cfg['synthetic_regression_base']+4,8,4,True);zs=A.synthetic_source(replay,synthetic)
        for row in R.L.synthetic_regression(model,0,4,cfg):
            used=zs if row['control']=='full' else torch.zeros_like(zs) if row['control']=='reset' else zs[torch.arange(len(zs))^1]
            A.query_check(replay,row,used,synthetic['query'],cfg['regression_action_base']+10000+4,synthetic['side'].tolist(),synthetic['quality'].tolist(),synthetic['target'].tolist())
        changed=copy.deepcopy(native['rows'][0]);changed['used'][0][0]+=.001
        with self.assertRaises(AssertionError):A.query_check(replay,changed,z,[d['query'] for d in data],cfg['regression_action_base'],changed['side'],changed['quality'],changed['target'])

    def test_frozen_rules_reconstructed_and_json_roundtrip(self):
        endpoint=R.L.P.L3.read(R.L.OUT/'endpoint.json');primary=C.plain(R.L.decide(endpoint,R.CONFIG));independent=C.plain(A1.independent_decide(endpoint,R.CONFIG))
        self.assertEqual(primary,independent);self.assertEqual(primary['verdict'],'FAIL')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'verdict.json';C.save(path,primary);self.assertEqual(primary,R.L.P.L3.read(path))

if __name__=='__main__':unittest.main()
