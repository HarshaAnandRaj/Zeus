import copy,unittest
from unittest.mock import patch
import torch
from core.native_memory_adapter import consolidate
from core.native_body_agent import NativeBodyAgent
from training import run_lmb2 as R,lmb2_contract as K
from training.dual_body_replay import Replay,canonical


class DualReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        cls.directory=R.OUT/'0_demonstration_a'
        cls.source=torch.load(cls.directory/'source_4.pt',weights_only=False)['model']
        cls.data=R.load_collection(cls.directory/'collection_4.jsonl.gz')

    def replay(self,row=None):
        row=self.data[0] if row is None else row
        return Replay(self.source).body(row['records'],row['preparation'],K.CONFIG['collection_action_base']+4000,
            K.CONFIG['body_horizon'],row['inherited'],True)

    def test_known_accumulating_rejection_complete_exact_without_model_methods(self):
        with patch.object(NativeBodyAgent,'act',side_effect=AssertionError('model method forbidden')),patch.object(NativeBodyAgent,'observe',side_effect=AssertionError('model method forbidden')):
            summary=self.replay()
        self.assertEqual(summary['ticks'],len(self.data[0]['records']));self.assertGreaterEqual(summary['ticks'],151)

    def test_source_storage_exact_in_single_and_batch_modes(self):
        model=R.trained_model(0,'demonstration');episodes=[d['preparation'] for d in self.data]
        for batch in ([episodes[0]],episodes):
            state,_=consolidate(model.store,batch);replay=Replay(self.source)
            self.assertTrue(torch.equal(state['z'],replay.native_starts(batch)))

    def test_trace_tampering_cannot_be_used_as_recurrent_input(self):
        for key in ('probability','h','z','action','reward','label'):
            row=copy.deepcopy(self.data[0]);target=row['records'][150]
            if key in ('probability','h','z'):target[key][0]+=.001
            elif key=='reward':target[key]+=.001
            else:target[key]=(target[key]+1)%6
            with self.subTest(key=key),self.assertRaises(AssertionError):self.replay(row)

    def test_independent_math_corruption_is_rejected(self):
        import numpy as np
        with patch('training.dual_body_replay.numpy_gru',side_effect=lambda w,p,x,h:np.zeros_like(h)):
            with self.assertRaisesRegex(AssertionError,'local independent recurrent mathematics'):self.replay()

    def test_invalid_public_fields_masked_and_nonbinary_validity_rejected(self):
        a=[[.85,.95,.5,0.,0.,0.,0.,0.]];b=copy.deepcopy(a);b[0][5:]=[123.,456.,789.]
        self.assertTrue(torch.equal(canonical(a),canonical(b)))
        replay=Replay(self.source);h=torch.zeros(1,32);z=torch.zeros(1,8)
        pa,ha=replay.decide(a,h,z);pb,hb=replay.decide(b,h,z)
        self.assertTrue(torch.equal(pa,pb) and torch.equal(ha,hb))
        b[0][4]=.5
        with self.assertRaises(AssertionError):canonical(b)

    def test_incomplete_or_trailing_records_rejected(self):
        for changed in (self.data[0]['records'][:-1],self.data[0]['records']+[self.data[0]['records'][-1]]):
            row=copy.deepcopy(self.data[0]);row['records']=changed
            with self.assertRaises((AssertionError,StopIteration)):self.replay(row)

if __name__=='__main__':unittest.main()
