import unittest,torch
from training import probe_lmb5_interior_state as P

class ProbeTests(unittest.TestCase):
    def test_real_pre_action_states_and_fixed_weights(self):
        torch.set_num_threads(1)
        cfg=P.R.K.CONFIG|dict(body_horizon=32);prep=P.R.preparations(229813000,4,cfg)[0]
        body,_=P.R.S.initial(0);before_hash=P.M.L.P.L3.tree_hash(body.state_dict())
        for mode in ('inherited','empty'):
            rows=[];result=P.R.operate(body,0,'warm',prep,mode,cfg,emit=rows.append)
            before=dict(h=[0.]*32,z=result['initial_z'],previous=-1,previous_reward=0.,previous_done=True);samples=[]
            for row in rows:
                samples.append(dict(observation=row['observation'],state=before,probability=row['probability']))
                before=dict(h=row['h'],z=row['z'],previous=row['action'],previous_reward=row['reward'],previous_done=row['body_done'])
            distributions,error=P.distributions(body,samples)
            self.assertLess(error,2e-5)
            # Empty birth can acquire a nonzero store by a later real inspection.
            if mode=='empty':
                self.assertTrue(torch.equal(distributions['intact'][0],distributions['z_zero'][0]))
            samples[0]['probability']=[1.,0.,0.,0.,0.,0.]
            with self.assertRaises(AssertionError):P.distributions(body,samples)
        self.assertEqual(before_hash,P.M.L.P.L3.tree_hash(body.state_dict()))

if __name__=='__main__':unittest.main()
