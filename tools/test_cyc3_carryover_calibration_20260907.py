"""CYC3 mechanics on nonregistered or purely synthetic conditions only."""
import copy
import unittest
from core.embodiment import EmbodiedWorldV2
from tools import cyc3_carryover_calibration_20260907 as C

class CYC3Tests(unittest.TestCase):
    def test_preparation_pairs_match_current_observation_and_timing(self):
        a,b=C.prepare(7,2),C.prepare(7,6)
        self.assertEqual(a['observation'],b['observation'])
        self.assertEqual(a['boundary']['body']['age'],16)
        self.assertEqual(a['cache'].keys(),b['cache'].keys())
        for key in a['cache']:self.assertEqual(a['cache'][key]['tick'],b['cache'][key]['tick'])
        self.assertNotEqual(a['cache'],b['cache'])
        self.assertEqual(C.snapshot(C.restore(a['boundary'])),a['boundary'])
        for key in ['resources','capacity','step_count','ambient_phase']:
            self.assertEqual(a['before_body_match'][key],a['boundary'][key])

    def test_erase_swap_only_memory_and_preserve_current_sensing(self):
        a,b=C.prepare(11,2),C.prepare(11,6)
        for mode in ['intact','erased','swapped']:
            w,cache=C.fork_memory(a,b,mode)
            self.assertEqual(C.snapshot(w),a['boundary'])
            self.assertEqual(cache['4']['resource'],w.observation()[3])
            self.assertEqual(cache['4']['tick'],16)
            if mode=='erased':self.assertEqual(list(cache),['4'])
            if mode=='swapped':self.assertEqual(cache,b['cache'])

    def test_controller_visibility_and_content(self):
        obs=(.104,.95,.5,.0028,.5)
        cache={'2':dict(resource=.6,tick=16),'6':dict(resource=.01,tick=16)}
        before=copy.deepcopy(cache)
        self.assertEqual(int(C.choose(obs,16,cache)[0]),1)
        self.assertEqual(cache,before)
        cache={'6':dict(resource=.6,tick=16),'2':dict(resource=.01,tick=16)}
        self.assertEqual(int(C.choose(obs,16,cache)[0]),2)
        self.assertEqual(before['2']['resource'],.6)
        self.assertEqual(C.estimate({},8,16),.4)

    def test_search_death_witness_cap_and_no_parent_mutation(self):
        w=EmbodiedWorldV2(seed=13);w.body.energy=.06;w.resources=[0.]*9
        fixture=dict(boundary=C.snapshot(w));before=copy.deepcopy(fixture)
        self.assertEqual(C.alternative_search(fixture,0,horizon=3)['status'],'EXHAUSTIVE_NO_SURVIVOR')
        self.assertEqual(fixture,before)
        w.body.energy=.9;fixture=dict(boundary=C.snapshot(w))
        self.assertEqual(C.alternative_search(fixture,0,horizon=3)['status'],'COUNTEREXAMPLE')
        self.assertEqual(C.alternative_search(fixture,0,horizon=3,cap=1)['status'],'UNVERIFIED_CAP')

    def test_pair_unit_and_binary_failure(self):
        pairs=[]
        for _ in range(128):
            os=[]
            for side in [0,1]:
                controls={c:dict(survived_256=(c=='intact'),survived_512=(c=='intact'),age=512,
                                first_action='move_left') for c in ['intact','erased','swapped']}
                correct=side+1
                os.append(dict(controls=controls,correct_first_action=correct,searches=[
                    dict(first_action=a,status='EXHAUSTIVE_NO_SURVIVOR',expanded_transitions=1)
                    for a in range(6) if a!=correct]))
            pairs.append(dict(orientations=os))
        r=C.adjudicate(pairs)
        self.assertEqual(r['verdict'],'PASS')
        self.assertEqual(r['bars']['intact_pair_survival_512']['n'],128)
        pairs[0]['orientations'][0]['searches'][0]['status']='COUNTEREXAMPLE'
        self.assertEqual(C.adjudicate(pairs)['verdict'],'FAIL')

if __name__=='__main__':unittest.main()
