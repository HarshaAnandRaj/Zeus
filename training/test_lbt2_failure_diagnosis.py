"""Physical reference fixtures check diagnostic meaning; neural fields are symbolic."""
import copy,unittest
from unittest.mock import patch
from training import diagnose_lbt2_failures as D
from training.calibrate_lifetime_world_v2 import physical


class ScarceBirthAnatomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=D.R.CONFIG|dict(base=991000,ecologies=4,trials=1,energies=[.12,.85],horizon=16)
        cls.preps=D.R.episodes(cls.config)

    def fixture(self,prep,policy):
        world=D.A.world_for(prep['seed'],3,prep['energy'],self.config);records=[]
        teacher=D.T.initial_teacher(prep['bodies'][0][2]['next_observation'])
        z=[.1]*8;feeds=repairs=inspections=bad=0
        for tick in range(self.config['horizon']):
            action=D.T.action(world.observation(),teacher) if policy=='teacher' else 0
            before=physical(world.snapshot());effect=world.step(action);after=physical(world.snapshot())
            teacher=D.T.observe(effect,teacher);probability=[float(i==action) for i in range(6)]
            records.append(dict(trial=0,index=prep['index'],energy=prep['energy'],tick=tick,
                **D.P.public_record(effect,tick,self.config['horizon']),probability=probability,h=[0.]*32,z=z.copy(),
                audit_tool_before=before['tool'],audit_tool_after=after['tool'],audit_physical_before=before,audit_physical_after=after))
            feeds+=action==3 and effect.after.energy>effect.before.energy;repairs+=after['tool']>before['tool'];inspections+=action==4
            bad+=action==3 and effect.after.integrity<effect.before.integrity-world.config.integrity_decay
            if effect.terminated:break
        natural=dict(trial=0,index=prep['index'],energy=prep['energy'],seed=prep['seed'],side=prep['side'],quality=prep['quality'],
            target=prep['target'],ticks=tick+1,survived=not effect.terminated,initial_z=z.copy(),written_z=z.copy(),final_z=z.copy(),
            storage_distance=0.,fast_reset=True,feeding=feeds,repairs=repairs,inspections=inspections,bad_harvest=bad,
            final_energy=effect.after.energy,final_integrity=effect.after.integrity,first_action=records[0]['action'],
            first_correct=records[0]['action']==prep['target'],quality_probability=.99 if prep['quality'] else .01,quality_correct=True)
        return natural,records

    def test_recall_and_actuation_are_distinct_with_known_physical_latencies(self):
        total=D.group()
        for prep in self.preps[:4]:
            natural,records=self.fixture(prep,'teacher');summary,stats,fatal=D.anatomy(prep,natural,records,0,self.config)
            self.assertTrue(summary['survived']);self.assertIsNone(fatal)
            self.assertEqual(summary['first_event_steps']['safe_patch'],2)
            self.assertEqual(summary['first_event_steps']['safe_harvest'],3)
            self.assertEqual(summary['first_event_steps']['positive_feeding'],3)
            self.assertTrue(summary['first_argmax_correct']);self.assertEqual(stats['teacher_sample_agreement'],16)
            D.merge(total,stats)
        self.assertEqual(total['bodies'],4);self.assertEqual(total['steps'],64)
        natural,records=self.fixture(self.preps[0],'wait');summary,stats,fatal=D.anatomy(self.preps[0],natural,records,0,self.config)
        self.assertTrue(summary['initial_quality_correct']);self.assertFalse(summary['first_sample_correct'])
        self.assertFalse(summary['survived']);self.assertEqual(summary['ticks'],9);self.assertEqual(summary['cause'],'energy_only')
        self.assertIsNone(summary['first_event_steps']['positive_feeding']);self.assertEqual(stats['no_safe_harvest'],1)
        self.assertEqual(len(fatal['last16']),9);self.assertEqual(fatal['last16'][-1]['tick'],8)

    def test_physics_public_consequence_endpoint_and_case_tampering_reject(self):
        prep=self.preps[0];natural,records=self.fixture(prep,'teacher')
        for kind in ('physical','public','summary','case','reward'):
            n=copy.deepcopy(natural);rows=copy.deepcopy(records)
            if kind=='physical':rows[1]['audit_physical_after']['energy']+=.01
            elif kind=='public':rows[1]['next_observation'][0]+=.01
            elif kind=='summary':n['feeding']+=1
            elif kind=='case':rows[1]['energy']=.85
            else:rows[1]['reward']+=.01
            with self.subTest(kind=kind),self.assertRaises(AssertionError):D.anatomy(prep,n,rows,0,self.config)

    def test_unqualified_or_incomplete_audit_cannot_authorize_extraction(self):
        for receipt in (dict(status='FAIL',verdict='FAIL'),dict(status='PASS',verdict='PASS'),
                        dict(status='PASS',verdict='FAIL',pillar_promotion=True)):
            with patch.object(D.R,'verify',return_value={}),patch.object(D.M.L.P.L3,'read',return_value=receipt):
                with self.assertRaises(AssertionError):D.prerequisites()


if __name__=='__main__':unittest.main()
