import gzip,json,tempfile,unittest
from pathlib import Path
from core.lineage_energy_ecology import BirthResourceEcology,BirthResources
from core.lineage_ecology import LineageEcology
from training import calibrate_birth_resource_memory as R
from training import audit_birth_resource_memory as A


class BirthResourceTests(unittest.TestCase):
    def test_versioned_initial_resource_change_and_first_body_identity(self):
        ecology=BirthResourceEcology(seed=740000,resources=BirthResources(.12));base=LineageEcology(seed=740000)
        self.assertEqual(ecology.body(0).snapshot(),base.body(0).snapshot())
        actual=ecology.body(1).snapshot();expected=base.body(1).snapshot();expected['energy']=.12;expected['config']['initial_energy']=.12
        self.assertEqual(actual,expected)
        with self.assertRaises(ValueError):BirthResources(float('nan'))
        with self.assertRaises(ValueError):BirthResources(.02)

    def test_reference_public_reinspection_remains_an_action(self):
        config=R.CONFIG|dict(horizon=16)
        known=R.run_one(740000,.12,'carry',config=config);forgot=R.run_one(740000,.12,'forget',config=config)
        self.assertEqual(known['bodies'][0],forgot['bodies'][0]);self.assertEqual(known['bodies'][1]['inspections'],0)
        self.assertGreater(forgot['bodies'][1]['inspections'],0)

    def test_independent_replay_rejects_tampered_executed_action(self):
        config=R.CONFIG|dict(horizon=16)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.jsonl.gz'
            with R.trace_writer(path) as stream:expected=R.run_one(740000,.12,'carry',stream,config)
            with gzip.open(path,'rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
            actual,count=A.replay_one(iter(rows),740000,.12,'carry',config)
            self.assertEqual(actual,expected);self.assertEqual(count,len(rows))
            rows[0]['action']=0
            with self.assertRaises(AssertionError):A.replay_one(iter(rows),740000,.12,'carry',config)

    def test_independent_all_reference_policies_and_clustered_gates(self):
        config=R.CONFIG|dict(base=740000,ecologies=2,horizon=16,bootstrap_draws=20)
        results=[]
        with tempfile.TemporaryDirectory() as directory:
            for energy in config['energies']:
                for index in range(config['ecologies']):
                    for control in config['controls']:
                        path=Path(directory)/f'{energy}_{index}_{control}.gz'
                        with R.trace_writer(path) as stream:expected=R.run_one(config['base']+index,energy,control,stream,config)
                        with gzip.open(path,'rt',encoding='utf-8') as stream:
                            trace=(json.loads(line) for line in stream)
                            actual,count=A.replay_one(trace,config['base']+index,energy,control,config)
                            self.assertIsNone(next(trace,None))
                        self.assertEqual(expected,actual);results.append(actual)
        self.assertEqual(R.decide(results,config),A.independent_decide(results,config))

if __name__=='__main__':unittest.main()
