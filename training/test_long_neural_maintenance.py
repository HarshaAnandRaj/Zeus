import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from training import run_long_neural_maintenance as R,audit_long_neural_maintenance as A
from training.audit_lifetime_calibration import close

class LongMaintenanceCampaignTests(unittest.TestCase):
    def fixture_rows(self):
        # Abstract gate fixtures, never physics evidence or campaign outputs.
        rows=[]
        for trial in range(4):
            for index in range(128):
                side=(index//2)%2;q=int(index%2==side)
                for mode in R.CONFIG['modes']:
                    for control in R.CONFIG['controls']:
                        rows.append(dict(trial=trial,index=index,mode=mode,control=control,side=side,quality=q,
                            survived=control=='enabled',ticks=4096 if control=='enabled' else 400,repairs=23 if control=='enabled' else 0,
                            feeding=741 if control=='enabled' else 20,quality_correct=True,storage_distance=0.,fast_reset=True,
                            written_z=[1.]*8,initial_z=[1.]*8 if mode=='inherited' else [0.]*8))
        return rows

    def test_every_cell_integer_boundary_and_independent_rules(self):
        rows=self.fixture_rows();cell=[r for r in rows if (r['trial'],r['mode'],r['control'],r['side'],r['quality'])==(3,'inherited','enabled',1,0)]
        self.assertEqual(len(cell),32)
        for r in cell[:3]:r['survived']=False
        primary=R.decide(rows);self.assertEqual(primary['verdict'],'PASS');close(primary,A.independent_decide(rows))
        cell[3]['survived']=False
        primary=R.decide(rows);self.assertEqual(primary['verdict'],'FAIL');self.assertFalse(primary['gates']['survival_3_inherited_1_0'])
        close(primary,A.independent_decide(rows))
        with self.assertRaises(AssertionError):R.decide(rows[:-1]+[copy.deepcopy(rows[0])])

    def test_real_public_preparation_in_new_factory_and_changed_source_rejected(self):
        config=R.CONFIG|dict(base=988000,ecologies=4,trials=1,horizon=16);data=R.episodes(config)
        for prep in data:self.assertEqual(A.preparation(prep,config),24)
        self.assertEqual({(p['side'],p['quality']) for p in data},{(s,q) for s in (0,1) for q in (0,1)})
        changed=copy.deepcopy(data[0]);changed['bodies'][0][2]['next_observation'][7]=1-changed['bodies'][0][2]['next_observation'][7]
        with self.assertRaises(AssertionError):A.preparation(changed,config)

    def test_missing_qualification_cannot_create_experiment_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'unlaunched'
            with patch.object(R,'OUT',target),patch.object(R,'prerequisites',side_effect=AssertionError('parent unqualified')):
                with self.assertRaises(AssertionError):R.prepare()
            self.assertFalse(target.exists())

if __name__=='__main__':unittest.main()
