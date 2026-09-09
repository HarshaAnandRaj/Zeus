from dataclasses import replace
import json
import unittest

from core.lifetime_world import LifetimeAction as A
from core.lifetime_world_v2 import QualityConfig, QualityWorld, QualityInterface, QualityObservation
from training.calibrate_lifetime_world_v2 import action_for


class QualityWorldTests(unittest.TestCase):
    def world(self): return QualityWorld(seed=7, changing=True)

    def test_inspection_reveals_current_quality_with_explicit_mask(self):
        w = self.world(); interface = QualityInterface(w)
        self.assertEqual(len(interface.observation().values()), 8)
        self.assertFalse(interface.observation().prediction_mask()[-1])
        w.step(A.LEFT); w.step(A.LEFT); out = w.step(A.INSPECT)
        self.assertEqual(out.after.resource_quality, w.snapshot()['quality'][0])
        self.assertTrue(out.after.prediction_mask()[-1])
        self.assertFalse(w.step(A.WAIT).after.prediction_mask()[-1])

    def test_contaminated_stock_costs_integrity_and_provides_no_energy(self):
        w = self.world(); snapshot = w.snapshot()
        snapshot.update(position=0, quality=[0,1]); w = QualityWorld.restore(snapshot)
        before = w.snapshot(); row = w.step(A.HARVEST); c = w.config
        amount = min(c.extraction_limit, before['resources'][0])
        self.assertAlmostEqual(row.after.energy, before['energy']-c.metabolism-c.harvest_cost)
        self.assertAlmostEqual(row.after.integrity, before['integrity']-c.integrity_decay-c.contamination_damage*amount/c.extraction_limit)

    def test_safe_stock_has_normal_conversion_without_poisoning(self):
        w = self.world(); snapshot = w.snapshot(); snapshot.update(position=0, quality=[1,0]); w = QualityWorld.restore(snapshot)
        before = w.snapshot(); c=w.config; row=w.step(A.HARVEST)
        amount=min(c.extraction_limit,before['resources'][0])
        self.assertAlmostEqual(row.after.energy,min(1.,before['energy']-c.metabolism-c.harvest_cost+amount*(c.efficiency_floor+c.efficiency_gain*before['tool'])))
        self.assertAlmostEqual(row.after.integrity,before['integrity']-c.integrity_decay)

    def test_switch_preserves_body_stock_and_has_no_free_quality_flag(self):
        w = self.world(); s = w.snapshot(); s.update(tick=219, switches=[220,460,700])
        w=QualityWorld.restore(s); stable=dict(s); stable['switches']=[]; stable=QualityWorld.restore(stable)
        self.assertEqual(w.step(A.WAIT),stable.step(A.WAIT))
        left,right=w.snapshot(),stable.snapshot()
        self.assertEqual(left['quality'],right['quality'][::-1])
        for k in ('energy','integrity','tool','resources','position','rates'): self.assertEqual(left[k],right[k])
        self.assertEqual(left['switch_index'],1)

    def test_json_resume_exactly_preserves_event_and_sensor(self):
        w=self.world(); w.step(A.INSPECT); s=w.snapshot(); s.update(tick=219,switches=[220,460,700])
        w=QualityWorld.restore(s); clone=QualityWorld.restore(json.loads(json.dumps(s)))
        for action in (A.WAIT,A.LEFT,A.LEFT,A.INSPECT,A.RIGHT,A.RIGHT,A.MAINTAIN):
            self.assertEqual(w.step(action),clone.step(action)); self.assertEqual(w.snapshot(),clone.snapshot())

    def test_invalid_state_and_legacy_snapshot_rejected(self):
        for patch in ({'quality':[1,1]}, {'switch_index':1}, {'switches':[1,2,3]}, {'version':'old'}):
            with self.assertRaises(ValueError): QualityWorld.restore(self.world().snapshot()|patch)
        with self.assertRaises(ValueError): replace(QualityConfig(), extraction_limit=0.)

    def test_frozen_map_stops_writes_without_a_world_event_input(self):
        obs=QualityObservation(.8,.9,0.,.5,True,.5,.9,1.)
        state={}; action_for('frozen_map',obs,159,state)
        contradicted=replace(obs,resource_quality=0.)
        action_for('frozen_map',contradicted,160,state)
        self.assertEqual(state['quality'][0],1)
        action_for('public_memory',contradicted,160,state)
        self.assertEqual(state['quality'][0],0)

    def test_map_erasure_preserves_only_current_reading(self):
        obs=QualityObservation(.8,.9,.5,0.,False,0.,0.,0.)
        state={'quality':{0:1,4:0}}
        action_for('current_inspection',obs,0,state)
        self.assertEqual(state['quality'],{})


if __name__=='__main__': unittest.main()
