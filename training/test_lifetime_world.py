"""Lifetime mechanics checks, not a learned-agent evaluation."""
from dataclasses import replace
import json
import unittest

from core.lifetime_world import (LifetimeAction as A, LifetimeConfig, LifetimeWorld,
                                 LifetimeInterface, SENSOR_NAMES, ACTION_NAMES)


class LifetimeWorldTests(unittest.TestCase):
    def world(self, **kwargs):
        return LifetimeWorld(seed=7, changing=True, **kwargs)

    def test_public_interface_is_versioned_and_has_no_audit_channel(self):
        interface = LifetimeInterface(self.world())
        self.assertEqual(len(interface.observation().values()), 7)
        self.assertEqual(len(SENSOR_NAMES), 7); self.assertEqual(len(ACTION_NAMES), 6)
        self.assertEqual(ACTION_NAMES[5], 'maintain')
        self.assertFalse(hasattr(interface, 'snapshot'))
        self.assertFalse(hasattr(interface.observation(), 'change_tick'))

    def test_inspection_is_post_step_masked_and_expires_on_next_step(self):
        w = self.world()
        w.step(A.LEFT); w.step(A.LEFT)
        before = w.snapshot(); row = w.step(A.INSPECT)
        self.assertAlmostEqual(row.after.energy, before['energy'] - w.config.metabolism - w.config.inspect_cost)
        self.assertEqual(row.after.precise_resource, w.snapshot()['resources'][0])
        self.assertEqual(row.after.tool_condition, w.snapshot()['tool'])
        self.assertEqual(w.observation(), w.observation())
        self.assertTrue(all(row.after.prediction_mask()))
        nxt = w.step(A.WAIT)
        self.assertFalse(nxt.after.inspection_valid)
        self.assertEqual(nxt.after.prediction_mask()[-2:], (False, False))
        self.assertEqual(nxt.after.values()[-2:], (0., 0.))

    def test_resource_and_energy_accounting_with_wear(self):
        w = self.world(); w.step(A.LEFT); w.step(A.LEFT)
        s = w.snapshot(); c = w.config
        row = w.step(A.HARVEST); amount = min(c.extraction_limit, s['resources'][0])
        remaining = s['resources'][0] - amount
        self.assertAlmostEqual(w.snapshot()['resources'][0], remaining + s['rates'][0] * (s['capacity'][0] - remaining))
        expected = s['energy'] - c.metabolism - c.harvest_cost + amount * (c.efficiency_floor + c.efficiency_gain * s['tool'])
        self.assertAlmostEqual(row.after.energy, min(1., expected))
        self.assertAlmostEqual(w.snapshot()['tool'], s['tool'] - c.harvest_wear)

    def test_endpoint_moves_and_failed_actions_still_cost(self):
        w = self.world(); w.step(A.LEFT); w.step(A.LEFT)
        s = w.snapshot(); row = w.step(A.LEFT)
        self.assertEqual(row.after.position, 0.)
        self.assertAlmostEqual(row.after.energy, s['energy'] - w.config.metabolism - w.config.move_cost)
        q = w.snapshot()['tool']; w.step(A.MAINTAIN)
        self.assertEqual(q, w.snapshot()['tool'])

    def test_workshop_repair_changes_future_capability(self):
        w = self.world(); c = w.config; s = w.snapshot()
        s.update(tool=.2, integrity=.5)
        w = LifetimeWorld.restore(s); row = w.step(A.MAINTAIN)
        self.assertAlmostEqual(w.snapshot()['tool'], .2 + c.tool_repair)
        self.assertAlmostEqual(row.after.integrity, .5 + c.integrity_repair - c.integrity_decay)

    def test_event_changes_rates_without_reset_or_early_sensor_cue(self):
        c = replace(LifetimeConfig(), change_min=3, change_max=3)
        changed = LifetimeWorld(seed=7, changing=True, config=c)
        stable = LifetimeWorld(seed=7, changing=False, config=c)
        for action in (A.LEFT, A.LEFT, A.INSPECT):
            self.assertEqual(changed.step(action), stable.step(action))
        cs, ss = changed.snapshot(), stable.snapshot()
        self.assertEqual(cs['rates'], ss['rates'][::-1]); self.assertTrue(cs['changed'])
        for key in ('tick', 'position', 'energy', 'integrity', 'tool', 'resources'):
            self.assertEqual(cs[key], ss[key])
        changed.step(A.WAIT); stable.step(A.WAIT)
        self.assertNotEqual(changed.snapshot()['resources'], stable.snapshot()['resources'])

    def test_snapshot_json_roundtrip_preserves_event_sensor_and_rng(self):
        c = replace(LifetimeConfig(), change_min=4, change_max=4)
        w = self.world(config=c); w.step(A.INSPECT)
        clone = LifetimeWorld.restore(json.loads(json.dumps(w.snapshot())))
        self.assertEqual(w.snapshot(), clone.snapshot())
        for action in (A.LEFT, A.LEFT, A.HARVEST, A.INSPECT, A.RIGHT, A.RIGHT, A.MAINTAIN):
            self.assertEqual(w.step(action), clone.step(action))
            self.assertEqual(w.snapshot(), clone.snapshot())

    def test_invalid_actions_and_dead_steps_do_not_mutate(self):
        w = self.world(); original = w.snapshot()
        for action in (99, -1, .5, True, 'left'):
            with self.assertRaises(ValueError): w.step(action)
            self.assertEqual(original, w.snapshot())
        s = w.snapshot(); s['energy'] = 0.; dead = LifetimeWorld.restore(s)
        with self.assertRaises(RuntimeError): dead.step(A.HARVEST)
        self.assertEqual(s, dead.snapshot())

    def test_snapshot_rejects_wrong_version_and_inconsistent_event(self):
        for replacement in ({'version': 'legacy'}, {'tick': -1}, {'resources': [float('nan'), .5]},
                            {'changed': True}, {'position': 9}, {'tool': -1}):
            with self.assertRaises(ValueError): LifetimeWorld.restore(self.world().snapshot() | replacement)

    def test_configuration_validation(self):
        for replacement in ({'fast_recovery': .001}, {'metabolism': float('nan')},
                            {'change_min': 0}, {'resource_bins': 1.5}, {'initial_energy': 0.}):
            with self.assertRaises(ValueError): replace(LifetimeConfig(), **replacement)


if __name__ == '__main__': unittest.main()
