"""Untrained development fixtures only; no campaign endpoint is opened."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from core.anchored_body_agent import AnchoredBodyAgent
from core.native_body_agent import NativeBodyAgent
from core.protected_lineage_agent import ProtectedLineageAgent
from core.lineage_ecology import LineageConfig, LineageEcology
from core.native_memory_adapter import consolidate, public_body_records
from training import native_body_operation as O
from training.anchored_body_replay import Replay, MODEL_VERSION
from training.native_physical_replay import CheckedWorld


def development_source():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(230803000)
        base = SimpleNamespace(store=ProtectedLineageAgent(), fast=nn.GRUCell(17, 32),
            reinstate=nn.Linear(8, 32, bias=False), gate=nn.Linear(40, 32), actor=nn.Linear(32, 6))
        return NativeBodyAgent(SimpleNamespace(base=base, reader=nn.Linear(8, 2)), 'untrained-replay-development-230803000')


def candidate(mode):
    model = AnchoredBodyAgent(development_source(), mode)
    with torch.no_grad():
        model.anchor.weight.copy_(torch.sin(torch.arange(256).reshape(32, 8).float()) * .17)
        model.anchor.bias.copy_(torch.linspace(-.08, .08, 32))
    return model.eval()


class AnchorReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        cls.preps = []; cls.ecologies = []
        for index in range(4):
            ecology = LineageEcology(seed=230813000 + index, config=LineageConfig(4, 64)); side = index // 2
            bodies = [public_body_records(ecology.body(c), side, 8, inspect=c == 0) for c in range(3)]
            cue = bodies[0][2]['next_observation']; quality = int(cue[7])
            cls.preps.append(dict(seed=ecology.seed, side=side, quality=quality,
                target=1 + (side if quality else 1 - side), bodies=bodies, query=list(ecology.body(3).observation().values())))
            cls.ecologies.append(ecology)
        cls.fixtures = []
        for mode in ('current', 'recurrent'):
            model = candidate(mode)
            for index, (prep, ecology) in enumerate(zip(cls.preps, cls.ecologies)):
                for inherited in (True, False):
                    records = []; seed = 230823000 + index
                    summary = O.operate(model, prep, ecology.body(3), action_seed=seed, horizon=64,
                        inherited=inherited, emit=records.append)
                    cls.fixtures.append(dict(mode=mode, prep=prep, ecology=ecology, records=records,
                        inherited=inherited, seed=seed, summary=summary, weights=copy.deepcopy(model.state_dict())))

    def replay_body(self, fixture, records=None, weights=None):
        records = fixture['records'] if records is None else records
        weights = fixture['weights'] if weights is None else weights
        replay = Replay(weights, expected_mode=fixture['mode'])
        world = CheckedWorld(fixture['ecology'].body(3), records)
        summary = replay.body(records, fixture['prep'], fixture['seed'], 64, fixture['inherited'], world=world)
        world.complete()
        return summary, replay

    def test_whole_fresh_bodies_both_modes_and_memory_controls_without_model_methods(self):
        with patch.object(AnchoredBodyAgent, 'step_inputs', side_effect=AssertionError('production method forbidden')), \
                patch.object(NativeBodyAgent, 'act', side_effect=AssertionError('production method forbidden')), \
                patch.object(NativeBodyAgent, 'observe', side_effect=AssertionError('production method forbidden')), \
                patch.object(ProtectedLineageAgent, 'observe', side_effect=AssertionError('production method forbidden')):
            for fixture in self.fixtures:
                with self.subTest(mode=fixture['mode'], seed=fixture['prep']['seed'], inherited=fixture['inherited']):
                    actual, replay = self.replay_body(fixture)
                    expected = dict(seed=fixture['prep']['seed'], **{k: fixture['summary'][k] for k in actual if k != 'seed'})
                    self.assertEqual(actual, expected)
                    self.assertEqual(replay.decisions, len(fixture['records']))
                    self.assertLess(replay.max_probability_error, 2e-5); self.assertLess(replay.max_state_error, 1e-4)

    @torch.no_grad()
    def test_vectorized_readout_and_native_consolidation_from_computed_state(self):
        for mode in ('current', 'recurrent'):
            model = candidate(mode); replay = Replay(model.state_dict(), expected_mode=mode)
            prepared, _ = consolidate(model.store, self.preps); z = replay.native_starts(self.preps)
            self.assertTrue(torch.equal(z, prepared['z']))
            state = model.initial(4, z)
            state['h'] = torch.linspace(-.6, .6, 128).reshape(4, 32)
            state['previous'] = torch.tensor([-1, 2, 3, 4]); state['previous_reward'] = torch.tensor([0., .02, -.03, .01])
            state['previous_done'] = torch.tensor([True, False, False, True])
            values = torch.tensor([p['query'] for p in self.preps])
            values[1, 2] = .25; values[2, 2] = .75
            values[3, 4:] = torch.tensor([1., .45, .7, 1.])
            for _ in range(5):
                expected, acted = model.logits(values, state)
                actual, h = replay.decide(values, state['h'], state['z'], state['previous'], state['previous_reward'], state['previous_done'])
                self.assertTrue(torch.equal(actual, expected.softmax(-1))); self.assertTrue(torch.equal(h, acted['h']))
                state = acted | dict(previous=torch.tensor([2, 0, 1, 3]), previous_done=torch.zeros(4, dtype=torch.bool))
            self.assertEqual(replay.decisions, 20)

    def test_saved_neural_public_and_physical_corruption_cannot_advance_replay(self):
        for mode in ('current', 'recurrent'):
            fixture = next(f for f in self.fixtures if f['mode'] == mode and f['inherited'])
            for field in ('h', 'z', 'probability', 'reward', 'action', 'next_observation', 'audit_physical_after'):
                records = copy.deepcopy(fixture['records']); row = records[2]
                if field in ('h', 'z', 'probability', 'next_observation'): row[field][0] += .01
                elif field == 'audit_physical_after': row[field]['energy'] += .01
                elif field == 'action': row[field] = (row[field] + 1) % 6
                else: row[field] += .01
                with self.subTest(mode=mode, field=field), self.assertRaises(AssertionError):
                    self.replay_body(fixture, records)
            for records in (fixture['records'][:-1], fixture['records'] + [fixture['records'][-1]]):
                with self.assertRaises((AssertionError, StopIteration, IndexError)): self.replay_body(fixture, records)

    def test_independent_anchor_and_output_math_corruption_rejected(self):
        fixture = self.fixtures[0]
        with patch('training.anchored_body_replay.numpy_anchor', side_effect=lambda w, x, r, mode: np.zeros_like(r)):
            with self.assertRaisesRegex(AssertionError, 'local independent recurrent mathematics'): self.replay_body(fixture)
        replay = Replay(fixture['weights'])
        replay.nw['actor.weight'] = replay.nw['actor.weight'].copy(); replay.nw['actor.weight'][0] += 1.
        with self.assertRaisesRegex(AssertionError, 'local independent output mathematics'):
            replay.decide([fixture['prep']['query']], torch.zeros(1, 32), torch.zeros(1, 8))

    def test_metadata_and_valid_mode_mutation_require_external_campaign_binding(self):
        fixture = self.fixtures[0]; original = fixture['weights']
        for replacement in ({}, dict(version='wrong', mode='current', parent_hash='x'),
                dict(version=MODEL_VERSION, mode='unknown', parent_hash='x'),
                dict(version=MODEL_VERSION, mode='current', parent_hash='x', extra=True)):
            changed = copy.deepcopy(original); changed['_extra_state'] = replacement
            with self.assertRaises(AssertionError): Replay(changed)
        changed = copy.deepcopy(original); changed['_extra_state']['mode'] = 'recurrent'
        with self.assertRaisesRegex(AssertionError, 'campaign anchor mode'): Replay(changed, expected_mode='current')
        with self.assertRaisesRegex(AssertionError, 'campaign anchor parent'): Replay(original, expected_parent_hash='different-parent')
        # A self-consistent declaration alone cannot establish which mode a campaign promised.
        unbound = Replay(changed)
        with self.assertRaisesRegex(AssertionError, 'complete hardware probability'):
            unbound.body(fixture['records'], fixture['prep'], fixture['seed'], 64, world=fixture['ecology'].body(3))
        with self.assertRaises(ValueError): candidate('current').load_state_dict(changed)
        changed = copy.deepcopy(original); changed['store._extra_state']['mode'] = 'every_step'
        with self.assertRaisesRegex(AssertionError, 'unsupported consolidation'): Replay(changed)
        changed = copy.deepcopy(original); changed['anchor.weight'] = changed['anchor.weight'][:, :-1]
        with self.assertRaisesRegex(AssertionError, 'tensor shape'): Replay(changed)
        changed = copy.deepcopy(original); changed['anchor.bias'][0] = float('nan')
        with self.assertRaisesRegex(AssertionError, 'nonfinite checkpoint'): Replay(changed)


if __name__ == '__main__': unittest.main()
