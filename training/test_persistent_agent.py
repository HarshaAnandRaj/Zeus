"""Synthetic implementation checks only; no Zeus training or survival assay."""
import copy
from dataclasses import replace
import unittest

import torch

from core.persistent_agent import AgentConfig, PersistentAgent
from core.persistent_session import PersistentSession
from training.persistent_learning import (
    LossSettings, SequenceBatch, collect_segment, generalized_advantages,
    sequence_loss, update_segment,
)


class PersistentAgentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.model = PersistentAgent().double()
        self.generator = torch.Generator().manual_seed(81)
        self.settings = LossSettings(.9, .8, .5, 1., .01)

    def observations(self, n=6):
        return torch.linspace(.1, .9, n * 5, dtype=torch.float64).reshape(n, 5)

    def segment(self, *, count=5, greedy=False, terminated=False, truncated=False):
        session = PersistentSession(self.model); session.begin_episode()
        obs = self.observations(count + 1); rows = []
        for i in range(count):
            session.act(obs[i], generator=self.generator, greedy=greedy)
            rows.append(session.record_outcome(obs[i + 1], reward=.2 + .01 * i,
                        terminated=terminated and i == count - 1,
                        truncated=truncated and i == count - 1))
        return session, rows

    def test_chunking_keeps_exact_state_and_outputs(self):
        session, rows = self.segment()
        whole = sequence_loss(self.model, SequenceBatch.from_transitions(rows), self.settings)
        split = sequence_loss(self.model, SequenceBatch.from_transitions(rows[2:]), self.settings)
        self.assertTrue(torch.equal(whole['final_state'], split['final_state']))
        self.assertTrue(torch.equal(session.state, whole['final_state']))
        before = session.state; session.detach()
        self.assertTrue(torch.equal(before, session.state))
        self.assertGreater(float(before.norm()), 0.)

    def test_chunk_detaches_past_but_keeps_gradients_inside_sequence(self):
        _, rows = self.segment()
        batch = SequenceBatch.from_transitions(rows)
        state = batch.initial_state.clone().requires_grad_()
        observation = batch.observation.clone().requires_grad_()
        losses = sequence_loss(self.model, replace(batch, initial_state=state, observation=observation), self.settings)
        losses['final_state'].sum().backward()
        self.assertIsNone(state.grad)
        self.assertGreater(float(observation.grad[0].abs().sum()), 0.)

    def test_new_episode_resets_prior_state_and_requires_start_marker(self):
        obs = self.observations(1)
        a = self.model.step(obs, torch.tensor([-1]), torch.ones(1, 32).double(), torch.tensor([True]))
        b = self.model.step(obs, torch.tensor([-1]), self.model.initial_state(1), torch.tensor([True]))
        self.assertTrue(torch.equal(a.state, b.state))
        with self.assertRaisesRegex(ValueError, 'start marker'):
            self.model.step(obs, torch.tensor([1]), self.model.initial_state(1), torch.tensor([True]))

    def test_inputs_include_body_and_previous_executed_action(self):
        obs = self.observations(1); state = self.model.initial_state(1); starts = torch.tensor([False])
        a = self.model.step(obs, torch.tensor([1]), state, starts)
        b = self.model.step(obs, torch.tensor([2]), state, starts)
        changed = obs.clone(); changed[0, 0] += .1
        c = self.model.step(changed, torch.tensor([1]), state, starts)
        self.assertFalse(torch.equal(a.state, b.state)); self.assertFalse(torch.equal(a.state, c.state))
        self.assertTrue(torch.equal(a.logits, self.model.actor(a.state)))
        self.assertTrue(torch.equal(a.value, self.model.critic(a.state).squeeze(-1)))
        self.assertFalse(torch.equal(a.predictions[:, 1], a.predictions[:, 2]))

    def test_each_objective_reaches_shared_recurrence(self):
        _, rows = self.segment(); batch = SequenceBatch.from_transitions(rows)
        for objective, head in [('actor', self.model.actor), ('value', self.model.critic),
                                ('prediction', self.model.transition)]:
            self.model.zero_grad(set_to_none=True)
            loss = sequence_loss(self.model, batch, self.settings)
            self.assertFalse(loss['advantages'].requires_grad)
            self.assertFalse(loss['value_targets'].requires_grad)
            loss[objective].backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.recurrence.parameters()), 0.)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.parameters()), 0.)

    def test_terminal_truncation_and_chunk_bootstrap_by_hand(self):
        values = torch.tensor([[2.], [3.]], requires_grad=True)
        next_values = torch.tensor([[3.], [10.]], requires_grad=True)
        rewards = torch.tensor([[1.], [2.]])
        none = torch.zeros(2, 1, dtype=torch.bool)
        terminal = torch.tensor([[False], [True]])
        _, death = generalized_advantages(rewards, values, next_values, terminal, none, gamma=.5, gae_lambda=1.)
        _, cut = generalized_advantages(rewards, values, next_values, none, terminal, gamma=.5, gae_lambda=1.)
        _, chunk = generalized_advantages(rewards, values, next_values, none, none, gamma=.5, gae_lambda=1.)
        torch.testing.assert_close(death, torch.tensor([[2.], [2.]]))
        torch.testing.assert_close(cut, torch.tensor([[4.5], [7.]]))
        torch.testing.assert_close(chunk, cut)
        self.assertFalse(death.requires_grad)
        # An internal reset cannot leak a later episode's large reward backward.
        internal = torch.tensor([[True], [False]])
        adv, _ = generalized_advantages(rewards, values, next_values, none, internal, gamma=.5, gae_lambda=1.)
        self.assertEqual(float(adv[0]), .5)

    def test_joint_synthetic_update_requires_refresh_and_rejects_stale_data(self):
        session, rows = self.segment(); batch = SequenceBatch.from_transitions(rows)
        before = copy.deepcopy(self.model.state_dict())
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        metrics = update_segment(session, optimizer, batch, self.settings, max_grad_norm=1.)
        self.assertEqual(metrics['revision'], 1)
        for prefix in ('recurrence.', 'actor.', 'critic.', 'transition.'):
            self.assertTrue(any(not torch.equal(v, before[k]) for k, v in self.model.state_dict().items() if k.startswith(prefix)))
        with self.assertRaisesRegex(ValueError, 'stale'):
            sequence_loss(self.model, batch, self.settings)
        with self.assertRaisesRegex(RuntimeError, 'refresh_state'):
            session.act(rows[-1].next_observation, generator=self.generator)
        state = self.model.initial_state(1)
        with torch.no_grad():
            for row in rows:
                state = self.model.step(row.observation[None], torch.tensor([row.previous_action]),
                                        state, torch.tensor([row.starts])).state
        session.refresh_state()
        self.assertTrue(torch.equal(state, session.state))
        # Replay neither consumes the final outcome twice nor chooses a new action.
        generator_state = self.generator.get_state().clone()
        session.refresh_state()
        self.assertTrue(torch.equal(generator_state, self.generator.get_state()))
        _, output = session.act(rows[-1].next_observation, generator=self.generator)
        expected = self.model.step(rows[-1].next_observation[None], torch.tensor([rows[-1].action]),
                                   state, torch.tensor([False]))
        self.assertTrue(torch.equal(output.state, expected.state))

    def test_optimizer_cannot_silently_leave_core_frozen(self):
        session, rows = self.segment(); batch = SequenceBatch.from_transitions(rows)
        optimizer = torch.optim.Adam(self.model.actor.parameters())
        with self.assertRaisesRegex(ValueError, 'every trainable'):
            update_segment(session, optimizer, batch, self.settings, max_grad_norm=1.)

    def test_pending_action_blocks_updates_before_any_parameter_change(self):
        session, rows = self.segment(); batch = SequenceBatch.from_transitions(rows)
        session.act(rows[-1].next_observation, generator=self.generator)
        before = copy.deepcopy(self.model.state_dict())
        optimizer = torch.optim.AdamW(self.model.parameters())
        with self.assertRaisesRegex(RuntimeError, 'pending action outcome'):
            update_segment(session, optimizer, batch, self.settings, max_grad_norm=1.)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_greedy_data_cannot_be_used_as_on_policy_samples(self):
        _, rows = self.segment(greedy=True)
        with self.assertRaisesRegex(ValueError, 'sampled behavior'):
            SequenceBatch.from_transitions(rows)

    def test_reordered_or_spliced_history_is_rejected(self):
        _, rows = self.segment()
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            SequenceBatch.from_transitions([rows[0], rows[2]])
        with self.assertRaisesRegex(ValueError, 'episode boundaries'):
            SequenceBatch.from_transitions([replace(rows[0], terminated=True), rows[1]])
        with self.assertRaisesRegex(ValueError, 'revisions'):
            SequenceBatch.from_transitions([rows[0], replace(rows[1], revision=7)])

    def test_behavior_replay_catches_untracked_parameter_changes(self):
        _, rows = self.segment()
        with torch.no_grad():
            self.model.actor.bias[0] += 1
        with self.assertRaisesRegex(ValueError, 'behavior mismatch'):
            sequence_loss(self.model, SequenceBatch.from_transitions(rows), self.settings)

    def test_session_enforces_outcome_order_and_explicit_resets(self):
        session = PersistentSession(self.model); obs = self.observations(2)
        with self.assertRaises(RuntimeError): session.act(obs[0], generator=self.generator)
        session.begin_episode()
        with self.assertRaises(RuntimeError): session.begin_episode()
        session.act(obs[0], generator=self.generator)
        with self.assertRaises(RuntimeError): session.act(obs[1], generator=self.generator)
        with self.assertRaises(RuntimeError): session.refresh_state()
        row = session.record_outcome(obs[1], reward=0., terminated=False)
        with self.assertRaises(ValueError): session.act(obs[0], generator=self.generator)
        session.end_episode(); session.begin_episode()
        self.assertEqual(float(session.state.norm()), 0.)
        self.assertGreater(float(row.state_after.norm()), 0.)

    def test_trace_and_returned_tensors_do_not_alias_live_state(self):
        session = PersistentSession(self.model); session.begin_episode(); obs = self.observations(2)
        _, output = session.act(obs[0], generator=self.generator)
        state = session.state
        output.state.zero_(); output.logits.zero_()
        row = session.record_outcome(obs[1], reward=.1, terminated=False)
        self.assertTrue(torch.equal(row.state_after, state))
        row.state_after.zero_(); row.next_observation.zero_()
        self.assertTrue(torch.equal(session.state, state))
        session.act(obs[1], generator=self.generator)

    def test_collector_chunk_cut_carries_history_without_hidden_world_access(self):
        class StubWorld:
            def __init__(self): self.tick = 0
            def observation(self): return [.2 + .01 * self.tick, .8, .5, .3, .5]
            def viable(self): return self.tick < 4
            def step(self, action):
                self.tick += 1
                return dict(action=action, reward=.2)
            @property
            def resources(self): raise AssertionError('privileged input access')
        world = StubWorld(); session = PersistentSession(self.model); session.begin_episode()
        first = collect_segment(session, world, steps=2, reward_fn=lambda e: e['reward'], generator=self.generator)
        second = collect_segment(session, world, steps=5, reward_fn=lambda e: e['reward'], generator=self.generator)
        self.assertEqual(len(first), 2); self.assertEqual(len(second), 2)
        self.assertFalse(first[-1].terminated); self.assertFalse(first[-1].truncated)
        self.assertFalse(second[0].starts); self.assertTrue(second[-1].terminated)
        self.assertTrue(torch.equal(first[-1].state_after, second[0].state_before))
        SequenceBatch.from_transitions(first + second)

    def test_bad_inputs_rejected_before_session_state_changes(self):
        session = PersistentSession(self.model); session.begin_episode()
        for obs in ([.5] * 4, [float('nan')] * 5, [2.] * 5):
            with self.assertRaises(ValueError): session.act(obs, generator=self.generator)
        self.assertEqual(float(session.state.norm()), 0.)
        with self.assertRaises(ValueError): AgentConfig(0)
        with self.assertRaises(ValueError): LossSettings(.9, .8, float('nan'), 1., .01)

    def test_default_float32_path_and_batched_resets(self):
        model = PersistentAgent()
        obs = torch.full((2, 5), .5)
        state = torch.ones(2, 32)
        output = model.step(obs, torch.tensor([-1, 3]), state, torch.tensor([True, False]))
        reset = model.step(obs[:1], torch.tensor([-1]), model.initial_state(1), torch.tensor([True]))
        torch.testing.assert_close(output.state[:1], reset.state)
        self.assertEqual(output.state.dtype, torch.float32)
        self.assertEqual(output.predictions.shape, (2, 6, 5))
        self.assertTrue(torch.isfinite(output.predictions).all())

    def test_model_checkpoint_and_explicit_sampling_generator_reproduce(self):
        session, rows = self.segment()
        saved = copy.deepcopy(self.model.state_dict()); clone = PersistentAgent().double()
        clone.load_state_dict(saved)
        other = PersistentSession(clone); other.begin_episode()
        generator = torch.Generator().manual_seed(81)
        for row in rows:
            action, output = other.act(row.observation, generator=generator)
            self.assertEqual(action, row.action)
            self.assertTrue(torch.equal(output.logits[0], row.behavior_logits))
            other.record_outcome(row.next_observation, reward=row.reward, terminated=False)
        self.assertTrue(torch.equal(other.state, session.state))


if __name__ == '__main__':
    unittest.main()
