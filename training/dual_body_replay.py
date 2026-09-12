"""Complete CPU float32 operator replay plus independent local NumPy mathematics.

No model methods or recorded recurrent states are used to advance the replayer.
The GRU primitive is shared with PyTorch: mathematical independence is supplied
by the separate NumPy check at every transition, not by claiming a second engine.
"""
import numpy as np
import torch
from torch.nn import functional as F
from core.lineage_ecology import LineageEcology, LineageConfig
from core.lifetime_world_v2 import QualityConfig

VERSION = 'dual-public-body-replay-v1-20260913'
PROBABILITY_LIMIT = 2e-5
STATE_LIMIT = 1e-4


def canonical(values):
    obs = torch.as_tensor(values, dtype=torch.float32).clone()
    assert obs.ndim == 2 and obs.shape[1] == 8 and torch.isfinite(obs).all()
    assert ((obs[:, 4] == 0) | (obs[:, 4] == 1)).all()
    obs[:, 5:] = torch.where(obs[:, 4:5].bool(), obs[:, 5:], 0.)
    return obs


def numpy_gru(weights, prefix, inputs, previous):
    gi = inputs @ weights[prefix + '.weight_ih'].T + weights[prefix + '.bias_ih']
    gh = previous @ weights[prefix + '.weight_hh'].T + weights[prefix + '.bias_hh']
    ir, iz, inn = np.split(gi, 3, axis=1)
    hr, hz, hn = np.split(gh, 3, axis=1)
    reset = 1 / (1 + np.exp(-(ir + hr)))
    update = 1 / (1 + np.exp(-(iz + hz)))
    candidate = np.tanh(inn + reset * hn)
    return candidate + update * (previous - candidate)


class Replay:
    def __init__(self, weights):
        self.w = {k: v.detach().cpu().clone() for k, v in weights.items() if isinstance(v, torch.Tensor)}
        assert all(v.dtype == torch.float32 for k, v in self.w.items() if k.endswith(('weight', 'bias', 'weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')))
        self.nw = {k: v.numpy() for k, v in self.w.items()}
        self.decisions = self.writes = 0
        self.max_probability_error = self.max_state_error = 0.

    def _state_check(self, actual, independent):
        error = float(np.max(np.abs(actual.numpy() - independent)))
        assert np.isfinite(error) and error < STATE_LIMIT, 'local independent recurrent mathematics rejected'
        self.max_state_error = max(self.max_state_error, error)

    def gru(self, prefix, inputs, previous):
        result = torch._VF.gru_cell(inputs, previous, *(self.w[prefix + '.' + k] for k in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')))
        self._state_check(result, numpy_gru(self.nw, prefix, inputs.numpy(), previous.numpy()))
        return result

    def decide(self, values, h, z, previous=-1, reward=0., done=True):
        obs = canonical(values); n = len(obs)
        prev = torch.as_tensor(previous, dtype=torch.long).broadcast_to((n,))
        starts = prev == -1
        assert ((prev >= -1) & (prev < 6)).all()
        code = F.one_hot(prev.clamp_min(0), 6).float().masked_fill(starts[:, None], 0)
        rr = torch.as_tensor(reward, dtype=torch.float32).broadcast_to((n,))
        dd = torch.as_tensor(done, dtype=torch.bool).broadcast_to((n,))
        inputs = torch.cat((obs, code, rr[:, None], dd[:, None].float(), starts[:, None].float()), 1)
        before = h.masked_fill(starts[:, None], 0)
        next_h = self.gru('fast', inputs, before)
        gate = F.linear(torch.cat((next_h, z), 1), self.w['gate.weight'], self.w['gate.bias']).sigmoid()
        logits = F.linear(next_h + gate * F.linear(z, self.w['reinstate.weight']), self.w['actor.weight'], self.w['actor.bias'])
        probability = logits.softmax(-1)
        # All NumPy operations start at this replayer's computed pre-state.
        nh = numpy_gru(self.nw, 'fast', inputs.numpy(), before.numpy()); nz = z.numpy()
        ng = 1 / (1 + np.exp(-(np.concatenate((nh, nz), 1) @ self.nw['gate.weight'].T + self.nw['gate.bias'])))
        nl = (nh + ng * (nz @ self.nw['reinstate.weight'].T)) @ self.nw['actor.weight'].T + self.nw['actor.bias']
        ne = np.exp(nl - nl.max(1, keepdims=True)); npb = ne / ne.sum(1, keepdims=True)
        error = float(np.max(np.abs(probability.numpy() - npb)))
        assert np.isfinite(error) and error < PROBABILITY_LIMIT, 'local independent output mathematics rejected'
        self.max_probability_error = max(self.max_probability_error, error); self.decisions += n
        return probability, next_h

    def observe(self, values, action, reward, next_values, done, z):
        before = canonical(values); after = canonical(next_values); n = len(before)
        action = torch.as_tensor(action, dtype=torch.long).broadcast_to((n,))
        eligible = (action == 4) & after[:, 4].bool() & ((after[:, 2] == 0) | (after[:, 2] == 1))
        if not eligible.any(): return z
        rr = torch.as_tensor(reward, dtype=torch.float32).broadcast_to((n,))
        dd = torch.as_tensor(done, dtype=torch.bool).broadcast_to((n,))
        inputs = torch.cat((before, F.one_hot(action, 6).float(), rr[:, None], after, dd[:, None].float()), 1)
        update = self.gru('store.slow', inputs, z)
        self.writes += int(eligible.sum())
        return torch.where(eligible[:, None], update, z)

    def native_starts(self, episodes):
        assert episodes and all(len(e['bodies']) == 3 for e in episodes)
        z = torch.zeros(len(episodes), 8)
        for cycle in range(3):
            lengths = {len(e['bodies'][cycle]) for e in episodes}; assert len(lengths) == 1
            for tick in range(next(iter(lengths))):
                rows = [e['bodies'][cycle][tick] for e in episodes]
                z = self.observe([r['observation'] for r in rows], [r['action'] for r in rows], [r['reward'] for r in rows],
                    [r['next_observation'] for r in rows], [r['body_done'] for r in rows], z)
        return z

    def body(self, trace, prep, action_seed, horizon, inherited=True, public_labels=False, world=None):
        # These are computed from public source records, never saved neural states.
        z = self.native_starts([prep]); z = z if inherited else torch.zeros_like(z)
        h = torch.zeros(1, 32); previous = -1; rr = 0.; done = True
        world = world if world is not None else LineageEcology(seed=prep['seed'], config=LineageConfig(4, 8)).body(3)
        rng = torch.Generator().manual_seed(action_seed); physics = QualityConfig()
        cue = prep['bodies'][0][2]['next_observation']
        safe = (int(cue[2]) if cue[7] else 1 - int(cue[2])) if inherited else None
        tool = cue[6] if inherited else None
        feeding = repairs = inspections = bad = 0
        trace = iter(trace)
        for tick in range(horizon):
            row = next(trace); obs = world.observation(); position = obs.position
            if public_labels:
                if safe is None: label = 1 if position > 0 else 4
                elif (tool is not None and tool < .75) or obs.integrity < .8: label = 2 if position < .5 else 1 if position > .5 else 5
                elif position < float(safe): label = 2
                elif position > float(safe): label = 1
                else: label = 3 if obs.energy < .5 else 0
                assert row['label'] == label, 'public label rejected'
            probability, h = self.decide([obs.values()], h, z, previous, rr, done)
            assert torch.equal(probability[0], torch.tensor(row['probability'], dtype=torch.float32)), 'complete hardware probability replay rejected'
            chosen = int(torch.multinomial(probability, 1, generator=rng))
            assert chosen == row['action'], 'independently sampled action rejected'
            before_tool = world.snapshot()['tool']; effect = world.step(chosen); after_tool = world.snapshot()['tool']
            rr = (-1. if effect.terminated else .01) + .1 * (effect.after.energy - effect.before.energy) + .1 * (effect.after.integrity - effect.before.integrity)
            done = effect.terminated or tick + 1 == horizon
            expected = dict(observation=list(effect.before.values()), action=chosen, reward=rr, next_observation=list(effect.after.values()),
                body_done=done, terminated=effect.terminated, audit_tool_before=before_tool, audit_tool_after=after_tool)
            for key, value in expected.items(): assert row[key] == value, 'public physics rejected: ' + key
            if 'tick' in row: assert row['tick'] == tick
            z = self.observe([effect.before.values()], chosen, rr, [effect.after.values()], done, z)
            for key, actual in (('h', h), ('z', z)):
                assert torch.equal(actual[0], torch.tensor(row[key], dtype=torch.float32)), 'complete hardware state replay rejected: ' + key
            if chosen == 3 and tool is not None: tool = max(0., tool - physics.harvest_wear)
            if chosen == 5 and effect.after.position == .5 and tool is not None: tool = min(1., tool + physics.tool_repair)
            if chosen == 4 and effect.after.inspection_valid and effect.after.position in (0, 1):
                side = int(effect.after.position); safe = side if effect.after.resource_quality else 1 - side; tool = effect.after.tool_condition
            feeding += chosen == 3 and effect.after.energy > effect.before.energy
            repairs += chosen == 5 and after_tool > before_tool; inspections += chosen == 4
            bad += chosen == 3 and effect.after.integrity < effect.before.integrity - world.config.integrity_decay
            previous = chosen
            if done: break
        assert done and next(trace, None) is None, 'incomplete or trailing body stream'
        return dict(seed=prep['seed'], ticks=tick + 1, survived=not effect.terminated, feeding=feeding, repairs=repairs,
            inspections=inspections, bad_harvest=bad, energy=effect.after.energy, integrity=effect.after.integrity)

    def receipt(self):
        return dict(decisions=self.decisions, eligible_writes=self.writes, max_local_probability_error=self.max_probability_error,
            max_local_state_error=self.max_state_error, shared_engine='PyTorch CPU float32 GRU/linear/sampling operators',
            independent_scope='public causal routing, complete trajectory, NumPy local neural mathematics and public physics')
