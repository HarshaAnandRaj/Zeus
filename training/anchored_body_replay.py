"""Candidate anchor replay with complete CPU state and local NumPy checks.

The public observation/write/body routes are inherited from the fixed replayer.
Neither production model methods nor recorded neural states advance this replay.
NumPy is a local arithmetic check from the computed previous state, not a claim
of long-trajectory equivalence between execution engines.
"""
import numpy as np
import torch
from torch.nn import functional as F

from training.dual_body_replay import (
    Replay as PublicReplay, canonical, numpy_gru, PROBABILITY_LIMIT,
)

VERSION = 'anchored-public-body-replay-v1-20260913'
MODEL_VERSION = 'anchored-body-state-candidate-v1-20260913'
STORE_VERSION = 'protected-lineage-agent-v1-20260912'
MODES = ('current', 'recurrent')


def numpy_anchor(weights, inputs, recurrent, mode):
    """Independent local bounded anchor, with the campaign mode explicit."""
    assert mode in MODES, 'undeclared anchor mode'
    evidence = inputs[:, :8] if mode == 'current' else recurrent[:, :8]
    adjustment = np.tanh(evidence @ weights['anchor.weight'].T + weights['anchor.bias'])
    return recurrent + (1 - np.abs(recurrent)) * adjustment


class Replay(PublicReplay):
    def __init__(self, weights, *, expected_mode=None, expected_parent_hash=None):
        extra = weights.get('_extra_state')
        assert isinstance(extra, dict) and set(extra) == {'version', 'parent_hash', 'mode'}, 'anchor checkpoint metadata rejected'
        assert extra['version'] == MODEL_VERSION and extra['mode'] in MODES, 'anchor checkpoint version/mode rejected'
        assert isinstance(extra['parent_hash'], str) and extra['parent_hash'], 'anchor checkpoint parent identity rejected'
        if expected_mode is not None:
            assert expected_mode in MODES and extra['mode'] == expected_mode, 'campaign anchor mode rejected'
        if expected_parent_hash is not None:
            assert extra['parent_hash'] == expected_parent_hash, 'campaign anchor parent rejected'
        assert weights.get('store._extra_state') == dict(
            version=STORE_VERSION, fast_size=32, slow_size=8, mode='protected',
        ), 'unsupported consolidation checkpoint'
        self.mode = extra['mode']; self.parent_hash = extra['parent_hash']
        super().__init__(weights)
        shapes = {
            'anchor.weight': (32, 8), 'anchor.bias': (32,),
            'fast.weight_ih': (96, 17), 'fast.weight_hh': (96, 32),
            'fast.bias_ih': (96,), 'fast.bias_hh': (96,),
            'gate.weight': (32, 40), 'gate.bias': (32,),
            'reinstate.weight': (32, 8), 'actor.weight': (6, 32), 'actor.bias': (6,),
            'store.slow.weight_ih': (24, 24), 'store.slow.weight_hh': (24, 8),
            'store.slow.bias_ih': (24,), 'store.slow.bias_hh': (24,),
        }
        for key, shape in shapes.items():
            assert key in self.w and self.w[key].shape == shape, 'checkpoint tensor shape rejected: ' + key
        assert all(torch.isfinite(value).all() for value in self.w.values()), 'nonfinite checkpoint rejected'

    @torch.no_grad()
    def decide(self, values, h, z, previous=-1, reward=0., done=True):
        obs = canonical(values); n = len(obs)
        for value, width in ((h, 32), (z, 8)):
            assert value.shape == (n, width) and value.dtype == torch.float32 and value.device.type == 'cpu'
            assert torch.isfinite(value).all(), 'nonfinite input state rejected'
        prev = torch.as_tensor(previous, dtype=torch.long).broadcast_to((n,))
        starts = prev == -1
        assert ((prev >= -1) & (prev < 6)).all()
        code = F.one_hot(prev.clamp_min(0), 6).float().masked_fill(starts[:, None], 0)
        rr = torch.as_tensor(reward, dtype=torch.float32).broadcast_to((n,))
        dd = torch.as_tensor(done, dtype=torch.bool).broadcast_to((n,))
        assert torch.isfinite(rr).all()
        inputs = torch.cat((obs, code, rr[:, None], dd[:, None].float(), starts[:, None].float()), 1)
        before = h.masked_fill(starts[:, None], 0)
        recurrent = self.gru('fast', inputs, before)
        evidence = inputs[:, :8] if self.mode == 'current' else recurrent[:, :8]
        adjustment = F.linear(evidence, self.w['anchor.weight'], self.w['anchor.bias']).tanh()
        next_h = recurrent + (1 - recurrent.abs()) * adjustment
        gate = F.linear(torch.cat((next_h, z), 1), self.w['gate.weight'], self.w['gate.bias']).sigmoid()
        logits = F.linear(next_h + gate * F.linear(z, self.w['reinstate.weight']), self.w['actor.weight'], self.w['actor.bias'])
        probability = logits.softmax(-1)

        nr = numpy_gru(self.nw, 'fast', inputs.numpy(), before.numpy())
        nh = numpy_anchor(self.nw, inputs.numpy(), nr, self.mode)
        self._state_check(next_h, nh)
        nz = z.numpy()
        ng = 1 / (1 + np.exp(-(np.concatenate((nh, nz), 1) @ self.nw['gate.weight'].T + self.nw['gate.bias'])))
        nl = (nh + ng * (nz @ self.nw['reinstate.weight'].T)) @ self.nw['actor.weight'].T + self.nw['actor.bias']
        ne = np.exp(nl - nl.max(1, keepdims=True)); npb = ne / ne.sum(1, keepdims=True)
        error = float(np.max(np.abs(probability.numpy() - npb)))
        assert np.isfinite(error) and error < PROBABILITY_LIMIT, 'local independent output mathematics rejected'
        self.max_probability_error = max(self.max_probability_error, error); self.decisions += n
        return probability, next_h

    def receipt(self):
        return super().receipt() | dict(replay_version=VERSION, anchor_mode=self.mode,
            parent_hash=self.parent_hash, anchor_math='r + (1 - abs(r)) * tanh(W evidence + b)')
