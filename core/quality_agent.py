"""Fresh learner adapter for the public quality-world interface."""
import hashlib
import json

import torch

from core.persistent_agent import PersistentAgent, AgentConfig
from core.persistent_session import PersistentSession
from core.lifetime_world_v2 import QualityObservation, INTERFACE_VERSION, QualityInterface

LEARNER_VERSION = 'quality-recurrent-learner-v1-20260909'


def model_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
    return digest.hexdigest()


class QualityAgent(PersistentAgent):
    observation_dim = 8

    def get_extra_state(self):
        return dict(learner=LEARNER_VERSION, interface=INTERFACE_VERSION,
                    actions=QualityInterface.action_names, sensors=QualityInterface.sensor_names,
                    hidden_size=self.config.hidden_size)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('incompatible learner/interface checkpoint')

    @staticmethod
    def canonical_observation(observation):
        if observation.shape[-1] != 8:
            raise ValueError('quality observation requires eight values')
        valid = observation[..., 4]
        if not ((valid == 0) | (valid == 1)).all():
            raise ValueError('inspection validity must be binary')
        return torch.cat((observation[..., :5],
                          torch.where(valid[..., None].bool(), observation[..., 5:],
                                      torch.zeros_like(observation[..., 5:]))), dim=-1)

    def step(self, observation, previous_action, state, starts):
        return super().step(self.canonical_observation(observation), previous_action, state, starts)

    def prediction_loss(self, prediction, target):
        target = self.canonical_observation(target.detach())
        mask = torch.cat((torch.ones_like(target[..., :5]),
                          target[..., 4:5].expand_as(target[..., 5:])), dim=-1)
        square = (prediction - target).square()
        # Each transition has equal weight regardless of whether inspection occurred.
        return ((square * mask).sum(-1) / mask.sum(-1)).mean()


class QualitySession(PersistentSession):
    def __init__(self, model: QualityAgent, *, fixed_weights=False):
        if not isinstance(model, QualityAgent):
            raise ValueError('quality session requires an explicitly versioned learner')
        super().__init__(model)
        self.fixed_weights = fixed_weights
        if fixed_weights:
            if any(p.requires_grad for p in model.parameters()):
                raise ValueError('evaluation model must have gradients disabled')
            self._fixed_hash = model_hash(model)

    def act(self, observation, *, generator, greedy=False):
        if not isinstance(observation, QualityObservation):
            raise ValueError('only a public QualityObservation can enter the session')
        result = super().act(observation.values(), generator=generator, greedy=greedy)
        if self.fixed_weights:
            self._history.clear()
        return result

    def record_outcome(self, observation, **kwargs):
        if not isinstance(observation, QualityObservation):
            raise ValueError('outcome must be a public QualityObservation')
        return super().record_outcome(observation.values(), **kwargs)

    def assert_update_ready(self):
        if self.fixed_weights:
            raise RuntimeError('fixed-weight evaluation cannot optimize')
        return super().assert_update_ready()

    def refresh_state(self):
        if self.fixed_weights:
            raise RuntimeError('history reconstruction is forbidden during evaluation')
        return super().refresh_state()

    def erase_history(self):
        if not self.fixed_weights or self._pending is not None:
            raise RuntimeError('history control requires an evaluation decision boundary')
        self._state = self.model.initial_state(1)
        self._history.clear()
        # Keep the actual previous action and pending current observation; this
        # control removes recurrent history, not the public one-step interface.

    def verify_fixed_weights(self):
        if not self.fixed_weights or model_hash(self.model) != self._fixed_hash:
            raise RuntimeError('evaluation weights changed')
