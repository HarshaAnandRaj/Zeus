"""Small recurrent actor with an unlabeled persistent intention channel.

Mirrors the frozen Agent mechanics exactly, except: each tick the policy also
samples a 4-way intention from its own head, and last tick's intention (one-hot)
enters next tick's GRU input (17 -> 21 dims). Slots carry no prescribed meaning
and no direct reward; whatever they mean, the policy must learn it. Frozen
Agent/World sources are untouched.
"""
import torch
from torch import nn
from torch.nn import functional as F


VERSION = "encephalon-agent-intent-v1-20260921"
INTENT_SLOTS = 4


class AgentIntent(nn.Module):
    def __init__(self, sensory_source="recurrent", width=32):
        super().__init__()
        if sensory_source != "recurrent" or width < 9:
            raise ValueError("E1-D uses the recurrent route at width >= 9")
        self.sensory_source, self.width = sensory_source, width
        self.context = nn.GRUCell(17 + INTENT_SLOTS, width)
        self.sense = nn.Linear(9, width)
        self.gate = nn.Linear(width, width)
        self.actor = nn.Linear(width, 6)
        self.intent = nn.Linear(width, INTENT_SLOTS)
        self.value = nn.Linear(width, 1)
        self.consequence = nn.Linear(width + 6, 3)

    def initial(self, batch):
        param = next(self.parameters())
        return dict(h=param.new_zeros(batch, self.width),
                    previous=torch.full((batch,), -1, dtype=torch.long, device=param.device),
                    reward=param.new_zeros(batch),
                    intent=torch.zeros((batch,), dtype=torch.long, device=param.device))

    def forward(self, observations, state):
        if observations.ndim != 2 or observations.shape[1] != 9:
            raise ValueError("nine public sensors required")
        start = state["previous"] < 0
        previous = F.one_hot(state["previous"].clamp_min(0), 6).to(observations.dtype)
        previous = previous.masked_fill(start[:, None], 0)
        held = F.one_hot(state["intent"], INTENT_SLOTS).to(observations.dtype)
        held = held.masked_fill(start[:, None], 0)
        inputs = torch.cat((observations, previous, state["reward"][:, None],
                            start[:, None].to(observations.dtype), held), -1)
        context = self.context(inputs, state["h"])
        sense_input = context[:, :9]
        fused = context + self.gate(context).sigmoid() * self.sense(sense_input).tanh()
        return (self.actor(fused), self.intent(fused), self.value(fused).squeeze(-1),
                fused, state | dict(h=context))

    def predict(self, fused, action):
        encoded = F.one_hot(action, 6).to(fused.dtype)
        return self.consequence(torch.cat((fused, encoded), -1))

    @staticmethod
    def observe(state, action, reward, intent):
        return state | dict(previous=action, reward=reward, intent=intent)

    @staticmethod
    def detach(state):
        return {name: value.detach() for name, value in state.items()}
