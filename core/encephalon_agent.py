"""Small fresh recurrent actor: current sensing, context, value and consequence heads."""
import torch
from torch import nn
from torch.nn import functional as F


VERSION = "encephalon-agent-mechanics-v1-20260919"


class Agent(nn.Module):
    def __init__(self, sensory_source="observation", width=32):
        super().__init__()
        if sensory_source not in ("observation", "recurrent") or width < 9:
            raise ValueError("invalid sensory route or width")
        self.sensory_source, self.width = sensory_source, width
        self.context = nn.GRUCell(17, width)
        self.sense = nn.Linear(9, width)
        self.gate = nn.Linear(width, width)
        self.actor = nn.Linear(width, 6)
        self.value = nn.Linear(width, 1)
        self.consequence = nn.Linear(width + 6, 3)

    def initial(self, batch):
        param = next(self.parameters())
        return dict(h=param.new_zeros(batch, self.width),
                    previous=torch.full((batch,), -1, dtype=torch.long, device=param.device),
                    reward=param.new_zeros(batch))

    def forward(self, observations, state):
        if observations.ndim != 2 or observations.shape[1] != 9:
            raise ValueError("nine public sensors required")
        start = state["previous"] < 0
        previous = F.one_hot(state["previous"].clamp_min(0), 6).to(observations.dtype)
        previous = previous.masked_fill(start[:, None], 0)
        inputs = torch.cat((observations, previous, state["reward"][:, None],
                            start[:, None].to(observations.dtype)), -1)
        context = self.context(inputs, state["h"])
        sense_input = observations if self.sensory_source == "observation" else context[:, :9]
        fused = context + self.gate(context).sigmoid() * self.sense(sense_input).tanh()
        return self.actor(fused), self.value(fused).squeeze(-1), fused, state | dict(h=context)

    def predict(self, fused, action):
        encoded = F.one_hot(action, 6).to(fused.dtype)
        return self.consequence(torch.cat((fused, encoded), -1))

    @staticmethod
    def observe(state, action, reward):
        return state | dict(previous=action, reward=reward)

    @staticmethod
    def detach(state):
        return {name: value.detach() for name, value in state.items()}

