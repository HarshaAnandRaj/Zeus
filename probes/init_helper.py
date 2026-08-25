import torch

from common import device


def seeded_reset(model, noise=0.1):
    g = torch.Generator(device="cpu").manual_seed(1234)
    model.reset_state(noise=noise, generator=g)
