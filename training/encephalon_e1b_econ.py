"""E1-B learner with only finite-world stock renewal changed."""
import copy
import hashlib

import numpy as np
import torch

from core.encephalon_agent import Agent
from core.encephalon_resources import ResourceWorld, Resources
from training import encephalon_e1b_contract as B
from training import encephalon_e1b_econ_contract as K
from training.encephalon_e1 import sample, tree_hash
from training.encephalon_e1b import Fit as BFit, physical, restore_world
from training.encephalon_learning import viability_reward
from training.run_encephalon_e0 import encoded


class Fit(BFit):
    def __init__(self, margin, width, lineage, config=None, development=False):
        if margin not in ("medium", "generous") or width not in K.WIDTHS:
            raise ValueError("only new medium/generous finite fits are trained")
        self.margin = margin
        self.renewal = K.MARGINS[margin]
        super().__init__(f"finite_{width}", lineage, B.CONFIG if config is None else config,
                         development)
        self.trace = hashlib.sha256(b"E1-B-ECON training chain").hexdigest()

    def new_world(self, lane=None):
        actual_lane = self.created if lane is None else lane
        world = super().new_world(lane)
        if actual_lane < self.config["original_lanes"]:
            return world
        assert isinstance(world, ResourceWorld) and world.mode == "finite"
        index = self.births[actual_lane] - 1
        seed = (self.base + actual_lane * self.config["lane_seed_stride"]
                + (index + actual_lane) % self.config["lane_seed_stride"])
        changed = ResourceWorld(seed=seed, mode="finite", energy=world.energy,
                                integrity=world.integrity,
                                resources=Resources(renewal=self.renewal))
        assert changed.repair_side == world.repair_side
        return changed

    def snapshot(self):
        return super().snapshot() | dict(version=K.VERSION, margin=self.margin,
                                         renewal_per_patch=self.renewal)

    @classmethod
    def restore(cls, packet):
        if packet["version"] != K.VERSION or packet["renewal_per_patch"] != K.MARGINS[packet["margin"]]:
            raise ValueError("incompatible E1-B-ECON checkpoint")
        width = packet["config"]["width"]
        fit = cls(packet["margin"], width, packet["lineage"], packet["config"], packet["development"])
        fit.agent.load_state_dict(packet["model"], strict=True)
        fit.optimizer.load_state_dict(packet["optimizer"])
        fit.worlds = [restore_world(w) for w in packet["worlds"]]
        fit.state = {k: v.clone() for k, v in packet["controller"].items()}
        fit.sampler.bit_generator.state = copy.deepcopy(packet["sampler"])
        fit.needs.bit_generator.state = copy.deepcopy(packet["needs"])
        torch.set_rng_state(packet["torch_rng"])
        fit.update, fit.created, fit.trace = packet["update"], packet["created"], packet["trace"]
        fit.births, fit.history = list(packet["births"]), copy.deepcopy(packet["history"])
        assert sum(fit.births) == fit.created
        assert all(not isinstance(w, ResourceWorld) or w.resources.renewal == fit.renewal
                   for w in fit.worlds)
        return fit


@torch.no_grad()
def evaluate(model, margin, width, lineage, profile, control, *, count=64,
             horizon=4096, base=None, sampler_seed=None):
    if margin not in K.MARGINS or width not in K.WIDTHS or profile not in K.PROFILES or control not in K.CONTROLS:
        raise ValueError("invalid endpoint identity")
    base = K.CONFIG["heldout_base"] if base is None else base
    sampler_seed = K.seed(lineage, profile) if sampler_seed is None else sampler_seed
    renewal = K.MARGINS[margin]
    agent = Agent("recurrent", width).double().eval()
    agent.load_state_dict(model, strict=True)
    energy, integrity = B.CONFIG["profiles"][profile]
    worlds = [ResourceWorld(seed=base+j, mode="finite", energy=energy,
                            integrity=integrity, repair_enabled=control != "repair_disabled",
                            resources=Resources(renewal=renewal)) for j in range(count)]
    initial = [w.snapshot() for w in worlds]
    sampler = np.random.Generator(np.random.PCG64(sampler_seed))
    state = agent.initial(count)
    actions, anchors = [], []
    counts = np.zeros((count, 6), dtype=np.int64)
    feeds, repairs, empty = [np.zeros(count, dtype=np.int64) for _ in range(3)]
    food = np.zeros((count, 2), dtype=np.int64)
    digest = hashlib.sha256()
    for tick in range(horizon):
        live = torch.tensor([w.viable() for w in worlds], dtype=torch.bool)
        if not live.any():
            break
        obs = torch.tensor([w.observation().values() for w in worlds], dtype=torch.float64)
        logits, value, _, following = agent(obs, state)
        if tick % 64 == 0:
            anchors.append(dict(tick=tick, h=following["h"][:4].tolist(),
                                logits=logits[:4].tolist(), value=value[:4].tolist()))
        action = sample(logits.softmax(-1), sampler)
        recorded = action.clone()
        recorded[~live] = -1
        rewards = []
        for j, world in enumerate(worlds):
            if not live[j]:
                rewards.append(0.)
                continue
            a = int(action[j]); old_e, old_i, old_p = world.energy, world.integrity, world.position
            step = world.step(a)
            counts[j, a] += 1
            if a == 3:
                gained = world.energy - max(0, old_e - 10)
                feeds[j] += gained > 0
                empty[j] += gained == 0
                if gained:
                    food[j, old_p // 4] += gained
            if a == 5:
                repairs[j] += world.integrity > max(0, old_i - 3)
            rewards.append(viability_reward(step))
        actions.append(recorded.tolist())
        digest.update(encoded([recorded.tolist(), [physical(w) for w in worlds]]) + b"\n")
        observed = agent.observe(following, action, torch.tensor(rewards, dtype=torch.float64))
        state = {key: torch.where(live[:, None] if observed[key].ndim == 2 else live,
                                  observed[key], state[key]) for key in state}
    return dict(version=K.VERSION, margin=margin, renewal_per_patch=renewal,
                width=width, lineage=lineage, profile=profile, control=control,
                sampler_seed=sampler_seed, initial=initial, horizon=horizon,
                actions=actions, anchors=anchors, action_counts=counts.tolist(),
                feeding=feeds.tolist(), repairs=repairs.tolist(), empty_feeds=empty.tolist(),
                food_energy=food.tolist(), ticks=[w.tick for w in worlds],
                survived=[w.viable() and w.tick == horizon for w in worlds],
                final=[w.snapshot() for w in worlds],
                controller={k: v.tolist() for k, v in state.items()},
                trace_sha256=digest.hexdigest())
