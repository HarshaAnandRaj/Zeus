"""Own-action learning and unfiltered evaluation for the E1-B factorial comparison."""
import copy
import hashlib
import numpy as np
import torch

from core.encephalon_agent import Agent
from core.encephalon_world import World
from core.encephalon_resources import ResourceWorld, VERSION as RESOURCE_VERSION
from training import encephalon_e1b_contract as K
from training.encephalon_e1 import Fit as OriginalFit, collect, sample, tree_hash
from training.encephalon_learning import loss, viability_reward
from training.run_encephalon_e0 import encoded, physical as original_physical


def arm_config(arm, config):
    c = copy.deepcopy(config)
    mode, width = c["arms"][arm]
    c.update(arm=arm, training_mode=mode, width=width)
    return c


def make_world(ecology, seed, energy, integrity, repair=True):
    if ecology == "original":
        return World(seed=seed, full_visibility=True, energy=energy, integrity=integrity, repair_enabled=repair)
    return ResourceWorld(seed=seed, mode=ecology, energy=energy, integrity=integrity, repair_enabled=repair)


def restore_world(packet):
    return ResourceWorld.restore(packet) if packet["version"] == RESOURCE_VERSION else World.restore(packet)


def physical(w):
    if isinstance(w, World): return ["original", *original_physical(w)]
    l = w.ledger
    return ["resource", w.energy, w.integrity, w.position, w.tick, w.repair_side,
            *w.stocks, *l["food"], *l["supplied"], *l["overflow"], l["spent"], l["wear"], l["repaired"]]


class Fit(OriginalFit):
    def __init__(self, arm, lineage, config=None, development=False):
        c = arm_config(arm, K.CONFIG if config is None else config)
        self.arm, self.births = arm, [0] * c["batch"]
        super().__init__("recurrent", lineage, c, development)
        self.trace = hashlib.sha256(b"E1-B training chain").hexdigest()

    def new_world(self, lane=None):
        c = self.config
        if lane is None:
            assert self.created < c["batch"]
            lane = self.created
        profile = list(c["profiles"].values())[int(self.needs.integers(len(c["profiles"])))]
        index = self.births[lane]
        assert index < c["lane_seed_stride"], "world seed role exhausted"
        seed = self.base + lane * c["lane_seed_stride"] + (index + lane) % c["lane_seed_stride"]
        self.births[lane] += 1
        self.created += 1
        ecology = "original" if lane < c["original_lanes"] else c["training_mode"]
        return make_world(ecology, seed, *profile)

    def advance(self, checker=None):
        c = self.config
        batch, state, bootstrap = collect(self.agent, self.worlds, self.state, self.sampler, c["rollout"], checker)
        total, metrics = loss(batch, bootstrap)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradients = {name: sum(float(p.grad.square().sum()) for p in module.parameters()) ** .5
                     for name, module in self.agent.named_children()}
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), c["gradient_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in self.agent.parameters()):
            raise FloatingPointError("nonfinite learned parameter")
        self.update += 1
        self.state = self.agent.detach(state)
        metrics.update(update=self.update, total=float(total.detach()), gradient_norm=float(grad_norm),
                       module_gradient_norms=gradients,
                       original_live_decisions=int(batch["alive"][:, :c["original_lanes"]].sum()),
                       resource_live_decisions=int(batch["alive"][:, c["original_lanes"]:].sum()),
                       deaths=sum(not w.viable() for w in self.worlds),
                       censored=sum(w.viable() and w.tick >= c["training_horizon"] for w in self.worlds))
        self.trace = tree_hash([self.trace, batch["actions"], batch["reward"],
                               [w.snapshot() for w in self.worlds], metrics])
        self.history.append(metrics)
        fresh = self.agent.initial(c["batch"])
        for lane, world in enumerate(self.worlds):
            if not world.viable() or world.tick >= c["training_horizon"]:
                self.worlds[lane] = self.new_world(lane)
                for key in self.state: self.state[key][lane] = fresh[key][lane]
        return metrics

    def snapshot(self):
        return super().snapshot() | dict(version=K.VERSION, arm=self.arm, births=self.births.copy())

    @classmethod
    def restore(cls, packet):
        if packet["version"] != K.VERSION: raise ValueError("incompatible E1-B checkpoint")
        fit = cls(packet["arm"], packet["lineage"], packet["config"], packet["development"])
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
        return fit


@torch.no_grad()
def evaluate(model, arm, lineage, ecology, profile, control, *, config=None, development=False):
    c = K.CONFIG if config is None else config
    n, horizon = c["endpoint_bodies"], c["endpoint_horizon"]
    agent = Agent("recurrent", c["arms"][arm][1]).double()
    agent.load_state_dict(model, strict=True); agent.eval()
    e, i = c["profiles"][profile]
    base = c["development_base"] if development else c["heldout_base"]
    seed = K.endpoint_seed(c, lineage, ecology, profile, development)
    sampler = np.random.Generator(np.random.PCG64(seed))
    worlds = [make_world(ecology, base + j, e, i, control != "repair_disabled") for j in range(n)]
    initial = [w.snapshot() for w in worlds]
    state = agent.initial(n)
    actions, anchors, probes = [], [], []
    counts = np.zeros((n, 6), dtype=np.int64)
    feeds, repairs = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    food = np.zeros((n, 2), dtype=np.int64)
    crossing = np.zeros(n, dtype=np.int64); last_station = np.full(n, -1)
    departures = np.zeros((n, 3), dtype=np.int64)  # count, summed energy, summed integrity before departure
    empty_feeds = np.zeros(n, dtype=np.int64)
    digest = hashlib.sha256()
    for tick in range(horizon):
        alive = torch.tensor([w.viable() for w in worlds], dtype=torch.bool)
        if not alive.any(): break
        obs = torch.tensor([w.observation().values() for w in worlds], dtype=torch.float64)
        logits, value, _, following = agent(obs, state)
        if tick % 64 == 0:
            anchors.append(dict(tick=tick, h=following["h"][:4].tolist(), logits=logits[:4].tolist(), value=value[:4].tolist()))
        if control == "trained" and ecology == "finite" and tick in c["probe_ticks"]:
            altered = []
            for stock in (0., 1.):
                public = obs.clone(); public[:, 3] = stock
                altered.append(agent(public, state)[0].softmax(-1))
            for lane in range(min(c["probe_lanes"], n)):
                if alive[lane]:
                    probes.append(dict(tick=tick, lane=lane, empty=altered[0][lane].tolist(), full=altered[1][lane].tolist()))
        action = sample(logits.softmax(-1), sampler)
        recorded = action.clone(); recorded[~alive] = -1
        rewards = []
        for j, w in enumerate(worlds):
            if not alive[j]: rewards.append(0.); continue
            a = int(action[j]); old_e, old_i, old_p = w.energy, w.integrity, w.position
            step = w.step(a)
            counts[j, a] += 1
            if a == 3:
                gain = w.energy - max(0, old_e - 10)
                feeds[j] += gain > 0
                empty_feeds[j] += gain == 0
                if gain: food[j, old_p // 4] += gain
            if a == 5: repairs[j] += w.integrity > max(0, old_i - 3)
            if old_p != w.position:
                if old_p in (0, 4): departures[j] += (1, old_e, old_i)
                if w.position in (0, 4):
                    station = w.position // 4
                    crossing[j] += last_station[j] >= 0 and last_station[j] != station
                    last_station[j] = station
            rewards.append(viability_reward(step))
        actions.append(recorded.tolist())
        digest.update(encoded([recorded.tolist(), [physical(w) for w in worlds]]) + b"\n")
        observed = agent.observe(following, action, torch.tensor(rewards, dtype=torch.float64))
        state = {key: torch.where(alive[:, None] if observed[key].ndim == 2 else alive, observed[key], state[key]) for key in state}
    return dict(arm=arm, lineage=lineage, ecology=ecology, profile=profile, control=control,
                development=development, sampler_seed=seed, initial=initial, horizon=horizon,
                actions=actions, anchors=anchors, stock_probes=probes, action_counts=counts.tolist(),
                feeding=feeds.tolist(), repairs=repairs.tolist(), food_energy=food.tolist(),
                crossings=crossing.tolist(), departures=departures.tolist(), empty_feeds=empty_feeds.tolist(),
                ticks=[w.tick for w in worlds], survived=[w.viable() and w.tick == horizon for w in worlds],
                final=[w.snapshot() for w in worlds], controller={k: v.tolist() for k, v in state.items()},
                trace_sha256=digest.hexdigest())
