"""E1-D learning with a persistent intention channel (intent arm) or plain Agent.

Arm "intent" uses AgentIntent (4 unlabeled slots, entropy weight 0); arm
"nointent" uses the frozen Agent (entropy weight 0, fresh fit: an E1-C
entropy00 replication by design). Per tick the RNG draws actions for all
lanes first, then intents for all lanes (intent arm only); eval conditions
transform only the HELD intent, never the draw stream, so replay stays exact.
"""
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch

from core.encephalon_agent import Agent
from core.encephalon_agent_intent import AgentIntent, INTENT_SLOTS
from core.encephalon_world import World
from training import encephalon_e1d_contract as K
from training.encephalon_learning import discounted_returns, viability_reward
from training.run_encephalon_e0 import encoded, physical

ROUTE = "recurrent"
INTENT_PERM = (2, 0, 3, 1)


def deterministic():
    torch.set_num_threads(K.CONFIG["threads"])
    torch.use_deterministic_algorithms(True)


def tree_hash(value):
    """Hash values, not torch archive timestamps or object identities."""
    digest = hashlib.sha256()

    def visit(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous().numpy()
            digest.update(encoded(["tensor", str(a.dtype), list(a.shape)]))
            digest.update(a.tobytes())
        elif isinstance(v, np.ndarray):
            digest.update(encoded(["array", str(v.dtype), list(v.shape)]))
            digest.update(v.tobytes())
        elif isinstance(v, dict):
            digest.update(b"dict")
            for key in sorted(v, key=str):
                visit(key); visit(v[key])
        elif isinstance(v, (list, tuple)):
            digest.update(encoded([type(v).__name__, len(v)]))
            for item in v: visit(item)
        else:
            digest.update(encoded(v))
        digest.update(b";")
    visit(value)
    return digest.hexdigest()


def sample(probability, generator):
    """Unmodified categorical distribution, one independent uniform per lane."""
    p = probability.detach().cpu().numpy()
    cumulative = np.cumsum(p, axis=-1)
    cumulative[:, -1] = 1.
    uniforms = generator.random(len(p))
    actions = (uniforms[:, None] >= cumulative).sum(axis=-1)
    return torch.from_numpy(actions.astype(np.int64))


def has_intent(agent):
    return isinstance(agent, AgentIntent)


def collect(agent, worlds, state, generator, steps, intent_condition="live", checker=None):
    if intent_condition not in ("live", "clamped", "permuted"):
        raise ValueError("unknown intent condition")
    fields = ("log_probability", "intent_log_probability", "value", "reward", "ended", "alive", "entropy",
              "prediction", "target", "actions", "intents")
    batch = {key: [] for key in fields}
    for _ in range(steps):
        alive = torch.tensor([w.viable() for w in worlds], dtype=torch.bool)
        obs = torch.tensor([w.observation().values() for w in worlds], dtype=torch.float64)
        if has_intent(agent):
            logits, intent_logits, value, fused, following = agent(obs, state)
        else:
            logits, value, fused, following = agent(obs, state)
            intent_logits = None
        if checker is not None:
            checker(agent, obs, state, logits, value, fused, following)
        probability, log_all = logits.softmax(-1), logits.log_softmax(-1)
        action = sample(probability, generator)
        if intent_logits is not None:
            intent_log_all = intent_logits.log_softmax(-1)
            intent = sample(intent_logits.softmax(-1), generator)
            intent_lp = intent_log_all.gather(1, intent[:, None]).squeeze(1)
            held = intent.clone()
            if intent_condition == "clamped":
                held = torch.zeros_like(intent)
            elif intent_condition == "permuted":
                held = torch.tensor([INTENT_PERM[int(i)] for i in intent], dtype=torch.long)
        else:
            intent = held = torch.full(action.shape, -1, dtype=torch.long)
            intent_lp = torch.zeros(action.shape, dtype=torch.float64)
        rewards = [viability_reward(w.step(int(action[i]))) if alive[i] else 0.
                   for i, w in enumerate(worlds)]
        reward = torch.tensor(rewards, dtype=torch.float64)
        ended = torch.tensor([not w.viable() for w in worlds])
        target = torch.tensor([w.observation().values()[:3] for w in worlds], dtype=torch.float64)
        row = dict(log_probability=log_all.gather(1, action[:, None]).squeeze(1),
                   intent_log_probability=intent_lp,
                   value=value, reward=reward, ended=ended, alive=alive,
                   entropy=-(probability * log_all).sum(-1),
                   prediction=agent.predict(fused, action), target=target, actions=action,
                   intents=intent)
        for key in fields: batch[key].append(row[key])
        if has_intent(agent):
            observed = agent.observe(following, action, reward, held)
        else:
            observed = agent.observe(following, action, reward)
        state = {key: torch.where(alive[:, None] if observed[key].ndim == 2 else alive,
                                  observed[key], state[key]) for key in state}
    obs = torch.tensor([w.observation().values() for w in worlds], dtype=torch.float64)
    with torch.no_grad():
        out = agent(obs, state)
        bootstrap = out[2] if has_intent(agent) else out[1]
        bootstrap = bootstrap * torch.tensor([w.viable() for w in worlds])
    return {key: torch.stack(items) for key, items in batch.items()}, state, bootstrap


def loss_d(batch, bootstrap, entropy_weight):
    """E1 actor-critic formula with parameterized entropy coefficient."""
    returns = discounted_returns(batch["reward"], batch["ended"], bootstrap)
    weight = batch["alive"].to(torch.float64)
    count = weight.sum().clamp_min(1)
    advantage = returns - batch["value"]
    joint_log_prob = batch["log_probability"] + batch["intent_log_probability"]
    policy = -(joint_log_prob * advantage.detach() * weight).sum() / count
    value = (advantage.square() * weight).sum() / count
    prediction = ((batch["prediction"] - batch["target"]).square().mean(-1) * weight).sum() / count
    entropy = (batch["entropy"] * weight).sum() / count
    total = policy + .5 * value + .1 * prediction - entropy_weight * entropy
    return total, dict(policy=float(policy.detach()), value=float(value.detach()),
                       prediction=float(prediction.detach()), entropy=float(entropy.detach()),
                       live_decisions=int(weight.sum()))


def make_agent(arm, width):
    if arm == "intent":
        return AgentIntent("recurrent", width).double()
    if arm == "nointent":
        return Agent("recurrent", width).double()
    raise ValueError(f"unknown E1-D arm {arm}")


class Fit:
    def __init__(self, arm, lineage, config=None, development=False):
        self.config = copy.deepcopy(K.CONFIG if config is None else config)
        c = self.config
        if arm not in c["arms"]: raise ValueError(f"unknown E1-D arm {arm}")
        self.arm, self.lineage, self.development = arm, lineage, development
        base = c["development_base"] if development else c["train_base"]
        self.base = base
        torch.manual_seed(base + c["initialization_offset"] + lineage)
        self.agent = make_agent(arm, c["width"])
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=c["learning_rate"],
                                          betas=tuple(c["adam_betas"]), eps=c["adam_eps"])
        self.sampler = np.random.Generator(np.random.PCG64(base + c["sampling_offset"] + lineage))
        self.needs = np.random.Generator(np.random.PCG64(base + c["needs_offset"] + lineage))
        self.created = 0
        self.worlds = [self.new_world() for _ in range(c["batch"])]
        self.state = self.agent.initial(c["batch"])
        self.update = 0
        self.history = []
        self.trace = hashlib.sha256(b"E1-D training chain").hexdigest()

    def new_world(self):
        profile = list(self.config["profiles"].values())[int(self.needs.integers(3))]
        seed = self.base + self.created % self.config["world_seed_count"]
        self.created += 1
        return World(seed=seed, full_visibility=True, energy=profile[0], integrity=profile[1])

    def advance(self, checker=None):
        c = self.config
        batch, state, bootstrap = collect(self.agent, self.worlds, self.state, self.sampler,
                                          c["rollout"], "live", checker)
        total, metrics = loss_d(batch, bootstrap, c["entropy_weight"])
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradients = {}
        for name, module in self.agent.named_children():
            gradients[name] = sum(float(p.grad.square().sum()) for p in module.parameters()) ** .5
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), c["gradient_clip"],
                                                   error_if_nonfinite=True)
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in self.agent.parameters()):
            raise FloatingPointError("nonfinite learned parameter")
        self.update += 1
        self.state = self.agent.detach(state)
        metrics.update(update=self.update, total=float(total.detach()),
                       gradient_norm=float(grad_norm), module_gradient_norms=gradients,
                       deaths=sum(not w.viable() for w in self.worlds),
                       censored=sum(w.viable() and w.tick >= c["training_horizon"] for w in self.worlds))
        if has_intent(self.agent):
            import math as _math
            counts = batch["intents"][batch["alive"]].bincount(minlength=c["intent_slots"]).double()
            probs = counts / counts.sum().clamp_min(1)
            metrics["intent_bits"] = float(-(probs * (probs + 1e-12).log()).sum())
        self.trace = tree_hash([self.trace, batch["actions"], batch["intents"], batch["reward"],
                               [w.snapshot() for w in self.worlds], metrics])
        self.history.append(metrics)
        fresh = self.agent.initial(c["batch"])
        for i, world in enumerate(self.worlds):
            if not world.viable() or world.tick >= c["training_horizon"]:
                self.worlds[i] = self.new_world()
                for key in self.state: self.state[key][i] = fresh[key][i]
        return metrics

    def snapshot(self):
        return dict(version=K.VERSION, config=self.config, arm=self.arm, lineage=self.lineage,
                    development=self.development, update=self.update, created=self.created,
                    model=copy.deepcopy(self.agent.state_dict()),
                    optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    worlds=[w.snapshot() for w in self.worlds],
                    controller={k: v.clone() for k, v in self.state.items()},
                    sampler=copy.deepcopy(self.sampler.bit_generator.state),
                    needs=copy.deepcopy(self.needs.bit_generator.state),
                    torch_rng=torch.get_rng_state().clone(), trace=self.trace,
                    history=copy.deepcopy(self.history))

    @classmethod
    def restore(cls, packet):
        if packet["version"] != K.VERSION:
            raise ValueError("incompatible checkpoint")
        fit = cls(packet["arm"], packet["lineage"], packet["config"], packet["development"])
        fit.agent.load_state_dict(packet["model"], strict=True)
        fit.optimizer.load_state_dict(packet["optimizer"])
        fit.worlds = [World.restore(w) for w in packet["worlds"]]
        fit.state = {k: v.clone() for k, v in packet["controller"].items()}
        fit.sampler.bit_generator.state = packet["sampler"]
        fit.needs.bit_generator.state = packet["needs"]
        torch.set_rng_state(packet["torch_rng"])
        fit.update, fit.created, fit.trace = packet["update"], packet["created"], packet["trace"]
        fit.history = copy.deepcopy(packet["history"])
        return fit


def write_checkpoint(path, packet):
    path = Path(path)
    temporary = path.with_suffix(".pending")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("xb") as stream:
        torch.save(dict(payload=packet, sha256=tree_hash(packet)), stream)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.rename(path)


def read_checkpoint(path):
    envelope = torch.load(path, map_location="cpu", weights_only=False)
    if set(envelope) != {"payload", "sha256"} or tree_hash(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("checkpoint content hash mismatch")
    return envelope["payload"]


@torch.no_grad()
def evaluate(model, arm, lineage, profile, control, intent_condition="live",
             *, count=None, horizon=None, development=False):
    c = K.CONFIG
    count = c["endpoint_bodies"] if count is None else count
    horizon = c["endpoint_horizon"] if horizon is None else horizon
    agent = make_agent(arm, c["width"])
    agent.load_state_dict(model, strict=True)
    agent.eval()
    e, i = c["profiles"][profile]
    base = c["development_base"] if development else c["heldout_base"]
    seed = base + c["endpoint_sampling_offset"] + lineage * 10 + list(c["profiles"]).index(profile)
    sampler = np.random.Generator(np.random.PCG64(seed))
    worlds = [World(seed=base + j, full_visibility=True, energy=e, integrity=i,
                    repair_enabled=control != "repair_disabled") for j in range(count)]
    initial = [w.snapshot() for w in worlds]
    state = agent.initial(count)
    actions, intents, anchors, counts = [], [], [], np.zeros((count, 6), dtype=np.int64)
    intent_counts = np.zeros((count, c["intent_slots"]), dtype=np.int64)
    feeding, repairs = np.zeros(count, dtype=np.int64), np.zeros(count, dtype=np.int64)
    digest = hashlib.sha256()
    for _ in range(horizon):
        alive = torch.tensor([w.viable() for w in worlds], dtype=torch.bool)
        if not alive.any(): break
        obs = torch.tensor([w.observation().values() for w in worlds], dtype=torch.float64)
        if has_intent(agent):
            logits, intent_logits, value, _, following = agent(obs, state)
        else:
            logits, value, _, following = agent(obs, state)
        if len(actions) % 64 == 0:
            anchors.append(dict(tick=len(actions), h=following["h"][:4].tolist(),
                                logits=logits[:4].tolist(), value=value[:4].tolist()))
        action = sample(logits.softmax(-1), sampler)
        recorded = action.clone(); recorded[~alive] = -1
        if has_intent(agent):
            intent = sample(intent_logits.softmax(-1), sampler)
            held = intent.clone()
            if intent_condition == "clamped":
                held = torch.zeros_like(intent)
            elif intent_condition == "permuted":
                held = torch.tensor([INTENT_PERM[int(v)] for v in intent], dtype=torch.long)
            logged = intent.clone(); logged[~alive] = -1
            intents.append(logged.tolist())
            for j in range(count):
                if alive[j]: intent_counts[j, int(intent[j])] += 1
        else:
            intent = held = torch.full(action.shape, -1, dtype=torch.long)
            intents.append(intent.tolist())
        rewards = []
        for j, world in enumerate(worlds):
            if alive[j]:
                step = world.step(int(action[j]))
                counts[j, int(action[j])] += 1
                feeding[j] += step.after.energy > step.before.energy
                repairs[j] += step.after.integrity > step.before.integrity
                rewards.append(viability_reward(step))
            else: rewards.append(0.)
        actions.append(recorded.tolist())
        digest.update(encoded([recorded.tolist(), [physical(w) for w in worlds]]) + b"\n")
        if has_intent(agent):
            observed = agent.observe(following, action, torch.tensor(rewards, dtype=torch.float64), held)
        else:
            observed = agent.observe(following, action, torch.tensor(rewards, dtype=torch.float64))
        state = {key: torch.where(alive[:, None] if observed[key].ndim == 2 else alive,
                                  observed[key], state[key]) for key in state}
    return dict(arm=arm, route=ROUTE, lineage=lineage, profile=profile, control=control,
                intent_condition=intent_condition, development=development, sampler_seed=seed,
                initial=initial, horizon=horizon, actions=actions, intents=intents, anchors=anchors,
                action_counts=counts.tolist(), intent_counts=intent_counts.tolist(),
                feeding=feeding.tolist(), repairs=repairs.tolist(), ticks=[w.tick for w in worlds],
                survived=[w.viable() and w.tick == horizon for w in worlds],
                final=[w.snapshot() for w in worlds], controller={k: v.tolist() for k, v in state.items()},
                trace_sha256=digest.hexdigest())
