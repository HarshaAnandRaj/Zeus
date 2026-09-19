"""Own-action actor-critic mechanics; not an E1 functional training campaign."""
import torch


def viability_reward(step):
    return ((-1.0 if step.terminated else .01)
            + .1 * (step.after.energy - step.before.energy)
            + .1 * (step.after.integrity - step.before.integrity))


def discounted_returns(rewards, ended, bootstrap, gamma=.99):
    result = []
    future = bootstrap
    for t in reversed(range(len(rewards))):
        future = rewards[t] + gamma * future * (~ended[t]).to(rewards.dtype)
        result.append(future)
    return torch.stack(result[::-1])


def collect(agent, worlds, state, generator, steps=32):
    if not worlds or steps < 1:
        raise ValueError("nonempty rollout required")
    batch = {key: [] for key in ("log_probability", "value", "reward", "ended",
                                "alive", "entropy", "prediction", "target", "actions")}
    for _ in range(steps):
        alive = torch.tensor([w.viable() for w in worlds], dtype=torch.bool)
        observation = torch.tensor([w.observation().values() for w in worlds], dtype=torch.float32)
        logits, value, fused, next_state = agent(observation, state)
        log_all = logits.log_softmax(-1)
        probability = logits.softmax(-1)
        action = torch.multinomial(probability, 1, generator=generator).squeeze(1)
        rewards = []
        for i, world in enumerate(worlds):
            rewards.append(viability_reward(world.step(int(action[i]))) if alive[i] else 0.)
        reward = torch.tensor(rewards, dtype=torch.float32)
        ended = torch.tensor([not w.viable() for w in worlds], dtype=torch.bool)
        target = torch.tensor([w.observation().values()[:3] for w in worlds], dtype=torch.float32)
        values = dict(log_probability=log_all.gather(1, action[:, None]).squeeze(1), value=value,
                      reward=reward, ended=ended, alive=alive,
                      entropy=-(probability * log_all).sum(-1), prediction=agent.predict(fused, action),
                      target=target, actions=action)
        for key, item in values.items():
            batch[key].append(item)
        observed = agent.observe(next_state, action, reward)
        state = {key: torch.where(alive[:, None], observed[key], state[key])
                 if observed[key].ndim == 2 else torch.where(alive, observed[key], state[key])
                 for key in state}
    observation = torch.tensor([w.observation().values() for w in worlds], dtype=torch.float32)
    with torch.no_grad():
        _, bootstrap, _, _ = agent(observation, state)
        bootstrap *= torch.tensor([w.viable() for w in worlds])
    return {key: torch.stack(items) for key, items in batch.items()}, state, bootstrap


def loss(batch, bootstrap):
    returns = discounted_returns(batch["reward"], batch["ended"], bootstrap)
    weight = batch["alive"].to(torch.float32)
    count = weight.sum().clamp_min(1)
    advantage = returns - batch["value"]
    policy = -(batch["log_probability"] * advantage.detach() * weight).sum() / count
    value = (advantage.square() * weight).sum() / count
    prediction = ((batch["prediction"] - batch["target"]).square().mean(-1) * weight).sum() / count
    entropy = (batch["entropy"] * weight).sum() / count
    total = policy + .5 * value + .1 * prediction - .01 * entropy
    return total, dict(policy=float(policy.detach()), value=float(value.detach()),
                       prediction=float(prediction.detach()), entropy=float(entropy.detach()),
                       live_decisions=int(weight.sum()))

