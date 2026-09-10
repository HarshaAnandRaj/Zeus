"""Historical replay and isolated update decomposition; no new policy campaign."""
import copy
import gzip
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from core.quality_agent import QualityAgent, QualitySession
from core.lifetime_world_v2 import QualityWorld, QualityInterface
from training.persistent_learning import SequenceBatch, sequence_loss, update_segment
from training import run_ql2 as R
from training import ql2_contract as K


@torch.no_grad()
def policy(model, batch):
    state = batch.initial_state
    output = []
    for t in range(len(batch.actions)):
        result = model.step(batch.observation[t], batch.previous_action[t], state, batch.starts[t])
        state = result.state
        output.append(result.logits.softmax(-1)[0])
    return torch.stack(output)


def branch(model, optimizer, batch, mode):
    clone = copy.deepcopy(model)
    opt = torch.optim.AdamW(clone.parameters(), lr=.0003, weight_decay=.01, foreach=False, fused=False)
    opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    losses = sequence_loss(clone, batch, K.SETTINGS)
    opt.zero_grad(set_to_none=True)
    losses['total'].backward()
    # Keep every parameter participating so momentum/weight decay are matched.
    if mode == 'actor':
        gradients = torch.autograd.grad(sequence_loss(clone, batch, K.SETTINGS)['actor'],
                                       tuple(clone.parameters()), allow_unused=True)
        for p, g in zip(clone.parameters(), gradients):
            p.grad = torch.zeros_like(p) if g is None else g.clone()
    elif mode == 'momentum_only':
        for p in clone.parameters():
            p.grad.zero_()
    torch.nn.utils.clip_grad_norm_(clone.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return policy(clone, batch)


def diagnose(model, optimizer, batch, consequences, categories):
    losses = sequence_loss(model, batch, K.SETTINGS)
    vectors = {}
    for name, weight in [('actor', 1.), ('value', .5), ('prediction', 1.), ('entropy', -.02)]:
        gradients = torch.autograd.grad(losses[name] * weight,
                                       tuple(model.recurrence.parameters()), retain_graph=True)
        vectors[name] = torch.cat([g.flatten() for g in gradients])
    actor = vectors['actor']; auxiliary = sum(vectors[k] for k in ('value', 'prediction', 'entropy'))
    cosine = float(torch.dot(actor, auxiliary) / (actor.norm() * auxiliary.norm()).clamp_min(1e-20))
    before = policy(model, batch)
    policies = {mode: branch(model, optimizer, batch, mode)
                for mode in ('joint', 'actor', 'momentum_only')}
    rewards = torch.tensor(consequences, dtype=before.dtype)
    immediate_advantage = rewards.gather(1, batch.actions)[:, 0] - (before * rewards).sum(-1)
    advantages = losses['advantages'].detach()[:, 0]
    rows = []
    for t, category in enumerate(categories):
        chosen = int(batch.actions[t, 0])
        rows.append(dict(category=category, reward=float(batch.rewards[t, 0]),
            advantage=float(advantages[t]), immediate_advantage=float(immediate_advantage[t]),
            chosen_probability_delta={k: float(p[t, chosen] - before[t, chosen]) for k, p in policies.items()},
            immediate_expected_reward_delta={k: float(((p[t] - before[t]) * rewards[t]).sum()) for k, p in policies.items()}))
    return dict(steps=rows, weighted_core_norms={k: float(v.norm()) for k, v in vectors.items()},
                actor_auxiliary_cosine=cosine)


def run(trial, arm):
    directory = R.OUT / f'training/{trial}_{arm}_a'
    payload, completion = R.checked_checkpoint(directory)
    model = QualityAgent(); model.load_state_dict(payload['initial'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01, foreach=False, fused=False)
    generator = torch.Generator().manual_seed(K.INITIALIZATIONS[trial] + 1000)
    diagnostics = []; all_updates = 0; all_steps = 0
    with gzip.open(directory / 'trace.jsonl.gz', 'rt') as stream:
        events = iter(json.loads(line) for line in stream)
        for entry in events:
            assert entry['kind'] == 'start'
            phase = entry['phase']; episode = entry['episode']
            world = QualityWorld.restore(entry['world']); interface = QualityInterface(world)
            session = QualitySession(model); session.begin_episode(); update_index = 0
            done = False
            while not done:
                selected = int(model.revision) % 16 == 0
                rows = []; consequences = []; categories = []
                for _ in range(K.CHUNK):
                    event = next(events); assert event['kind'] == 'step'
                    action, out = session.act(interface.observation(), generator=generator)
                    assert action == event['action']
                    assert torch.equal(out.logits, torch.tensor(event['logits']))
                    if selected:
                        snap = world.snapshot()
                        consequences.append([K.reward(QualityWorld.restore(snap).step(a)) for a in range(6)])
                    effect = interface.step(action)
                    assert K.reward(effect) == event['reward']
                    rows.append(session.record_outcome(effect.after, reward=event['reward'],
                        terminated=event['terminated'], truncated=event['truncated']))
                    category = ('feeding' if action == 3 and effect.after.energy > effect.before.energy
                                else 'harmful_patch_harvest' if action == 3 and world.snapshot()['position'] in (0, 4)
                                and effect.after.integrity < effect.before.integrity - .00201 else 'other')
                    categories.append(category); all_steps += 1
                    if event['terminated'] or event['truncated']:
                        done = True; break
                batch = SequenceBatch.from_transitions(rows)
                if selected:
                    diagnostics.append(dict(trial=trial, arm=arm, phase=phase, episode=episode,
                        revision=int(model.revision), **diagnose(model, optimizer, batch, consequences, categories)))
                metrics = update_segment(session, optimizer, batch, K.SETTINGS, max_grad_norm=1.)
                assert metrics == payload['episodes'][episode]['updates'][update_index]
                update_index += 1; all_updates += 1
                if not done:
                    session.refresh_state()
            end = next(events); assert end['kind'] == 'end'
            assert json.loads(json.dumps(world.snapshot())) == end['episode']['final_world']
    assert R.Q.tree_hash(model.state_dict()) == R.Q.tree_hash(payload['final'])
    assert R.Q.tree_hash(optimizer.state_dict()) == R.Q.tree_hash(payload['optimizer'])
    assert torch.equal(generator.get_state(), payload['sampling_rng'])
    print(f'Exact replay: {trial} {arm}: {all_updates} updates, {all_steps} steps', flush=True)
    return dict(trial=trial, arm=arm, updates=all_updates, steps=all_steps,
                source_sha=R.sha(Path(__file__)),
                checkpoint_sha=completion['checkpoint_sha'], exact_final=True, diagnostics=diagnostics)


def summarize(runs):
    summaries = []
    for run in runs:
        for phase in range(3):
            chunks = [d for d in run['diagnostics'] if d['phase'] == phase]
            for category in ('feeding', 'harmful_patch_harvest', 'other', 'all'):
                rows = [r for d in chunks for r in d['steps'] if category == 'all' or r['category'] == category]
                if not rows:
                    continue
                mean = lambda f: float(np.mean([f(r) for r in rows]))
                summaries.append(dict(trial=run['trial'], arm=run['arm'], phase=phase, category=category,
                    n=len(rows), positive_advantage=mean(lambda r: r['advantage'] > 0),
                    mean_advantage=mean(lambda r: r['advantage']), mean_reward=mean(lambda r: r['reward']),
                    mean_immediate_advantage=mean(lambda r: r['immediate_advantage']),
                    chosen_probability_delta={k: mean(lambda r: r['chosen_probability_delta'][k])
                                              for k in ('joint', 'actor', 'momentum_only')},
                    immediate_expected_reward_delta={k: mean(lambda r: r['immediate_expected_reward_delta'][k])
                                              for k in ('joint', 'actor', 'momentum_only')}))
    return summaries


def main():
    R.Q.configure(); R.verify(R.read(R.OUT / 'manifest.json'))
    output = ROOT / 'runs/ql2_credit_diagnosis_20260910'
    output.mkdir(exist_ok=True)
    runs = []
    for trial in range(4):
        for arm in K.TRAIN_ARMS:
            target = output / f'{trial}_{arm}.json'
            if target.exists():
                result = R.read(target)
                assert result['source_sha'] == R.sha(Path(__file__))
                assert result['checkpoint_sha'] == R.sha(R.OUT / f'training/{trial}_{arm}_a/checkpoint.pt')
            else:
                result = run(trial, arm); R.save(target, result)
            runs.append(result)
    R.verify(R.read(R.OUT / 'manifest.json'))
    result = dict(grade='Post-hoc historical diagnosis, not a survival endpoint',
                  source_sha=R.sha(Path(__file__)), exact_replay=all(r['exact_final'] for r in runs),
                  updates=sum(r['updates'] for r in runs), steps=sum(r['steps'] for r in runs),
                  sampled_updates=sum(len(r['diagnostics']) for r in runs), summaries=summarize(runs))
    R.save(ROOT / 'zeus_sandbox/universe/reports/ql2_credit_diagnosis_20260910.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'summaries'}), flush=True)


if __name__ == '__main__':
    main()
