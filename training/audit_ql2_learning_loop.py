"""Post-hoc connectivity audit; isolated copies, no campaign/checkpoint mutation."""
import copy
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from core.quality_agent import QualityAgent, QualitySession
from core.lifetime_world_v2 import QualityInterface
from training.persistent_learning import SequenceBatch, sequence_loss, update_segment
from training import ql2_contract as K
from training import run_ql2 as R


def main():
    R.Q.configure()
    R.verify(R.read(R.OUT / 'manifest.json'))
    results = []
    for trial, seed in enumerate(K.INITIALIZATIONS):
        for arm in K.TRAIN_ARMS:
            directory = R.OUT / f'training/{trial}_{arm}_a'
            payload, _ = R.checked_checkpoint(directory)
            model = QualityAgent(); model.load_state_dict(payload['initial'])
            session = QualitySession(model); session.begin_episode()
            interface = QualityInterface(K.start_world(K.TRAIN_BASE, arm, 0))
            generator = torch.Generator().manual_seed(seed + 1000)
            rows = []
            for _ in range(K.CHUNK):
                session.act(interface.observation(), generator=generator)
                effect = interface.step(session._pending[4])
                rows.append(session.record_outcome(effect.after, reward=K.reward(effect),
                                                   terminated=effect.terminated))
                if effect.terminated:
                    break
            with gzip.open(directory / 'trace.jsonl.gz', 'rt') as stream:
                saved = []
                for line in stream:
                    entry = json.loads(line)
                    if entry['kind'] == 'step':
                        saved.append(entry)
                    if len(saved) == len(rows):
                        break
            assert len(saved) == len(rows)
            for row, entry in zip(rows, saved):
                assert row.action == entry['action'] and row.reward == entry['reward']
                assert torch.equal(row.behavior_logits, torch.tensor(entry['logits'][0]))
            batch = SequenceBatch.from_transitions(rows)
            losses = sequence_loss(model, batch, K.SETTINGS)
            norms = {}
            for name in ('actor', 'value', 'prediction'):
                gradients = torch.autograd.grad(losses[name], tuple(model.recurrence.parameters()),
                                                retain_graph=True)
                norms[name] = float(torch.cat([g.flatten() for g in gradients]).norm())
                assert norms[name] > 0
            # Same actual sequence; alter only its final reward on an isolated copy.
            altered = copy.deepcopy(session)
            changed_rewards = batch.rewards.clone(); changed_rewards[-1] += 1.
            changed_batch = replace(batch, rewards=changed_rewards)
            actor_params = tuple(model.actor.parameters())
            original_gradient = torch.cat([g.flatten() for g in torch.autograd.grad(
                losses['actor'], actor_params)])
            altered_loss = sequence_loss(model, changed_batch, K.SETTINGS)
            changed_gradient = torch.cat([g.flatten() for g in torch.autograd.grad(
                altered_loss['actor'], actor_params)])
            gradient_delta = float((changed_gradient - original_gradient).norm())
            assert gradient_delta > 0
            before = copy.deepcopy(model.state_dict())
            metrics = []
            for instance, data in ((session, batch), (altered, changed_batch)):
                optimizer = torch.optim.AdamW(instance.model.parameters(), lr=.0003,
                                             weight_decay=.01, foreach=False, fused=False)
                metrics.append(update_segment(instance, optimizer, data, K.SETTINGS,
                                              max_grad_norm=1.))
                instance.refresh_state()
            # Check against the historical first update, not merely a synthetic pass.
            assert metrics[0] == payload['episodes'][0]['updates'][0]
            delta = lambda a, b, prefix: float(torch.cat([
                (a[k] - b[k]).flatten() for k in a if k.startswith(prefix + '.')]).norm())
            groups = ('recurrence', 'actor', 'critic', 'transition')
            update_changes = {g: delta(model.state_dict(), before, g) for g in groups}
            historical_changes = {g: delta(payload['final'], payload['initial'], g) for g in groups}
            assert all(v > 0 for v in (*update_changes.values(), *historical_changes.values()))
            obs = torch.tensor([rows[-1].next_observation.tolist()])
            previous = torch.tensor([rows[-1].action]); starts = torch.tensor([False])
            with torch.no_grad():
                policies = [s.model.step(obs, previous, s.state, starts).logits.softmax(-1)
                            for s in (session, altered)]
            tv = float((policies[0] - policies[1]).abs().sum() / 2)
            assert tv > 0
            results.append(dict(trial=trial, arm=arm, steps=len(rows),
                historical_first_update_exact=True, core_gradient_norms=norms,
                first_update_parameter_delta=update_changes,
                historical_parameter_delta=historical_changes,
                reward_intervention_actor_gradient_delta=gradient_delta,
                reward_intervention_next_policy_tv=tv,
                historical_revision=int(payload['final']['revision'])))
    output = dict(grade='Post-hoc mechanical connectivity, not functional competence',
                  passed=True, checks=results,
                  limits=['First chunk reproduced for each of eight training arms; not every update replayed.',
                          'Reward intervention is artificial and isolated; no survival claim.',
                          'Prediction head influences actor only through shared training parameters.',
                          'BPTT is truncated at 64 steps; evaluation has fixed weights.'])
    target = ROOT / 'zeus_sandbox/universe/reports/ql2_learning_loop_audit_20260910.json'
    R.save(target, output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
