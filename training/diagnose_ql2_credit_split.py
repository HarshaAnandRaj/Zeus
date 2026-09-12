"""Separate sequence-gradient interference from historical AdamW moments."""
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from training import diagnose_ql2_credit as D
from training.persistent_learning import sequence_loss


def flat(values, parameters):
    return torch.cat([(torch.zeros_like(p) if v is None else v).flatten()
                      for v, p in zip(values, parameters)])


def split(model, optimizer, batch, consequences, categories):
    params = tuple(model.parameters())
    losses = sequence_loss(model, batch, D.K.SETTINGS)
    actor_g = flat(torch.autograd.grad(losses['actor'], params), params).detach()
    state = batch.initial_state
    logps = []
    for t in range(len(batch.actions)):
        out = model.step(batch.observation[t], batch.previous_action[t], state, batch.starts[t])
        state = out.state
        logps.append(out.logits.log_softmax(-1)[0, batch.actions[t, 0]])
    before = D.policy(model, batch)
    policies = {}
    for mode in ('historical', 'zero_first_moment', 'fresh_adam', 'plain_sgd'):
        clone = copy.deepcopy(model)
        if mode == 'plain_sgd':
            opt = torch.optim.SGD(clone.parameters(), lr=.0003)
        else:
            opt = torch.optim.AdamW(clone.parameters(), lr=.0003, weight_decay=.01,
                                    foreach=False, fused=False)
            if mode != 'fresh_adam':
                opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                if mode == 'zero_first_moment':
                    for s in opt.state.values():
                        s['exp_avg'].zero_()
        gs = torch.autograd.grad(sequence_loss(clone, batch, D.K.SETTINGS)['actor'],
                                 tuple(clone.parameters()), allow_unused=True)
        for p, g in zip(clone.parameters(), gs):
            p.grad = torch.zeros_like(p) if g is None else g.clone()
        torch.nn.utils.clip_grad_norm_(clone.parameters(), 1., error_if_nonfinite=True)
        opt.step(); policies[mode] = D.policy(clone, batch)
    rows = []
    for t, category in enumerate(categories):
        if category == 'other':
            continue
        score_g = flat(torch.autograd.grad(logps[t], params, retain_graph=True, allow_unused=True), params)
        advantage = float(losses['advantages'][t, 0])
        own = float(advantage * score_g.square().sum() / len(batch.actions))
        aggregate = float(-torch.dot(score_g, actor_g))
        action = int(batch.actions[t, 0])
        rows.append(dict(category=category, advantage=advantage, own_gradient_slope=own,
            aggregate_gradient_slope=aggregate, other_transitions_slope=aggregate-own,
            probability_delta={k:float(p[t, action]-before[t, action]) for k,p in policies.items()}))
    return dict(steps=rows, sampled_contexts=len(batch.actions))


def main():
    D.R.Q.configure(); D.R.verify(D.R.read(D.R.OUT/'manifest.json'))
    out = ROOT/'runs/ql2_credit_split_20260912'; out.mkdir(exist_ok=True)
    source = D.R.sha(Path(__file__))
    dependency = D.R.sha(Path(D.__file__))
    D.diagnose = split
    results = []
    for trial in range(4):
        for arm in D.K.TRAIN_ARMS:
            path = out/f'{trial}_{arm}.json'
            if path.exists():
                run = D.R.read(path)
                assert run['split_source_sha'] == source and run['source_sha'] == dependency
                assert run['checkpoint_sha'] == D.R.sha(D.R.OUT/f'training/{trial}_{arm}_a/checkpoint.pt')
            else:
                run = D.run(trial, arm); run['split_source_sha'] = source; D.R.save(path, run)
            results.append(run)
    groups = []
    for run in results:
        for category in ('feeding','harmful_patch_harvest'):
            rows = [r for c in run['diagnostics'] for r in c['steps'] if r['category']==category]
            positive = category=='feeding'
            signed = [r for r in rows if (r['advantage']>0 if positive else r['advantage']<0)]
            desired = lambda x: x>0 if positive else x<0
            groups.append(dict(trial=run['trial'],arm=run['arm'],category=category,n=len(rows),signed_n=len(signed),
                aggregate_follows_credit=sum(desired(r['aggregate_gradient_slope']) for r in signed),
                update_follows_credit={k:sum(desired(r['probability_delta'][k]) for r in signed)
                                      for k in ('historical','zero_first_moment','fresh_adam','plain_sgd')}))
    D.R.save(ROOT/'zeus_sandbox/universe/reports/ql2_credit_split_20260912.json',
        dict(grade='Post-hoc local diagnostic, no functional endpoint', source_sha=source,
             replay_dependency_sha=dependency, exact_replay=True,
             updates=sum(r['updates'] for r in results),steps=sum(r['steps'] for r in results),groups=groups))
    print(json.dumps(groups),flush=True)


if __name__=='__main__':
    main()
