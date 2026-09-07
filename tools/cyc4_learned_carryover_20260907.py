"""CYC4 bounded supervised recurrent cache distillation; protocol 9c09057."""
from __future__ import annotations
import copy
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import cyc3_carryover_calibration_20260907 as C

OUT = ROOT / 'runs/cyc4_20260907'
CONFIG = dict(train_seed=202674000, train_pairs=64, eval_seed=202675000,
              eval_pairs=128, init_seed=20260991, shuffle_seed=20260992,
              bootstrap_seed=20260993, bootstrap_samples=10000, epochs=200,
              batch=16, lr=.001, hidden=32, input_dim=10, output_dim=9,
              early_weight=8., horizon=512, device='cpu', dtype='float32', threads=1)
CONTROLS = ('intact', 'erased', 'swapped', 'untrained')


def configure():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def exact(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def features(obs):
    # Deliberately no bodily history, label, time, action or invisible resources.
    x = [float(obs[3])] + [0.] * 9
    position = int(round(obs[4] * 8))
    assert 0 <= position < 9
    x[1 + position] = 1.
    return x


class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(10, 32, batch_first=True)
        self.readout = nn.Linear(32, 9)

    def forward(self, x, hidden=None):
        state, hidden = self.gru(x, hidden)
        return torch.sigmoid(self.readout(state)), hidden


def choose(obs, tick, predictions):
    assert len(predictions) == 9 and all(np.isfinite(predictions))
    cache = {str(i): dict(resource=float(v), tick=tick) for i, v in enumerate(predictions)}
    return C.choose(obs, tick, cache)


def data_rows(preparation, teacher):
    observations = [list(C.restore(preparation['initial']).observation())]
    observations += [r['after'] for r in preparation['exposure']]
    observations[16] = preparation['observation']
    observations += [r['after'] for r in teacher['trace']]
    cache = {}; xs = []; ys = []; ws = []
    for tick, obs in enumerate(observations):
        C.observe(cache, obs, tick)
        xs.append(features(obs)); ys.append([C.estimate(cache, cell, tick) for cell in range(9)])
        ws.append(8. if tick <= 16 else 1.)
    valid = len(xs)
    assert valid <= 513
    while len(xs) < 513:
        xs.append(xs[-1]); ys.append(ys[-1]); ws.append(0.)
    return xs, ys, ws, valid


def make_data():
    episodes = []; xs = []; ys = []; ws = []
    for offset in range(CONFIG['train_pairs']):
        seed = CONFIG['train_seed'] + offset
        for target in (2, 6):
            p = C.prepare(seed, target); teacher = C.rollout(p, p, 'intact')
            x, y, w, valid = data_rows(p, teacher)
            episodes.append(dict(preparation=p, teacher=teacher, valid=valid))
            xs.append(x); ys.append(y); ws.append(w)
        if (offset + 1) % 16 == 0: print('training data pairs', offset + 1, flush=True)
    return dict(x=torch.tensor(xs), y=torch.tensor(ys), weight=torch.tensor(ws), episodes=episodes)


def fit(data):
    torch.manual_seed(CONFIG['init_seed'])
    model = Memory(); initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                                 weight_decay=.01, amsgrad=False, foreach=False, fused=False)
    generator = torch.Generator().manual_seed(CONFIG['shuffle_seed'])
    x, y, weight = data['x'], data['y'], data['weight']
    losses = []; updates = 0
    for epoch in range(CONFIG['epochs']):
        order = torch.randperm(len(x), generator=generator); numerator = denominator = 0.
        for start in range(0, len(x), CONFIG['batch']):
            idx = order[start:start + CONFIG['batch']]
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(x[idx])
            w = weight[idx].unsqueeze(-1)
            total = ((prediction - y[idx]).square() * w).sum(); norm = w.sum() * 9
            loss = total / norm
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); updates += 1
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            numerator += total.item(); denominator += norm.item()
        losses.append(numerator / denominator)
        if (epoch + 1) % 10 == 0:
            print('epoch', epoch + 1, 'updates', updates, 'weighted_mse', losses[-1], flush=True)
    with torch.no_grad(): prediction, _ = model(x)
    return dict(initial=initial, final=copy.deepcopy(model.state_dict()), optimizer=optimizer.state_dict(),
                losses=losses, updates=updates, teacher_predictions=prediction,
                initial_hash=state_hash(initial), final_hash=state_hash(model.state_dict()))


@torch.no_grad()
def history(model, prepared):
    obs = [list(C.restore(prepared['initial']).observation())]
    obs += [r['after'] for r in prepared['exposure'][:15]]
    _, hidden = model(torch.tensor([[features(o) for o in obs]]))
    return hidden


@torch.no_grad()
def neural_rollout(model, prepared, donor, condition):
    world = C.restore(prepared['boundary'])
    hidden = (torch.zeros(1, 1, 32) if condition == 'erased' else
              history(model, donor if condition == 'swapped' else prepared))
    initial_hidden = hidden.tolist(); trace = []; survival256 = False
    while world.viable() and world.body.age < 512:
        x = features(world.observation())
        prediction, hidden = model(torch.tensor([[x]]), hidden)
        values = prediction[0, 0].tolist()
        action, decision = choose(world.observation(), world.step_count, values)
        row = C.step_account(world, action)
        row.update(input=x, prediction=values, hidden=hidden.tolist(), decision=decision)
        trace.append(row)
        if world.body.age == 256: survival256 = world.viable()
    # Consume the last physical observation, including terminal observations.
    last_x = features(world.observation())
    last_prediction, hidden = model(torch.tensor([[last_x]]), hidden)
    return dict(condition=condition, initial_physical=prepared['boundary'], initial_hidden=initial_hidden,
                final_physical=C.snapshot(world), final_hidden=hidden.tolist(), final_input=last_x,
                final_prediction=last_prediction[0, 0].tolist(), trace=trace,
                first_action=trace[0]['action'], age=world.body.age, survived_256=survival256,
                survived_512=world.body.age == 512 and world.viable(), failure_cause=C.cause(world),
                cycles=C.closures([4] + [r['position_after'] for r in trace],
                                  [r['harvested_resource'] for r in trace]))


def campaign(checkpoint):
    trained = Memory().eval(); trained.load_state_dict(checkpoint['final'])
    untrained = Memory().eval(); untrained.load_state_dict(checkpoint['initial'])
    pairs = []
    for index in range(CONFIG['eval_pairs']):
        seed = CONFIG['eval_seed'] + index
        left, right = C.prepare(seed, 2), C.prepare(seed, 6)
        assert left['observation'] == right['observation']
        orientations = []
        for prepared, donor in ((left, right), (right, left)):
            correct = 1 if prepared['rich_target'] == 2 else 2
            controls = {c: neural_rollout(untrained if c == 'untrained' else trained, prepared, donor, c)
                        for c in CONTROLS}
            teacher = C.rollout(prepared, donor, 'intact')
            searches = [C.alternative_search(prepared, a) for a in range(6) if a != correct]
            orientations.append(dict(preparation=prepared, correct_first_action=correct,
                                     controls=controls, teacher=teacher, searches=searches))
        pairs.append(dict(seed=seed, orientations=orientations))
        if (index + 1) % 16 == 0: print('endpoint pairs', index + 1, flush=True)
    return pairs


def adjudicate(pairs, data):
    assert len(pairs) == CONFIG['eval_pairs']
    assert [p['seed'] for p in pairs] == list(range(CONFIG['eval_seed'], CONFIG['eval_seed'] + 128))
    for p in pairs:
        assert len(p['orientations']) == 2
        assert {o['correct_first_action'] for o in p['orientations']} == {1, 2}
        for o in p['orientations']:
            assert len(o['searches']) == 5
            assert {s['first_action'] for s in o['searches']} == set(range(6)) - {o['correct_first_action']}
    n = len(pairs); bars = {}; summaries = {}
    for horizon, threshold in ((256, .90), (512, .80)):
        k = sum(all(o['controls']['intact'][f'survived_{horizon}'] for o in p['orientations']) for p in pairs)
        ci = C.wilson(k, n)
        bars[f'intact_pair_survival_{horizon}'] = dict(successes=k, n=n, point=k/n, ci95=ci,
                                                      threshold=threshold, passed=ci[0] >= threshold)
    draws = np.random.Generator(np.random.PCG64(CONFIG['bootstrap_seed'])).integers(0, n, (10000, n))
    for control in CONTROLS[1:]:
        delta = np.array([np.mean([int(o['controls']['intact']['survived_512']) -
                                   int(o['controls'][control]['survived_512']) for o in p['orientations']])
                          for p in pairs])
        ci = np.quantile(delta[draws].mean(1), [.025, .975]).tolist()
        bars['intact_minus_' + control] = dict(point=float(delta.mean()), ci95=ci,
                                               threshold=.30, passed=ci[0] >= .30)
    searches = [s for p in pairs for o in p['orientations'] for s in o['searches']]
    bars['information_necessity'] = dict(passed=all(s['status'] == 'EXHAUSTIVE_NO_SURVIVOR' for s in searches),
        total=len(searches), expanded_transitions=sum(s['expanded_transitions'] for s in searches),
        counterexamples=sum(s['status'] == 'COUNTEREXAMPLE' for s in searches),
        capped=sum(s['status'] == 'UNVERIFIED_CAP' for s in searches))
    train_ok = sum(e['teacher']['survived_512'] for e in data['episodes'])
    eval_ok = sum(o['teacher']['survived_512'] for p in pairs for o in p['orientations'])
    bars['explicit_teacher'] = dict(training_survivors=train_ok, evaluation_survivors=eval_ok,
                                    passed=train_ok == 128 and eval_ok == 256)
    for control in CONTROLS:
        episodes = [o['controls'][control] for p in pairs for o in p['orientations']]
        summaries[control] = dict(worlds=len(episodes), survival_256=sum(e['survived_256'] for e in episodes),
            survival_512=sum(e['survived_512'] for e in episodes), mean_absolute_age=float(np.mean([e['age'] for e in episodes])),
            first_action_correct=sum(o['controls'][control]['first_action'] ==
                C.Action(o['correct_first_action']).name.lower() for p in pairs for o in p['orientations']))
    passed = all(b['passed'] for b in bars.values())
    return dict(verdict='PASS' if passed else 'FAIL', bars=bars, summaries=summaries,
                failed_requirements=[k for k, v in bars.items() if not v['passed']],
                consequence='retain_supervised_component_close' if passed else 'eliminate_this_recipe_close',
                automatic_followup=False, pillar_promotion=False)


def manifest():
    paths = ['core/embodiment.py', 'tools/cycle_forensics_20260907.py',
             'tools/cyc3_carryover_calibration_20260907.py', 'tools/cyc4_learned_carryover_20260907.py',
             'tools/test_cyc4_learned_carryover_20260907.py', 'docs/cyc4_learned_carryover_protocol_20260907.md',
             'zeus_sandbox/universe/reports/cyc3_carryover_calibration_verdict_20260907.json',
             'zeus_sandbox/universe/reports/cyc3_completion_audit_20260907.json']
    return dict(config=CONFIG, sources={p: C.sha(ROOT/p) for p in paths},
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__))


def main():
    configure(); OUT.mkdir(exist_ok=False); m = manifest(); C.save(OUT/'manifest.json', m)
    try:
        assert m['sources']['core/embodiment.py'] == '984f0c2253204d57688ea972064aaccd684f4fc006bbcd378752b343c745b3b2'
        C.verify(m); data = make_data(); torch.save(data, OUT/'training_data.pt')
        assert data['x'].shape == (128, 513, 10)
        print('training twin a', flush=True); a = fit(data); torch.save(a, OUT/'twin_a.pt'); C.verify(m)
        print('training twin b', flush=True); b = fit(data); torch.save(b, OUT/'twin_b.pt'); C.verify(m)
        assert exact(a, b) and a['updates'] == 1600, 'training twins or budget differ'
        print('evaluation twin a', flush=True); ea = campaign(a); C.save(OUT/'campaign_a.json.gz', ea); C.verify(m)
        print('evaluation twin b', flush=True); eb = campaign(b); C.save(OUT/'campaign_b.json.gz', eb); C.verify(m)
        assert ea == eb, 'evaluation twins differ'
        outcome = adjudicate(ea, data)
        result = dict(kind='CYC4', **outcome, exact_twins=True, manifest=m,
                      initial_hash=a['initial_hash'], final_hash=a['final_hash'],
                      updates=a['updates'], final_training_mse=a['losses'][-1],
                      artifacts={p.name:C.sha(p) for p in OUT.iterdir() if p.is_file()})
        C.save(OUT/'verdict.json', result); print(json.dumps(outcome), flush=True)
    except Exception as exc:
        C.save(OUT/'invalid.json', dict(status='INVALID_STOP', error=repr(exc), manifest=m)); raise


if __name__ == '__main__': main()
