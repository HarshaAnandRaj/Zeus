"""Independent QL1 endpoint replay. Never trains, selects, or changes a model."""
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from core.quality_agent import QualityAgent, model_hash
from core.lifetime_world_v2 import QualityWorld, QualityConfig
from dataclasses import asdict
from training.audit_lifetime_calibration_v2 import balance
from training.audit_lifetime_calibration import close, normalize
from training import quality_learning_contract as K
from training import run_quality_learning as R


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sensors(s):
    patch = s['position'] // 4
    local = s['resources'][patch] if s['position'] in (0, 4) else 0.
    quality = s['quality'][patch] if s['position'] in (0, 4) else 0.
    valid = s['inspection']
    return [s['energy'], s['integrity'], s['position']/4,
            math.floor(local * s['config']['resource_bins']) / s['config']['resource_bins'],
            float(valid), local if valid else 0., s['tool'] if valid else 0., float(quality) if valid else 0.]


@torch.no_grad()
def audit_episode(records, expected, weights, sampling_seed):
    key = {k: expected[k] for k in ('trial', 'seed', 'changing', 'arm')}
    model = QualityAgent(); model.load_state_dict(weights); model.eval().requires_grad_(False)
    identity = model_hash(model)
    world = QualityWorld(seed=key['seed'], changing=key['changing'])
    start = next(records)
    require(start == dict(kind='start', **key, world=normalize(world.snapshot()), model_hash=identity), 'initial state/model mismatch')
    state = model.initial_state(1); previous = -1
    generator = torch.Generator().manual_seed(sampling_seed)
    ticks = 0; inspections = 0; unsafe = 0; total = 0.
    while world.viable() and ticks < K.HORIZON:
        row = next(records); ticks += 1
        require(row['kind'] == 'step' and row['tick'] == ticks and all(row[k] == v for k,v in key.items()), 'step identity/order')
        before = world.snapshot(); obs = sensors(before)
        require(row['observation'] == obs == list(world.observation().values()), 'input sensors')
        if key['arm'] == 'reset_history': state = model.initial_state(1)
        output = model.step(torch.tensor([obs], dtype=torch.float32), torch.tensor([previous]), state, torch.tensor([previous == -1]))
        action = int(torch.multinomial(output.logits.softmax(-1), 1, generator=generator))
        require(type(row['action']) is int and row['action'] == action, 'sampled action')
        for name in ('state', 'logits', 'value', 'predictions'):
            require(row[name] == getattr(output, name).tolist(), 'model output ' + name)
        state = output.state; previous = action
        effect = world.step(action); after = world.snapshot()
        # Scalar balances are independently arranged; the world supplies exact replay.
        close(balance(before, action, before['config']), after)
        next_obs = sensors(after)
        require(row['next_observation'] == next_obs == list(effect.after.values()), 'output sensors')
        require(row['mask'] == [True]*5 + [after['inspection']]*3, 'inspection mask')
        terminal = min(after['energy'], after['integrity']) <= after['config']['death_threshold']
        require(type(row['terminated']) is bool and row['terminated'] == terminal == effect.terminated, 'termination')
        reward = (-1. if terminal else .01) + .1*(after['energy']-before['energy']) + .1*(after['integrity']-before['integrity'])
        close(reward, row['reward']); total += reward
        inspections += action == 4
        unsafe += action == 3 and before['position'] in (0,4) and before['quality'][before['position']//4] == 0 and before['resources'][before['position']//4] > 0
    summary = dict(**key, survived=world.viable(), ticks=ticks, inspections=inspections,
                   unsafe_harvests=unsafe, reward=total, final_world=normalize(world.snapshot()))
    require(next(records) == dict(kind='end', episode=summary), 'trace summary')
    require(summary == expected, 'results summary')
    require(model_hash(model) == identity, 'audit weights changed')
    return ticks


def decision(rows):
    """Recompute registered statistics without calling the runner adjudicator."""
    lookup = {}
    for r in rows:
        require(type(r['survived']) is bool, 'nonbinary survival')
        key = (r['trial'], r['seed'], r['changing'], r['arm'])
        require(key not in lookup, 'duplicate episode'); lookup[key] = r['survived']
    expected = {(t,s,c,a) for t in range(4) for s in K.EVALUATION_SEEDS for c in (False,True) for a in K.ARMS}
    require(set(lookup) == expected, 'endpoint coverage')
    counts = {str(t):{str(c):{a:sum(lookup[t,s,c,a] for s in K.EVALUATION_SEEDS) for a in K.ARMS} for c in (False,True)} for t in range(4)}
    rng = np.random.default_rng(202687000)
    model_draws = rng.integers(0,4,(10000,4)); world_draws = rng.integers(0,64,(10000,64))
    effects = {}
    for arm in ('reset_history', 'initial_model'):
        values = [[int(lookup[t,s,True,'intact']) - int(lookup[t,s,True,arm]) for s in K.EVALUATION_SEEDS] for t in range(4)]
        matrix = np.asarray(values)
        replicates = [float(matrix[np.ix_(m,w)].mean()) for m,w in zip(model_draws,world_draws)]
        low, high = np.quantile(replicates,[.025,.975]); mean = float(matrix.mean())
        effects[arm] = dict(matrix=values, mean=mean, bounds=[float(low),float(high)], passed=bool(mean >= .10 and low > 0))
    absolute = all(counts[str(t)][str(c)]['intact'] >= 58 for t in range(4) for c in (False,True))
    learned = absolute and effects['initial_model']['passed']
    return dict(counts=counts, effects=effects, learned_viability='PASS' if learned else 'FAIL',
                learned_history_benefit='PASS' if learned and effects['reset_history']['passed'] else 'FAIL', pillar_promotion=False)


def main():
    require(not sys.flags.optimize, 'audit requires Python assertions enabled')
    R.configure(); manifest = R.read(R.OUT/'manifest.json'); R.verify(manifest)
    require(not list(R.OUT.glob('*_invalid.json')), 'invalid campaign marker')
    require(R.read(R.OUT/'training/completion.json') == dict(passed=True, twins=4, sources=manifest['sources']), 'training completion')
    parents = []
    for trial, seed in enumerate(K.INITIALIZATIONS):
        a, ac = R.checked_checkpoint(R.OUT/f'training/{trial}_a')
        b, bc = R.checked_checkpoint(R.OUT/f'training/{trial}_b')
        require(ac['logical_hash'] == bc['logical_hash'] and ac['trace_sha'] == bc['trace_sha'], 'twin identity')
        require(a['init_seed'] == b['init_seed'] == seed, 'initialization identity')
        torch.manual_seed(seed); initial = QualityAgent()
        require(R.tree_hash(a['initial']) == R.tree_hash(initial.state_dict()), 'fresh initialization')
        require([(e['seed'],e['changing']) for e in a['episodes']] == [(s,K.episode_changing(i)) for i,s in enumerate(K.TRAIN_SEEDS)], 'development coverage')
        parents.append(a)
    complete = R.read(R.OUT/'evaluation/completion.json')
    require(complete == dict(results_sha=R.sha(R.OUT/'evaluation/results.json'), trace_sha=R.sha(R.OUT/'evaluation/trace.jsonl.gz')), 'evaluation hashes')
    rows = R.read(R.OUT/'evaluation/results.json')['episodes']; verdict = decision(rows)
    expected_order = [(t,s,c,a) for t in range(4) for s in K.EVALUATION_SEEDS for c in (False,True) for a in K.ARMS]
    require([(r['trial'],r['seed'],r['changing'],r['arm']) for r in rows] == expected_order, 'episode order')
    count = 0
    with gzip.open(R.OUT/'evaluation/trace.jsonl.gz','rt',encoding='utf-8') as f:
        records = (json.loads(line) for line in f)
        for row in rows:
            trial = row['trial']; index = K.EVALUATION_SEEDS.index(row['seed'])
            weights = parents[trial]['initial' if row['arm'] == 'initial_model' else 'final']
            count += audit_episode(records, row, weights, 202686000 + trial*1000 + 2*index + int(row['changing']))
        require(next(records, None) is None, 'extra trace records')
    require(R.read(R.OUT/'provisional_verdict.json') == dict(**verdict, independent_audit_required=True), 'decision mismatch')
    R.verify(manifest)
    R.save(R.OUT/'audit.json', dict(passed=True, episodes=len(rows), transitions=count,
        checks=['source and artifact identities', 'exact development twins and fresh initializations',
                'all seeded endpoint starts', 'checkpoint action and recurrent state replay without session runner',
                'independent physical balances and public sensors', 'reward termination summaries and complete coverage',
                'independent crossed bootstrap and binary decisions'],
        provisional_sha=R.sha(R.OUT/'provisional_verdict.json'), evaluation=complete, verdict=verdict))
    print(json.dumps(dict(passed=True,episodes=len(rows),transitions=count,verdict=verdict)),flush=True)


if __name__ == '__main__':
    main()
