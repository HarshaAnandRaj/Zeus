"""Full deterministic replay plus independently arranged scalar balances."""
from __future__ import annotations

from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.lifetime_world import LifetimeWorld
from training import calibrate_lifetime_world as C


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def normalize(value): return json.loads(json.dumps(value))


def close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: close(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): close(x, y)
    elif isinstance(a, float):
        assert math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-12, (a, b)
    else:
        assert a == b, (a, b)


def independent_balance(before, action, config):
    c = config
    result = dict(before)
    position = before['position']; extracted = 0.
    if action in (1, 2): position = max(0, min(4, position + (-1 if action == 1 else 1)))
    stock = list(before['resources'])
    if action == 3 and position in (0, 4):
        index = position // 4
        extracted = min(c['extraction_limit'], stock[index])
        stock[index] -= extracted
    action_cost = {0: 0., 1: c['move_cost'], 2: c['move_cost'], 3: c['harvest_cost'],
                   4: c['inspect_cost'], 5: c['maintain_cost']}[action]
    energy = math.fsum((before['energy'], -c['metabolism'], -action_cost,
                        extracted * (c['efficiency_floor'] + c['efficiency_gain'] * before['tool'])))
    workshop = action == 5 and position == 2
    repair = c['integrity_repair'] if workshop else 0.
    rest = c['wait_recovery'] if action == 0 and energy > c['wait_energy_min'] else 0.
    integrity = math.fsum((before['integrity'], repair, rest, -c['integrity_decay'],
                           -c['starvation_damage'] * max(0., c['starvation_level'] - energy)))
    tool = before['tool'] + (c['tool_repair'] if workshop else 0.) - (c['harvest_wear'] if action == 3 else 0.)
    clip = lambda v: max(0., min(1., v))
    resources = [clip(math.fsum((r, rate * (capacity - r))))
                 for r, rate, capacity in zip(stock, before['rates'], before['capacity'])]
    rates = list(before['rates']); tick = before['tick'] + 1
    changed = before['changed']
    if tick == before['change_tick']:
        rates = [rates[1], rates[0]]; changed = True
    result.update(position=position, resources=resources, rates=rates, changed=changed,
                  energy=clip(energy), integrity=clip(integrity), tool=clip(tool),
                  tick=tick, inspection=action == 4)
    return result


def main():
    completion = read(C.OUT / 'completion.json'); manifest = completion['manifest']
    for p, digest in manifest['sources'].items(): assert C.sha(ROOT / p) == digest
    for name, digest in completion['artifacts'].items(): assert C.sha(C.OUT / name) == digest
    data = read(C.OUT / 'results.json'); episodes = data['episodes']
    expected_keys = [(seed, changing, policy) for seed in C.SEEDS for changing in (False, True) for policy in C.POLICIES]
    assert [(e['seed'], e['changing'], e['policy']) for e in episodes] == expected_keys
    assert manifest['seeds'] == list(C.SEEDS) and manifest['horizon'] == C.HORIZON and manifest['policies'] == list(C.POLICIES)
    assert manifest['config'] == asdict(C.LifetimeConfig()) and manifest['new_neural_forward_passes'] == 0
    count = 0
    with gzip.open(C.OUT / 'traces.jsonl.gz', 'rt', encoding='utf-8') as f:
        for ep in episodes:
            world = LifetimeWorld.restore(ep['initial'])
            assert normalize(world.snapshot()) == normalize(LifetimeWorld(seed=ep['seed'], changing=ep['changing']).snapshot())
            policy_state = {}; energy = 0.; post_alive = 0; actions = {a.name.lower(): 0 for a in C.A}
            for tick in range(ep['ticks']):
                row = json.loads(next(f))
                assert (row['seed'], row['changing'], row['policy']) == (ep['seed'], ep['changing'], ep['policy'])
                before = C.physical(world.snapshot()); assert before == row['before']
                reference = {k: before[k] for k in ('resources', 'rates', 'capacity', 'tool')} if ep['policy'] == 'informed' else None
                action = C.action_for(ep['policy'], world.observation(), tick, policy_state, reference)
                assert int(action) == row['action']
                independent = independent_balance(before, int(action), manifest['config'])
                close(independent, row['after'])
                public = world.step(action)
                assert C.physical(world.snapshot()) == row['after']
                assert list(public.after.values()) == row['observation']
                assert list(public.after.prediction_mask()) == row['prediction_mask']
                assert public.terminated == row['terminated']
                # Public sensor construction is also checked without the world helper.
                s = row['after']; local = s['resources'][s['position'] // 4] if s['position'] in (0, 4) else 0.
                expected_obs = [s['energy'], s['integrity'], s['position'] / 4,
                    math.floor(local * manifest['config']['resource_bins']) / manifest['config']['resource_bins'],
                    float(s['inspection']), local if s['inspection'] else 0., s['tool'] if s['inspection'] else 0.]
                assert expected_obs == row['observation']
                assert row['prediction_mask'] == [True] * 5 + [s['inspection']] * 2
                assert row['terminated'] == (min(s['energy'], s['integrity']) <= manifest['config']['death_threshold'])
                if public.terminated: assert tick == ep['ticks'] - 1
                actions[action.name.lower()] += 1; energy += s['energy']
                if ep['changing'] and s['tick'] > s['change_tick'] and not public.terminated: post_alive += 1
                count += 1
            assert normalize(world.snapshot()) == ep['final']
            assert ep['survived'] == world.viable() == ep['time_limited']
            assert ep['ticks'] == C.HORIZON or not ep['survived']
            assert ep['actions'] == actions and ep['post_change_alive_ticks'] == post_alive
            close(ep['mean_energy'], energy / ep['ticks'])
        assert not f.read().strip()
    for condition in ('stable', 'changing'):
        for policy in C.POLICIES:
            rows = [e for e in episodes if e['changing'] == (condition == 'changing') and e['policy'] == policy]
            expected = dict(survivors=sum(e['survived'] for e in rows), n=32,
                mean_ticks=sum(e['ticks'] for e in rows) / 32, mean_energy=sum(e['mean_energy'] for e in rows) / 32)
            close(expected, data['aggregates'][condition][policy])
    a = data['aggregates']
    bars = dict(reference_feasible=all(a[k]['informed']['survivors'] >= 29 for k in a),
        constant_actions_fail=all(a[k][p]['survivors'] == 0 for k in a for p in C.POLICIES if p.startswith('constant_')),
        survival_headroom_over_simple_controllers=a['changing']['informed']['survivors'] - max(a['changing'][p]['survivors'] for p in ('reactive','periodic','reactive_inspect')) >= 7)
    assert data['bars'] == bars and data['verdict'] == ('PASS' if all(bars.values()) else 'FAIL')
    assert not any(data[k] for k in ('inspection_utility_established', 'acquired_memory_utility_established', 'training_authorized', 'automatic_followup'))
    assert all(C.sha(ROOT / p) == digest for p, digest in manifest['sources'].items())
    audit = dict(passed=True, episodes=len(episodes), transitions=count,
        checks=dict(source_artifact_identity=True,all_seeded_initializations=True,all_controller_choices=True,
                    all_deterministic_replays=True,independent_scalar_balances=True,public_sensors_masks_termination=True,
                    all_episode_aggregates_and_bars=True),
        balance_absolute_tolerance=1e-12, completion_sha=C.sha(C.OUT / 'completion.json'), audit_sha=C.sha(Path(__file__)))
    C.save(C.OUT / 'audit.json', audit); print(json.dumps(audit), flush=True)


if __name__ == '__main__': main()
