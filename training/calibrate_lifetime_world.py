"""Frozen first lifetime-world calibration. No neural model or training."""
from __future__ import annotations

from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.lifetime_world import LifetimeAction as A, LifetimeWorld, LifetimeInterface, LifetimeConfig

OUT = ROOT / 'runs/lifetime_calibration_v1_20260909'
SEEDS = tuple(range(202680000, 202680032))
HORIZON = 1024
POLICIES = ('informed', 'reactive', 'periodic', 'reactive_inspect') + tuple('constant_' + a.name.lower() for a in A)
# Explicit route from the workshop, independent of observations or events.
ROUTE = (A.LEFT, A.LEFT) + (A.HARVEST,) * 12 + (A.RIGHT, A.RIGHT, A.MAINTAIN, A.MAINTAIN,
          A.RIGHT, A.RIGHT) + (A.HARVEST,) * 12 + (A.LEFT, A.LEFT, A.MAINTAIN, A.MAINTAIN)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, allow_nan=False, separators=(',', ':'))


def toward(position, target): return A.LEFT if target < position else A.RIGHT


def action_for(name, obs, tick, state, privileged=None):
    if name.startswith('constant_'): return A[name[len('constant_'):].upper()]
    if name == 'periodic': return ROUTE[tick % len(ROUTE)]
    position = round(obs.position * 4)
    if name == 'informed':
        q = privileged['tool']
        if position == 2 and (q < .85 or obs.integrity < .90) and obs.energy > .25:
            return A.MAINTAIN
        if (q < .35 or obs.integrity < .65) and obs.energy > .55 and position != 2:
            return toward(position, 2)
        if obs.energy >= .90: return A.WAIT
        target = 0 if privileged['rates'][0] * privileged['capacity'][0] >= privileged['rates'][1] * privileged['capacity'][1] else 4
        if position in (0, 4) and privileged['resources'][position // 4] >= .025:
            return A.HARVEST
        return A.HARVEST if position == target else toward(position, target)
    # Public controllers receive no private field, rates, event flag or schedule.
    if obs.inspection_valid:
        state['tool'] = obs.tool_condition
    if name == 'reactive_inspect' and position == 2 and tick - state.get('inspected', -32) >= 32:
        state['inspected'] = tick
        return A.INSPECT
    needs_tool = name == 'reactive_inspect' and state.get('tool', 1.) < .80
    if position == 2 and (obs.integrity < .90 or needs_tool) and obs.energy > .25:
        state['tool'] = min(1., state.get('tool', 1.) + .40)
        return A.MAINTAIN
    if obs.integrity < .65 and obs.energy > .55 and position != 2:
        return toward(position, 2)
    if obs.energy >= .90: return A.WAIT
    if obs.coarse_resource >= .25: return A.HARVEST
    direction = state.setdefault('direction', -1)
    if position == 0: direction = 1
    if position == 4: direction = -1
    state['direction'] = direction
    return A.LEFT if direction < 0 else A.RIGHT


def physical(snapshot):
    return {k: v for k, v in snapshot.items() if k not in ('rng_state', 'config', 'version', 'interface')}


def main():
    OUT.mkdir(exist_ok=False)
    paths = ['core/lifetime_world.py', 'training/test_lifetime_world.py',
             'training/calibrate_lifetime_world.py', 'docs/lifetime_calibration_v1_protocol_20260909.md']
    manifest = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    sources={p: sha(ROOT / p) for p in paths}, config=asdict(LifetimeConfig()),
                    seeds=SEEDS, horizon=HORIZON, policies=POLICIES, new_neural_forward_passes=0)
    save(OUT / 'manifest.json', manifest)
    summaries = []
    try:
        with gzip.open(OUT / 'traces.jsonl.gz', 'wt', encoding='utf-8') as trace:
            for seed in SEEDS:
                for changing in (False, True):
                    for name in POLICIES:
                        world = LifetimeWorld(seed=seed, changing=changing)
                        interface = LifetimeInterface(world); state = {}; counts = {a.name.lower(): 0 for a in A}
                        initial = world.snapshot(); energy_sum = 0.; post_alive = 0
                        for tick in range(HORIZON):
                            before = world.snapshot()
                            reference = {k: before[k] for k in ('resources', 'rates', 'capacity', 'tool')} if name == 'informed' else None
                            action = action_for(name, interface.observation(), tick, state, reference)
                            public = interface.step(action); after = world.snapshot()
                            counts[action.name.lower()] += 1; energy_sum += public.after.energy
                            if changing and after['tick'] > after['change_tick'] and not public.terminated: post_alive += 1
                            trace.write(json.dumps(dict(seed=seed, changing=changing, policy=name,
                                before=physical(before), action=int(action), after=physical(after),
                                observation=public.after.values(), prediction_mask=public.after.prediction_mask(),
                                terminated=public.terminated), separators=(',', ':'), allow_nan=False) + '\n')
                            if public.terminated: break
                        summaries.append(dict(seed=seed, changing=changing, policy=name, initial=initial,
                            final=world.snapshot(), survived=world.viable(), ticks=world.snapshot()['tick'],
                            time_limited=world.viable(), mean_energy=energy_sum / world.snapshot()['tick'],
                            post_change_alive_ticks=post_alive, actions=counts))
                print('calibrated seed', seed, flush=True)
        aggregates = {}
        for changing in (False, True):
            key = 'changing' if changing else 'stable'
            aggregates[key] = {}
            for name in POLICIES:
                rows = [r for r in summaries if r['changing'] == changing and r['policy'] == name]
                aggregates[key][name] = dict(survivors=sum(r['survived'] for r in rows), n=len(rows),
                    mean_ticks=sum(r['ticks'] for r in rows) / len(rows),
                    mean_energy=sum(r['mean_energy'] for r in rows) / len(rows))
        feasible = all(aggregates[k]['informed']['survivors'] >= 29 for k in aggregates)
        constants_fail = all(aggregates[k][p]['survivors'] == 0 for k in aggregates for p in POLICIES if p.startswith('constant_'))
        reference = aggregates['changing']['informed']['survivors']
        best_simple = max(aggregates['changing'][p]['survivors'] for p in ('reactive', 'periodic', 'reactive_inspect'))
        headroom = reference - best_simple >= 7
        bars = dict(reference_feasible=feasible, constant_actions_fail=constants_fail,
                    survival_headroom_over_simple_controllers=headroom)
        result = dict(kind='LIFETIME_V1_CALIBRATION', aggregates=aggregates, episodes=summaries, bars=bars,
                      verdict='PASS' if all(bars.values()) else 'FAIL',
                      inspection_utility_established=False, acquired_memory_utility_established=False,
                      training_authorized=False, automatic_followup=False)
        save(OUT / 'results.json', result)
        assert all(sha(ROOT / p) == s for p, s in manifest['sources'].items())
        save(OUT / 'completion.json', dict(manifest=manifest, artifacts={p.name: sha(p) for p in OUT.iterdir() if p.is_file()}))
        print(json.dumps(dict(bars=bars, verdict=result['verdict'], aggregates=aggregates)), flush=True)
    except Exception as exc:
        save(OUT / 'invalid.json', dict(error=repr(exc))); raise


if __name__ == '__main__': main()
