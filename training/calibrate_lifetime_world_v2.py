"""Quality-world calibration: public information, revision and map controls."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.lifetime_world_v2 import QualityWorld, QualityInterface, QualityConfig
from core.lifetime_world import LifetimeAction as A
from training import calibrate_lifetime_world as V1

OUT = ROOT / 'runs/lifetime_calibration_v2_20260909'
SEEDS = tuple(range(202681000, 202681032))
HORIZON = 1024
POLICIES = ('informed', 'public_memory', 'frozen_map', 'current_inspection', 'reactive', 'periodic') + tuple('constant_' + a.name.lower() for a in A)
sha, save, physical = V1.sha, V1.save, V1.physical


def action_for(name, obs, tick, state, privileged=None):
    if name in ('reactive', 'periodic') or name.startswith('constant_'):
        return V1.action_for(name, obs, tick, state)
    position = round(obs.position * 4)
    memory = state.setdefault('quality', {})
    if name == 'current_inspection': memory.clear()
    writable = name != 'frozen_map' or tick < 160
    if writable and obs.inspection_valid and position in (0, 4):
        memory[position] = int(obs.resource_quality)
    if obs.inspection_valid: state['tool'] = obs.tool_condition
    # Bodily consequences can contradict an old quality estimate. This is a
    # supplied diagnostic rule, not a learned discovery or privileged event cue.
    if writable and state.get('last_action') == int(A.HARVEST) and obs.integrity < state.get('last_integrity', 0.) - .05:
        site = state['last_position']; memory[site] = 0
        if memory.get(4 - site) == 0: memory.pop(4 - site)
    q = privileged['tool'] if name == 'informed' else state.get('tool', .90)
    if name == 'informed': memory = {0: privileged['quality'][0], 4: privileged['quality'][1]}
    if position == 2 and (q < .85 or obs.integrity < .90) and obs.energy > .25:
        action = A.MAINTAIN
    elif (q < .35 or obs.integrity < .65) and obs.energy > .55 and position != 2:
        action = V1.toward(position, 2)
    elif obs.energy >= .90:
        action = A.WAIT
    elif position in (0, 4):
        if position not in memory:
            action = A.INSPECT
        elif memory[position] == 1:
            action = A.HARVEST
        else:
            state['direction'] = 1 if position == 0 else -1
            action = A.RIGHT if position == 0 else A.LEFT
    else:
        known = [p for p in (0, 4) if memory.get(p) == 1]
        if known:
            action = V1.toward(position, known[0])
        else:
            action = A.LEFT if state.get('direction', -1) < 0 else A.RIGHT
    state.update(last_action=int(action), last_integrity=obs.integrity, last_position=position)
    if action == A.HARVEST: state['tool'] = max(0., q - .008)
    if action == A.MAINTAIN and position == 2: state['tool'] = min(1., q + .40)
    return action


def decisions(aggregates):
    stable, changing = aggregates['stable'], aggregates['changing']
    memory = changing['public_memory']['survivors']
    physical = all(aggregates[c]['informed']['survivors'] >= 29 for c in aggregates)
    usable = all(aggregates[c]['public_memory']['survivors'] >= 29 for c in aggregates)
    revision = stable['frozen_map']['survivors'] >= 29 and memory - changing['frozen_map']['survivors'] >= 7
    simple = memory - max(changing[p]['survivors'] for p in ('reactive', 'periodic')) >= 7
    constants = all(aggregates[c][p]['survivors'] == 0 for c in aggregates for p in POLICIES if p.startswith('constant_'))
    # A distinct efficiency question: retaining the map may save paid inspections
    # without increasing survival relative to repeated current sensing.
    efficiency = all(aggregates[c]['public_memory']['survivors'] >= aggregates[c]['current_inspection']['survivors']
        and aggregates[c]['public_memory']['inspections'] <= .5 * aggregates[c]['current_inspection']['inspections']
        and aggregates[c]['current_inspection']['inspections'] > 0 for c in aggregates)
    return dict(informed_feasible=physical, public_information_usable=usable,
                revision_beats_frozen_map=revision, beats_reactive_and_fixed_route=simple,
                constants_fail=constants, map_saves_inspections=efficiency)


def main():
    OUT.mkdir(exist_ok=False)
    paths = ['core/lifetime_world.py', 'core/lifetime_world_v2.py',
             'training/calibrate_lifetime_world.py', 'training/calibrate_lifetime_world_v2.py',
             'training/test_lifetime_world_v2.py', 'docs/lifetime_calibration_v2_protocol_20260909.md']
    manifest = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        sources={p: sha(ROOT / p) for p in paths}, config=asdict(QualityConfig()), seeds=SEEDS,
        horizon=HORIZON, policies=POLICIES, new_neural_forward_passes=0)
    save(OUT / 'manifest.json', manifest); episodes = []
    try:
        with gzip.open(OUT / 'traces.jsonl.gz', 'wt', encoding='utf-8') as f:
            for seed in SEEDS:
                for changing in (False, True):
                    for name in POLICIES:
                        world = QualityWorld(seed=seed, changing=changing); interface = QualityInterface(world)
                        state = {}; counts = {a.name.lower(): 0 for a in A}; initial = world.snapshot()
                        unsafe = 0; energy = 0.
                        for tick in range(HORIZON):
                            before = world.snapshot()
                            reference = {k: before[k] for k in ('quality','tool')} if name == 'informed' else None
                            action = action_for(name, interface.observation(), tick, state, reference)
                            bad = action == A.HARVEST and before['position'] in (0,4) and before['quality'][before['position']//4] == 0 and before['resources'][before['position']//4] > 0
                            public = interface.step(action); after = world.snapshot()
                            counts[action.name.lower()] += 1; unsafe += int(bad); energy += public.after.energy
                            f.write(json.dumps(dict(seed=seed, changing=changing, policy=name,
                                before=physical(before), action=int(action), after=physical(after),
                                observation=public.after.values(), prediction_mask=public.after.prediction_mask(),
                                terminated=public.terminated, unsafe_harvest=bool(bad)), separators=(',',':')) + '\n')
                            if public.terminated: break
                        episodes.append(dict(seed=seed, changing=changing, policy=name, initial=initial,
                            final=world.snapshot(), ticks=after['tick'], survived=world.viable(), time_limited=world.viable(),
                            actions=counts, unsafe_harvests=unsafe, mean_energy=energy / after['tick']))
                print('calibrated seed', seed, flush=True)
        aggregates = {}
        for condition in ('stable','changing'):
            aggregates[condition] = {}
            for name in POLICIES:
                rows = [r for r in episodes if r['changing'] == (condition == 'changing') and r['policy'] == name]
                aggregates[condition][name] = dict(n=len(rows), survivors=sum(r['survived'] for r in rows),
                    mean_ticks=sum(r['ticks'] for r in rows)/len(rows), mean_energy=sum(r['mean_energy'] for r in rows)/len(rows),
                    inspections=sum(r['actions']['inspect'] for r in rows), unsafe_harvests=sum(r['unsafe_harvests'] for r in rows))
        bars = decisions(aggregates)
        result = dict(aggregates=aggregates, episodes=episodes, bars=bars,
            verdict='PASS' if all(bars.values()) else 'FAIL',
            memory_survival_advantage_over_current_inspection=aggregates['changing']['public_memory']['survivors'] - aggregates['changing']['current_inspection']['survivors'],
            learned_adaptation_established=False, automatic_followup=False)
        save(OUT / 'results.json', result)
        assert all(sha(ROOT / p) == digest for p,digest in manifest['sources'].items())
        save(OUT / 'completion.json', dict(manifest=manifest, artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
        print(json.dumps(dict(bars=bars, verdict=result['verdict'], aggregates=aggregates)), flush=True)
    except Exception as exc:
        save(OUT / 'invalid.json', dict(error=repr(exc))); raise


if __name__ == '__main__': main()
