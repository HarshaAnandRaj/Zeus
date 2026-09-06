"""Read-only diagnostic replay; see the committed dated design, not a gate."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from core.embodiment import Action, EmbodiedWorldV2
from training.evaluate_quotient_policy import load_campaign, campaigns_exact, failure_cause
from training.evaluate_viability_quotient import load_artifact
from training.train_quotient_policy import build_quotient, make_policy, quotient_step, decision_features, continuing_viability_reward

REPORTS = ROOT / 'zeus_sandbox/universe/reports'
OUT = REPORTS / 'qv1_postmortem_20260906.json'


def read(name):
    return json.loads((REPORTS / name).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def repertoire(counts):
    total = sum(counts.values())
    fractions = {a: counts.get(a, 0) / total for a in [x.name.lower() for x in Action]}
    return {'counts': dict(counts), 'fractions': fractions,
            'observed_actions': sum(v > 0 for v in counts.values()),
            'entropy_effective_actions': float(np.exp(-sum(p*np.log(p) for p in fractions.values() if p)))}


def km(episodes):
    s, rows = 1., []
    for age in sorted({e['age'] for e in episodes}):
        risk = sum(e['age'] >= age for e in episodes)
        deaths = sum(e['age'] == age and not e['survived_512'] for e in episodes)
        censored = sum(e['age'] == age and e['survived_512'] for e in episodes)
        s *= 1-deaths/risk
        rows.append({'age': age, 'at_risk': risk, 'deaths': deaths, 'censored': censored, 'survival': s})
    return {'table': rows, 'median': next((r['age'] for r in rows if r['survival'] <= .5), None),
            'maximum_age': max(e['age'] for e in episodes),
            'survival_at': {t: next((r['survival'] for r in reversed(rows) if r['age'] <= t), 1.) for t in [64,128,256,512]}}


def initial(seed):
    w = EmbodiedWorldV2(seed=seed)
    return {'seed': seed, 'observation': list(w.observation()), 'capacity': w.capacity,
            'resources': w.resources, 'ambient_phase': w.ambient_phase}


@torch.no_grad()
def replay(arm, campaign, parent, seeds, sample_seeds, decoding):
    quotient = build_quotient(arm, parent)
    policy = make_policy().eval()
    policy.load_state_dict(campaign['arms'][arm]['policy_state_dict'])
    before = sha(campaign['parent_qv0_path'])
    worlds = [EmbodiedWorldV2(seed=s) for s in seeds]
    n = len(worlds)
    state = quotient.initial_state(n)
    previous = torch.tensor([w.observation() for w in worlds], dtype=state.dtype)
    previous_action = None
    generators = [torch.Generator().manual_seed(s) for s in sample_seeds]
    active = torch.ones(n, dtype=torch.bool)
    traces = [[] for _ in worlds]
    rewards = [0.] * n
    for tick in range(256):
        if not active.any():
            break
        obs = torch.tensor([w.observation() for w in worlds], dtype=state.dtype)
        proposed = quotient_step(quotient, state, obs, previous, previous_action,
                                 retain_history=campaign['arms'][arm]['retain_history'])
        state = torch.where(active[:, None], proposed, state)
        logits = policy(decision_features(quotient, state))
        actions = logits.argmax(-1)
        if decoding == 'sampled':
            probs = logits.softmax(-1)
            for i in torch.nonzero(active).flatten().tolist():
                actions[i] = torch.multinomial(probs[i], 1, generator=generators[i])[0]
        for i in torch.nonzero(active).flatten().tolist():
            w = worlds[i]
            a = int(actions[i])
            old = list(w.observation())
            resource = w.resources[w.body.position]
            amount = min(.13, resource) if a == Action.HARVEST else 0.
            cost = .008 if a in (Action.MOVE_LEFT, Action.MOVE_RIGHT) else .012 if a == Action.REGULATE else .006 if a == Action.SPEAK else 0.
            penalty = .006 if a == Action.HARVEST and amount < .03 else 0.
            expected_unclipped = old[0] - .018 - cost - penalty + .8*amount
            effect = w.step(a)
            expected = min(1., max(0., expected_unclipped))
            if abs(expected-w.body.energy) > 1e-12:
                raise RuntimeError('energy accounting mismatch')
            traces[i].append({'tick': tick+1, 'before': old, 'after': list(w.observation()),
                              'action': effect['action'], 'state': state[i].tolist(), 'logits': logits[i].tolist(),
                              'harvested_resource': amount, 'basal_cost': .018, 'action_cost': cost,
                              'empty_harvest_penalty': penalty, 'clipping_adjustment': expected-expected_unclipped,
                              'resource_renewal_at_current_cell': .008*(w.capacity[w.body.position]-(resource-amount)) if a not in (1,2) else None,
                              'viable': effect['viable']})
            rewards[i] += continuing_viability_reward(effect)
            active[i] = effect['viable']
        previous, previous_action = obs, actions
    episodes = [{'seed': seeds[i], 'sample_seed': sample_seeds[i], 'age': w.body.age,
                 'survived': w.viable() and w.body.age == 256, 'reward': rewards[i],
                 'failure_cause': failure_cause(w), 'selected_actions': dict(Counter(t['action'] for t in traces[i])),
                 'trace': traces[i]} for i,w in enumerate(worlds)]
    assert before == sha(campaign['parent_qv0_path'])
    return {'mean_age': float(np.mean([e['age'] for e in episodes])),
            'survival': sum(e['survived'] for e in episodes), 'worlds': n,
            'mean_reward': float(np.mean(rewards)),
            'repertoire': repertoire(sum((Counter(e['selected_actions']) for e in episodes), Counter())),
            'episodes': episodes}


def main():
    if OUT.exists():
        raise RuntimeError('refuse to replace existing diagnostic')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = read('qv1_retention_lineage_verdict_20260906.json')
    left, right = [load_campaign(Path(source[k])) for k in ['left_campaign','right_campaign']]
    assert campaigns_exact(left,right)
    parent = load_artifact(Path(left['parent_qv0_path']))
    paths = [Path(source[k]) for k in ['left_campaign','right_campaign']] + [Path(left['parent_qv0_path'])]
    paths += [REPORTS / n for n in ['qv1_retention_lineage_verdict_20260906.json','pol2_endogenous_action_verdict_20260905.json','dyn1_resilience_20260905.json','embodiment_v2_calibration_20260905.json']]
    paths += [ROOT / 'docs/qv1_postmortem_design_20260906.md', Path(__file__)]
    paths += [Path(m.__file__) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).suffix == '.py' and (str(ROOT / 'core') in str(m.__file__) or str(ROOT / 'training') in str(m.__file__))]
    identities = {str(p.relative_to(ROOT)): sha(p) for p in set(paths)}
    d1 = {}
    for arm, data in left['arms'].items():
        d1[arm] = {'training_rows': data['training']}
        for label, rows in [('first20',data['training'][:20]),('last20',data['training'][-20:])]:
            counts = sum((Counter(r['action_counts']) for r in rows),Counter())
            d1[arm][label] = {'repertoire': repertoire(counts), 'mean_age': float(np.mean([r['mean_age'] for r in rows])), 'survival_fraction': float(np.mean([r['survival_fraction'] for r in rows]))}
    d2 = {c: km(v['episodes']) for c,v in source['conditions'].items()}
    training = [initial(202661000+i) for i in range(1280)]
    held = [initial(202670000+i) for i in range(64)]
    def full_key(row):
        return json.dumps({k:v for k,v in row.items() if k!='seed'},sort_keys=True)
    x = np.array([r['observation']+r['capacity']+r['resources']+[r['ambient_phase']] for r in training])
    y = np.array([r['observation']+r['capacity']+r['resources']+[r['ambient_phase']] for r in held])
    d3 = {'training_initials':training, 'heldout_initials':held, 'seed_overlap':len({r['seed'] for r in training}&{r['seed'] for r in held}), 'full_initial_overlap':len({full_key(r) for r in training}&{full_key(r) for r in held}),
          'feature_order':['energy','integrity','temperature','local_resource','location']+[f'capacity_{i}' for i in range(9)]+[f'resource_{i}' for i in range(9)]+['ambient_phase'],
          'train_min':x.min(0).tolist(),'train_max':x.max(0).tolist(),'heldout_min':y.min(0).tolist(),'heldout_max':y.max(0).tolist(),
          'standardized_mean_difference':((y.mean(0)-x.mean(0))/np.sqrt((x.var(0)+y.var(0))/2)).tolist()}
    indices = [i*1279//63 for i in range(64)]
    d4 = {}
    for arm in left['arms']:
        d4[arm] = {}
        for split, seeds, samples in [('train',[202661000+i for i in indices],[202660000+i for i in indices]),('heldout',[r['seed'] for r in held],[202690000+i for i in range(64)])]:
            for decoding in ['greedy','sampled']:
                cell = replay(arm,left,parent,seeds,samples,decoding)
                d4[arm][f'{split}_{decoding}'] = cell
                print(json.dumps({'arm':arm,'split':split,'decoding':decoding,'age':cell['mean_age'],'survival':cell['survival']}),flush=True)
        for a,b in zip(d4[arm]['heldout_greedy']['episodes'],source['conditions'][arm]['episodes']):
            assert a['seed']==b['seed'] and a['age']==min(256,b['age']) and a['survived']==b['reached_256']
            if b['age']<=256:
                assert a['selected_actions']==b['selected_actions'] and abs(a['reward']-b['reward'])<1e-9 and a['failure_cause']==b['failure_cause']
    rng = np.random.Generator(np.random.PCG64(20260960))
    draws = rng.integers(0,64,(10000,64))
    contrasts = {}
    for arm,cells in d4.items():
        contrasts[arm] = {}
        for split in ['train','heldout']:
            contrasts[arm][split] = {}
            for key in ['age','survived','reward']:
                diff = np.array([float(a[key])-float(b[key]) for a,b in zip(cells[split+'_sampled']['episodes'],cells[split+'_greedy']['episodes'])])
                contrasts[arm][split][key] = {'sampled_minus_greedy':float(diff.mean()),'ci95':np.quantile(diff[draws].mean(1),[.025,.975]).tolist()}
    episodes = d4['inherited_recurrent']['heldout_greedy']['episodes']
    d5 = []
    for e in episodes:
        ts = e['trace']
        d5.append({'seed':e['seed'],'age':e['age'],'initial_energy':ts[0]['before'][0],'final_energy':ts[-1]['after'][0],
                   'basal_cost':sum(t['basal_cost'] for t in ts),'action_cost':sum(t['action_cost'] for t in ts),
                   'harvested_energy':.8*sum(t['harvested_resource'] for t in ts),
                   'empty_harvest_penalty':sum(t['empty_harvest_penalty'] for t in ts),
                   'clipping_adjustment':sum(t['clipping_adjustment'] for t in ts),
                   'first_resource_below_0_03':next((t['tick'] for t in ts if t['after'][3]<.03),None),
                   'movement_count':sum(t['action'] in ['move_left','move_right'] for t in ts)})
    result = {'grade':'diagnostic; no changed formal verdict or continuation authority','source_hashes':identities,
              'exact_campaign_twins':True,'greedy_original_replay_verified':True,'D1':d1,'D2':d2,'D3':d3,'D4':d4,'D4_contrasts':contrasts,'D5':d5}
    assert identities == {str(p.relative_to(ROOT)):sha(p) for p in set(paths)}
    with OUT.open('x') as f:
        json.dump(result,f,allow_nan=False)
    print(json.dumps({'output':str(OUT),'sha256':sha(OUT)}),flush=True)


if __name__ == '__main__':
    main()
