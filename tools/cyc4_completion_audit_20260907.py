"""Independent CYC4 artifact, local-information, physical replay and gate audit."""
import copy
from collections import Counter
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import cyc4_learned_carryover_20260907 as M
from tools import cyc3_completion_audit_20260907 as A
C = M.C
RUN = M.OUT


def input_row(obs):
    return [float(obs[3])] + [float(i == round(8 * obs[4])) for i in range(9)]


def estimates(cache, tick):
    return [.4 if str(i) not in cache else min(1., max(0., .575 +
        (cache[str(i)]['resource'] - .575) * .992 ** (tick - cache[str(i)]['tick']))) for i in range(9)]


def preparation(p):
    assert C.prepare(p['seed'], p['rich_target']) == p
    w = C.restore(p['initial']); cache = {}; A.record(cache, w)
    observations = [list(w.observation())]; targets = [estimates(cache, 0)]
    assert len(p['exposure']) == 16
    for action, row in zip(C.EXPOSURE, p['exposure']):
        assert C.step_account(w, C.Action(action)) == row
        A.record(cache, w); observations.append(list(w.observation())); targets.append(estimates(cache, w.step_count))
    assert C.snapshot(w) == p['before_body_match']
    w.body.energy = .104; w.body.integrity = .95; w.body.temperature = .5
    assert C.snapshot(w) == p['boundary'] and cache == p['cache']
    observations[16] = list(w.observation())
    return observations, targets


def teacher_replay(p, e):
    w = C.restore(p['boundary']); cache = copy.deepcopy(p['cache'])
    assert e['initial_physical'] == p['boundary'] and e['initial_cache'] == cache
    observations = []; targets = []
    for row in e['trace']:
        assert w.viable()
        action = A.independent_choice(w.observation(), w.step_count, cache)
        assert C.Action(action).name.lower() == row['action']
        actual = C.step_account(w, C.Action(action))
        assert actual == {k:v for k,v in row.items() if k != 'decision'}
        A.record(cache, w); observations.append(list(w.observation())); targets.append(estimates(cache, w.step_count))
    assert C.snapshot(w) == e['final_physical'] and cache == e['final_cache']
    check_endpoint(w, e)
    return observations, targets


def check_endpoint(w, episode):
    assert episode['age'] == 16 + len(episode['trace']) == w.body.age
    assert episode['survived_256'] == any(r['tick'] == 256 and r['viable'] for r in episode['trace'])
    assert episode['survived_512'] == (w.body.age == 512 and w.viable())
    assert episode['first_action'] == episode['trace'][0]['action']
    assert episode['failure_cause'] == C.cause(w)
    assert episode['cycles'] == C.closures([4] + [r['position_after'] for r in episode['trace']],
                                         [r['harvested_resource'] for r in episode['trace']])


@torch.no_grad()
def neural_replay(model, p, donor, condition, episode):
    assert episode['condition'] == condition and episode['initial_physical'] == p['boundary']
    source = donor if condition == 'swapped' else p
    obs = [list(C.restore(source['initial']).observation())] + [r['after'] for r in source['exposure'][:15]]
    if condition == 'erased': hidden = torch.zeros(1, 1, 32)
    else: _, hidden = model(torch.tensor([[input_row(o) for o in obs]]))
    assert hidden.tolist() == episode['initial_hidden']
    world = C.restore(p['boundary']); cache = copy.deepcopy(p['cache'])
    mse_sum = 0.; disagreements = 0; tail = Counter()
    for row in episode['trace']:
        assert world.viable()
        x = input_row(world.observation()); prediction, hidden = model(torch.tensor([[x]]), hidden)
        values = prediction[0, 0].tolist()
        assert x == row['input'] and values == row['prediction'] and hidden.tolist() == row['hidden']
        synthetic = {str(i): dict(resource=v, tick=world.step_count) for i, v in enumerate(values)}
        action = A.independent_choice(world.observation(), world.step_count, synthetic)
        assert C.Action(action).name.lower() == row['action']
        A.record(cache, world)
        target = estimates(cache, world.step_count)
        mse_sum += sum((v-t)**2 for v,t in zip(values, target)) / 9
        disagreements += action != A.independent_choice(world.observation(), world.step_count, cache)
        measured = C.step_account(world, C.Action(action))
        assert measured == {k:v for k,v in row.items() if k not in ('input', 'prediction', 'hidden', 'decision')}
    x = input_row(world.observation()); prediction, hidden = model(torch.tensor([[x]]), hidden)
    assert x == episode['final_input'] and prediction[0, 0].tolist() == episode['final_prediction']
    assert hidden.tolist() == episode['final_hidden'] and C.snapshot(world) == episode['final_physical']
    check_endpoint(world, episode)
    if not episode['survived_512']:
        tail.update(r['action'] for r in episode['trace'][-20:])
    return dict(steps=len(episode['trace']), squared_error_sum=mse_sum,
                explicit_choice_disagreements=disagreements, death_tail_actions=dict(tail))


def independent_statistics(pairs, data):
    n = len(pairs); expected = {}; z = 1.959963984540054
    for horizon, threshold in ((256,.90), (512,.80)):
        k = sum(all(o['controls']['intact'][f'survived_{horizon}'] for o in p['orientations']) for p in pairs)
        rate = k/n; d = 1+z*z/n
        ci = [max(0., (rate+z*z/(2*n))/d-z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/d),
              min(1., (rate+z*z/(2*n))/d+z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/d)]
        expected[f'intact_pair_survival_{horizon}'] = dict(successes=k,n=n,point=rate,ci95=ci,
                                                         threshold=threshold,passed=ci[0]>=threshold)
    draws = np.random.Generator(np.random.PCG64(20260993)).integers(0,n,(10000,n))
    for control in ('erased','swapped','untrained'):
        delta = np.array([sum(int(o['controls']['intact']['survived_512'])-
                             int(o['controls'][control]['survived_512']) for o in p['orientations'])/2 for p in pairs])
        ci = np.quantile(delta[draws].mean(axis=1), [.025,.975]).tolist()
        expected['intact_minus_'+control] = dict(point=float(delta.mean()),ci95=ci,threshold=.30,passed=ci[0]>=.30)
    searches = [s for p in pairs for o in p['orientations'] for s in o['searches']]
    expected['information_necessity'] = dict(passed=all(s['status']=='EXHAUSTIVE_NO_SURVIVOR' for s in searches),
        total=len(searches), expanded_transitions=sum(s['expanded_transitions'] for s in searches),
        counterexamples=sum(s['status']=='COUNTEREXAMPLE' for s in searches),capped=sum(s['status']=='UNVERIFIED_CAP' for s in searches))
    ktrain = sum(e['teacher']['survived_512'] for e in data['episodes'])
    keval = sum(o['teacher']['survived_512'] for p in pairs for o in p['orientations'])
    expected['explicit_teacher'] = dict(training_survivors=ktrain,evaluation_survivors=keval,passed=ktrain==128 and keval==256)
    return expected


def main():
    M.configure(); report = A.read(RUN/'verdict.json'); checks = {}
    checks['source_hashes'] = all(A.digest(ROOT/p)==h for p,h in report['manifest']['sources'].items())
    checks['artifact_hashes'] = all(A.digest(RUN/p)==h for p,h in report['artifacts'].items())
    checks['exact_campaign_content'] = A.digest(RUN/'campaign_a.json.gz',True)==A.digest(RUN/'campaign_b.json.gz',True)
    data = torch.load(RUN/'training_data.pt',weights_only=False,map_location='cpu')
    a = torch.load(RUN/'twin_a.pt',weights_only=False,map_location='cpu')
    b = torch.load(RUN/'twin_b.pt',weights_only=False,map_location='cpu')
    checks['exact_weights_optimizer_predictions_and_losses'] = M.exact(a,b)
    torch.manual_seed(20260991); random_model = M.Memory().eval()
    checks['initialization_and_saved_tensor_hashes'] = M.exact(random_model.state_dict(),a['initial']) and all(
        M.state_hash(a[k])==a[k+'_hash']==report[k+'_hash'] for k in ('initial','final'))
    checks['fixed_training_budget'] = a['updates']==1600 and len(a['losses'])==200 and all(
        s['step'].item()==1600 for s in a['optimizer']['state'].values())
    assert len(data['episodes'])==128
    for i, e in enumerate(data['episodes']):
        p=e['preparation']; assert p['seed']==202674000+i//2 and p['rich_target']==(2 if i%2==0 else 6)
        obs, targets = preparation(p); after, ys = teacher_replay(p,e['teacher']); obs+=after; targets+=ys
        valid=len(obs); assert valid==e['valid']
        weights=[8. if t<=16 else 1. for t in range(valid)]
        while len(obs)<513: obs.append(obs[-1]); targets.append(targets[-1]); weights.append(0.)
        assert torch.equal(data['x'][i],torch.tensor([input_row(o) for o in obs]))
        assert torch.equal(data['y'][i],torch.tensor(targets)) and torch.equal(data['weight'][i],torch.tensor(weights))
    checks['all_training_physics_labels_inputs_and_masks'] = True
    model=M.Memory().eval(); model.load_state_dict(a['final'])
    with torch.no_grad(): pred,_=model(data['x'])
    checks['final_teacher_predictions'] = torch.equal(pred,a['teacher_predictions'])
    final_mse=float(((pred-data['y']).square()*data['weight'].unsqueeze(-1)).sum()/(9*data['weight'].sum()))
    del b, pred
    pairs=A.read(RUN/'campaign_a.json.gz'); assert len(pairs)==128
    stats={c:dict(steps=0,squared_error_sum=0.,explicit_choice_disagreements=0,death_tail_actions=Counter(),causes=Counter()) for c in M.CONTROLS}
    search_count=nodes=0
    for i,pair in enumerate(pairs):
        assert pair['seed']==202675000+i and len(pair['orientations'])==2
        os=pair['orientations']; left,right=[o['preparation'] for o in os]
        assert left['observation']==right['observation'] and left['cache'].keys()==right['cache'].keys()
        assert all(left['cache'][k]['tick']==right['cache'][k]['tick'] for k in left['cache'])
        assert {k for k in left['cache'] if left['cache'][k]!=right['cache'][k]}=={'2','6'}
        for oi,o in enumerate(os):
            p=o['preparation']; donor=os[1-oi]['preparation']
            assert p['seed']==pair['seed'] and p['rich_target']==(2 if oi==0 else 6)
            preparation(p); teacher_replay(p,o['teacher'])
            assert o['correct_first_action']==oi+1
            assert len(o['searches'])==5 and {s['first_action'] for s in o['searches']}==set(range(6))-{oi+1}
            for search in o['searches']:
                if search['status']=='EXHAUSTIVE_NO_SURVIVOR':
                    impossible, expanded=A.bfs_certificate(p['boundary'],search['first_action'])
                    assert impossible and expanded==search['expanded_transitions']; nodes+=expanded
                elif search['status']=='COUNTEREXAMPLE':
                    w=C.restore(p['boundary']); assert search['witness'][0]==search['first_action'] and len(search['witness'])==12
                    for action in search['witness']: assert w.viable(); w.step(action)
                    assert w.viable()
                else: assert search['status']=='UNVERIFIED_CAP'
                search_count+=1
            assert set(o['controls'])==set(M.CONTROLS)
            for condition,e in o['controls'].items():
                diagnostic=neural_replay(random_model if condition=='untrained' else model,p,donor,condition,e)
                for key in ('steps','squared_error_sum','explicit_choice_disagreements'): stats[condition][key]+=diagnostic[key]
                stats[condition]['death_tail_actions'].update(diagnostic['death_tail_actions'])
                stats[condition]['causes'].update([e['failure_cause']])
        if (i+1)%16==0: print('audited pairs',i+1,flush=True)
    checks['matched_histories_and_physical_forks'] = True
    checks['all_1024_neural_episodes_and_interventions'] = True
    checks['all_local_inputs_predictions_states_actions_and_physics'] = True
    checks['all_384_training_and_fresh_explicit_teacher_episodes'] = True
    checks['all_1280_search_certificates'] = search_count==1280
    checks['independent_pair_statistics'] = M.exact(independent_statistics(pairs,data),report['bars'])
    expected_pass=all(b['passed'] for b in report['bars'].values())
    checks['binary_verdict_and_no_promotion'] = report['verdict']==('PASS' if expected_pass else 'FAIL') and not report['automatic_followup'] and not report['pillar_promotion']
    recomputed=M.adjudicate(pairs,data)
    checks['reported_summaries_and_failed_requirements'] = all(recomputed[k]==report[k] for k in ('summaries','failed_requirements','consequence'))
    checks['post_audit_source_hashes'] = all(A.digest(ROOT/p)==h for p,h in report['manifest']['sources'].items())
    for s in stats.values(): s['on_policy_nominal_resource_mse']=s['squared_error_sum']/s['steps']
    result=dict(kind='CYC4_COMPLETION_AUDIT',passed=all(checks.values()),checks=checks,
                verdict=report['verdict'],final_checkpoint_teacher_weighted_mse=final_mse,
                diagnostic_only=stats,search_nodes=nodes,source_verdict_sha=A.digest(RUN/'verdict.json'),
                audit_source_sha=A.digest(Path(__file__)))
    C.save(RUN/'completion_audit.json',result)
    print(json.dumps(result),flush=True); assert result['passed']


if __name__=='__main__': main()
