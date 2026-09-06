"""CAP1 diagnostic capture and metrics; no optimization or gate replacement."""
import copy
import gzip
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.embodiment import Action, EmbodiedWorldV2
from core.model import ZeusCore, ZeusConfig
from training.evaluate_homeostatic_policy_v2 import _load_artifact, _load_model, _state_logits
from training.train_homeostatic_policy import viability_reward
from training import evaluate_dynamics_resilience as dyn
from tools.qv1_postmortem_20260906 import repertoire, sha, read, REPORTS

OUT = ROOT / 'runs/cap1_20260906'
MANIFEST = ROOT / 'docs/registrations/cap1_manifest.json'


def save(path, value):
    data = json.dumps(value,allow_nan=False).encode()
    with path.open('xb') as f:
        f.write(gzip.compress(data,mtime=0) if path.suffix=='.gz' else data)


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def dimension(states):
    x = np.asarray(states,dtype=np.float64)
    x = x-x.mean(0)
    gram = x@x.T if len(x)<=x.shape[1] else x.T@x
    eig = np.linalg.eigvalsh(gram).clip(0)[::-1]
    total = eig.sum()
    if total < 1e-20:
        return {'participation_ratio':0.,'rank90':0}
    return {'participation_ratio':float(total**2/(eig@eig)),
            'rank90':int(np.searchsorted(eig.cumsum(),.9*total)+1)}


def grid(observation):
    e,i,t,r,p = observation
    assert e>.05 and i>.05 and .05<t<.95
    scaled = [(e-.05)/.95,(i-.05)/.95,(t-.05)/.9,r]
    return tuple(min(9,max(0,int(v*10))) for v in scaled)+(int(round(p*8)),)


def capacity(outputs, viable):
    n = len({tuple(np.round(row,8)) for row,v in zip(outputs,viable) if v})
    return float(np.log2(n)) if n else 0.


def channel(world):
    before = copy.deepcopy(world.__dict__)
    outputs, viable = [], []
    for action in Action:
        clone = copy.deepcopy(world)
        effect = clone.step(action)
        outputs.append(list(effect['after']))
        viable.append(effect['viable'])
    assert world.__dict__ == before
    responses = np.array(outputs)-np.array(outputs[0])
    return {'outputs':outputs,'viable':viable,'capacity_bits':capacity(outputs,viable),
            'response_rank':int(np.linalg.matrix_rank(responses,tol=1e-8))}


def calibration():
    assert dimension([[1,1],[1,1]]) == {'participation_ratio':0.,'rank90':0}
    one = dimension([[-1,-1],[0,0],[1,1]])
    assert abs(one['participation_ratio']-1)<1e-10 and one['rank90']==1
    iso = dimension([[1,0],[-1,0],[0,1],[0,-1]])
    assert abs(iso['participation_ratio']-2)<1e-10 and iso['rank90']==2
    for n in [1,2,6]:
        assert capacity([[i] for i in range(n)],[True]*n)==float(np.log2(n))
    for a in Action:
        w = EmbodiedWorldV2(seed=20260961)
        c = channel(w)
        effect = w.step(a)
        assert c['outputs'][a]==list(effect['after']) and c['viable'][a]==effect['viable']
    return {'constant_rank_one_isotropic':True,'constant_two_six_output_channels':True,'clone_purity_and_selected_transition':True}


def register():
    if MANIFEST.exists():
        raise RuntimeError('manifest exists')
    paths = list((ROOT/'core').rglob('*.py'))+list((ROOT/'training').glob('*.py'))
    paths += [Path(__file__),ROOT/'tools/qv1_postmortem_20260906.py',ROOT/'docs/capability_components_protocol_20260906.md']
    paths += [REPORTS/n for n in ['pol2_endogenous_action_verdict_20260905.json','dyn1_resilience_20260905.json','embodiment_v2_calibration_20260905.json','qv1_postmortem_20260906.json.gz']]
    pol=read('pol2_endogenous_action_verdict_20260905.json')
    paths += [Path(pol['source_checkpoint']),Path(pol['artifacts']['a']['path']),Path(pol['artifacts']['b']['path']),ROOT/'corpus/data/tokenizer/bpe_8192.json']
    save(MANIFEST,{'artifacts':{str(p.relative_to(ROOT)):sha(p) for p in paths},'torch':str(torch.__version__),'numpy':np.__version__})


def check():
    manifest=load(MANIFEST)
    assert manifest['torch']==str(torch.__version__) and manifest['numpy']==np.__version__
    for p,h in manifest['artifacts'].items():
        if sha(ROOT/p)!=h:
            raise RuntimeError('source drift '+p)
    return sha(MANIFEST)


@torch.no_grad()
def capture_pol2():
    target=OUT/'pol2.json.gz'
    if target.exists():
        raise RuntimeError('capture already exists')
    pol=read('pol2_endogenous_action_verdict_20260905.json')
    cal=read('embodiment_v2_calibration_20260905.json')
    artifact=_load_artifact(Path(pol['artifacts']['a']['path']))
    model=_load_model(Path(pol['source_checkpoint']),artifact,device='cpu',trained=True)
    result={}
    for mode in ['normal','fixed_rest','fixed_harvest']:
        episodes=[]
        for index,seed in enumerate(pol['conditions']['eval_seeds']):
            world=EmbodiedWorldV2(seed=seed)
            if mode=='normal':
                model.reset_state()
            states,observations,actions,channels=[],[],[],[]
            reward=0.
            for tick in range(256):
                obs=list(world.observation())
                if mode=='normal':
                    model.sense_body(obs,emit_readout=False)
                    logits=_state_logits(model,'normal',None)
                    a=int(logits.argmax())
                    states.append(model.S.tolist())
                else:
                    a=int(Action.REST if mode=='fixed_rest' else Action.HARVEST)
                observations.append(obs)
                ch=channel(world)
                channels.append(ch)
                actions.append(a)
                effect=world.step(a)
                assert list(effect['after'])==ch['outputs'][a] and effect['viable']==ch['viable'][a]
                reward+=viability_reward(effect)
                if not effect['viable']:
                    break
            counts=dict(Counter(Action(a).name.lower() for a in actions))
            original=(pol['evaluations']['normal'] if mode=='normal' else cal['results'][mode])['episodes'][index]
            assert original['seed']==seed and original['age']==world.body.age and original['completed']==(world.viable() and world.body.age==256)
            assert original['selected_actions' if mode=='normal' else 'actions']==counts and abs(original['reward']-reward)<1e-9
            episodes.append({'seed':seed,'age':world.body.age,'completed':world.viable() and world.body.age==256,
                             'reward':reward,'states':states,'observations':observations,'actions':actions,'channels':channels})
            if (index+1)%8==0:
                print(json.dumps({'phase':'CAP1_POL2','mode':mode,'worlds':index+1,'total':64}),flush=True)
        result[mode]=episodes
    save(target,{'source_manifest':check(),'original_replay_verified':True,'conditions':result})


class Observer:
    def __init__(self,model):
        object.__setattr__(self,'inner',model)
        object.__setattr__(self,'states',[])
    def __getattr__(self,name):
        return getattr(self.inner,name)
    def __setattr__(self,name,value):
        setattr(self.inner,name,value)
    def step(self,*args,**kwargs):
        result=self.inner.step(*args,**kwargs)
        self.states.append(self.inner.S.detach().cpu().clone())
        return result


@torch.no_grad()
def capture_dyn1():
    target=OUT/'dyn1.json.gz'
    if target.exists():
        raise RuntimeError('capture already exists')
    original=read('dyn1_resilience_20260905.json')
    checkpoint=Path(original['artifact'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    result={}
    for arm in ['trained','random_init']:
        records=[]
        trained=ZeusCore.load(checkpoint,device='cpu') if arm=='trained' else None
        for index,seed in enumerate(dyn.SEEDS):
            if arm=='random_init':
                torch.manual_seed(seed)
                model=ZeusCore(ZeusConfig(**payload['config']),tokenizer_path=payload.get('tokenizer')).eval()
            else:
                model=trained
            observer=Observer(model)
            row=dyn.evaluate_pair(observer,seed)
            old=original[arm]['records'][index]
            assert len(observer.states)==640 and row['pre_kick_state_equal']
            for branch,start,end in [('control',320,384),('perturbed',576,640)]:
                for k,v in row[branch]['features'].items():
                    expected=old[branch]['features'][k]
                    assert (v==expected if k=='effective_rank_90' else np.isclose(v,expected,rtol=1e-5,atol=1e-7)),(arm,seed,branch,k,v,expected)
                row[branch]['states']=torch.stack(observer.states[start:end]).tolist()
            records.append(row)
            print(json.dumps({'phase':'CAP1_DYN1','arm':arm,'seeds':index+1,'total':16}),flush=True)
        result[arm]=records
    save(target,{'source_manifest':check(),'original_feature_replay_verified':True,'conditions':result})


def metrics(e):
    counts=Counter(Action(a).name.lower() for a in e['actions'])
    bins={grid(o) for o in e['observations']}
    return {'seed':e['seed'],'age':e['age'],'repertoire':repertoire(counts),
            'occupied_viable_cells':len(bins),'viable_grid_fraction':len(bins)/90000,
            'dimension':dimension(e['states']) if e['states'] else {'participation_ratio':0.,'rank90':0},
            'mean_viable_capacity_bits':float(np.mean([c['capacity_bits'] for c in e['channels']])),
            'mean_local_response_rank':float(np.mean([c['response_rank'] for c in e['channels']]))}


def summarize():
    target=REPORTS/'capability_components_20260906.json'
    if target.exists():
        raise RuntimeError('summary exists')
    pol=load(OUT/'pol2.json.gz')
    dy=load(OUT/'dyn1.json.gz')
    qv=load(REPORTS/'qv1_postmortem_20260906.json.gz')
    pm={k:[metrics(e) for e in eps] for k,eps in pol['conditions'].items()}
    qm={}
    for arm,cells in qv['D4'].items():
        qm[arm]={}
        for cell,data in cells.items():
            episodes=[]
            for e in data['episodes']:
                w=EmbodiedWorldV2(seed=e['seed'])
                channels=[]
                for t in e['trace']:
                    assert list(w.observation())==t['before']
                    ch=channel(w)
                    a=int(Action[t['action'].upper()])
                    effect=w.step(a)
                    assert list(effect['after'])==t['after']==ch['outputs'][a]
                    channels.append(ch)
                derived={'seed':e['seed'],'age':e['age'],'states':[t['state'] for t in e['trace']],
                         'observations':[t['before'] for t in e['trace']],
                         'actions':[int(Action[t['action'].upper()]) for t in e['trace']], 'channels':channels}
                episodes.append(metrics(derived))
                # Publish causal outputs, not just the channel scalar.
                e['capability_channels']=channels
            qm[arm][cell]=episodes
            print(json.dumps({'phase':'CAP1_QV1_metrics','arm':arm,'cell':cell}),flush=True)
    save(OUT/'qv1_channels.json.gz',{'source_manifest':check(),'conditions':{a:{c:[{'seed':e['seed'],'channels':e['capability_channels']} for e in d['episodes']] for c,d in cells.items()} for a,cells in qv['D4'].items()}})
    dm={a:[{'seed':e['seed'],**{b:dimension(e[b]['states']) for b in ['control','perturbed']}} for e in eps] for a,eps in dy['conditions'].items()}
    draws=np.random.Generator(np.random.PCG64(20260961)).integers(0,64,(10000,64))
    comparisons={}
    for control in ['fixed_rest','fixed_harvest']:
        values={'repertoire':[], 'coverage_cells':[], 'capacity_bits':[], 'common_prefix_coverage_cells':[]}
        for i in range(64):
            a,b=pm['normal'][i],pm[control][i]
            values['repertoire'].append(a['repertoire']['entropy_effective_actions']-b['repertoire']['entropy_effective_actions'])
            values['coverage_cells'].append(a['occupied_viable_cells']-b['occupied_viable_cells'])
            values['capacity_bits'].append(a['mean_viable_capacity_bits']-b['mean_viable_capacity_bits'])
            n=min(pol['conditions'][k][i]['age'] for k in ['normal','fixed_rest','fixed_harvest'])
            values['common_prefix_coverage_cells'].append(len({grid(o) for o in pol['conditions']['normal'][i]['observations'][:n]})-len({grid(o) for o in pol['conditions'][control][i]['observations'][:n]}))
        comparisons[control]={}
        for name,vs in values.items():
            x=np.array(vs,dtype=float)
            comparisons[control][name]={'mean_difference':float(x.mean()),'ci95':np.quantile(x[draws].mean(1),[.025,.975]).tolist()}
    separation=all(v['repertoire']['ci95'][0]>0 for v in comparisons.values())
    result={'status':'MEASUREMENT COMPONENTS VALIDATED' if separation else 'NOT READY',
            'class':'O','continuation_authority':False,'source_manifest':check(),'calibration':calibration(),
            'capture_fidelity':{'POL2':pol['original_replay_verified'],'DYN1':dy['original_feature_replay_verified'],'QV1':True},
            'POL2':pm,'QV1':qm,'DYN1':dm,'POL2_paired_comparisons':comparisons,
            'limits':['Coverage is a declared viable observation grid, not a reachable viability kernel.',
                      'Capacity is the one-step body/world channel, not learned policy access to it.',
                      'Response rank is one-step, not long-horizon controllability.',
                      'DYN1 has no body/action channel: viability and empowerment are not applicable.',
                      'No combined score, pillar pass, or follow-up experiment is licensed.'],
            'capture_hashes':{p.name:sha(p) for p in OUT.glob('*.gz')}}
    save(target,result)
    print(json.dumps({'status':result['status'],'comparisons':comparisons}),flush=True)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    phase=sys.argv[1]
    if phase=='register':
        register()
        return
    check()
    calibration()
    OUT.mkdir(exist_ok=True)
    {'pol2':capture_pol2,'dyn1':capture_dyn1,'summarize':summarize}[phase]()


if __name__=='__main__':
    main()
