"""Explicit QL1 phases; no training/evaluation is performed at import.

Artifacts are exclusive. A partial phase is preserved and never silently retried.
There is no mid-life resume in this pilot runner.
"""
from contextlib import contextmanager
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from core.quality_agent import QualityAgent, QualitySession, model_hash
from core.lifetime_world_v2 import QualityWorld, QualityInterface
from training.persistent_learning import SequenceBatch, update_segment
from training import quality_learning_contract as K

OUT = ROOT / 'runs/ql1_20260909'
SOURCES = ('core/persistent_agent.py', 'core/persistent_session.py', 'core/quality_agent.py',
           'core/lifetime_world.py', 'core/lifetime_world_v2.py', 'training/persistent_learning.py',
           'training/quality_learning_contract.py', 'training/run_quality_learning.py',
           'training/test_quality_learning.py', 'docs/ql1_learning_protocol_20260909.md')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, separators=(',', ':'), allow_nan=False)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def tree_hash(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(b'tensor'); visit(str(x.dtype)); visit(tuple(x.shape))
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            h.update(b'dict')
            for key in sorted(x, key=lambda k: (type(k).__name__, repr(k))): visit(key); visit(x[key])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode()); visit(len(x))
            for item in x: visit(item)
        else:
            h.update((type(x).__name__ + ':' + json.dumps(x, allow_nan=False) + ';').encode())
    visit(value)
    return h.hexdigest()


@contextmanager
def trace_writer(path):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='\n') as stream:
                yield stream


def log(stream, value):
    stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')


def configure():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def verify(manifest):
    assert all(sha(ROOT / name) == digest for name, digest in manifest['sources'].items())
    assert manifest['config'] == json.loads(json.dumps(K.registered_config()))
    assert manifest['torch'] == torch.__version__ and manifest['numpy'] == np.__version__


def prepare():
    if OUT.exists():
        manifest = read(OUT / 'manifest.json'); verify(manifest); return manifest
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', *SOURCES], cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError('commit all registered sources before preparing QL1')
    OUT.mkdir()
    manifest = dict(config=K.registered_config(), sources={p: sha(ROOT / p) for p in SOURCES},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        torch=torch.__version__, numpy=np.__version__)
    save(OUT / 'manifest.json', manifest); return read(OUT / 'manifest.json')


def train_one(directory, init_seed):
    directory.mkdir()
    torch.manual_seed(init_seed); model = QualityAgent()
    initial = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k,v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=K.LEARNING_RATE, weight_decay=.01, foreach=False, fused=False)
    generator = torch.Generator().manual_seed(init_seed + 1000)
    episodes = []
    with trace_writer(directory / 'trace.jsonl.gz') as trace:
        for index, seed in enumerate(K.TRAIN_SEEDS):
            world = QualityWorld(seed=seed, changing=K.episode_changing(index)); interface = QualityInterface(world)
            session = QualitySession(model); session.begin_episode(); tick = 0; score = 0.; updates = []
            log(trace, dict(kind='start', episode=index, world=world.snapshot(), model_revision=int(model.revision)))
            while interface.viable() and tick < K.HORIZON:
                rows = []
                for _ in range(min(K.CHUNK, K.HORIZON - tick)):
                    before_state = session.state
                    action, output = session.act(interface.observation(), generator=generator)
                    effect = interface.step(action); r = K.reward(effect); tick += 1; score += r
                    rows.append(session.record_outcome(effect.after, reward=r, terminated=effect.terminated,
                        truncated=tick == K.HORIZON and not effect.terminated))
                    log(trace, dict(kind='step', episode=index, tick=tick, action=action, reward=r,
                        observation=effect.before.values(), next_observation=effect.after.values(),
                        mask=effect.after.prediction_mask(), state_before=before_state.tolist(), state=output.state.tolist(),
                        logits=output.logits.tolist(), value=output.value.tolist(), predictions=output.predictions.tolist(),
                        terminated=effect.terminated))
                    if effect.terminated: break
                updates.append(update_segment(session, optimizer, SequenceBatch.from_transitions(rows),
                                               K.SETTINGS, max_grad_norm=K.MAX_GRAD_NORM))
                if interface.viable() and tick < K.HORIZON: session.refresh_state()
            episode = dict(index=index, seed=seed, changing=K.episode_changing(index), ticks=tick,
                           survived=interface.viable(), reward=score, updates=updates, final_world=world.snapshot())
            episodes.append(episode); log(trace, dict(kind='end', episode=episode))
            if (index + 1) % 16 == 0: print('development', init_seed, index + 1, flush=True)
    payload = dict(initial=initial, final=model.state_dict(), optimizer=optimizer.state_dict(),
                   sampling_rng=generator.get_state(), episodes=episodes, init_seed=init_seed)
    torch.save(payload, directory / 'checkpoint.pt')
    save(directory / 'completion.json', dict(logical_hash=tree_hash(payload), model_hash=model_hash(model),
        trace_sha=sha(directory / 'trace.jsonl.gz'), checkpoint_sha=sha(directory / 'checkpoint.pt')))


def checked_checkpoint(directory):
    complete = read(directory / 'completion.json')
    assert sha(directory / 'checkpoint.pt') == complete['checkpoint_sha']
    assert sha(directory / 'trace.jsonl.gz') == complete['trace_sha']
    payload = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert tree_hash(payload) == complete['logical_hash']
    return payload, complete


def train(manifest):
    training = OUT / 'training'; training.mkdir()
    for index, seed in enumerate(K.INITIALIZATIONS):
        for twin in K.TWINS:
            verify(manifest); train_one(training / f'{index}_{twin}', seed)
        _, a = checked_checkpoint(training / f'{index}_a'); _, b = checked_checkpoint(training / f'{index}_b')
        assert a['logical_hash'] == b['logical_hash'] and a['trace_sha'] == b['trace_sha'], 'exact twin mismatch'
    save(training / 'completion.json', dict(passed=True, twins=4, sources=manifest['sources']))


def evaluate(manifest):
    assert read(OUT / 'training/completion.json')['passed']
    # Verify all twins before opening any held-out lifetime.
    parents = []
    for i in range(4):
        parent, a = checked_checkpoint(OUT / f'training/{i}_a'); _, b = checked_checkpoint(OUT / f'training/{i}_b')
        assert a['logical_hash'] == b['logical_hash'] and a['trace_sha'] == b['trace_sha']
        parents.append(parent)
    directory = OUT / 'evaluation'; directory.mkdir(); rows = []
    with trace_writer(directory / 'trace.jsonl.gz') as trace:
        for trial, parent in enumerate(parents):
            for index, seed in enumerate(K.EVALUATION_SEEDS):
                for changing in (False, True):
                    for arm in K.ARMS:
                        model = QualityAgent(); model.load_state_dict(parent['initial' if arm == 'initial_model' else 'final'])
                        model.eval().requires_grad_(False); session = QualitySession(model, fixed_weights=True); session.begin_episode()
                        generator = torch.Generator().manual_seed(202686000 + trial * 1000 + 2 * index + int(changing))
                        world = QualityWorld(seed=seed, changing=changing); interface = QualityInterface(world)
                        key = dict(trial=trial, seed=seed, changing=changing, arm=arm)
                        log(trace, dict(kind='start', **key, world=world.snapshot(), model_hash=model_hash(model)))
                        inspections = 0; unsafe = 0; score = 0.; tick = 0
                        while interface.viable() and tick < K.HORIZON:
                            if arm == 'reset_history': session.erase_history()
                            action, output = session.act(interface.observation(), generator=generator)
                            before = world.snapshot()
                            # Audit-only contamination count: never an actor input or reward bonus.
                            if action == 3 and before['position'] in (0,4) and before['quality'][before['position']//4] == 0 and before['resources'][before['position']//4] > 0: unsafe += 1
                            effect = interface.step(action); r = K.reward(effect); score += r; tick += 1
                            inspections += action == 4
                            session.record_outcome(effect.after, reward=r, terminated=effect.terminated,
                                truncated=tick == K.HORIZON and not effect.terminated)
                            log(trace, dict(kind='step', **key, tick=tick, action=action, reward=r,
                                observation=effect.before.values(), next_observation=effect.after.values(),
                                mask=effect.after.prediction_mask(), state=output.state.tolist(), logits=output.logits.tolist(),
                                value=output.value.tolist(), predictions=output.predictions.tolist(), terminated=effect.terminated))
                        session.verify_fixed_weights()
                        row = dict(**key, survived=interface.viable(), ticks=tick, inspections=inspections,
                                   unsafe_harvests=unsafe, reward=score, final_world=world.snapshot())
                        rows.append(row); log(trace, dict(kind='end', episode=row))
                print('evaluated', trial, seed, flush=True)
    save(directory / 'results.json', dict(episodes=rows))
    verify(manifest)
    save(directory / 'completion.json', dict(results_sha=sha(directory / 'results.json'), trace_sha=sha(directory / 'trace.jsonl.gz')))


def adjudicate(rows):
    if any(type(r['survived']) is not bool for r in rows):
        raise ValueError('survival outcomes must be booleans')
    expected = {(trial, seed, changing, arm) for trial in range(4) for seed in K.EVALUATION_SEEDS for changing in (False,True) for arm in K.ARMS}
    keyed = {(r['trial'],r['seed'],r['changing'],r['arm']):r for r in rows}
    if len(rows) != len(expected) or set(keyed) != expected: raise ValueError('incomplete/duplicated endpoint')
    counts = {str(trial): {str(changing): {arm:sum(keyed[trial,seed,changing,arm]['survived'] for seed in K.EVALUATION_SEEDS)
               for arm in K.ARMS} for changing in (False,True)} for trial in range(4)}
    viable = all(counts[str(t)][str(c)]['intact'] >= 58 for t in range(4) for c in (False,True))
    rng = np.random.default_rng(202687000); mi = rng.integers(0,4,(10000,4)); wi = rng.integers(0,64,(10000,64))
    effects = {}
    for control in ('reset_history','initial_model'):
        matrix = np.array([[int(keyed[t,s,True,'intact']['survived'])-int(keyed[t,s,True,control]['survived']) for s in K.EVALUATION_SEEDS] for t in range(4)])
        boot = matrix[mi[:,:,None],wi[:,None,:]].mean((1,2)); bounds = np.quantile(boot,[.025,.975])
        effects[control] = dict(matrix=matrix.tolist(), mean=float(matrix.mean()), bounds=bounds.tolist(),
                               passed=bool(matrix.mean() >= .10 and bounds[0] > 0))
    learning = viable and effects['initial_model']['passed']
    history = learning and effects['reset_history']['passed']
    return dict(counts=counts,effects=effects,learned_viability='PASS' if learning else 'FAIL',
                learned_history_benefit='PASS' if history else 'FAIL', pillar_promotion=False)


def finalize(manifest):
    c = read(OUT / 'evaluation/completion.json')
    assert c['results_sha'] == sha(OUT / 'evaluation/results.json') and c['trace_sha'] == sha(OUT / 'evaluation/trace.jsonl.gz')
    result = adjudicate(read(OUT / 'evaluation/results.json')['episodes']); verify(manifest)
    save(OUT / 'provisional_verdict.json', dict(**result, independent_audit_required=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','train','evaluate','finalize'))
    args = parser.parse_args(); configure(); manifest = prepare()
    if args.phase != 'prepare':
        try: globals()[args.phase](manifest)
        except Exception as exc:
            path = OUT / f'{args.phase}_invalid.json'
            if not path.exists(): save(path, dict(error=repr(exc)))
            raise


if __name__ == '__main__': main()
