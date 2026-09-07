"""Post-result replay audit, not a new endpoint or a functional qualification."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.model import ZeusCore


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def js(a, b):
    arrays = [v.double().numpy() for v in (a, b)]
    p, q = [np.exp(v - np.max(v)) / np.exp(v - np.max(v)).sum() for v in arrays]
    mid = (p + q) / 2
    return float(sum(np.sum(v * (np.log(np.maximum(v, 1e-300)) - np.log(np.maximum(mid, 1e-300)))) for v in (p, q)) / 2)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = ROOT / 'runs/sn2_20260907'
    report = json.loads((folder / 'report.json').read_text())
    checks = {}
    for name, expected in report['manifest']['sources'].items():
        assert sha(ROOT / name) == expected, name
    assert sha(folder / 'observations.pt') == report['observation_sha']
    checks['source_and_observation_hashes'] = True
    saved = torch.load(folder / 'observations.pt', map_location='cpu', weights_only=False)
    assert len(saved) == len(report['rows']) == 10
    torch.manual_seed(report['manifest']['construction_seed'])
    initial_buffers = torch.load(folder / 'initial_buffers.pt', map_location='cpu', weights_only=True)
    assert sha(folder / 'initial_buffers.pt') == report['initial_buffers_sha']
    model = ZeusCore.load(str(ROOT / report['manifest']['config']['checkpoint']), 'cpu').eval()
    with torch.no_grad():
        for item, row in zip(saved, report['rows']):
            assert (item['prompt'], item['seed']) == (row['prompt'], row['seed'])
            st = item['snapshot']
            for name, buffer in model.named_buffers():
                buffer.copy_(initial_buffers[name])
                assert torch.equal(buffer, initial_buffers[name])
            model.reset_state(.12, torch.Generator('cpu').manual_seed(item['seed']))
            model.E_hist.zero_()
            model.deploy_self_source = False
            ids = model.encode(item['prompt'])
            if not report['manifest']['config']['skip_pad_window'] and ids:
                model.pad_window(ids[0])
            model.ingest(ids)
            replay = model.snapshot_runtime()
            for key, value in st.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, replay[key]), key
                else:
                    assert value == replay[key], key
            for mode, expected in item['logits'].items():
                s = torch.zeros_like(st['S']) if mode in ('zero_s', 'zero_both', 'self_source') else st['S']
                h = None if mode == 'self_source' else torch.zeros_like(st['H']) if mode in ('zero_h', 'zero_both') else st['H']
                tokens = st['E_hist'].flip(0) if mode == 'reverse_tokens' else st['E_hist']
                last = tokens[-1] if mode == 'reverse_tokens' else st['last_e']
                actual = model.readout(s, h, last, tokens, st['anchor_vec'])
                assert torch.equal(actual, expected), (item['prompt'], item['seed'], mode)
                assert int(actual.argmax()) == row['argmax'][mode]
                if mode != 'coupled':
                    assert abs(js(item['logits']['coupled'], actual) - row['js'][mode]) < 1e-12
        checks['all_ten_preparations_replayed'] = True
        checks['all_sixty_direct_readout_logits_exact'] = True
        checks['independent_numpy_js_and_argmax'] = True
    count = 0
    for i, pair in enumerate(report['initialization_pairs']):
        a, b = saved[2*i:2*i+2]
        for key in ('E_hist', 'last_e', 'anchor_vec'):
            u, v = a['snapshot'][key], b['snapshot'][key]
            assert torch.equal(u, v) if isinstance(u, torch.Tensor) else u == v
        assert torch.equal(a['logits']['self_source'], b['logits']['self_source'])
        divergence = js(a['logits']['coupled'], b['logits']['coupled'])
        assert abs(divergence - pair['initialization_js']) < 1e-12
        differs = int(a['logits']['coupled'].argmax()) != int(b['logits']['coupled'].argmax())
        assert differs == pair['initialization_argmax_differs']
        count += differs
    assert count == report['summary']['initialization_argmax_disagreements']
    checks['paired_token_identity_and_self_source_invariance'] = True
    full = np.stack([a['logits']['coupled'].double().numpy() for a in saved]).ravel()
    zero = np.stack([a['logits']['zero_both'].double().numpy() for a in saved]).ravel()
    residual = full - zero
    quantities = dict(full=np.var(full), zero_both=np.var(zero), residual=np.var(residual),
                      covariance=np.mean((zero-zero.mean())*(residual-residual.mean())))
    for name, value in quantities.items():
        assert np.isclose(value, report['variance'][name], rtol=1e-12, atol=1e-12), name
    assert np.isclose(quantities['full'], quantities['zero_both']+quantities['residual']+2*quantities['covariance'])
    checks['independent_variance_covariance_identity'] = True
    for name, expected in report['manifest']['sources'].items():
        assert sha(ROOT / name) == expected, name
    checks['post_audit_sources'] = True
    output = dict(kind='SN2_COMPLETION_AUDIT', passed=True, diagnostic_only=True, checks=checks,
                  report_sha=sha(folder / 'report.json'), audit_source_sha=sha(Path(__file__)))
    with (folder / 'completion_audit.json').open('x') as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
