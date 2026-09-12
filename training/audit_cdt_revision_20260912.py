"""Read-only counterexamples for the current external CDT diagnostic code."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CDT = ROOT.parent / 'Configuration Drift Hypothesis'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def phase_preserving_surrogate(y, seed):
    spectrum = np.fft.rfft(y, axis=0)
    phases = np.random.default_rng(seed).uniform(0, 2*np.pi, len(spectrum))
    phases[0] = 0
    if len(y) % 2 == 0:
        phases[-1] = 0
    return np.fft.irfft(spectrum * np.exp(1j*phases)[:, None], n=len(y), axis=0)


def main():
    sys.path.insert(0, str(CT := CDT))
    probe = module('review_gamma', CT / 'gamma_probe.py')
    capacity = module('review_capacity', CT / 'scale_capacity.py')
    captured = []
    original_rate = probe._distant_rate
    probe._distant_rate = lambda y, eps, tau: captured.append(y.copy()) or 0.
    rows = []
    for n in (255, 256):
        t = np.arange(n)
        y = np.column_stack([np.sin(2*np.pi*5*t/n), np.cos(2*np.pi*5*t/n)])
        y += np.array([-2., 3.])
        probe._phase_surr_rate(y, .1, 8, np.random.default_rng(20260912))
        old = captured[-1]; fixed = phase_preserving_surrogate(y, 20260912)
        spectrum = np.fft.rfft(y, axis=0)
        cross = spectrum[:, :, None] * spectrum[:, None, :].conj()
        measures = {}
        for name, value in [('original', y), ('current_control', old), ('corrected_comparison', fixed)]:
            f = np.fft.rfft(value, axis=0)
            measures[name] = dict(correlation=float(np.corrcoef(value.T)[0, 1]),
                centered_rank=int(np.linalg.matrix_rank(value-value.mean(0))), mean=value.mean(0).tolist(),
                max_power_error=float(np.max(abs(abs(f)**2-abs(spectrum)**2))),
                max_cross_spectrum_error=float(np.max(abs(f[:, :, None]*f[:, None, :].conj()-cross))))
        assert measures['original']['centered_rank'] == 2
        assert measures['current_control']['centered_rank'] == 1
        assert measures['corrected_comparison']['centered_rank'] == 2
        assert measures['current_control']['max_cross_spectrum_error'] > 1
        assert measures['corrected_comparison']['max_cross_spectrum_error'] < 1e-8
        assert np.allclose(fixed.mean(0), y.mean(0))
        rows.append(dict(n=n, measures=measures))
    probe._distant_rate = original_rate
    floors = []
    for h in (4, 8, 16, 32, 64):
        for radius in (.4, 1.5):
            score = capacity.probe_alive(h, radius, 20260912)
            assert score == 1/300
            floors.append(dict(hidden=h, radius=radius, score=score))
    sources = ['gamma_probe.py', 'scale_capacity.py', 'alive_reward_demo.py',
               'zeus_adaptive_dim.py', 'configuration_drift_theory.md', 'configuration_drift_theorem.md']
    result = dict(grade='Deterministic implementation counterexamples, not a Zeus endpoint',
        sources={s: hashlib.sha256((CDT/s).read_bytes()).hexdigest() for s in sources},
        phase_control=rows, capacity_probe=floors,
        phase_cross_spectrum_claim='FAIL', capacity_floor_identifiability='FAIL',
        corrected_comparison_invariants='PASS', external_files_modified=False,
        limitation='Does not recompute Zeus gamma scores or validate a replacement null calibration.')
    target = ROOT / 'zeus_sandbox/universe/reports/cdt_revision_audit_20260912.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
