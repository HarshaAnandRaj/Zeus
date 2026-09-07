"""Mechanics only: synthetic data/nonregistered world seeds, no fitting."""
import unittest
import torch
from tools import cyc4_learned_carryover_20260907 as M


class Mechanics(unittest.TestCase):
    def test_features_exclude_body_and_preserve_local_information(self):
        a = [.1, .9, .5, .28, .25]
        b = [.8, .1, .2, .28, .25]
        self.assertEqual(M.features(a), M.features(b))
        self.assertEqual(M.features(a), [.28, 0., 0., 1., 0., 0., 0., 0., 0., 0.])

    def test_teacher_data_timing_targets_and_padding(self):
        p = M.C.prepare(71, 2)
        # A truncated fixture exercises masking, without treating it as a valid endpoint.
        teacher = dict(trace=[])
        x, y, w, valid = M.data_rows(p, teacher)
        self.assertEqual(valid, 17)
        self.assertEqual(len(x), 513)
        self.assertEqual(w[:17], [8.] * 17)
        self.assertEqual(w[17:], [0.] * 496)
        self.assertEqual(x[16], M.features(p['observation']))
        self.assertEqual(y[16], [M.C.estimate(p['cache'], i, 16) for i in range(9)])

    def test_memory_gradient_reaches_early_history(self):
        torch.manual_seed(7); model = M.Memory()
        x = torch.randn(2, 17, 10, requires_grad=True)
        prediction, _ = model(x)
        prediction[:, -1].sum().backward()
        self.assertGreater(x.grad[:, 0].abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_sequential_and_sequence_recurrence_agree(self):
        torch.manual_seed(9); model = M.Memory().eval(); x = torch.randn(1, 17, 10)
        with torch.no_grad():
            full, h = model(x); state = None; outputs = []
            for i in range(17):
                p, state = model(x[:, i:i+1], state); outputs.append(p)
        torch.testing.assert_close(full, torch.cat(outputs, 1), atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(h, state, atol=1e-7, rtol=1e-6)

    def test_selective_history_forks(self):
        a, b = M.C.prepare(73, 2), M.C.prepare(73, 6)
        torch.manual_seed(11); model = M.Memory().eval()
        histories = [M.history(model, p) for p in (a, b)]
        self.assertFalse(torch.equal(*histories))
        self.assertEqual(M.features(a['observation']), M.features(b['observation']))
        for condition, expected in [('intact', histories[0]), ('erased', torch.zeros(1, 1, 32)), ('swapped', histories[1])]:
            result = M.neural_rollout(model, a, b, condition)
            self.assertEqual(result['initial_hidden'], expected.tolist())
            self.assertEqual(result['initial_physical'], a['boundary'])
            self.assertEqual(result['trace'][0]['input'], M.features(a['observation']))

    def test_supplied_controller_reads_predicted_resources(self):
        obs = [.104, .95, .5, .0028, .5]; pred = [.01] * 9
        pred[2] = .6
        self.assertEqual(int(M.choose(obs, 16, pred)[0]), 1)
        pred[2], pred[6] = pred[6], pred[2]
        self.assertEqual(int(M.choose(obs, 16, pred)[0]), 2)
        obs[2] = .8
        self.assertEqual(int(M.choose(obs, 16, pred)[0]), 4)

    def test_pair_level_binary_gate_and_controls(self):
        pairs = []
        for i in range(128):
            orientations = []
            for correct in (1, 2):
                controls = {c: dict(survived_256=c == 'intact', survived_512=c == 'intact',
                    age=512 if c == 'intact' else 20, first_action='move_left') for c in M.CONTROLS}
                orientations.append(dict(correct_first_action=correct, controls=controls,
                    teacher=dict(survived_512=True), searches=[dict(first_action=a,
                    status='EXHAUSTIVE_NO_SURVIVOR', expanded_transitions=1)
                    for a in range(6) if a != correct]))
            pairs.append(dict(seed=M.CONFIG['eval_seed'] + i, orientations=orientations))
        data = dict(episodes=[dict(teacher=dict(survived_512=True)) for _ in range(128)])
        result = M.adjudicate(pairs, data)
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(result['bars']['intact_pair_survival_512']['n'], 128)
        for p in pairs:
            p['orientations'][1]['controls']['intact']['survived_512'] = False
        result = M.adjudicate(pairs, data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['bars']['intact_pair_survival_512']['successes'], 0)


if __name__ == '__main__': unittest.main()
