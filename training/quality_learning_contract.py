"""Fixed QL1 first-pilot settings. Importing this file launches nothing."""
from training.persistent_learning import LossSettings

VERSION = 'ql1-fixed-development-and-evaluation-20260909'
INITIALIZATIONS = (202684000, 202684001, 202684002, 202684003)
TRAIN_SEEDS = tuple(range(202682000, 202682256))
EVALUATION_SEEDS = tuple(range(202683000, 202683064))
TWINS = ('a', 'b')
HORIZON = 1024
CHUNK = 64
LEARNING_RATE = .0003
MAX_GRAD_NORM = 1.
SETTINGS = LossSettings(gamma=.995, gae_lambda=.95, value_weight=.5,
                        prediction_weight=1., entropy_weight=.02)
ARMS = ('intact', 'reset_history', 'initial_model')


def reward(effect):
    """Externally supplied viability objective using public body readings only."""
    before, after = effect.before, effect.after
    return ((-1. if effect.terminated else .01)
            + .1 * (after.energy - before.energy)
            + .1 * (after.integrity - before.integrity))


def episode_changing(index):
    # All parameters remain fixed across development; no easier training physics.
    return bool(index % 2)


def registered_config():
    from dataclasses import asdict
    return dict(version=VERSION, initializations=INITIALIZATIONS, train_seeds=TRAIN_SEEDS,
                evaluation_seeds=EVALUATION_SEEDS, twins=TWINS, horizon=HORIZON, chunk=CHUNK,
                learning_rate=LEARNING_RATE, max_grad_norm=MAX_GRAD_NORM,
                loss=asdict(SETTINGS), arms=ARMS, hidden_size=32, device='cpu', dtype='float32',
                training_sampling='raw_categorical', evaluation_sampling='raw_categorical',
                train_sampling_offset=1000, evaluation_sampling_base=202686000,
                bootstrap_seed=202687000, bootstrap_draws=10000)
