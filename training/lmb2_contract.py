"""Fresh learner-history motor correction, separate from closed LMB1."""
from training.lmb1_contract import CONFIG as BASE
VERSION='lmb2-learner-history-correction-v1-20260913'
CONFIG=BASE|dict(arms=('learner_history','demonstration'),updates=96,refresh=12,collection_batch=8,
    training_base=211013000,calibration_base=211023000,training_cue_base=211033000,
    evaluation_base=211113000,regression_base=211123000,synthetic_regression_base=211223000,
    batch_base=211313000,collection_action_base=211343000,action_base=211413000,
    regression_action_base=211423000,bootstrap_seed=211513000,attribution_min=.02)
