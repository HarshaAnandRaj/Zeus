"""Fresh public grounding versus matched detached-body auxiliary control."""
from training.run_lmb3 import CONFIG as BASE
VERSION='lmb4-public-state-grounding-v1-20260913'
CONFIG=BASE|dict(arms=('grounded','detached'),updates=96,refresh=12,collection_batch=8,own_batch=8,
    training_base=214013000,own_public_base=214023000,training_cue_base=214033000,head_initial_base=214043000,
    calibration_base=214063000,evaluation_base=214113000,regression_base=214123000,
    synthetic_regression_base=214223000,batch_base=214313000,own_batch_base=214323000,
    collection_action_base=214343000,action_base=214413000,regression_action_base=214423000,
    bootstrap_seed=214513000,grounding_weight=.25,attribution_min=.02)
