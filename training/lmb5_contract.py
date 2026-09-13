"""Prospective scarce-birth exposure comparison; no compute on import."""
from training.scarce_birth_training import CONFIG as FIT
CONFIG=FIT|dict(trials=4,arms=['balanced','original'],twins=['a','b'],calibration_base=219063000,
    calibration_ecologies=64,evaluation_base=219113000,evaluation_ecologies=128,regression_base=219123000,
    regression_ecologies=128,action_base=219413000,regression_action_base=219423000,bootstrap_seed=219513000,
    bootstrap_draws=10000,survival_min=.90,feed_min=16.,repair_min=.5,quality_min=.90,
    attribution_min=.02,readout_effect_min=.40)
