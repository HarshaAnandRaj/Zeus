"""LMB1 fixed real-body motor prerequisite; no compute on import."""
CONFIG=dict(trials=4,training_base=209013000,training_ecologies=128,
    calibration_base=209023000,calibration_ecologies=128,evaluation_base=209113000,evaluation_ecologies=128,
    regression_base=209123000,regression_ecologies=256,synthetic_regression_base=209223000,synthetic_n=512,
    training_cue_base=209033000,synthetic_delays=(64,128),body_horizon=256,
    updates=384,batch=8,chunk=32,lr=.003,clip=1.,cue_weight=.25,
    action_weights=(1.,8.,8.,4.,16.,16.),batch_base=209313000,action_base=209413000,
    regression_action_base=209423000,bootstrap_seed=209513000,bootstrap_draws=10000,
    survival_min=.90,feeding_mean_min=16.,repair_mean_min=.5,initial_effect_min=.50,
    memory_accuracy_min=.90,quality_min=.90,twins=('a','b'))


def batch_indices(update,n,config=CONFIG):
    import torch
    return torch.randint(n,(config['batch'],),generator=torch.Generator().manual_seed(config['batch_base']+update))
