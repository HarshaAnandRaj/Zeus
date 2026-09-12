"""LCM5 fixed repair budget and data roles; no compute on import."""
CONFIG=dict(trials=4,training_base=208013000,training_ecologies=512,
    development_base=208023000,development_ecologies=128,evaluation_base=208113000,evaluation_ecologies=512,
    synthetic_base=208213000,synthetic_n=512,synthetic_delays=(64,128),
    initialization_base=208313000,batch_base=208513000,action_base=208413000,
    synthetic_action_base=208423000,bootstrap_seed=208613000,bootstrap_draws=10000,
    updates=384,batch=64,lr=.03,clip=1.,accuracy_min=.90,recall_min=.90,bin_min=32,
    quality_reset_margin=.30,quality_opposite_margin=.60,
    arms=('inherited','no_write'),twins=('a','b'))


def native_config(base,ecologies):
    from training.lcm4_compatibility_contract import CONFIG as P
    return P|dict(ecology_base=base,ecologies=ecologies,action_base=CONFIG['action_base'],bootstrap_seed=CONFIG['bootstrap_seed'])


def batch_indices(update,n,config=CONFIG):
    import torch
    return torch.randint(n,(config['batch'],),generator=torch.Generator().manual_seed(config['batch_base']+update))
