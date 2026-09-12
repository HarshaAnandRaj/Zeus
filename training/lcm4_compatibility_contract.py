"""LCM4-C fixed native adapter development qualification; no training on import."""
CONFIG=dict(trials=4,ecology_base=207012000,ecologies=512,cycles=4,body_horizon=8,
    parent_arm='bridge_raw',action_base=207112000,bootstrap_seed=207212000,
    bootstrap_draws=10000,accuracy_min=.90,recall_min=.90,opposite_direction_min=.80,
    reset_effect_min=.30,opposite_effect_min=.60,twins=('a','b'),controls=('full','reset','opposite'))


def donor_indices(n):
    if n%4:raise ValueError('whole adjacent opposite-ecology/patch clusters required')
    return [i^2 for i in range(n)]
