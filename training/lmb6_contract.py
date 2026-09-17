"""Prospective LMB6 roles and rules; importing this module runs no experiment."""
from training.scarce_birth_training import CONFIG as FIT

CONFIG=FIT|dict(training_base=231014000,batch_base=231314000,
    trials=4,arms=['current','recurrent','legacy'],twins=['a','b'],
    calibration_base=231064000,calibration_ecologies=64,
    evaluation_base=231114000,evaluation_ecologies=128,
    regression_base=231124000,regression_ecologies=128,
    action_base=231414000,regression_action_base=231424000,
    bootstrap_seed=231514000,bootstrap_draws=10000,
    survival_min=.90,feed_min=16.,repair_min=.5,quality_min=.90,
    attribution_min=.02,attribution_energies=[.12,.20],readout_effect_min=.40)
SCOPE='Fresh anchored-state motor/readout prerequisite only; no native memory-benefit or emergence verdict'


def seed_roles(config=CONFIG):
    """Half-open intervals, including every per-parent action RNG offset."""
    return {
        'training':(config['training_base'],config['training_base']+config['training_ecologies']),
        'calibration':(config['calibration_base'],config['calibration_base']+config['calibration_ecologies']),
        'evaluation':(config['evaluation_base'],config['evaluation_base']+config['evaluation_ecologies']),
        'regression':(config['regression_base'],config['regression_base']+config['regression_ecologies']),
        'batch':(config['batch_base'],config['batch_base']+config['updates']),
        'action':(config['action_base'],config['action_base']+1000*(config['trials']-1)+config['evaluation_ecologies']),
        'regression_action':(config['regression_action_base'],config['regression_action_base']+1000*(config['trials']-1)+3),
        'bootstrap':(config['bootstrap_seed'],config['bootstrap_seed']+1),
    }


def validate(config=CONFIG):
    assert config['trials']==4 and config['arms']==['current','recurrent','legacy'] and config['twins']==['a','b']
    assert config['energies']==[.12,.20,.35,.85] and config['attribution_energies']==[.12,.20]
    assert config['batch']==8 and config['evaluation_ecologies']==config['regression_ecologies']
    assert all(type(config[k]) is int and config[k]>=4 and config[k]%4==0
        for k in ('training_ecologies','calibration_ecologies','evaluation_ecologies','regression_ecologies'))
    roles=seed_roles(config)
    for i,(name,(lo,hi)) in enumerate(roles.items()):
        assert type(lo) is int and type(hi) is int and 0<=lo<hi
        for other,(a,b) in list(roles.items())[i+1:]:
            assert hi<=a or b<=lo,(name,other,'RNG role collision')
    return roles
