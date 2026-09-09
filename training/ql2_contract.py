"""QL2 fixed exposure schedule; no campaign runs at import."""
from core.lifetime_world_v2 import QualityWorld
from training.quality_learning_contract import SETTINGS, reward
INITIALIZATIONS=tuple(range(202694000,202694004))
TRAIN_BASE=202692000
EVALUATION_SEEDS=tuple(range(202793000,202793064))
TRAIN_ARMS=('curriculum','ordinary')
ARMS=('curriculum','ordinary','initial_model','reset_history')
TWINS=('a','b')
PHASE_STEPS=(4096,4096,8192)
HORIZON=1024
CHUNK=64

def start_world(seed,arm,phase):
    if arm not in TRAIN_ARMS or phase not in (0,1,2):raise ValueError('unregistered training condition')
    world=QualityWorld(seed=seed,changing=False)
    if arm=='curriculum' and phase<2:
        snap=world.snapshot();safe=snap['quality'].index(1)*4
        snap['position']=safe if phase==0 else (1 if safe==0 else 3)
        world=QualityWorld.restore(snap)
    return world

def config():
    from dataclasses import asdict
    return dict(version='ql2-start-exposure-20260910',initializations=INITIALIZATIONS,train_base=TRAIN_BASE,
        evaluation_seeds=EVALUATION_SEEDS,train_arms=TRAIN_ARMS,arms=ARMS,twins=TWINS,
        phase_steps=PHASE_STEPS,horizon=HORIZON,chunk=CHUNK,loss=asdict(SETTINGS),
        lr=.0003,weight_decay=.01,gradient_clip=1.,train_sampling_offset=1000,
        evaluation_sampling_base=202696000,bootstrap_seed=202697000,bootstrap_draws=10000)

