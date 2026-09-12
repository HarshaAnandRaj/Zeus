"""Frozen public-cue, nuisance, reset and inherited-action assay contract."""
import torch

VERSION='lcm2-protected-recall-assay-v1-20260912'
ARMS=('protected','every_step','no_write')
CONTROLS=('full','reset','shuffle','trained_no_write','every_step','initial')
CONFIG=dict(trials=4,batch=64,updates=192,train_delays=(8,16),evaluation_delays=(64,128),
    cycles=4,fast_size=32,slow_size=8,lr=.003,clip=1.,quality_weight=.2,
    initialization_base=205212000,training_base=205012000,evaluation_base=205112000,
    evaluation_action_base=205412000,evaluation_n=512,bootstrap_seed=205512000,
    bootstrap_draws=10000,accuracy_min=.90,recall_min=.90,reset_effect_min=.30,
    shuffle_effect_min=.60,opposite_direction_min=.80,arms=ARMS,controls=CONTROLS,twins=('a','b'))


def episodes(seed,batch,delay,*,paired=False):
    """Public synthetic transitions, not a QualityWorld physical rollout.

    Endpoint pairs share side/nuisance and have opposite inspected quality.
    Labels never enter the actor. No nuisance action is an inspection.
    """
    if batch<1 or delay<1 or (paired and batch%2):raise ValueError('invalid episode shape')
    rng=torch.Generator().manual_seed(seed);lanes=batch//2 if paired else batch
    def expand(x):return x.repeat_interleave(2,0) if paired else x
    sides=expand(torch.randint(0,2,(lanes,),generator=rng))
    qualities=torch.arange(batch)%2 if paired else torch.randint(0,2,(batch,),generator=rng)
    cue=torch.zeros(batch,8);cue[:,:2]=expand(.75+.2*torch.rand(lanes,2,generator=rng))
    cue[:,2]=sides;cue[:,3]=expand(.25+.5*torch.rand(lanes,generator=rng));cue[:,4]=1
    cue[:,5:7]=expand(.6+.3*torch.rand(lanes,2,generator=rng));cue[:,7]=qualities
    before=cue.clone();before[:,4:]=0
    count=(CONFIG['cycles']-1)*delay
    nuisance=expand(torch.rand(lanes,count,8,generator=rng));nuisance[:,:,4:]=0
    nuisance[:,:,:2]=.5+.45*nuisance[:,:,:2]
    nuisance[:,:,2]=torch.round(nuisance[:,:,2]*4)/4
    actions=expand(torch.randint(0,5,(lanes,count),generator=rng));actions=torch.where(actions==4,5,actions)
    query=torch.tensor([.85,.95,.5,0.,0.,0.,0.,0.]).expand(batch,-1).clone()
    safe=torch.where(qualities.bool(),sides,1-sides);target=torch.where(safe==0,1,2)
    return dict(before=before,cue=cue,nuisance=nuisance,actions=actions,query=query,
                side=sides,quality=qualities,target=target,delay=delay)
