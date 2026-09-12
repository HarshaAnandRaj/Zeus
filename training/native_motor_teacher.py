"""Explicit public motor demonstrations; no seed, hidden quality or audit access."""
from core.lifetime_world_v2 import QualityConfig

VERSION='public-motor-teacher-v1-20260913'
PHYSICS=QualityConfig()


def initial_teacher(public_cue=None):
    if public_cue is None:return dict(safe=None,tool=None)
    if not public_cue[4] or public_cue[2] not in (0,1):raise ValueError('public patch inspection required')
    side=int(public_cue[2]);safe=side if public_cue[7] else 1-side
    return dict(safe=safe,tool=public_cue[6])


def action(observation,state):
    position=observation.position
    if state['safe'] is None:return 1 if position>0 else 4
    if (state['tool'] is not None and state['tool']<.75) or observation.integrity<.8:
        return 2 if position<.5 else 1 if position>.5 else 5
    destination=float(state['safe'])
    if position<destination:return 2
    if position>destination:return 1
    return 3 if observation.energy<.5 else 0


def observe(step,state):
    result=state.copy();after=step.after
    if int(step.action)==3 and result['tool'] is not None:result['tool']=max(0.,result['tool']-PHYSICS.harvest_wear)
    if int(step.action)==5 and after.position==.5 and result['tool'] is not None:result['tool']=min(1.,result['tool']+PHYSICS.tool_repair)
    if int(step.action)==4 and after.inspection_valid and after.position in (0,1):
        side=int(after.position);result['safe']=side if after.resource_quality else 1-side;result['tool']=after.tool_condition
    return result
