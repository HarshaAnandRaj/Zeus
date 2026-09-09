"""Versioned continuous-lifetime physics and a restricted public interface.

The world chooses no actions. Audit snapshots contain privileged information;
only PublicObservation/PublicStep belong at the actor boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import math
import random

WORLD_VERSION = 'lifetime-world-v1-20260909'
INTERFACE_VERSION = 'lifetime-seven-sensors-six-actions-v1'


class LifetimeAction(IntEnum):
    WAIT = 0
    LEFT = 1
    RIGHT = 2
    HARVEST = 3
    INSPECT = 4
    MAINTAIN = 5


ACTION_NAMES = tuple(a.name.lower() for a in LifetimeAction)
SENSOR_NAMES = ('energy', 'integrity', 'position', 'coarse_resource',
                'inspection_valid', 'precise_resource', 'tool_condition')


@dataclass(frozen=True)
class LifetimeConfig:
    metabolism: float = .012
    move_cost: float = .004
    harvest_cost: float = .002
    inspect_cost: float = .004
    maintain_cost: float = .025
    extraction_limit: float = .13
    efficiency_floor: float = .35
    efficiency_gain: float = .55
    harvest_wear: float = .008
    tool_repair: float = .40
    integrity_decay: float = .002
    integrity_repair: float = .22
    wait_recovery: float = .005
    wait_energy_min: float = .50
    starvation_level: float = .20
    starvation_damage: float = .04
    death_threshold: float = .02
    fast_recovery: float = .10
    slow_recovery: float = .015
    capacity_low: float = .80
    capacity_high: float = 1.
    initial_energy: float = .85
    initial_integrity: float = .95
    initial_tool: float = .90
    initial_fill_low: float = .65
    initial_fill_high: float = .90
    change_min: int = 320
    change_max: int = 640
    resource_bins: int = 4

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'nonfinite/nonnumeric configuration: {name}')
            if name not in ('change_min', 'change_max', 'resource_bins') and not 0 <= value <= 1:
                raise ValueError(f'configuration outside [0,1]: {name}')
        for name in ('change_min', 'change_max', 'resource_bins'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.change_min > self.change_max or self.capacity_low > self.capacity_high or self.initial_fill_low > self.initial_fill_high:
            raise ValueError('reversed configuration interval')
        if self.slow_recovery >= self.fast_recovery or self.efficiency_floor + self.efficiency_gain > 1:
            raise ValueError('invalid recovery or efficiency relation')
        if min(self.initial_energy, self.initial_integrity) <= self.death_threshold:
            raise ValueError('initial body must be viable')


@dataclass(frozen=True)
class PublicObservation:
    energy: float
    integrity: float
    position: float
    coarse_resource: float
    inspection_valid: bool
    precise_resource: float
    tool_condition: float

    def values(self) -> tuple[float, ...]:
        return (self.energy, self.integrity, self.position, self.coarse_resource,
                float(self.inspection_valid), self.precise_resource, self.tool_condition)

    def prediction_mask(self) -> tuple[bool, ...]:
        # Absent diagnostic readings are not zero-valued physical targets.
        return (True, True, True, True, True, self.inspection_valid, self.inspection_valid)


@dataclass(frozen=True)
class PublicStep:
    action: LifetimeAction
    before: PublicObservation
    after: PublicObservation
    terminated: bool


class LifetimeWorld:
    def __init__(self, *, seed: int, changing: bool, config: LifetimeConfig = LifetimeConfig()):
        if type(seed) is not int or type(changing) is not bool:
            raise ValueError('explicit integer seed and boolean changing required')
        self.config = config
        self._rng = random.Random(seed)
        self._capacity = [self._rng.uniform(config.capacity_low, config.capacity_high) for _ in range(2)]
        self._resources = [c * self._rng.uniform(config.initial_fill_low, config.initial_fill_high) for c in self._capacity]
        fast = self._rng.randrange(2)
        self._rates = [config.fast_recovery if i == fast else config.slow_recovery for i in range(2)]
        drawn_tick = self._rng.randint(config.change_min, config.change_max)
        self._change_tick = drawn_tick if changing else None
        self._changed = False
        self._tick = 0
        self._position = 2
        self._energy = config.initial_energy
        self._integrity = config.initial_integrity
        self._tool = config.initial_tool
        self._inspection = False

    @staticmethod
    def _clip(value):
        return max(0., min(1., value))

    def viable(self) -> bool:
        return min(self._energy, self._integrity) > self.config.death_threshold

    def observation(self) -> PublicObservation:
        resource = self._resources[self._position // 4] if self._position in (0, 4) else 0.
        coarse = math.floor(resource * self.config.resource_bins) / self.config.resource_bins
        return PublicObservation(self._energy, self._integrity, self._position / 4,
                                 coarse, self._inspection, resource if self._inspection else 0.,
                                 self._tool if self._inspection else 0.)

    def step(self, action: LifetimeAction | int) -> PublicStep:
        if not isinstance(action, LifetimeAction) and type(action) is not int:
            raise ValueError('action must use the lifetime enum or an integer index')
        action = LifetimeAction(action)
        if not self.viable():
            raise RuntimeError('cannot advance a terminated lifetime')
        c = self.config
        before = self.observation()
        self._inspection = False
        self._energy -= c.metabolism
        if action in (LifetimeAction.LEFT, LifetimeAction.RIGHT):
            self._energy -= c.move_cost
            self._position = max(0, min(4, self._position + (-1 if action == LifetimeAction.LEFT else 1)))
        elif action == LifetimeAction.HARVEST:
            self._energy -= c.harvest_cost
            amount = 0.
            if self._position in (0, 4):
                patch = self._position // 4
                amount = min(c.extraction_limit, self._resources[patch])
                self._resources[patch] -= amount
            self._energy += amount * (c.efficiency_floor + c.efficiency_gain * self._tool)
            self._tool -= c.harvest_wear
        elif action == LifetimeAction.INSPECT:
            self._energy -= c.inspect_cost
        elif action == LifetimeAction.MAINTAIN:
            self._energy -= c.maintain_cost
            if self._position == 2:
                self._tool += c.tool_repair
                self._integrity += c.integrity_repair
        elif action == LifetimeAction.WAIT and self._energy > c.wait_energy_min:
            self._integrity += c.wait_recovery
        self._integrity -= c.integrity_decay + c.starvation_damage * max(0., c.starvation_level - self._energy)
        self._resources = [self._clip(r + rate * (capacity - r)) for r, rate, capacity in zip(self._resources, self._rates, self._capacity)]
        self._energy = self._clip(self._energy)
        self._integrity = self._clip(self._integrity)
        self._tool = self._clip(self._tool)
        self._tick += 1
        if self._change_tick == self._tick:
            self._rates.reverse()
            self._changed = True
        self._inspection = action == LifetimeAction.INSPECT
        return PublicStep(action, before, self.observation(), not self.viable())

    def snapshot(self) -> dict:
        """Privileged audit/resume record. Never an actor observation."""
        return dict(version=WORLD_VERSION, interface=INTERFACE_VERSION, config=asdict(self.config),
                    capacity=self._capacity.copy(), resources=self._resources.copy(), rates=self._rates.copy(),
                    change_tick=self._change_tick, changed=self._changed, tick=self._tick,
                    position=self._position, energy=self._energy, integrity=self._integrity,
                    tool=self._tool, inspection=self._inspection, rng_state=self._rng.getstate())

    @classmethod
    def restore(cls, snapshot: dict):
        if snapshot['version'] != WORLD_VERSION or snapshot['interface'] != INTERFACE_VERSION:
            raise ValueError('incompatible lifetime snapshot')
        c = LifetimeConfig(**snapshot['config'])
        for name in ('capacity', 'resources', 'rates'):
            values = snapshot[name]
            if len(values) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError(f'invalid {name}')
        for name in ('energy', 'integrity', 'tool'):
            value = snapshot[name]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'invalid {name}')
        if type(snapshot['position']) is not int or not 0 <= snapshot['position'] <= 4:
            raise ValueError('invalid position')
        if type(snapshot['tick']) is not int or snapshot['tick'] < 0:
            raise ValueError('invalid tick')
        event = snapshot['change_tick']
        if event is not None and (type(event) is not int or not c.change_min <= event <= c.change_max):
            raise ValueError('invalid event tick')
        if type(snapshot['changed']) is not bool or type(snapshot['inspection']) is not bool:
            raise ValueError('invalid flags')
        if snapshot['changed'] != (event is not None and snapshot['tick'] >= event):
            raise ValueError('event position is inconsistent')
        if sorted(snapshot['rates']) != [c.slow_recovery, c.fast_recovery]:
            raise ValueError('invalid recovery coefficients')
        world = cls.__new__(cls)
        world.config = c
        for name in ('capacity', 'resources', 'rates'):
            setattr(world, '_' + name, list(snapshot[name]))
        for name in ('change_tick', 'changed', 'tick', 'position', 'energy', 'integrity', 'tool', 'inspection'):
            setattr(world, '_' + name, snapshot[name])
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, (tuple, list)) else value
        world._rng = random.Random()
        world._rng.setstate(tuples(snapshot['rng_state']))
        return world


class LifetimeInterface:
    """Public API contract, not protection against Python reflection."""
    __slots__ = ('__world',)
    version = INTERFACE_VERSION
    action_names = ACTION_NAMES
    sensor_names = SENSOR_NAMES

    def __init__(self, world: LifetimeWorld):
        self.__world = world

    def observation(self) -> PublicObservation:
        return self.__world.observation()

    def step(self, action: LifetimeAction | int) -> PublicStep:
        return self.__world.step(action)

    def viable(self) -> bool:
        return self.__world.viable()
