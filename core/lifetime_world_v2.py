"""Quality-changing lifetime world; v1 remains frozen as a negative baseline."""
from dataclasses import asdict, dataclass, fields

from core.lifetime_world import (LifetimeConfig, LifetimeWorld, LifetimeAction as A,
                                 PublicObservation, PublicStep, LifetimeInterface,
                                 SENSOR_NAMES)

WORLD_VERSION = 'lifetime-world-v2-quality-20260909'
INTERFACE_VERSION = 'lifetime-eight-sensors-six-actions-v2'


@dataclass(frozen=True)
class QualityConfig(LifetimeConfig):
    contamination_damage: float = .12

    def __post_init__(self):
        super().__post_init__()
        if self.extraction_limit <= 0:
            raise ValueError('positive extraction limit required')


@dataclass(frozen=True)
class QualityObservation(PublicObservation):
    resource_quality: float

    def values(self):
        return super().values() + (self.resource_quality,)

    def prediction_mask(self):
        return super().prediction_mask() + (self.inspection_valid,)


class QualityWorld(LifetimeWorld):
    def __init__(self, *, seed: int, changing: bool, config: QualityConfig = QualityConfig()):
        super().__init__(seed=seed, changing=False, config=config)
        if type(changing) is not bool:
            raise ValueError('changing must be boolean')
        # Both patches recover at the same rate. Information concerns quality,
        # not finding a faster patch or carrying over v1's supply imbalance.
        self._rates = [config.fast_recovery] * 2
        safe = self._rng.randrange(2)
        self._quality = [int(i == safe) for i in range(2)]
        drawn = [self._rng.randint(a, b) for a, b in ((220, 300), (460, 540), (700, 780))]
        self._switches = drawn if changing else []
        self._switch_index = 0

    def observation(self):
        base = super().observation()
        quality = self._quality[self._position // 4] if self._inspection and self._position in (0, 4) else 0.
        return QualityObservation(**asdict(base), resource_quality=float(quality))

    def step(self, action):
        if not isinstance(action, A) and type(action) is not int:
            raise ValueError('action must use the lifetime enum or an integer index')
        action = A(action)
        if not self.viable(): raise RuntimeError('cannot advance a terminated lifetime')
        c = self.config; before = self.observation(); self._inspection = False
        self._energy -= c.metabolism
        damage = 0.
        if action in (A.LEFT, A.RIGHT):
            self._energy -= c.move_cost
            self._position = max(0, min(4, self._position + (-1 if action == A.LEFT else 1)))
        elif action == A.HARVEST:
            self._energy -= c.harvest_cost
            if self._position in (0, 4):
                patch = self._position // 4
                amount = min(c.extraction_limit, self._resources[patch])
                self._resources[patch] -= amount
                self._energy += amount * self._quality[patch] * (c.efficiency_floor + c.efficiency_gain * self._tool)
                damage = c.contamination_damage * (1 - self._quality[patch]) * amount / c.extraction_limit
            self._tool -= c.harvest_wear
        elif action == A.INSPECT:
            self._energy -= c.inspect_cost
        elif action == A.MAINTAIN:
            self._energy -= c.maintain_cost
            if self._position == 2:
                self._tool += c.tool_repair; self._integrity += c.integrity_repair
        elif action == A.WAIT and self._energy > c.wait_energy_min:
            self._integrity += c.wait_recovery
        self._integrity -= damage + c.integrity_decay + c.starvation_damage * max(0., c.starvation_level - self._energy)
        self._resources = [self._clip(r + rate * (capacity - r)) for r, rate, capacity in zip(self._resources, self._rates, self._capacity)]
        self._energy = self._clip(self._energy); self._integrity = self._clip(self._integrity); self._tool = self._clip(self._tool)
        self._tick += 1
        if self._switch_index < len(self._switches) and self._tick == self._switches[self._switch_index]:
            self._quality.reverse(); self._switch_index += 1
        self._inspection = action == A.INSPECT
        return PublicStep(action, before, self.observation(), not self.viable())

    def snapshot(self):
        result = super().snapshot()
        result.update(version=WORLD_VERSION, interface=INTERFACE_VERSION,
                      quality=self._quality.copy(), switches=self._switches.copy(), switch_index=self._switch_index)
        return result

    @classmethod
    def restore(cls, snapshot):
        if snapshot['version'] != WORLD_VERSION or snapshot['interface'] != INTERFACE_VERSION:
            raise ValueError('incompatible quality-world snapshot')
        c = QualityConfig(**snapshot['config'])
        if c.extraction_limit <= 0: raise ValueError('positive extraction limit required')
        # Use v1's scalar/RNG validation on a copy with its expected rate pair;
        # v2's actual equal rates and event schedule are checked separately below.
        base = dict(snapshot)
        base.update(version='lifetime-world-v1-20260909', interface='lifetime-seven-sensors-six-actions-v1',
                    config={f.name: getattr(c, f.name) for f in fields(LifetimeConfig)},
                    rates=[c.slow_recovery, c.fast_recovery])
        restored = LifetimeWorld.restore(base)
        if snapshot['rates'] != [c.fast_recovery] * 2 or snapshot['change_tick'] is not None or snapshot['changed']:
            raise ValueError('invalid v2 recovery state')
        if sorted(snapshot['quality']) != [0, 1] or any(type(v) is not int for v in snapshot['quality']):
            raise ValueError('invalid quality state')
        schedule = snapshot['switches']
        if schedule and (len(schedule) != 3 or any(type(t) is not int or not a <= t <= b for t, (a, b) in zip(schedule, ((220,300),(460,540),(700,780))))):
            raise ValueError('invalid quality schedule')
        if type(snapshot['switch_index']) is not int or snapshot['switch_index'] != sum(t <= snapshot['tick'] for t in schedule):
            raise ValueError('invalid event position')
        world = cls.__new__(cls); world.__dict__.update(restored.__dict__); world.config = c
        world._rates = snapshot['rates'].copy(); world._quality = snapshot['quality'].copy()
        world._switches = schedule.copy(); world._switch_index = snapshot['switch_index']
        return world


class QualityInterface(LifetimeInterface):
    version = INTERFACE_VERSION
    sensor_names = SENSOR_NAMES + ('resource_quality',)
