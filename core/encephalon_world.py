"""Encephalon's two-need world. Integer physiology; no inherited model weights."""
from dataclasses import asdict, dataclass
from enum import IntEnum
import random


VERSION = "encephalon-world-v1-20260919"
SENSOR_NAMES = (
    "energy", "integrity", "position", "food_left", "food_right",
    "repair_left", "repair_right", "left_valid", "right_valid",
)


class Action(IntEnum):
    WAIT = 0
    LEFT = 1
    RIGHT = 2
    FEED = 3
    INSPECT = 4
    REPAIR = 5


@dataclass(frozen=True)
class Config:
    capacity: int = 1000
    metabolism: int = 7
    wear: int = 3
    move_cost: int = 4
    feed_cost: int = 3
    inspect_cost: int = 25
    repair_cost: int = 8
    food_gain: int = 420
    repair_gain: int = 450

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("physiology requires positive integers")
        if max(self.food_gain, self.repair_gain) > self.capacity:
            raise ValueError("restoration cannot exceed capacity")


@dataclass(frozen=True)
class Observation:
    energy: float
    integrity: float
    position: float
    food_left: float
    food_right: float
    repair_left: float
    repair_right: float
    left_valid: float
    right_valid: float

    def values(self):
        return tuple(getattr(self, name) for name in SENSOR_NAMES)


@dataclass(frozen=True)
class Transition:
    before: Observation
    action: Action
    after: Observation
    terminated: bool
    executed: bool


class World:
    def __init__(self, *, seed, changing=False, full_visibility=False,
                 repair_enabled=True, energy=850, integrity=900, config=Config()):
        if type(seed) is not int or seed < 0:
            raise ValueError("nonnegative integer seed required")
        if any(type(v) is not bool for v in (changing, full_visibility, repair_enabled)):
            raise ValueError("world flags must be booleans")
        self.config = config
        self.full_visibility = full_visibility
        self.repair_enabled = repair_enabled
        self.energy, self.integrity = energy, integrity
        self.position, self.tick, self.inspection_side = 2, 0, -1
        # Balanced contiguous seed blocks span two independent binary facts.
        self.initial_sides = [seed % 2, (seed // 2) % 2]
        self.food_side, self.repair_side = self.initial_sides
        rng = random.Random(seed)
        self.events = [[512 * i + rng.randint(-64, 64), (i - 1) % 2]
                       for i in range(1, 8)] if changing else []
        self._validate()

    def _validate(self):
        c = self.config
        for value in (self.energy, self.integrity):
            if type(value) is not int or not 0 <= value <= c.capacity:
                raise ValueError("invalid reserve")
        if type(self.position) is not int or not 0 <= self.position <= 4:
            raise ValueError("invalid position")
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("invalid clock")
        if type(self.inspection_side) is not int or self.inspection_side not in (-1, 0, 1):
            raise ValueError("invalid inspection marker")
        if self.inspection_side != -1 and self.position != self.inspection_side * 4:
            raise ValueError("inspection marker does not match station")
        if any(type(v) is not bool for v in (self.full_visibility, self.repair_enabled)):
            raise ValueError("invalid visibility or intervention flag")
        if (len(self.initial_sides) != 2 or any(type(v) is not int or v not in (0, 1)
                                               for v in self.initial_sides)):
            raise ValueError("invalid initial facts")
        previous = -1
        expected = self.initial_sides.copy()
        for event in self.events:
            if (len(event) != 2 or any(type(v) is not int for v in event)
                    or event[0] <= previous or event[1] not in (0, 1)):
                raise ValueError("invalid event schedule")
            previous = event[0]
            if event[0] <= self.tick:
                expected[event[1]] = 1 - expected[event[1]]
        if ([self.food_side, self.repair_side] != expected
                or any(type(v) is not int for v in (self.food_side, self.repair_side))):
            raise ValueError("facts do not match event history")

    def viable(self):
        return self.energy > 0 and self.integrity > 0

    def observation(self):
        valid = [self.full_visibility or self.inspection_side == side for side in (0, 1)]
        food = [float(valid[s] and self.food_side == s) for s in (0, 1)]
        repair = [float(valid[s] and self.repair_side == s) for s in (0, 1)]
        return Observation(self.energy / self.config.capacity,
                           self.integrity / self.config.capacity, self.position / 4,
                           *food, *repair, *map(float, valid))

    def step(self, action):
        if type(action) is not int and not isinstance(action, Action):
            raise ValueError("integer action required")
        action = Action(action)
        if not self.viable():
            raise RuntimeError("death is terminal")
        before = self.observation()
        c = self.config
        extra = (0, c.move_cost, c.move_cost, c.feed_cost, c.inspect_cost, c.repair_cost)
        self.energy = max(0, self.energy - c.metabolism - extra[action])
        self.integrity = max(0, self.integrity - c.wear)
        self.inspection_side = -1
        executed = self.viable()
        # Costs are paid first. A depleted body cannot borrow energy to feed.
        if executed:
            if action in (Action.LEFT, Action.RIGHT):
                self.position = min(4, max(0, self.position + (-1 if action == Action.LEFT else 1)))
            elif self.position in (0, 4):
                side = self.position // 4
                if action == Action.FEED and side == self.food_side:
                    self.energy = min(c.capacity, self.energy + c.food_gain)
                elif action == Action.REPAIR and side == self.repair_side and self.repair_enabled:
                    self.integrity = min(c.capacity, self.integrity + c.repair_gain)
                elif action == Action.INSPECT:
                    self.inspection_side = side
        self.tick += 1
        for when, kind in self.events:
            if when == self.tick:
                if kind == 0:
                    self.food_side = 1 - self.food_side
                else:
                    self.repair_side = 1 - self.repair_side
        return Transition(before, action, self.observation(), not self.viable(), executed)

    def snapshot(self):
        return dict(version=VERSION, config=asdict(self.config),
                    full_visibility=self.full_visibility, repair_enabled=self.repair_enabled,
                    energy=self.energy, integrity=self.integrity, position=self.position,
                    tick=self.tick, inspection_side=self.inspection_side,
                    initial_sides=self.initial_sides.copy(), food_side=self.food_side,
                    repair_side=self.repair_side, events=[event.copy() for event in self.events])

    @classmethod
    def restore(cls, snapshot):
        expected = {"version", "config", "full_visibility", "repair_enabled", "energy",
                    "integrity", "position", "tick", "inspection_side", "initial_sides",
                    "food_side", "repair_side", "events"}
        if set(snapshot) != expected or snapshot["version"] != VERSION:
            raise ValueError("incompatible snapshot")
        world = cls.__new__(cls)
        for name, value in snapshot.items():
            if name not in ("version", "config"):
                setattr(world, name, value)
        world.config = Config(**snapshot["config"])
        world.initial_sides = list(world.initial_sides)
        world.events = [list(event) for event in world.events]
        world._validate()
        return world

