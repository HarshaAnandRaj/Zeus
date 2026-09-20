"""E1-B external food stocks; original Encephalon physics remains unchanged."""
from dataclasses import asdict, dataclass

from core.encephalon_world import Action, Config, Observation, Transition


VERSION = "encephalon-resources-v1-20260920"


@dataclass(frozen=True)
class Resources:
    capacity: int = 420
    renewal: int = 4

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("positive integer resource parameters required")
        if self.capacity % 4 or self.renewal > self.capacity:
            raise ValueError("invalid stock strata or renewal")


class ResourceWorld:
    def __init__(self, *, seed, mode="finite", energy=850, integrity=900,
                 repair_enabled=True, config=Config(), resources=Resources()):
        if type(seed) is not int or seed < 0 or mode not in ("finite", "abundant"):
            raise ValueError("invalid resource-world identity")
        self.config, self.resources, self.mode = config, resources, mode
        self.energy, self.integrity = energy, integrity
        self.position, self.tick, self.repair_side = 2, 0, (seed // 2) % 2
        self.repair_enabled = repair_enabled
        q = resources.capacity
        strata = ((q, q), (q // 4, q), (q, q // 4), (q // 2, q // 2))
        self.stocks = list(strata[(seed // 4) % 4] if mode == "finite" else (q, q))
        self.ledger = dict(initial_energy=energy, initial_integrity=integrity,
                           initial_stocks=self.stocks.copy(), supplied=[0, 0],
                           overflow=[0, 0], food=[0, 0], spent=0, wear=0, repaired=0)
        self._validate()

    def _validate(self):
        c, r = self.config, self.resources
        if self.mode not in ("finite", "abundant") or type(self.repair_enabled) is not bool:
            raise ValueError("invalid resource mode or repair flag")
        for key, low, high in (("energy", 0, c.capacity), ("integrity", 0, c.capacity),
                               ("position", 0, 4), ("repair_side", 0, 1)):
            if type(getattr(self, key)) is not int or not low <= getattr(self, key) <= high:
                raise ValueError("invalid physical state")
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("invalid clock")
        if len(self.stocks) != 2 or any(type(q) is not int or not 0 <= q <= r.capacity for q in self.stocks):
            raise ValueError("invalid stock")
        if self.mode == "abundant" and self.stocks != [r.capacity] * 2:
            raise ValueError("abundant stocks must remain full")
        ledger = self.ledger
        if set(ledger) != {"initial_energy", "initial_integrity", "initial_stocks", "supplied",
                           "overflow", "food", "spent", "wear", "repaired"}:
            raise ValueError("incomplete physical ledger")
        for key in ("initial_stocks", "supplied", "overflow", "food"):
            if len(ledger[key]) != 2 or any(type(v) is not int or v < 0 for v in ledger[key]):
                raise ValueError("invalid patch ledger")
        for key in ("initial_energy", "initial_integrity", "spent", "wear", "repaired"):
            if type(ledger[key]) is not int or ledger[key] < 0:
                raise ValueError("invalid bodily ledger")
        if not all(q <= r.capacity for q in ledger["initial_stocks"]):
            raise ValueError("invalid starting stock")
        if max(ledger["initial_energy"], ledger["initial_integrity"]) > c.capacity:
            raise ValueError("invalid starting reserve")
        if self.energy != ledger["initial_energy"] + sum(ledger["food"]) - ledger["spent"]:
            raise ValueError("energy conservation failure")
        if self.integrity != ledger["initial_integrity"] + ledger["repaired"] - ledger["wear"]:
            raise ValueError("integrity accounting failure")
        for side in (0, 1):
            if self.stocks[side] != ledger["initial_stocks"][side] + ledger["supplied"][side] - ledger["food"][side]:
                raise ValueError("stock conservation failure")
            if self.mode == "finite" and ledger["supplied"][side] + ledger["overflow"][side] != r.renewal * self.tick:
                raise ValueError("renewal accounting failure")

    def viable(self):
        return self.energy > 0 and self.integrity > 0

    def observation(self):
        return Observation(self.energy / self.config.capacity, self.integrity / self.config.capacity,
                           self.position / 4, *(q / self.resources.capacity for q in self.stocks),
                           float(self.repair_side == 0), float(self.repair_side == 1), 1., 1.)

    def step(self, action):
        if type(action) is not int and not isinstance(action, Action):
            raise ValueError("integer action required")
        action = Action(action)
        if not self.viable():
            raise RuntimeError("death is terminal")
        c, r, ledger = self.config, self.resources, self.ledger
        before = self.observation()
        extra = (0, c.move_cost, c.move_cost, c.feed_cost, c.inspect_cost, c.repair_cost)
        spent, wear = min(self.energy, c.metabolism + extra[action]), min(self.integrity, c.wear)
        self.energy -= spent
        self.integrity -= wear
        ledger["spent"] += spent
        ledger["wear"] += wear
        executed = self.viable()
        if executed:
            if action in (Action.LEFT, Action.RIGHT):
                self.position = min(4, max(0, self.position + (-1 if action == Action.LEFT else 1)))
            elif self.position in (0, 4):
                side = self.position // 4
                if action == Action.FEED:
                    gained = min(c.food_gain, c.capacity - self.energy, self.stocks[side])
                    self.energy += gained
                    ledger["food"][side] += gained
                    if self.mode == "finite":
                        self.stocks[side] -= gained
                    else:
                        ledger["supplied"][side] += gained
                elif action == Action.REPAIR and side == self.repair_side and self.repair_enabled:
                    gained = min(c.repair_gain, c.capacity - self.integrity)
                    self.integrity += gained
                    ledger["repaired"] += gained
        # Both patches renew after the action, including the terminal physical tick.
        # Reaching a patch never adds a separate refill. Dead bodies cannot act again.
        if self.mode == "finite":
            for side in (0, 1):
                supplied = min(r.renewal, r.capacity - self.stocks[side])
                self.stocks[side] += supplied
                ledger["supplied"][side] += supplied
                ledger["overflow"][side] += r.renewal - supplied
        self.tick += 1
        return Transition(before, action, self.observation(), not self.viable(), executed)

    def snapshot(self):
        return dict(version=VERSION, config=asdict(self.config), resources=asdict(self.resources),
                    mode=self.mode, repair_enabled=self.repair_enabled, energy=self.energy,
                    integrity=self.integrity, position=self.position, tick=self.tick,
                    repair_side=self.repair_side, stocks=self.stocks.copy(),
                    ledger={k: v.copy() if isinstance(v, list) else v for k, v in self.ledger.items()})

    @classmethod
    def restore(cls, snapshot):
        keys = {"version", "config", "resources", "mode", "repair_enabled", "energy", "integrity",
                "position", "tick", "repair_side", "stocks", "ledger"}
        if set(snapshot) != keys or snapshot["version"] != VERSION:
            raise ValueError("incompatible resource snapshot")
        world = cls.__new__(cls)
        for key in keys - {"version", "config", "resources", "stocks", "ledger"}:
            setattr(world, key, snapshot[key])
        world.config, world.resources = Config(**snapshot["config"]), Resources(**snapshot["resources"])
        world.stocks = list(snapshot["stocks"])
        world.ledger = {k: v.copy() if isinstance(v, list) else v for k, v in snapshot["ledger"].items()}
        world._validate()
        return world
