"""Explicitly engineered public-observation references, never learned-agent evidence."""
from dataclasses import dataclass
from core.encephalon_world import Action as A


@dataclass
class Reference:
    mode: str = "adaptive"
    food_side: int | None = None
    repair_side: int | None = None
    route: tuple | None = None
    writes_enabled: bool = True

    def read(self, obs):
        if not self.writes_enabled:
            return
        for side, valid in enumerate((obs.left_valid, obs.right_valid)):
            if valid:
                food = side if (obs.food_left, obs.food_right)[side] else 1 - side
                repair = side if (obs.repair_left, obs.repair_right)[side] else 1 - side
                if self.mode != "frozen" or self.food_side is None:
                    self.food_side = food
                if self.mode != "frozen" or self.repair_side is None:
                    self.repair_side = repair

    def act(self, obs):
        if self.mode == "passive":
            return A.WAIT
        if self.mode == "reinspect" and self.route is None:
            self.food_side = self.repair_side = None
        self.read(obs)
        if self.mode == "fixed_left":
            self.food_side = self.repair_side = 0
        if self.route is not None:
            kind, side = self.route
        else:
            food_needed = obs.energy < .60
            repair_needed = obs.integrity < .60 and self.mode != "no_repair"
            if not food_needed and not repair_needed:
                return A.WAIT
            kind = 0 if food_needed and (not repair_needed or obs.energy / .007 <= obs.integrity / .003) else 1
            side = self.food_side if kind == 0 else self.repair_side
            if side is None:
                return A.INSPECT if obs.position in (0., 1.) else A.LEFT
            if self.mode == "reinspect":
                self.route = (kind, side)
        if obs.position < side:
            return A.RIGHT
        if obs.position > side:
            return A.LEFT
        return A.FEED if kind == 0 else A.REPAIR

    def observe(self, step):
        self.read(step.after)
        if not self.writes_enabled:
            return
        if step.executed and step.before.position in (0., 1.):
            side = int(step.before.position)
            if step.action in (A.FEED, A.REPAIR):
                self.route = None
                if self.mode not in ("frozen", "fixed_left"):
                    if step.action == A.FEED:
                        self.food_side = side if step.after.energy > step.before.energy else 1 - side
                    else:
                        self.repair_side = side if step.after.integrity > step.before.integrity else 1 - side

