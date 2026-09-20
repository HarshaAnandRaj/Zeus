"""Public scripted calibration only; no action from this module trains Zeus."""
from dataclasses import dataclass


@dataclass
class ResourceReference:
    resident: bool = False
    route: tuple | None = None

    def act(self, observation):
        o = observation.values()
        repair = 0 if o[5] else 1
        if self.route is None:
            # Keep a transit reserve without harvesting every tiny renewal.
            food_needed, repair_needed = o[0] < (.65 if self.resident else .40), o[1] < .60
            if self.resident and o[2] != repair:
                return 1 if o[2] > repair else 2
            if not food_needed and not repair_needed:
                return 0
            kind = 3 if food_needed and (not repair_needed or round(o[0] * 1000) * 3 <= round(o[1] * 1000) * 7) else 5
            if kind == 5 or self.resident:
                side = repair
            else:
                # Public stock, less the energy needed to reach it; tie to the left.
                side = max(range(2), key=lambda s: 420 * o[3 + s] - 44 * abs(o[2] - s))
            self.route = (kind, side)
        kind, side = self.route
        if o[2] != side:
            return 1 if o[2] > side else 2
        self.route = None
        return kind

    def state(self):
        return dict(resident=self.resident, route=list(self.route) if self.route else None)
