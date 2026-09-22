"""Independent, integer-only E1-B finite ecology with variable renewal."""
from training.encephalon_resource_physics import reference_action


def initial(seed, energy, integrity, renewal, repair=True):
    assert renewal in (4, 6, 10)
    stocks = list(((420, 420), (105, 420), (420, 105), (210, 210))[(seed // 4) % 4])
    return dict(version="encephalon-resources-v1-20260920",
                config=dict(capacity=1000, metabolism=7, wear=3, move_cost=4,
                            feed_cost=3, inspect_cost=25, repair_cost=8,
                            food_gain=420, repair_gain=450),
                resources=dict(capacity=420, renewal=renewal), mode="finite",
                repair_enabled=repair, energy=energy, integrity=integrity,
                position=2, tick=0, repair_side=(seed // 2) % 2, stocks=stocks,
                ledger=dict(initial_energy=energy, initial_integrity=integrity,
                            initial_stocks=stocks.copy(), supplied=[0, 0],
                            overflow=[0, 0], food=[0, 0], spent=0, wear=0,
                            repaired=0))


def observe(s):
    return [s["energy"] / 1000, s["integrity"] / 1000, s["position"] / 4,
            s["stocks"][0] / 420, s["stocks"][1] / 420,
            float(s["repair_side"] == 0), float(s["repair_side"] == 1), 1., 1.]


def balance(s):
    led = s["ledger"]
    assert s["energy"] == led["initial_energy"] + sum(led["food"]) - led["spent"]
    assert s["integrity"] == led["initial_integrity"] + led["repaired"] - led["wear"]
    for side in (0, 1):
        assert s["stocks"][side] == led["initial_stocks"][side] + led["supplied"][side] - led["food"][side]
        assert 0 <= s["stocks"][side] <= 420
        assert led["supplied"][side] + led["overflow"][side] == s["resources"]["renewal"] * s["tick"]


def transition(s, action):
    assert type(action) is int and action in range(6)
    assert s["energy"] > 0 and s["integrity"] > 0 and s["mode"] == "finite"
    before = observe(s)
    cost = 7 + (4 if action in (1, 2) else 3 if action == 3 else
                25 if action == 4 else 8 if action == 5 else 0)
    led = s["ledger"]
    spent = min(cost, s["energy"])
    worn = min(3, s["integrity"])
    s["energy"] -= spent
    s["integrity"] -= worn
    led["spent"] += spent
    led["wear"] += worn
    executed = s["energy"] > 0 and s["integrity"] > 0
    if executed:
        if action == 1:
            s["position"] = max(0, s["position"] - 1)
        elif action == 2:
            s["position"] = min(4, s["position"] + 1)
        elif s["position"] in (0, 4):
            side = s["position"] // 4
            if action == 3:
                gained = min(420, 1000 - s["energy"], s["stocks"][side])
                s["energy"] += gained
                s["stocks"][side] -= gained
                led["food"][side] += gained
            elif action == 5 and side == s["repair_side"] and s["repair_enabled"]:
                gained = min(450, 1000 - s["integrity"])
                s["integrity"] += gained
                led["repaired"] += gained
    renewal = s["resources"]["renewal"]
    for side in (0, 1):
        supplied = min(renewal, 420 - s["stocks"][side])
        s["stocks"][side] += supplied
        led["supplied"][side] += supplied
        led["overflow"][side] += renewal - supplied
    s["tick"] += 1
    balance(s)
    return before, observe(s), not executed, executed


def physical(s):
    led = s["ledger"]
    return ["resource", s["energy"], s["integrity"], s["position"], s["tick"],
            s["repair_side"], *s["stocks"], *led["food"], *led["supplied"],
            *led["overflow"], led["spent"], led["wear"], led["repaired"]]
