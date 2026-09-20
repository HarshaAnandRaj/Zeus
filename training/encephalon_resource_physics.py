"""Independent integer implementation for E1-B, without calling the primary world."""
def initial(seed, mode, energy, integrity, repair=True):
    q = 420
    strata = ((q, q), (q // 4, q), (q, q // 4), (q // 2, q // 2))
    stocks = list(strata[(seed // 4) % 4] if mode == "finite" else (q, q))
    return dict(version="encephalon-resources-v1-20260920",
                config=dict(capacity=1000, metabolism=7, wear=3, move_cost=4, feed_cost=3,
                            inspect_cost=25, repair_cost=8, food_gain=420, repair_gain=450),
                resources=dict(capacity=420, renewal=4), mode=mode, repair_enabled=repair,
                energy=energy, integrity=integrity, position=2, tick=0, repair_side=(seed // 2) % 2,
                stocks=stocks, ledger=dict(initial_energy=energy, initial_integrity=integrity,
                    initial_stocks=stocks.copy(), supplied=[0, 0], overflow=[0, 0], food=[0, 0],
                    spent=0, wear=0, repaired=0))


def observe(s):
    return [s["energy"] / 1000, s["integrity"] / 1000, s["position"] / 4,
            s["stocks"][0] / 420, s["stocks"][1] / 420,
            float(s["repair_side"] == 0), float(s["repair_side"] == 1), 1., 1.]


def balance(s):
    l = s["ledger"]
    assert s["energy"] == l["initial_energy"] + sum(l["food"]) - l["spent"]
    assert s["integrity"] == l["initial_integrity"] + l["repaired"] - l["wear"]
    for i in (0, 1):
        assert s["stocks"][i] == l["initial_stocks"][i] + l["supplied"][i] - l["food"][i]
        assert 0 <= s["stocks"][i] <= 420
        if s["mode"] == "finite": assert l["supplied"][i] + l["overflow"][i] == 4 * s["tick"]
        else: assert s["stocks"][i] == 420 and l["overflow"][i] == 0


def transition(s, action):
    assert type(action) is int and action in range(6) and s["energy"] > 0 and s["integrity"] > 0
    before = observe(s)
    cost = 7
    if action == 1 or action == 2: cost += 4
    elif action == 3: cost += 3
    elif action == 4: cost += 25
    elif action == 5: cost += 8
    l = s["ledger"]
    l["spent"] += min(cost, s["energy"])
    l["wear"] += min(3, s["integrity"])
    s["energy"] = max(0, s["energy"] - cost)
    s["integrity"] = max(0, s["integrity"] - 3)
    executed = s["energy"] > 0 and s["integrity"] > 0
    if executed:
        if action == 1: s["position"] = max(0, s["position"] - 1)
        if action == 2: s["position"] = min(4, s["position"] + 1)
        if action in (3, 5) and s["position"] in (0, 4):
            side = s["position"] // 4
            if action == 3:
                amount = min(s["stocks"][side], 420, 1000 - s["energy"])
                l["food"][side] += amount
                s["energy"] += amount
                if s["mode"] == "finite": s["stocks"][side] -= amount
                else: l["supplied"][side] += amount
            elif side == s["repair_side"] and s["repair_enabled"]:
                amount = min(450, 1000 - s["integrity"])
                l["repaired"] += amount
                s["integrity"] += amount
    if s["mode"] == "finite":
        for side in (0, 1):
            total = s["stocks"][side] + 4
            spill = max(0, total - 420)
            l["overflow"][side] += spill
            l["supplied"][side] += 4 - spill
            s["stocks"][side] = min(total, 420)
    s["tick"] += 1
    balance(s)
    return before, observe(s), not executed, executed


def reference_action(o, memory):
    repair = int(o[6])
    route = memory["route"]
    if route is None:
        if memory["resident"] and o[2] != repair:
            return 1 if o[2] > repair else 2
        hungry, damaged = o[0] < (.65 if memory["resident"] else .40), o[1] < .60
        if not hungry and not damaged: return 0
        food_first = hungry and (not damaged or round(o[0] * 1000) * 3 <= round(o[1] * 1000) * 7)
        kind = 3 if food_first else 5
        side = repair
        if kind == 3 and not memory["resident"]:
            scores = [420 * o[3 + j] - 44 * abs(o[2] - j) for j in (0, 1)]
            side = int(scores[1] > scores[0])
        route = [kind, side]
        memory["route"] = route
    kind, side = route
    if o[2] < side: return 2
    if o[2] > side: return 1
    memory["route"] = None
    return kind


def causal_state(s, memory):
    # Absolute clock and cumulative ledgers do not enter time-homogeneous physics/policy.
    return (s["energy"], s["integrity"], s["position"], tuple(s["stocks"]),
            s["repair_side"], s["repair_enabled"], s["mode"], memory["resident"],
            tuple(memory["route"]) if memory["route"] is not None else None)
