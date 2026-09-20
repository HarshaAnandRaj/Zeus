"""Identity-ordered E1-B cells and prespecified simultaneous lineage contrasts."""
import math
from functools import lru_cache
import numpy as np

from training import encephalon_e1b_contract as K
from training.audit_encephalon_e1 import student_critical


@lru_cache(maxsize=8)
def critical(lineages, alpha, comparisons):
    return student_critical(lineages - 1, alpha / (2 * comparisons))


def identity(packet):
    return tuple(packet[k] for k in ("arm", "lineage", "ecology", "profile", "control"))


def decide(packets, config=None, development=False):
    c = K.CONFIG if config is None else config
    cells = {}
    for p in packets:
        key = identity(p)
        assert key not in cells, "duplicate endpoint cell"
        assert len(p["survived"]) == c["endpoint_bodies"] and p["horizon"] == c["endpoint_horizon"]
        assert p["development"] is development
        cells[key] = p
    expected = {(a, l, e, p, x) for a in c["arms"] for l in range(c["lineages"])
                for e in c["ecologies"] for p in c["profiles"] for x in c["controls"]}
    assert set(cells) == expected, "missing or extra endpoint cells"
    rate = lambda a, l, e, p, x="trained": sum(cells[a, l, e, p, x]["survived"]) / c["endpoint_bodies"]
    t = critical(c["lineages"], c["family_alpha"], c["family_comparisons"])
    contrasts = {}
    def compare(name, differences, bound=1.):
        values = np.array(differences, dtype=float)
        mean = float(values.mean())
        half = t * float(values.std(ddof=1)) / math.sqrt(len(values))
        contrasts[name] = dict(differences=values.tolist(), mean=mean, lower=max(-bound, mean-half),
                               upper=min(bound, mean+half), margin=c["benefit_margin"],
                               verdict="PASS" if mean-half > c["benefit_margin"] else "FAIL")
    for a in c["arms"]:
        for e in c["ecologies"]:
            for p in c["profiles"]:
                compare(f"learning/{a}/{e}/{p}", [rate(a,l,e,p)-rate(a,l,e,p,"untrained") for l in range(c["lineages"])])
    for width in (32, 128):
        for p in c["profiles"]:
            compare(f"resource/{width}/{p}", [rate(f"finite_{width}",l,"finite",p)-rate(f"abundant_{width}",l,"finite",p) for l in range(c["lineages"])])
    for mode in ("abundant", "finite"):
        for e in c["ecologies"]:
            for p in c["profiles"]:
                compare(f"capacity/{mode}/{e}/{p}", [rate(f"{mode}_128",l,e,p)-rate(f"{mode}_32",l,e,p) for l in range(c["lineages"])])
    for p in c["profiles"]:
        compare(f"interaction/{p}", [(rate("finite_128",l,"finite",p)-rate("abundant_128",l,"finite",p))
                -(rate("finite_32",l,"finite",p)-rate("abundant_32",l,"finite",p)) for l in range(c["lineages"])], 2.)
    assert len(contrasts) == c["family_comparisons"]
    floors, qualified = {}, {}
    for a in c["arms"]:
        floors[a] = all(rate(a,l,e,p) >= c["survival_floor"] for l in range(c["lineages"]) for e in c["ecologies"] for p in c["profiles"])
        function = True
        for l in range(c["lineages"]):
            for e in c["ecologies"]:
                for p in c["profiles"]:
                    cell, disabled = cells[a,l,e,p,"trained"], cells[a,l,e,p,"repair_disabled"]
                    function &= not any(disabled["survived"]) and not any(disabled["repairs"])
                    function &= max(disabled["ticks"]) <= math.ceil(c["profiles"][p][1] / 3)
                    for alive, feed, repair, food in zip(cell["survived"], cell["feeding"], cell["repairs"], cell["food_energy"]):
                        function &= not alive or (feed > 0 and repair > 0 and (e != "finite" or min(food) > 0))
        learning = all(contrasts[f"learning/{a}/{e}/{p}"]["verdict"] == "PASS" for e in c["ecologies"] for p in c["profiles"])
        qualified[a] = bool(floors[a] and function and learning)
    selected = next((a for a in c["selection_priority"] if qualified[a]), None)
    summaries = []
    for key in sorted(cells):
        p = cells[key]
        summaries.append(dict(zip(("arm", "lineage", "ecology", "profile", "control"), key)) |
                         dict(survived=sum(p["survived"]), bodies=len(p["survived"]),
                              mean_ticks=float(np.mean(p["ticks"])), min_ticks=min(p["ticks"]), max_ticks=max(p["ticks"]),
                              feeding=sum(p["feeding"]), repairs=sum(p["repairs"]),
                              food_energy=np.sum(p["food_energy"],axis=0).tolist(),
                              action_counts=np.sum(p["action_counts"],axis=0).tolist()))
    return dict(e1b_verdict="PASS" if selected else "FAIL", selected_arm=selected,
                body_floor={a:"PASS" if v else "FAIL" for a,v in floors.items()},
                controller_qualification={a:"PASS" if v else "FAIL" for a,v in qualified.items()},
                resource_benefit={str(w):"PASS" if all(contrasts[f"resource/{w}/{p}"]["verdict"]=="PASS" for p in c["profiles"]) else "FAIL" for w in (32,128)},
                capacity_benefit={f"{m}/{e}":"PASS" if all(contrasts[f"capacity/{m}/{e}/{p}"]["verdict"]=="PASS" for p in c["profiles"]) else "FAIL" for m in ("abundant","finite") for e in c["ecologies"]},
                interaction_benefit="PASS" if all(contrasts[f"interaction/{p}"]["verdict"]=="PASS" for p in c["profiles"]) else "FAIL",
                critical_t=t, independent_lineages=c["lineages"], contrasts=contrasts, cells=summaries)
