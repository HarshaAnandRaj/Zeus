"""Geometry and fixed one-decision controller; no learned Zeus capability claim."""
import base64
import math
import numpy as np

from training import hegh0_contract as K


def pack(matrix):
    matrix = np.asarray(matrix, dtype="<f8", order="C")
    return dict(shape=list(matrix.shape), dtype="<f8",
                data=base64.b64encode(matrix.tobytes()).decode("ascii"))


def unpack(packet):
    if packet["dtype"] != "<f8":
        raise ValueError("unexpected geometry dtype")
    blob = base64.b64decode(packet["data"], validate=True)
    if len(blob) != 8 * math.prod(packet["shape"]):
        raise ValueError("geometry byte count mismatch")
    return np.frombuffer(blob, dtype="<f8").reshape(packet["shape"]).copy()


def sphere(raw):
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def geometry(raw, c):
    vectors = {d: sphere(raw[:, :d]) for d in c["dimensions"]}
    diagnostics, prices = {}, {}
    for d, x in vectors.items():
        gram = x @ x.T
        distances = np.linalg.norm(x[1:] - x[0], axis=1)
        mean = float(distances.mean())
        prices[d] = c["target_mean_cost"] * distances / mean
        diagnostics[str(d)] = dict(
            distances=distances.tolist(), cosines=gram[0, 1:].tolist(),
            cv=float(distances.std() / mean),
            maximum_norm_error=float(np.max(np.abs(np.diag(gram)-1))),
            minimum_gram_eigenvalue=float(np.linalg.eigvalsh(gram).min()),
        )
    low, high = c["dimensions"]
    lifted = np.zeros_like(vectors[high]); lifted[:, :low] = vectors[low]
    lifted_distances = np.linalg.norm(lifted[1:] - lifted[0], axis=1)
    assert np.allclose(lifted_distances, diagnostics[str(low)]["distances"], rtol=0, atol=c["numerical_atol"])
    wide = prices[low]
    return diagnostics, dict(
        wide=wide.tolist(),
        narrow=(c["target_mean_cost"] + c["contraction"]*(wide-c["target_mean_cost"])).tolist(),
        hd_priced=prices[high].tolist(), hd_relabel=wide.tolist(), lifted=wide.tolist(),
        constant=[c["target_mean_cost"]]*c["destinations"],
    )


def fixed_controller(costs, reserve, information, bonus, task, goal, c):
    estimates = costs if information == "known" else [c["target_mean_cost"]]*len(costs)
    destination = goal if task == "requested" else min(range(len(costs)), key=lambda j: (estimates[j], j))
    cutoff = c["service_value"] + bonus - c["local_value"]
    depart = estimates[destination] < min(reserve, cutoff)
    if not depart:
        return [-1, 0, 0, 0., reserve, c["local_value"], 0.]
    cost = costs[destination]
    success = cost < reserve  # cost paid first; zero reserve is terminal
    paid = min(cost, reserve)
    utility = c["service_value"]-cost if success else -1.
    return [destination, int(success), int(not success), paid, max(0., reserve-cost),
            utility, bonus if success else 0.]


def key(condition, reserve, information, bonus, task):
    return f"{condition}/{reserve:g}/{information}/{bonus:g}/{task}"


def summarize(actions):
    count = len(actions)
    return dict(bodies=count, departed=sum(a[0]>=0 for a in actions),
                successes=sum(a[1] for a in actions), deaths=sum(a[2] for a in actions),
                paid=float(sum(a[3] for a in actions)),
                final_energy=float(sum(a[4] for a in actions)),
                unbonused_utility=float(sum(a[5] for a in actions)),
                delivered_bonus=float(sum(a[6] for a in actions)),
                success_rate=sum(a[1] for a in actions)/count,
                death_rate=sum(a[2] for a in actions)/count)


def generate(index, c, development=False):
    base = c["development_base"] if development else c["heldout_base"]
    seed = base + index
    raw = np.random.Generator(np.random.PCG64(seed)).standard_normal(
        (c["destinations"]+1, max(c["dimensions"])))
    diagnostics, costs = geometry(raw, c)
    cases, thresholds, physical_access = {}, {}, {}
    margins = []
    for condition, values in costs.items():
        ordered = sorted(values)
        critical = lambda price: max(0., price-(c["service_value"]-c["local_value"]))
        q90 = ordered[math.ceil(.9*len(values))-1]
        thresholds[condition] = dict(
            first_known_infimum=critical(ordered[0]), first_hidden_infimum=critical(c["target_mean_cost"]),
            coverage90_known_infimum=critical(q90) if q90 < max(c["reserves"]) else None,
            coverage90_censored=q90 >= max(c["reserves"]),
        )
        for reserve in c["reserves"]:
            physical_access[f"{condition}/{reserve:g}"] = sum(v < reserve for v in values)/len(values)
            for information in c["information"]:
                for bonus in c["bonuses"]:
                    for task in c["tasks"]:
                        goals = range(c["destinations"]) if task == "requested" else [-1]
                        actions = [fixed_controller(values, reserve, information, bonus, task, g, c) for g in goals]
                        cases[key(condition,reserve,information,bonus,task)] = dict(actions=actions, **summarize(actions))
                    for value in values:
                        margins.extend((abs(value-reserve), abs(value-(c["service_value"]+bonus-c["local_value"]))))
    return dict(index=index, seed=seed, development=development, raw=pack(raw),
                geometry=diagnostics, costs=costs, thresholds=thresholds, physical_access=physical_access, cases=cases,
                minimum_boundary_margin=min(margins))


def student_cdf(value, degrees):
    # Integral of cos(theta)^(degrees-1), using its exact reduction recurrence.
    def integral(theta):
        exponent = degrees-1
        result = math.sin(theta) if exponent % 2 else theta
        start = 3 if exponent % 2 else 2
        for n in range(start, exponent+1, 2):
            result = math.sin(theta)*math.cos(theta)**(n-1)/n + (n-1)*result/n
        return result
    angle = math.atan(value/math.sqrt(degrees))
    return .5 + integral(angle)/(2*integral(math.pi/2))


def critical_t(c):
    target = 1-c["family_alpha"]/(2*c["family_comparisons"])
    lo, hi = 0., 128.
    for _ in range(80):
        mid = (lo+hi)/2
        if student_cdf(mid,c["maps"]-1) < target: lo=mid
        else: hi=mid
    return (lo+hi)/2


def contrast_values(m):
    def rate(a, reserve=1.1, info="known", bonus=.2, metric="success_rate"):
        return m["cases"][key(a,reserve,info,bonus,"requested")][metric]
    return dict(
        safe_access_known=rate("narrow")-rate("wide"),
        safe_access_hidden=rate("narrow",info="hidden")-rate("wide",info="hidden"),
        mortality_reduction_hidden=rate("wide",info="hidden",metric="death_rate")-rate("narrow",info="hidden",metric="death_rate"),
        bonus_interaction=(rate("narrow")-rate("narrow",bonus=0.))-(rate("wide")-rate("wide",bonus=0.)),
        safe_access_below_mean=rate("narrow",reserve=.9)-rate("wide",reserve=.9),
        hd_priced_safe_access=rate("hd_priced")-rate("wide"),
        lower_first_departure_bonus=m["thresholds"]["wide"]["first_known_infimum"]-m["thresholds"]["narrow"]["first_known_infimum"],
    )


def decide(maps, c):
    assert len(maps)==c["maps"] and sorted(m["index"] for m in maps)==list(range(c["maps"]))
    maps=sorted(maps,key=lambda m:m["index"])
    t=critical_t(c); observations=[contrast_values(m) for m in maps]; contrasts={}
    for name in observations[0]:
        values=np.array([row[name] for row in observations])
        mean=float(values.mean()); half=t*float(values.std(ddof=1))/math.sqrt(len(values))
        margin=c["first_departure_margin"] if name=="lower_first_departure_bonus" else c["benefit_margin"]
        contrasts[name]=dict(values=values.tolist(),mean=mean,lower=mean-half,upper=mean+half,margin=margin,
                             verdict="PASS" if mean-half>margin else "FAIL")
    assert len(contrasts)==c["family_comparisons"]
    geometry_summary={}
    for d in c["dimensions"]:
        cosines=np.array([m["geometry"][str(d)]["cosines"] for m in maps])
        geometry_summary[str(d)]=dict(mean_cv=float(np.mean([m["geometry"][str(d)]["cv"] for m in maps])),
            normalized_cosine_mean=float(math.sqrt(d)*cosines.mean()),
            normalized_cosine_second_moment=float(d*np.mean(cosines**2)))
    low,high=map(str,c["dimensions"])
    ratio=geometry_summary[high]["mean_cv"]/geometry_summary[low]["mean_cv"]
    geometry_pass=ratio<=c["geometry_cv_ratio_max"] and all(
        abs(g["normalized_cosine_mean"])<c["normalized_mean_limit"] and
        c["normalized_second_moment_range"][0] < g["normalized_cosine_second_moment"] < c["normalized_second_moment_range"][1]
        for g in geometry_summary.values())
    aggregates={}
    for name in maps[0]["cases"]:
        rows=[m["cases"][name] for m in maps]
        total={field:sum(r[field] for r in rows) for field in ("bodies","departed","successes","deaths","paid","final_energy","unbonused_utility","delivered_bonus")}
        total.update(success_rate=total["successes"]/total["bodies"],death_rate=total["deaths"]/total["bodies"])
        aggregates[name]=total
    return dict(geometry_verdict="PASS" if geometry_pass else "FAIL", geometry=geometry_summary,
                hd_to_low_cv_ratio=ratio,critical_t=t,contrasts=contrasts,aggregates=aggregates,
                affordance_verdict="PASS" if all(contrasts[n]["verdict"]=="PASS" for n in ("safe_access_known","safe_access_hidden","mortality_reduction_hidden")) else "FAIL",
                lower_departure_incentive_verdict=contrasts["lower_first_departure_bonus"]["verdict"],
                below_mean_access_verdict=contrasts["safe_access_below_mean"]["verdict"],
                recurrent_geometry_bridge="NOT_TESTED",learned_zeus_control="NOT_TESTED")
