"""Independent E1-B neural accumulation, physics, raw actions and verdict checks."""
import copy
import hashlib
import math
import statistics
import numpy as np

from training import encephalon_e1b_contract as K
from training import audit_encephalon_e0 as O
from training import encephalon_resource_physics as R
from training.audit_encephalon_e1 import Neural
from training.run_encephalon_e0 import encoded


def close(a, b, c):
    a, b = np.asarray(a), np.asarray(b)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(a, b, atol=c["neural_atol"], rtol=c["neural_rtol"])
    return float(np.max(np.abs(a-b))) if a.size else 0.


def initial(ecology, seed, energy, integrity, repair):
    if ecology != "original": return R.initial(seed, ecology, energy, integrity, repair)
    physics = dict(capacity=1000, metabolism=7, wear=3, move_cost=4, feed_cost=3,
                   inspect_cost=25, repair_cost=8, food_gain=420, repair_gain=450)
    return O.initial(seed, physics, visible=True, repair=repair, energy=energy, integrity=integrity)


def observe(s):
    return O.observe(s) if "food_side" in s else R.observe(s)


def physical(s):
    if "food_side" in s: return ["original", *O.physical(s)]
    keys = ("energy", "integrity", "position", "tick", "repair_side")
    out = ["resource"] + [s[k] for k in keys] + list(s["stocks"])
    for key in ("food", "supplied", "overflow"): out.extend(s["ledger"][key])
    out.extend(s["ledger"][key] for key in ("spent", "wear", "repaired"))
    return out


def step(s, action):
    if "food_side" not in s: return R.transition(s, action)
    row = O.transition(s, action)
    return row[1], row[4], row[5], row[6]


def probability(logits):
    z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def replay(packet, model, config=None, development=False):
    c = K.CONFIG if config is None else config
    n, width = c["endpoint_bodies"], c["arms"][packet["arm"]][1]
    assert packet["horizon"] == c["endpoint_horizon"] and packet["development"] is development
    assert packet["control"] in c["controls"] and packet["ecology"] in c["ecologies"]
    ecology, profile = packet["ecology"], packet["profile"]
    base = c["development_base"] if development else c["heldout_base"]
    # Independently arranged seed identity, not supplied by the raw packet.
    seed = base + c["endpoint_sampling_offset"] + 100 * packet["lineage"] + 10 * c["ecologies"].index(ecology) + list(c["profiles"]).index(profile)
    assert seed == packet["sampler_seed"]
    energy, integrity = c["profiles"][profile]
    worlds = [initial(ecology, base+j, energy, integrity, packet["control"] != "repair_disabled") for j in range(n)]
    assert worlds == packet["initial"]
    neural = Neural(model, "recurrent")
    state = dict(h=np.zeros((n,width)), previous=np.full(n,-1,dtype=np.int64), reward=np.zeros(n))
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = np.zeros((n,6),dtype=np.int64)
    feeding, repairs, empty = [np.zeros(n,dtype=np.int64) for _ in range(3)]
    food = np.zeros((n,2),dtype=np.int64)
    crossings = np.zeros(n,dtype=np.int64); last_station = np.full(n,-1)
    departures = np.zeros((n,3),dtype=np.int64)
    anchors = {row["tick"]:row for row in packet["anchors"]}
    assert len(anchors) == len(packet["anchors"]) and set(anchors) == set(range(0,len(packet["actions"]),64))
    probes = {(row["tick"],row["lane"]):row for row in packet["stock_probes"]}
    assert len(probes) == len(packet["stock_probes"])
    seen_probes = set(); maximum_error, margin = 0.,1.
    digest = hashlib.sha256()
    for tick, recorded in enumerate(packet["actions"]):
        alive = np.array([s["energy"] > 0 and s["integrity"] > 0 for s in worlds])
        assert alive.any() and tick < packet["horizon"]
        obs = np.array([observe(s) for s in worlds])
        logits, value, _, h = neural.forward(obs,state)
        if tick in anchors:
            a = anchors[tick]
            for x,y in ((a["h"],h[:4]),(a["logits"],logits[:4]),(a["value"],value[:4])):
                maximum_error = max(maximum_error,close(x,y,c))
        if packet["control"] == "trained" and ecology == "finite" and tick in c["probe_ticks"]:
            predictions = []
            for level in (0.,1.):
                altered = obs.copy(); altered[:,3] = level
                predictions.append(probability(neural.forward(altered,state)[0]))
            for lane in range(min(c["probe_lanes"],n)):
                if alive[lane]:
                    key = (tick,lane); seen_probes.add(key)
                    for name,p in zip(("empty","full"),predictions):
                        maximum_error=max(maximum_error,close(probes[key][name],p[lane],c))
        cumulative = np.cumsum(probability(logits),axis=1); cumulative[:,-1] = 1.
        uniforms = rng.random(n)
        expected = np.array([np.searchsorted(row,u,side="right") for row,u in zip(cumulative,uniforms)])
        margin=min(margin,float(np.min(np.abs(cumulative[alive,:-1]-uniforms[alive,None]))))
        expected[~alive]=-1
        assert expected.tolist()==recorded,"independent raw action disagrees"
        rewards=np.zeros(n)
        for lane,s in enumerate(worlds):
            if not alive[lane]:continue
            a=int(expected[lane]); old_e,old_i,old_p=s["energy"],s["integrity"],s["position"]
            before,after,ended,_=step(s,a)
            rewards[lane]=(-1. if ended else .01)+.1*(after[0]-before[0])+.1*(after[1]-before[1])
            counts[lane,a]+=1
            if a==3:
                gain=s["energy"]-max(0,old_e-10)
                feeding[lane]+=gain>0;empty[lane]+=gain==0
                if gain:food[lane,old_p//4]+=gain
            if a==5:repairs[lane]+=s["integrity"]>max(0,old_i-3)
            if old_p != s["position"]:
                if old_p in (0,4):departures[lane]+=[1,old_e,old_i]
                if s["position"] in (0,4):
                    destination=s["position"]//4
                    crossings[lane]+=last_station[lane]>=0 and last_station[lane]!=destination
                    last_station[lane]=destination
        digest.update(encoded([recorded,[physical(s) for s in worlds]])+b"\n")
        state["h"][alive]=h[alive];state["previous"][alive]=expected[alive];state["reward"][alive]=rewards[alive]
    assert seen_probes==set(probes)
    assert packet["final"]==worlds and digest.hexdigest()==packet["trace_sha256"]
    for name,value in (("action_counts",counts),("feeding",feeding),("repairs",repairs),("food_energy",food),
                       ("crossings",crossings),("departures",departures),("empty_feeds",empty)):
        assert packet[name]==value.tolist(),name
    ticks=[s["tick"] for s in worlds]
    survivors=[s["energy"]>0 and s["integrity"]>0 and s["tick"]==packet["horizon"] for s in worlds]
    assert packet["ticks"]==ticks and packet["survived"]==survivors
    assert len(packet["actions"])==packet["horizon"] or not any(s["energy"]>0 and s["integrity"]>0 for s in worlds)
    for key in state:maximum_error=max(maximum_error,close(packet["controller"][key],state[key],c))
    if packet["control"]=="repair_disabled":
        assert max(ticks)<=math.ceil(integrity/3) and not any(survivors) and not repairs.any()
    return dict(bodies=n,steps=sum(ticks),maximum_neural_error=maximum_error,minimum_cdf_boundary_margin=margin,
                stock_probes=len(probes))


def audit_decision(packets, decision, config=None, development=False):
    """Recompute rates, interval bounds and gates using separate scalar arithmetic."""
    c=K.CONFIG if config is None else config
    cells={tuple(p[k] for k in ("arm","lineage","ecology","profile","control")):p for p in packets}
    expected={(a,l,e,p,x) for a in c["arms"] for l in range(c["lineages"]) for e in c["ecologies"] for p in c["profiles"] for x in c["controls"]}
    assert len(cells)==len(packets) and set(cells)==expected
    for p in packets:
        assert len(p["survived"])==c["endpoint_bodies"] and p["horizon"]==c["endpoint_horizon"] and p["development"] is development
    rate=lambda a,l,e,p,x="trained":sum(cells[a,l,e,p,x]["survived"])/c["endpoint_bodies"]
    differences={}
    for a in c["arms"]:
        for e in c["ecologies"]:
            for p in c["profiles"]:differences[f"learning/{a}/{e}/{p}"]=[rate(a,l,e,p)-rate(a,l,e,p,"untrained") for l in range(c["lineages"])]
    for p in c["profiles"]:
        for w in (32,128):differences[f"resource/{w}/{p}"]=[rate(f"finite_{w}",l,"finite",p)-rate(f"abundant_{w}",l,"finite",p) for l in range(c["lineages"])]
        for m in ("abundant","finite"):
            for e in c["ecologies"]:differences[f"capacity/{m}/{e}/{p}"]=[rate(f"{m}_128",l,e,p)-rate(f"{m}_32",l,e,p) for l in range(c["lineages"])]
        differences[f"interaction/{p}"]=[wide-small for wide,small in zip(differences[f"resource/128/{p}"],differences[f"resource/32/{p}"])]
    assert set(differences)==set(decision["contrasts"]) and len(differences)==c["family_comparisons"]
    # Validate the supplied t quantile by an independent change-of-variable quadrature.
    df=c["lineages"]-1;t=decision["critical_t"]
    theta=np.linspace(math.atan(t/math.sqrt(df)),math.pi/2,100001)
    f=np.cos(theta)**(df-1)
    coefficient=math.gamma((df+1)/2)/(math.sqrt(math.pi)*math.gamma(df/2))
    tail=coefficient*((math.pi/2-theta[0])/100000)/3*(f[0]+f[-1]+4*f[1:-1:2].sum()+2*f[2:-1:2].sum())
    assert abs(tail-c["family_alpha"]/(2*c["family_comparisons"]))<1e-10,"critical quantile differs"
    passed={}
    for key,values in differences.items():
        r=decision["contrasts"][key];bound=2. if key.startswith("interaction/") else 1.
        mean=statistics.fmean(values);half=t*statistics.stdev(values)/math.sqrt(len(values))
        assert r["differences"]==values and r["margin"]==c["benefit_margin"]
        assert abs(r["mean"]-mean)<1e-12 and abs(r["lower"]-max(-bound,mean-half))<1e-12 and abs(r["upper"]-min(bound,mean+half))<1e-12
        passed[key]=mean-half>c["benefit_margin"]
        assert r["verdict"]==("PASS" if passed[key] else "FAIL")
    summaries=decision["cells"]
    assert [tuple(r[k] for k in ("arm","lineage","ecology","profile","control")) for r in summaries]==sorted(cells)
    for row,key in zip(summaries,sorted(cells)):
        p=cells[key]
        expected_row=dict(zip(("arm","lineage","ecology","profile","control"),key))|dict(
            survived=sum(p["survived"]),bodies=len(p["survived"]),mean_ticks=sum(p["ticks"])/len(p["ticks"]),
            min_ticks=min(p["ticks"]),max_ticks=max(p["ticks"]),feeding=sum(p["feeding"]),repairs=sum(p["repairs"]),
            food_energy=[sum(v[j] for v in p["food_energy"]) for j in range(2)],
            action_counts=[sum(v[j] for v in p["action_counts"]) for j in range(6)])
        assert row==expected_row
    qualified={}
    for a in c["arms"]:
        good=True;floor=True
        for l in range(c["lineages"]):
            for e in c["ecologies"]:
                for p in c["profiles"]:
                    cell,disabled=cells[a,l,e,p,"trained"],cells[a,l,e,p,"repair_disabled"]
                    floor &= sum(cell["survived"])>=math.ceil(c["survival_floor"]*c["endpoint_bodies"])
                    good &= not any(disabled["survived"]) and not any(disabled["repairs"]) and max(disabled["ticks"])<=math.ceil(c["profiles"][p][1]/3)
                    good &= passed[f"learning/{a}/{e}/{p}"]
                    good &= all(not alive or (f>0 and r>0 and (e!="finite" or min(q)>0)) for alive,f,r,q in zip(cell["survived"],cell["feeding"],cell["repairs"],cell["food_energy"]))
        assert decision["body_floor"][a]==("PASS" if floor else "FAIL")
        qualified[a]=bool(good and floor)
        assert decision["controller_qualification"][a]==("PASS" if qualified[a] else "FAIL")
    selected=next((a for a in c["selection_priority"] if qualified[a]),None)
    assert decision["selected_arm"]==selected and decision["e1b_verdict"]==("PASS" if selected else "FAIL")
    for w in (32,128):assert decision["resource_benefit"][str(w)]==("PASS" if all(passed[f"resource/{w}/{p}"] for p in c["profiles"]) else "FAIL")
    for m in ("abundant","finite"):
        for e in c["ecologies"]:assert decision["capacity_benefit"][f"{m}/{e}"]==("PASS" if all(passed[f"capacity/{m}/{e}/{p}"] for p in c["profiles"]) else "FAIL")
    assert decision["interaction_benefit"]==("PASS" if all(passed[f"interaction/{p}"] for p in c["profiles"]) else "FAIL")
    assert decision["independent_lineages"]==c["lineages"]
