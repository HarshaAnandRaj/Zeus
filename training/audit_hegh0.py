"""Independent scalar geometry, decisions, conservation and statistics for HEGH-0."""
import base64
import math
import statistics
import struct
import numpy as np

from training import hegh0_contract as K


def close(x, y, tolerance=1e-10):
    if isinstance(x, dict):
        assert isinstance(y, dict) and x.keys()==y.keys(), "mapping identity mismatch"
        for name in x: close(x[name],y[name],tolerance)
    elif isinstance(x, (list,tuple)):
        assert isinstance(y,(list,tuple)) and len(x)==len(y), "sequence identity mismatch"
        for a,b in zip(x,y):close(a,b,tolerance)
    elif type(x) is int:
        assert type(y) is int and x==y, (x,y)
    elif isinstance(x, float):
        assert isinstance(y,(int,float)) and math.isfinite(y) and abs(x-y)<=tolerance, (x,y)
    else:
        assert x==y, (x,y)


def geometry_reference(packet, c):
    raw=packet["raw"]
    rows,columns=c["destinations"]+1,max(c["dimensions"])
    assert raw["shape"]==[rows,columns] and raw["dtype"]=="<f8"
    blob=base64.b64decode(raw["data"],validate=True)
    values=struct.unpack("<"+"d"*(rows*columns),blob)
    assert all(math.isfinite(x) for x in values)
    # Verify the submitted random stream as provenance; geometry is recomputed below.
    generated=np.random.Generator(np.random.PCG64(packet["seed"])).standard_normal((rows,columns))
    assert generated.astype("<f8").tobytes()==blob, "random-stream mismatch"
    coords=[values[j*columns:(j+1)*columns] for j in range(rows)]
    measured,prices={},{}
    for dimension in c["dimensions"]:
        points=[]
        for row in coords:
            norm=math.sqrt(math.fsum(v*v for v in row[:dimension]))
            assert norm>0
            points.append([v/norm for v in row[:dimension]])
        cosine=[math.fsum(a*b for a,b in zip(points[0],point)) for point in points[1:]]
        distances=[math.sqrt(2-2*x) for x in cosine]
        avg=statistics.fmean(distances)
        matrix=np.array(points)
        eigenvalue=float(np.linalg.eigvalsh(matrix@matrix.T)[0])
        norm_error=max(abs(math.fsum(v*v for v in point)-1) for point in points)
        assert norm_error<=c["numerical_atol"] and eigenvalue>=-c["numerical_atol"]
        measured[str(dimension)]=dict(distances=distances,cosines=cosine,
            cv=statistics.pstdev(distances)/avg,maximum_norm_error=norm_error,
            minimum_gram_eigenvalue=eigenvalue)
        prices[dimension]=[c["target_mean_cost"]*x/avg for x in distances]
    low,high=c["dimensions"]
    broad=prices[low]
    tariffs=dict(wide=broad,narrow=[(1-c["contraction"])*c["target_mean_cost"]+c["contraction"]*x for x in broad],
        hd_priced=prices[high],hd_relabel=broad.copy(),lifted=broad.copy(),constant=[c["target_mean_cost"]]*c["destinations"])
    for row in tariffs.values():
        assert min(row)>0 and abs(statistics.fmean(row)-c["target_mean_cost"])<c["numerical_atol"]
    close(measured,packet["geometry"],c["numerical_atol"])
    close(tariffs,packet["costs"],c["numerical_atol"])
    return tariffs


def replay(packet,c,development=False):
    assert 0<=packet["index"]<c["maps"] and packet["development"] is development
    assert packet["seed"]==(c["development_base"] if development else c["heldout_base"])+packet["index"]
    tariffs=geometry_reference(packet,c)
    expected_keys=set(); thresholds={}; availability={}; margins=[]; bodies=0; maximum_error=0.
    for name,costs in tariffs.items():
        ordered=sorted(costs); ninety=ordered[math.ceil(.9*len(costs))-1]
        thresholds[name]=dict(first_known_infimum=max(0.,ordered[0]-c["service_value"]+c["local_value"]),
            first_hidden_infimum=max(0.,c["target_mean_cost"]-c["service_value"]+c["local_value"]),
            coverage90_known_infimum=max(0.,ninety-c["service_value"]+c["local_value"]) if ninety<max(c["reserves"]) else None,
            coverage90_censored=ninety>=max(c["reserves"]))
        for reserve in c["reserves"]:
            availability[f"{name}/{reserve:g}"]=sum(x<reserve for x in costs)/len(costs)
            for information in c["information"]:
                for bonus in c["bonuses"]:
                    ceiling=c["service_value"]+bonus-c["local_value"]
                    for cost in costs:margins.extend([abs(cost-reserve),abs(cost-ceiling)])
                    for task in c["tasks"]:
                        identity=f"{name}/{reserve:g}/{information}/{bonus:g}/{task}"
                        expected_keys.add(identity); wanted=[]
                        goals=range(c["destinations"]) if task=="requested" else [None]
                        for goal in goals:
                            destination=(min(range(len(costs)),key=lambda j:(costs[j],j)) if information=="known" else 0) if goal is None else goal
                            estimate=costs[destination] if information=="known" else c["target_mean_cost"]
                            if estimate>=reserve or estimate>=ceiling:
                                action=[-1,0,0,0.,reserve,c["local_value"],0.]
                            elif costs[destination]>=reserve:
                                action=[destination,0,1,reserve,0.,-1.,0.]
                            else:
                                expense=costs[destination]
                                action=[destination,1,0,expense,reserve-expense,c["service_value"]-expense,bonus]
                            error=abs(action[3]+action[4]-reserve)
                            maximum_error=max(maximum_error,error)
                            assert error<c["numerical_atol"]
                            wanted.append(action)
                        n=len(wanted);bodies+=n
                        sums=[math.fsum(x[j] for x in wanted) for j in range(1,7)]
                        summary=dict(actions=wanted,bodies=n,departed=sum(x[0]!=-1 for x in wanted),
                            successes=int(sums[0]),deaths=int(sums[1]),paid=sums[2],final_energy=sums[3],
                            unbonused_utility=sums[4],delivered_bonus=sums[5],success_rate=sums[0]/n,death_rate=sums[1]/n)
                        close(summary,packet["cases"][identity],c["numerical_atol"])
    assert set(packet["cases"])==expected_keys
    close(thresholds,packet["thresholds"],c["numerical_atol"])
    close(availability,packet["physical_access"],c["numerical_atol"])
    close(min(margins),packet["minimum_boundary_margin"],c["numerical_atol"])
    assert min(margins)>c["boundary_margin"], "numerically ambiguous decision boundary"
    for prefix in ("hd_relabel","lifted"):
        for identity,case in packet["cases"].items():
            if identity.startswith("wide/"):
                assert case==packet["cases"][prefix+identity[4:]],"geometry-only control changed decisions"
    return dict(bodies=bodies,maximum_conservation_error=maximum_error,minimum_boundary_margin=min(margins))


def cdf_quadrature(t,degrees):
    # Independent Simpson integration of Student's density, not the primary recurrence.
    coefficient=math.exp(math.lgamma((degrees+1)/2)-math.lgamma(degrees/2))/math.sqrt(degrees*math.pi)
    intervals=8192;h=t/intervals
    f=lambda x: coefficient*(1+x*x/degrees)**(-(degrees+1)/2)
    total=f(0)+f(t)
    total+=4*math.fsum(f(j*h) for j in range(1,intervals,2))
    total+=2*math.fsum(f(j*h) for j in range(2,intervals,2))
    return .5+h*total/3


def audit_decision(maps,decision,c):
    assert len(maps)==c["maps"] and {m["index"] for m in maps}==set(range(c["maps"]))
    maps=sorted(maps,key=lambda m:m["index"])
    t=decision["critical_t"]
    assert abs(cdf_quadrature(t,c["maps"]-1)-(1-c["family_alpha"]/(2*c["family_comparisons"])))<1e-10
    columns={name:[] for name in ("safe_access_known","safe_access_hidden","mortality_reduction_hidden","bonus_interaction","safe_access_below_mean","hd_priced_safe_access","lower_first_departure_bonus")}
    for m in maps:
        def rate(a,reserve=1.1,info="known",bonus=.2,death=False):
            row=m["cases"][f"{a}/{reserve:g}/{info}/{bonus:g}/requested"]
            return row["deaths" if death else "successes"]/row["bodies"]
        values=[rate("narrow")-rate("wide"),rate("narrow",info="hidden")-rate("wide",info="hidden"),
            rate("wide",info="hidden",death=True)-rate("narrow",info="hidden",death=True),
            rate("narrow")-rate("wide")-rate("narrow",bonus=0.)+rate("wide",bonus=0.),
            rate("narrow",reserve=.9)-rate("wide",reserve=.9),rate("hd_priced")-rate("wide"),
            m["thresholds"]["wide"]["first_known_infimum"]-m["thresholds"]["narrow"]["first_known_infimum"]]
        for name,value in zip(columns,values):columns[name].append(value)
    expected={}
    for name,values in columns.items():
        mean=statistics.fmean(values);half=t*statistics.stdev(values)/math.sqrt(len(values))
        margin=c["first_departure_margin"] if name=="lower_first_departure_bonus" else c["benefit_margin"]
        expected[name]=dict(values=values,mean=mean,lower=mean-half,upper=mean+half,margin=margin,
                            verdict="PASS" if mean-half>margin else "FAIL")
    close(expected,decision["contrasts"])
    geometry={}
    for d in c["dimensions"]:
        points=[x for m in maps for x in m["geometry"][str(d)]["cosines"]]
        geometry[str(d)]=dict(mean_cv=statistics.fmean(m["geometry"][str(d)]["cv"] for m in maps),
            normalized_cosine_mean=math.sqrt(d)*statistics.fmean(points),
            normalized_cosine_second_moment=d*statistics.fmean(x*x for x in points))
    close(geometry,decision["geometry"])
    ratio=geometry[str(c["dimensions"][1])]["mean_cv"]/geometry[str(c["dimensions"][0])]["mean_cv"]
    close(ratio,decision["hd_to_low_cv_ratio"])
    geo=(ratio<=c["geometry_cv_ratio_max"] and all(abs(g["normalized_cosine_mean"])<c["normalized_mean_limit"] and
        c["normalized_second_moment_range"][0]<g["normalized_cosine_second_moment"]<c["normalized_second_moment_range"][1] for g in geometry.values()))
    assert decision["geometry_verdict"]==("PASS" if geo else "FAIL")
    assert decision["affordance_verdict"]==("PASS" if all(expected[n]["verdict"]=="PASS" for n in ("safe_access_known","safe_access_hidden","mortality_reduction_hidden")) else "FAIL")
    assert decision["lower_departure_incentive_verdict"]==expected["lower_first_departure_bonus"]["verdict"]
    assert decision["below_mean_access_verdict"]==expected["safe_access_below_mean"]["verdict"]
    assert decision["recurrent_geometry_bridge"]==decision["learned_zeus_control"]=="NOT_TESTED"
    assert decision["aggregates"].keys()==maps[0]["cases"].keys()
    for name,row in decision["aggregates"].items():
        pieces=[m["cases"][name] for m in maps]
        for field in ("bodies","departed","successes","deaths","paid","final_energy","unbonused_utility","delivered_bonus"):
            value=(sum(p[field] for p in pieces) if field in ("bodies","departed","successes","deaths") else math.fsum(p[field] for p in pieces))
            close(value,row[field],1e-8)
        close(row["successes"]/row["bodies"],row["success_rate"])
        close(row["deaths"]/row["bodies"],row["death_rate"])


def exact_controls():
    from fractions import Fraction as F
    assert F(1)**2+F(-1)**2==2
    assert (F(1)/F('.01'))**2+F(-1)**2==10001
    wide=(F('.2'),F('1.8'));narrow=(F(1),F(1))
    assert sum(wide)/2==sum(narrow)/2==1
    assert min(wide)==F('.2') and min(narrow)==1
    assert sum(x<F('1.2') for x in wide)==1 and sum(x<F('1.2') for x in narrow)==2
    assert np.linalg.eigvalsh([[1,.9,.9],[.9,1,-.9],[.9,-.9,1]])[0]<0
    return dict(equal_distance_energy=[2,10001],equal_mean_departure_threshold=[.2,1.],
                affordable_destinations_at_1_2=[1,2],invalid_pairwise_gram_rejected=True)
