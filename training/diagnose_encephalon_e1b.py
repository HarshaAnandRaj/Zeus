"""Read-only accounting after independently verified E1-B; no fitting or new rollouts."""
from collections import defaultdict
import math
from pathlib import Path
import statistics
import numpy as np

from training import encephalon_e1b_contract as K
from training.audit_encephalon_e1 import unpack
from training.run_encephalon_e0 import read, save, sha, source_sha
from training.run_encephalon_e1 import load_gzip


def blank():
    return dict(bodies=0,survivors=0,ticks=0,energy_deaths=0,integrity_deaths=0,both_deaths=0,
                never_fed=0,never_repaired=0,actions=[0]*6,food_energy=[0,0],crossings=0,
                departure_count=0,departure_energy_sum=0,departure_integrity_sum=0,ineffective_feeds=0,
                actual_energy_spent=0,renewal_supplied=0,renewal_overflow=0,initial_body_and_stock=0,
                final_body_and_stock=0)


def include(row,packet,j):
    initial,final=packet["initial"][j],packet["final"][j]
    row["bodies"]+=1;row["survivors"]+=int(packet["survived"][j]);row["ticks"]+=packet["ticks"][j]
    row["energy_deaths"]+=final["energy"]==0;row["integrity_deaths"]+=final["integrity"]==0
    row["both_deaths"]+=final["energy"]==0 and final["integrity"]==0
    row["never_fed"]+=packet["feeding"][j]==0;row["never_repaired"]+=packet["repairs"][j]==0
    row["actions"]=[a+b for a,b in zip(row["actions"],packet["action_counts"][j])]
    row["food_energy"]=[a+b for a,b in zip(row["food_energy"],packet["food_energy"][j])]
    row["crossings"]+=packet["crossings"][j]
    n,e,i=packet["departures"][j]
    row["departure_count"]+=n;row["departure_energy_sum"]+=e;row["departure_integrity_sum"]+=i
    row["ineffective_feeds"]+=packet["empty_feeds"][j]
    row["actual_energy_spent"]+=initial["energy"]+sum(packet["food_energy"][j])-final["energy"]
    if packet["ecology"]=="finite":
        ledger=final["ledger"]
        # Entire environmental/body energy identity, including a clipped terminal charge.
        before=initial["energy"]+sum(initial["stocks"])
        after=final["energy"]+sum(final["stocks"])
        supplied=sum(ledger["supplied"]);overflow=sum(ledger["overflow"])
        assert before+supplied-ledger["spent"]==after
        assert supplied+overflow==8*final["tick"]
        assert ledger["spent"]==initial["energy"]+sum(packet["food_energy"][j])-final["energy"]
        row["renewal_supplied"]+=supplied;row["renewal_overflow"]+=overflow
        row["initial_body_and_stock"]+=before;row["final_body_and_stock"]+=after


def finish(key,row):
    t=row["ticks"]
    assert sum(row["actions"])==t
    cost=[0,4,4,3,25,8]
    extra=[a*c for a,c in zip(row["actions"],cost)]
    n=row["departure_count"]
    return dict(key=key,**row,survival_rate=row["survivors"]/row["bodies"],
                mean_ticks=t/row["bodies"],action_fractions=[x/t for x in row["actions"]],
                nominal_extra_cost_by_action=extra,nominal_extra_cost_per_tick=sum(extra)/t,
                actual_energy_cost_per_tick=row["actual_energy_spent"]/t,
                mean_energy_before_departure=row["departure_energy_sum"]/n if n else None,
                mean_integrity_before_departure=row["departure_integrity_sum"]/n if n else None,
                available_renewal_per_tick=8 if "/finite/" in key else None,
                renewal_overflow_per_tick=row["renewal_overflow"]/t)


def run():
    report=read(K.REPORT)
    assert report["evidence_verdict"]=="PASS" and not report["development"]
    c=report["manifest"]["config"]
    groups=defaultdict(blank);models=[];probes=defaultdict(list)
    for artifact in report["artifacts"]:
        path=K.ROOT/artifact["archive"]
        assert sha(path)==artifact["archive_sha256"]
        value=load_gzip(path);final=unpack(value["final"])
        history=final["history"]
        assert len(history)==c["updates"]
        model=dict(arm=artifact["arm"],lineage=artifact["lineage"],
                   live_training_decisions=sum(h["live_decisions"] for h in history),
                   original_live_decisions=sum(h["original_live_decisions"] for h in history),
                   resource_live_decisions=sum(h["resource_live_decisions"] for h in history),
                   created_bodies=final["created"],parameter_count=sum(t.numel() for t in final["model"].values()))
        for label,rows in (("first128",history[:128]),("last128",history[-128:])):
            model[label]={name:statistics.fmean(h[name] for h in rows) for name in ("entropy","prediction","policy","value","gradient_norm")}
            model[label]["entropy_fraction_of_maximum"]=model[label]["entropy"]/math.log(6)
            model[label]["module_gradient_norms"]={name:statistics.fmean(h["module_gradient_norms"][name] for h in rows) for name in rows[0]["module_gradient_norms"]}
        sampled_states=defaultdict(list)
        for p in value["endpoints"]:
            a,e,x,need=p["arm"],p["ecology"],p["control"],p["profile"]
            for j in range(len(p["survived"])):
                include(groups[f"{a}/{e}/{x}/all"],p,j)
                include(groups[f"{a}/{e}/{x}/{need}"],p,j)
                if x=="trained":
                    s=p["initial"][j]
                    layout=("shared" if s["food_side"]==s["repair_side"] else "separate") if e=="original" else "stock_"+"_".join(map(str,s["stocks"]))
                    include(groups[f"{a}/{e}/{x}/layout/{layout}"],p,j)
            for probe in p["stock_probes"]:
                probes[a].append(dict(lineage=p["lineage"],profile=need,tick=probe["tick"],lane=probe["lane"],
                    total_variation=.5*sum(abs(x-y) for x,y in zip(probe["empty"],probe["full"])),
                    feed_probability_change=probe["full"][3]-probe["empty"][3],
                    left_probability_change=probe["full"][1]-probe["empty"][1]))
            if x=="trained":
                for anchor in p["anchors"]:
                    for lane,h in enumerate(anchor["h"]):
                        if p["actions"][anchor["tick"]][lane]>=0:sampled_states[e].append(h)
        model["sampled_context_usage"]={}
        for ecology,states in sampled_states.items():
            matrix=np.asarray(states,dtype=float)
            centered=matrix-matrix.mean(axis=0)
            variance=np.linalg.svd(centered,compute_uv=False)**2
            total=float(variance.sum())
            normalized=variance/total if total else variance
            model["sampled_context_usage"][ecology]=dict(samples=len(matrix),ambient_width=matrix.shape[1],
                participation_ratio=total**2/float((variance**2).sum()) if total else 0.,
                axes_for_95_percent_variance=int(np.searchsorted(np.cumsum(normalized),.95)+1) if total else 0,
                activation_fraction_abs_above_95=float(np.mean(np.abs(matrix)>.95)),
                scope="Centered covariance of first four live lanes at 64-tick anchors; not representational capacity or controllability")
        models.append(model)
    assert len(models)==len(c["arms"])*c["lineages"]
    rows=[finish(key,value) for key,value in sorted(groups.items())]
    overall=[r for r in rows if r["key"].endswith("/all")]
    assert sum(r["bodies"] for r in overall)==report["independently_replayed_bodies"]
    assert sum(r["ticks"] for r in overall)==report["independently_replayed_steps"]
    output=dict(scope="Descriptive diagnosis; no new qualification or causal claim",source_report_sha256=sha(K.REPORT),
        diagnostic_source_sha256=source_sha(__file__),frozen_campaign_commit=report["manifest"]["commit"],
        e1b_verdict=report["e1b_verdict"],controller_qualification=report["controller_qualification"],
        resource_benefit=report["resource_benefit"],capacity_benefit=report["capacity_benefit"],
        interaction_benefit=report["interaction_benefit"],
        unique_fit_training_decisions=sum(m["live_training_decisions"] for m in models),
        training_decisions_across_exact_twins=2*sum(m["live_training_decisions"] for m in models),
        models=models,groups=rows,stock_probes=dict(probes),
        limitations=["One identical repeat analyzed; repeats are not independent models.",
                    "Subgroup rates, costs and stock sensitivity are descriptive, not additional gates.",
                    "Nominal costs include the entire final requested action; actual terminal charges can be clipped.",
                    "Finite renewal supplies at most8 energy per tick across both patches; initial stores can temporarily finance higher costs.",
                    "Context variance is sampled and affected by survival and state occupancy; low effective dimension does not prove redundant capacity.",
                    "A failed controller comparison does not prove capacity or resource pressure irrelevant under another learner."])
    target=K.REPORT.with_name(K.REPORT.stem+"_diagnosis.json")
    save(target,output)
    for row in overall:
        if "/trained/" in row["key"]:
            print(row["key"],f"{row['survivors']}/{row['bodies']}",f"energy_cost/tick={row['actual_energy_cost_per_tick']:.3f}",flush=True)


if __name__=="__main__":run()
