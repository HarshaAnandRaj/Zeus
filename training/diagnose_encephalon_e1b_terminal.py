"""Read saved E1-B terminal states; no new actions, fitting or qualification."""
from collections import defaultdict
import statistics

from training import encephalon_e1b_contract as K
from training.run_encephalon_e0 import read, save, sha, source_sha
from training.run_encephalon_e1 import load_gzip


def run():
    report = read(K.REPORT)
    assert report["evidence_verdict"] == "PASS" and not report["development"]
    cohorts = defaultdict(list)
    for artifact in report["artifacts"]:
        path = K.ROOT / artifact["archive"]
        assert sha(path) == artifact["archive_sha256"]
        for packet in load_gzip(path)["endpoints"]:
            if packet["ecology"] != "finite" or packet["control"] != "trained":
                continue
            for lane, state in enumerate(packet["final"]):
                item = dict(
                    position=state["position"], energy=state["energy"],
                    integrity=state["integrity"], stocks=state["stocks"],
                    ticks=packet["ticks"][lane],
                    last_action=packet["actions"][packet["ticks"][lane]-1][lane],
                    fed_at_both=min(packet["food_energy"][lane]) > 0,
                    crossed=packet["crossings"][lane] > 0,
                )
                cohorts[artifact["arm"] + "/all"].append(item)
                if state["energy"] == 0:
                    cohorts[artifact["arm"] + "/energy_death"].append(item)
    rows = []
    for key, items in sorted(cohorts.items()):
        assert items
        stocks = [sum(x["stocks"]) for x in items]
        station = [x for x in items if x["position"] in (0, 4)]
        rows.append(dict(
            key=key, bodies=len(items),
            terminal_positions=[sum(x["position"] == p for x in items) for p in range(5)],
            terminal_actions=[sum(x["last_action"] == a for x in items) for a in range(6)],
            mean_integrity=statistics.fmean(x["integrity"] for x in items),
            mean_body_energy=statistics.fmean(x["energy"] for x in items),
            mean_patch_energy=statistics.fmean(stocks),
            median_patch_energy=statistics.median(stocks),
            min_patch_energy=min(stocks), max_patch_energy=max(stocks),
            mean_larger_patch=statistics.fmean(max(x["stocks"]) for x in items),
            station_deaths=len(station),
            mean_local_stock_at_station=(statistics.fmean(x["stocks"][x["position"]//4] for x in station)
                                         if station else None),
            bodies_fed_at_both=sum(x["fed_at_both"] for x in items),
            bodies_crossed=sum(x["crossed"] for x in items),
            min_ticks=min(x["ticks"] for x in items),
            median_ticks=statistics.median(x["ticks"] for x in items),
            max_ticks=max(x["ticks"] for x in items),
        ))
    assert sum(r["bodies"] for r in rows if r["key"].endswith("/all")) == 6144
    output = dict(
        scope="Descriptive terminal-state accounting from the closed E1-B evidence",
        source_report_sha256=sha(K.REPORT), diagnostic_source_sha256=source_sha(__file__),
        rows=rows,
        limitations=[
            "Stocks include the terminal tick's four-unit renewal at each patch.",
            "Unused food at death does not prove it was still reachable before death.",
            "Final states are not a counterfactual rescue or a new survival gate.",
            "One of each exact pair is counted; deterministic twins are not independent evidence.",
        ],
    )
    save(K.REPORT.with_name(K.REPORT.stem + "_terminal_diagnosis.json"), output)
    for row in rows:
        if row["key"].endswith("/energy_death"):
            print(row, flush=True)


if __name__ == "__main__":
    run()
