"""Design search under uncertainty for one market.

For each scenario:
  1. Screen: one average day per design; drop clearly bad designs.
  2. Score: every remaining design on the same random weeks (common random numbers).
  3. Pick: lowest typical-week cost per package among designs that meet service in enough
     weeks and pay drivers above the floor (assumptions.yaml: objective).
  4. Final check: rerun the top few on fresh random weeks; the winner there is the recommendation.
Results are small files in scenarios/<market>/<scenario>/ for the app.

Usage:
  python -m engine.search --market msp --scenario all
  python -m engine.search --market msp --scenario holiday_peak --weeks 50   # quicker run
  python -m engine.search --confirm "3x small" "4x small" --weeks 1000      # recheck close calls
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from engine.config import REPO_ROOT, load_assumptions
from engine.randomness import average_day, draw_weeks
from engine.simulation import busiest_hub, design_grid, get_design, load_market, simulate_day, simulate_week

_M = None
_A = None
_WEEKS: dict = {}


def _init(market: str) -> None:
    global _M, _A
    _M, _A = load_market(market), load_assumptions()


def _weeks(stream: str, n: int):
    if (stream, n) not in _WEEKS:
        _WEEKS[(stream, n)] = draw_weeks(n, len(_M.site_rank), len(_A["schedule"]["linehaul_waves"]), _A, stream)
    return _WEEKS[(stream, n)]


def _score_design(job: tuple) -> list[dict]:
    name, scenario, stream, n = job
    d, sc = get_design(name, _A), _A["scenarios"][scenario]
    return [{**simulate_week(_M, d, w, _A, sc), "week": i, "hubs": d.label(), "n_hubs": len(d.hubs)}
            for i, w in enumerate(_weeks(stream, n))]


def screen(m, a, scenario: str) -> pd.DataFrame:
    """Fast average-day estimate for every design on the scenario's hardest day."""
    sc = a["scenarios"][scenario]
    day = sc.get("outage_day", "Mon")
    avg = average_day(len(m.site_rank), len(a["schedule"]["linehaul_waves"]), a)
    rows = []
    for d in design_grid(a):
        outage = busiest_hub(m, d) if sc.get("outage_day") else None
        r = simulate_day(m, d, day, a, avg, sc, outage_hub=outage)["metrics"]
        rows.append({"design": d.name, "screen_on_time": r["on_time_rate"],
                     "screen_cost": r["cost_total"] / max(r["delivered"], 1)})
    s = pd.DataFrame(rows)
    obj = a["objective"]
    ok_service = s["screen_on_time"] >= obj["screen_min_on_time"]
    best_cost = s.loc[ok_service, "screen_cost"].min() if ok_service.any() else s["screen_cost"].min()
    s["kept"] = ok_service & (s["screen_cost"] <= best_cost * obj["screen_max_cost_vs_best"])
    if not s["kept"].any():   # nothing passes: keep the best few by on-time so the search still runs
        s.loc[s["screen_on_time"].nlargest(5).index, "kept"] = True
    return s


def objective_for(a: dict, scenario: str | None) -> dict:
    """The targets for a kind of week: holiday peak weeks use objective.peak_targets in place of the normal ones."""
    obj = dict(a["objective"])
    if scenario and a["scenarios"].get(scenario, {}).get("peak"):
        obj.update(obj.get("peak_targets", {}))
    return obj


def summarize(weeks: pd.DataFrame, a: dict, scenario: str | None = None) -> pd.DataFrame:
    """Typical (P50) and bad-week (P90, or P10 where lower is worse) values per design, ranked."""
    obj = objective_for(a, scenario)
    if "extreme_weather_days" not in weeks:
        weeks = weeks.assign(extreme_weather_days=0)
    storm = weeks["extreme_weather_days"] > 0
    counted = weeks[~storm] if obj.get("exclude_extreme_weather_weeks", False) else weeks
    g = weeks.groupby("design")
    gs = weeks[storm].groupby("design")
    s = pd.DataFrame({
        "hubs": g["hubs"].first(),
        "n_hubs": g["n_hubs"].first(),
        "cost_p50": g["cost_per_package"].quantile(0.5),
        "cost_p90": g["cost_per_package"].quantile(0.9),
        "on_time_p50": g["on_time_rate"].quantile(0.5),
        "on_time_p10": g["on_time_rate"].quantile(0.1),
        # floor: share of counted weeks (storm weeks excluded when set) at or above the target
        "weeks_meeting_service": counted.groupby("design")["on_time_rate"].apply(
            lambda x: (x >= obj["on_time_target"]).mean()),
        "weeks_counted": counted.groupby("design").size(),
        # storm weeks, reported on their own
        "storm_week_share": storm.groupby(weeks["design"]).mean(),
        "storm_on_time_p50": gs["on_time_rate"].quantile(0.5),
        "storm_on_time_p10": gs["on_time_rate"].quantile(0.1),
        "storm_unclaimed_p50": gs["routes_unclaimed_share"].quantile(0.5),
        "storm_weeks_meeting_service": gs["on_time_rate"].apply(lambda x: (x >= obj["on_time_target"]).mean()),
        "driver_net_p50": g["driver_net_per_hour"].quantile(0.5),
        "driver_net_p10": g["driver_net_per_hour"].quantile(0.1),
        "unclaimed_p50": g["routes_unclaimed_share"].quantile(0.5),
        "unclaimed_p90": g["routes_unclaimed_share"].quantile(0.9),
        "peak_sort_utilization_p90": g["max_sort_utilization"].quantile(0.9),
        "km_per_package_p50": g["km_per_package"].quantile(0.5),
        "packages_per_stop_p50": g["packages_per_stop"].quantile(0.5),
        "surge_cost_p50": g["cost_driver_surge"].quantile(0.5),
        "floor_topup_p50": (g["cost_driver_floor_topup"].quantile(0.5) if "cost_driver_floor_topup" in weeks
                            else g["cost_driver_surge"].quantile(0.5) * 0),
        "surge_cost_p90": g["cost_driver_surge"].quantile(0.9),
        "overtime_cost_p90": g["cost_sort_overtime"].quantile(0.9),
        "routes_surged_p50": g["routes_surged_share"].quantile(0.5),
        "pickup_slot_delay_p50": g["pickup_slot_delay_minutes"].quantile(0.5),
        "weeks": g.size(),
    })
    s["meets_service"] = ((s["weeks_meeting_service"] >= obj["on_time_week_share"])
                          & (s["on_time_p50"] >= obj["on_time_typical_target"]))
    # With raise_base_to_floor on, every route is paid at least the floor, so the floor is a pay policy whose cost
    # is already in each layout's cost per package; it only acts as a pass/fail target when the raise is off.
    pay_policy = a.get("drivers", {}).get("raise_base_to_floor", False)
    s["meets_driver_floor"] = True if pay_policy else s["driver_net_p50"] >= obj["driver_net_per_hour_floor"]
    s["feasible"] = s["meets_service"] & s["meets_driver_floor"]
    # Feasible designs by typical cost; then the rest by how often they meet service, then cost.
    # Designs that miss a target: those within the tolerance of the best service count as tied on
    # service and go cheapest first; the rest follow by service, then cost.
    rest = s[~s["feasible"]].copy()
    rest["near_best_service"] = rest["weeks_meeting_service"] >= (
        rest["weeks_meeting_service"].max() - obj["infeasible_service_tolerance"] - 1e-9 if len(rest) else 0)
    rest = rest.sort_values(["near_best_service", "weeks_meeting_service", "cost_p50"], ascending=[False, False, True])
    near = rest[rest["near_best_service"]].sort_values("cost_p50")
    rest = pd.concat([near, rest[~rest["near_best_service"]]]).drop(columns="near_best_service")
    s = pd.concat([s[s["feasible"]].sort_values("cost_p50"), rest])
    s.insert(0, "rank", range(1, len(s) + 1))
    return s


def run_scenario(market: str, scenario: str, n_weeks: int | None, workers: int, pool: ProcessPoolExecutor) -> dict:
    a = load_assumptions()
    u = a["uncertainty"]
    n_search = n_weeks or u["weeks_search"]
    n_final = n_weeks or u["weeks_final"]
    t0 = time.perf_counter()

    m = load_market(market)
    scr = screen(m, a, scenario)
    kept = scr.loc[scr["kept"], "design"].tolist()
    t_screen = time.perf_counter() - t0

    jobs = [(d, scenario, "search", n_search) for d in kept]
    weeks = pd.DataFrame([row for rows in pool.map(_score_design, jobs) for row in rows])
    search = summarize(weeks, a, scenario)
    t_search = time.perf_counter() - t0

    top = search.index[: u["final_top_n"]].tolist()
    jobs = [(d, scenario, "final", n_final) for d in top]
    fweeks = pd.DataFrame([row for rows in pool.map(_score_design, jobs) for row in rows])
    final = summarize(fweeks, a, scenario)
    t_total = time.perf_counter() - t0

    out = REPO_ROOT / "scenarios" / market / scenario
    out.mkdir(parents=True, exist_ok=True)
    scr.to_csv(out / "screen.csv", index=False)
    weeks.to_parquet(out / "search_weeks.parquet", index=False)
    search.to_csv(out / "search_summary.csv")
    fweeks.to_parquet(out / "final_weeks.parquet", index=False)
    final.to_csv(out / "final_summary.csv")
    winner = final.index[0]
    result = {
        "market": market, "scenario": scenario, "description": a["scenarios"][scenario]["description"],
        "recommended": winner, "recommended_hubs": final.loc[winner, "hubs"],
        "recommended_feasible": bool(final.loc[winner, "feasible"]),
        "runners_up": final.index[1:].tolist(),
        "designs_screened": len(scr), "designs_scored": len(kept),
        "weeks_search": n_search, "weeks_final": n_final,
        "seconds": {"screen": round(t_screen, 1), "search": round(t_search - t_screen, 1),
                    "final": round(t_total - t_search, 1), "total": round(t_total, 1)},
        "workers": workers, "objective": a["objective"],
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return {"result": result, "screen": scr, "search": search, "final": final}


def year_round_table(frames: dict[str, pd.DataFrame], a: dict) -> pd.DataFrame:
    """One network for all scenarios: designs scored in every scenario, ranked by how many scenarios
    they meet the targets in, then by typical cost weighted by weeks per year. Near-ties on cost at the
    top go to the network that holds service best in its worst scenario."""
    weights = {k: v for k, v in a["weeks_per_year"].items() if k in frames}
    if not frames or not weights:
        return pd.DataFrame()
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    rows = []
    for d in common:
        row = {"design": d, "hubs": next(iter(frames.values())).loc[d, "hubs"]}
        for name, f in frames.items():
            row[f"{name}_cost_p50"] = f.loc[d, "cost_p50"]
            row[f"{name}_feasible"] = bool(f.loc[d, "feasible"])
            row[f"{name}_weeks_meeting_service"] = f.loc[d, "weeks_meeting_service"]
            row[f"{name}_storm_on_time_p50"] = f.loc[d, "storm_on_time_p50"]
        row["scenarios_meeting_targets"] = sum(row[f"{n}_feasible"] for n in frames)
        row["worst_weeks_meeting_service"] = min(row[f"{n}_weeks_meeting_service"] for n in frames)
        total = sum(weights.values()) or 1
        row["year_cost_per_package"] = sum(weights.get(n, 0) * row[f"{n}_cost_p50"] for n in frames) / total
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows).set_index("design").sort_values(
        ["scenarios_meeting_targets", "year_cost_per_package"], ascending=[False, True])
    top = t["scenarios_meeting_targets"] == t["scenarios_meeting_targets"].max()
    tie = a["objective"].get("year_round_cost_tie", 0.0)
    near = top & (t["year_cost_per_package"] <= t.loc[top, "year_cost_per_package"].min() * (1 + tie) + 1e-12)
    t = pd.concat([t[near].sort_values(["worst_weeks_meeting_service", "year_cost_per_package"],
                                       ascending=[False, True]), t[~near]])
    t.insert(0, "rank", range(1, len(t) + 1))
    return t


def year_round_choice(market: str, a: dict) -> pd.DataFrame:
    """Year-round pick from the saved search results (rechecked designs use their 1,000-week numbers)."""
    frames = {}
    for name in a["weeks_per_year"]:
        f = REPO_ROOT / "scenarios" / market / name / "search_summary.csv"
        c = REPO_ROOT / "scenarios" / market / name / "confirm_summary.csv"
        if f.exists():
            frames[name] = pd.read_csv(f, index_col="design")
            if c.exists():
                frames[name] = pd.read_csv(c, index_col="design").combine_first(frames[name])[frames[name].columns]
    if len(frames) < len(a["weeks_per_year"]):
        return pd.DataFrame()
    t = year_round_table(frames, a)
    t.to_csv(REPO_ROOT / "scenarios" / market / "year_round.csv")
    return t


def confirm(market: str, scenario: str, designs: list[str], n_weeks: int, pool: ProcessPoolExecutor) -> pd.DataFrame:
    """Recheck specific designs on many fresh weeks; adds a 90% confidence range on the share of
    weeks meeting the floor, so close pass/fail calls can be told apart from noise."""
    a = load_assumptions()
    jobs = [(d, scenario, "final", n_weeks) for d in designs]
    weeks = pd.DataFrame([row for rows in pool.map(_score_design, jobs) for row in rows])
    s = summarize(weeks, a, scenario)
    se = np.sqrt(s["weeks_meeting_service"] * (1 - s["weeks_meeting_service"]) / s["weeks_counted"])
    s["weeks_meeting_low90"] = s["weeks_meeting_service"] - 1.645 * se
    s["weeks_meeting_high90"] = s["weeks_meeting_service"] + 1.645 * se
    out = REPO_ROOT / "scenarios" / market / scenario
    s.to_csv(out / "confirm_summary.csv")
    return s


def print_scenario(r: dict) -> None:
    res, final, search = r["result"], r["final"], r["search"]
    print(f"\n{'=' * 100}\n{res['scenario']}: {res['description']}")
    print(f"Screened {res['designs_screened']} designs, scored {res['designs_scored']} on {res['weeks_search']} weeks, "
          f"final check of top {len(final)} on {res['weeks_final']} fresh weeks · {res['seconds']['total']:.0f}s")
    cols = {"hubs": "hubs", "cost_p50": "$/pkg typical", "cost_p90": "$/pkg bad wk",
            "on_time_p50": "on-time typ", "on_time_p10": "on-time bad wk", "weeks_meeting_service": "wks ≥ floor",
            "storm_on_time_p50": "storm wk on-time", "driver_net_p50": "driver net $/h", "unclaimed_p90": "unclaimed bad wk", "feasible": "meets targets"}
    fmt = {"$/pkg typical": "${:.2f}".format, "$/pkg bad wk": "${:.2f}".format, "on-time typ": "{:.1%}".format,
           "on-time bad wk": "{:.1%}".format, "wks ≥ floor": "{:.1%}".format, "storm wk on-time": "{:.1%}".format, "driver net $/h": "${:.2f}".format,
           "unclaimed bad wk": "{:.1%}".format}
    print("\nSearch (all scored designs, ranked):")
    print(search[list(cols)].rename(columns=cols).to_string(formatters=fmt))
    print("\nFinal check on fresh weeks:")
    print(final[list(cols)].rename(columns=cols).to_string(formatters=fmt))
    status = "meets all targets" if res["recommended_feasible"] else "BEST AVAILABLE, does not meet all targets"
    print(f"\nRecommended: {res['recommended']} ({res['recommended_hubs']}) — {status}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", default="msp")
    p.add_argument("--scenario", default="all")
    p.add_argument("--weeks", type=int, default=None, help="override weeks for search and final (quick runs)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--confirm", nargs="+", metavar="DESIGN", help="recheck these designs on fresh weeks")
    args = p.parse_args()

    a = load_assumptions()
    names = list(a["scenarios"]) if args.scenario == "all" else [args.scenario]
    t0 = time.perf_counter()
    if args.confirm:
        n = args.weeks or 1000
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.market,)) as pool:
            for name in names:
                s = confirm(args.market, name, args.confirm, n, pool)
                obj = objective_for(a, name)
                print(f"\n{name}: {n} fresh weeks · floor {obj['on_time_target']:.0%} in "
                      f"≥{obj['on_time_week_share']:.0%} of weeks, typical ≥{obj['on_time_typical_target']:.0%}")
                for d, r in s.iterrows():
                    print(f"  {d:18s} ${r.cost_p50:.2f} (${r.cost_p90:.2f})  on time {r.on_time_p50:.1%} "
                          f"(bad wk {r.on_time_p10:.1%})  weeks ≥ floor {r.weeks_meeting_service:.1%} "
                          f"[90% range {r.weeks_meeting_low90:.1%}–{r.weeks_meeting_high90:.1%}]  "
                          f"storm wks {r.storm_week_share:.0%} at {r.storm_on_time_p50:.1%}  "
                          f"{'MEETS' if r.feasible else 'misses'}")
        print(f"\n{time.perf_counter() - t0:.0f}s")
        return
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.market,)) as pool:
        for name in names:
            print_scenario(run_scenario(args.market, name, args.weeks, args.workers, pool))
    yr = year_round_choice(args.market, a)
    if len(yr):
        n = len(a["weeks_per_year"])
        print(f"\nOne network for the whole year (meets targets in all {n} scenarios, then lowest weighted cost):")
        cols = ["hubs", "scenarios_meeting_targets", "year_cost_per_package"] + [f"{k}_cost_p50" for k in a["weeks_per_year"]]
        print(yr[cols].head(6).to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nAll scenarios: {time.perf_counter() - t0:.0f}s with {args.workers} workers")


if __name__ == "__main__":
    main()
