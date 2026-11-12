"""Stress test: every app control at both ends of its range, combined extremes and random settings.

For each case it runs the instant estimate and a quick run, checks the numbers for sanity, and renders the Run and
Results pages with that result. Writes a JSON report to scenarios/msp/stress_report.json.

    python -m tools.stress_test                 # everything
    python -m tools.stress_test --only extremes  # one group: extremes, combined, random
    python -m tools.stress_test --limit 5        # first few cases (smoke test)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
import traceback

import numpy as np
import pandas as pd

from engine.config import REPO_ROOT, load_assumptions
from engine import live
from engine.search import objective_for
from engine.simulation import clock, get_design

A0 = load_assumptions()
STEPS = A0["app_controls"]["steps"]
CONTROLS = {c["path"]: c for s in STEPS for c in s["controls"]}


# ---------------------------------------------------------------- settings, built the way the app builds them

def get_path(d, path):
    for k in path.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def set_path(d, path, v):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[int(k)] if isinstance(d, list) else d[k]
    last = keys[-1]
    if isinstance(d, list):
        d[int(last)] = v
    else:
        d[last] = v


def settings(changes: dict) -> dict:
    """Same as the app's settings(): defaults plus changes; a new sort cut-off moves the route start with it."""
    a = copy.deepcopy(A0)
    for p, v in changes.items():
        set_path(a, p, v)
        if p == "schedule.sort_cutoff":
            gap = clock(A0["schedule"]["route_start"]) - clock(A0["schedule"]["sort_cutoff"])
            h = clock(v) + gap
            a["schedule"]["route_start"] = f"D {int(h) % 24:02d}:{int(round(h % 1 * 60)):02d}"
    return a


def ends(c: dict) -> list:
    """The values a visitor can reach at each end of a control."""
    kind = c.get("kind", "slider")
    if kind == "toggle":
        return [not get_path(A0, c["path"])]
    if kind == "clock":
        return [c["options"][0], c["options"][-1]]
    if kind == "range":
        return [[c["min"], c["min"]], [c["max"], c["max"]], [c["min"], c["max"]]]
    return [c["min"], c["max"]]


def random_value(c: dict, rng: random.Random):
    kind = c.get("kind", "slider")
    if kind == "toggle":
        return rng.random() < 0.5
    if kind == "clock":
        return rng.choice(c["options"])
    if kind == "range":
        lo, hi = sorted(rng.uniform(c["min"], c["max"]) for _ in range(2))
        return [round(lo, 3), round(hi, 3)]
    v = rng.uniform(c["min"], c["max"])
    return int(round(v)) if isinstance(c["min"], int) and isinstance(c["max"], int) else round(v, 4)


def cases(only: str | None, n_random: int, seed: int) -> list[tuple[str, dict]]:
    out = []
    if only in (None, "extremes"):
        for p, c in CONTROLS.items():
            for v in ends(c):
                extra = {}
                if c.get("requires") and not get_path(A0, c["requires"]):
                    extra[c["requires"]] = True            # a dependent slider only matters with its switch on
                out.append((f"{p} = {v}", {**extra, p: v}))
    if only in (None, "combined"):
        lo = {p: ends(c)[0] for p, c in CONTROLS.items() if c.get("kind", "slider") != "toggle"}
        hi = {p: ends(c)[-1] if c.get("kind") != "range" else ends(c)[1] for p, c in CONTROLS.items()
              if c.get("kind", "slider") != "toggle"}
        out += [("every control at its minimum", lo), ("every control at its maximum", hi),
                ("hardest: most volume, tightest targets, most disruption",
                 {"demand.network_share": 0.10, "demand.packages_per_household_per_day": 0.70,
                  "scenarios.holiday_peak.volume_multiplier": 2.5, "objective.on_time_typical_target": 0.995,
                  "objective.on_time_target": 0.99, "objective.on_time_week_share": 1.0,
                  "objective.peak_targets.on_time_typical_target": 0.995, "objective.peak_targets.on_time_target": 0.99,
                  "objective.peak_targets.on_time_week_share": 1.0, "uncertainty.truck_late_prob": 0.4,
                  "uncertainty.no_show_rate_range": [0.2, 0.2], "uncertainty.slow_day_prob": 0.3,
                  "uncertainty.extreme_weather_prob": 0.06, "weeks_per_year.holiday_peak": 10,
                  "weeks_per_year.hub_outage": 8}),
                ("easiest: least volume, loosest targets, no disruption",
                 {"demand.network_share": 0.01, "objective.on_time_typical_target": 0.90,
                  "objective.on_time_target": 0.85, "objective.on_time_week_share": 0.5,
                  "uncertainty.truck_late_prob": 0.0, "uncertainty.no_show_rate_range": [0.0, 0.0],
                  "uncertainty.slow_day_prob": 0.0, "uncertainty.extreme_weather_prob": 0.0,
                  "uncertainty.volume_week_sigma": 0.0}),
                ("only normal weeks count", {"weeks_per_year.holiday_peak": 0, "weeks_per_year.hub_outage": 0}),
                ("Dashers paid nothing", {"drivers.pay_base_per_route": 0, "drivers.pay_per_stop": 0.0,
                                          "drivers.pay_per_km": 0.0, "drivers.raise_base_to_floor": False}),
                ("tiny cars, short routes", {"routing.car_cargo_m3": 0.2832, "routing.max_route_hours": 2.0,
                                             "routing.max_packages_per_route": 30}),
                ("no vans or SUVs", {"drivers.vehicle_availability.1.1": 0.0, "drivers.vehicle_availability.2.1": 0.0,
                                     "drivers.vehicle_availability.3.1": 0.0})]
    if only in (None, "random"):
        rng = random.Random(seed)
        for i in range(n_random):
            pick = rng.sample(list(CONTROLS), k=rng.randint(4, 15))
            out.append((f"random #{i + 1}", {p: random_value(CONTROLS[p], rng) for p in pick}))
    return out


# ---------------------------------------------------------------- checks

NUMERIC = ["cost_p50", "cost_p90", "on_time_p50", "on_time_p10", "weeks_meeting_service", "driver_net_p50",
           "km_per_package_p50"]


def check_run(a: dict, r: dict) -> list[str]:
    """Problems with a finished run; an empty list means it passed."""
    bad = []
    if r["weeks_simulated"] <= 0 or r["packages_simulated"] <= 0:
        bad.append("simulated no weeks or no packages")
    for sc, f in r["frames"].items():
        if f.empty:
            bad.append(f"{sc}: no layouts scored")
            continue
        for col in NUMERIC:
            x = f[col].to_numpy(dtype=float)
            if not np.isfinite(x).all():
                bad.append(f"{sc}: {col} has NaN or infinite values")
        for col in ["on_time_p50", "on_time_p10", "weeks_meeting_service"]:
            if ((f[col] < -1e-9) | (f[col] > 1 + 1e-9)).any():
                bad.append(f"{sc}: {col} outside 0–1")
        if (f["cost_p50"] <= 0).any():
            bad.append(f"{sc}: a cost per package is zero or negative")
        if (f["cost_p90"] + 1e-9 < f["cost_p50"]).any():
            bad.append(f"{sc}: bad-week cost below typical-week cost")
        if (f["on_time_p10"] - 1e-9 > f["on_time_p50"]).any():
            bad.append(f"{sc}: bad-week OTD above typical-week OTD")
        obj = objective_for(a, sc)
        expect = (f["weeks_meeting_service"] >= obj["on_time_week_share"]) & (f["on_time_p50"] >= obj["on_time_typical_target"])
        if not a["drivers"].get("raise_base_to_floor"):
            expect &= f["driver_net_p50"] >= obj["driver_net_per_hour_floor"]
        if (expect.astype(bool) != f["feasible"].astype(bool)).any():
            bad.append(f"{sc}: pass/fail flag disagrees with the targets")
    y = r["year"]
    if y.empty:
        bad.append("no layout scored in every kind of week")
        return bad
    if not np.isfinite(y["year_cost_per_package"].to_numpy(dtype=float)).all():
        bad.append("year cost has NaN or infinite values")
    ok = y[y["meets_all"]]
    if len(ok):
        if not bool(y["meets_all"].iloc[0]):
            bad.append("a layout that misses a target ranks above one that meets every target")
        tie = a["objective"].get("year_round_cost_tie", 0.0)
        if y["year_cost_per_package"].iloc[0] > ok["year_cost_per_package"].min() * (1 + tie) + 1e-9:
            bad.append("the pick is not the cheapest layout that meets every target")
        if r.get("diagnosis"):
            bad.append("a diagnosis was given although a layout meets every target")
    else:
        if r.get("tiers_tried") != r.get("tiers_available"):
            bad.append("stopped before trying every group of larger networks although nothing passed")
        if not r.get("diagnosis"):
            bad.append("nothing meets the targets but no diagnosis was given")
        if y["otd_gap"].iloc[0] > y["otd_gap"].min() + 1e-12:
            bad.append("the closest layout is not the one with the smallest OTD gap")
    for d in y.index:
        get_design(d, a)
    return bad


# ---------------------------------------------------------------- app pages

def render_pages(a: dict, changes: dict, r: dict) -> list[str]:
    """Open the Run page with these settings (instant estimate, run times) and the Results page with this run."""
    from streamlit.testing.v1 import AppTest
    bad = []
    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=600).run()
    if at.exception:
        return [f"app start: {at.exception[0].message}"]
    vals = at.session_state["vals"]
    for p, v in changes.items():
        vals[p] = v
    at.session_state["vals"] = vals
    names = 3 + len(STEPS)                      # Overview, Core assumptions, Methodology, steps, then Run, Results
    for page in range(names):                   # every step page shows the changed values and their notes
        at.session_state["page"] = page
        at.run()
        if at.exception:
            bad.append(f"page {page}: {at.exception[0].message}")
    at.session_state["page"] = names            # Run page
    at.run()
    if at.exception:
        bad.append(f"Run page: {at.exception[0].message}")
    at.session_state["result"] = {"run": r, "base": live.load_baseline("msp", "quick", A0) or r, "a": a,
                                  "changes": changes, "mode": "quick"}
    at.session_state["page"] = names + 1        # Results page
    at.run()
    if at.exception:
        bad.append(f"Results page: {at.exception[0].message}")
    return bad


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["extremes", "combined", "random"])
    p.add_argument("--random", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int)
    p.add_argument("--no-pages", action="store_true")
    args = p.parse_args()
    todo = cases(args.only, args.random, args.seed)[: args.limit]
    report, t_all = [], time.perf_counter()
    for i, (name, changes) in enumerate(todo, 1):
        t0 = time.perf_counter()
        row = {"case": name, "changes": changes}
        try:
            a = settings(changes)
            si = live.screen_all(a)
            est = live.quick_estimate(a, si["screens"])
            r = live.run(a, "quick", screen_info=si)
            y = r["year"]
            row.update(pick=str(y.index[0]), hubs=str(y.iloc[0]["hubs"]), meets_all=bool(y["meets_all"].iloc[0]),
                       year_cost=round(float(y.iloc[0]["year_cost_per_package"]), 3),
                       tiers=f"{r['tiers_tried']}/{r['tiers_available']}", estimate=est["design"],
                       seconds=r["seconds"], predicted=round(live.estimate_seconds(si, "quick", a["app_controls"]["run"]), 1),
                       diagnosis=(r["diagnosis"] or {}).get("cause"), problems=check_run(a, r))
            if not args.no_pages:
                row["problems"] += render_pages(a, changes, r)
        except Exception as err:                     # a crash is a failure, recorded with its traceback
            row["problems"] = [f"crash: {err!r}"]
            row["traceback"] = traceback.format_exc()
        row["wall_seconds"] = round(time.perf_counter() - t0, 1)
        report.append(row)
        status = "PASS" if not row["problems"] else "FAIL"
        print(f"[{i}/{len(todo)}] {status} {name}: {row.get('hubs', '')} ${row.get('year_cost', '')} "
              f"meets={row.get('meets_all')} tiers={row.get('tiers')} {row.get('seconds')}s"
              + ("" if not row["problems"] else f"  -> {row['problems']}"), flush=True)
    out = REPO_ROOT / "scenarios" / "msp" / "stress_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    fails = [r for r in report if r["problems"]]
    print(f"\n{len(report) - len(fails)} of {len(report)} cases passed in {time.perf_counter() - t_all:.0f}s. "
          f"Report: {out}")


if __name__ == "__main__":
    main()
