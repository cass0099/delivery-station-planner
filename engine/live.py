"""Live runs for the app: any set of assumptions, one market.

Two modes:
- quick: screen every layout on an average day, then simulate the cheapest few from each kind of week on a
  short run of weeks. Seconds. Good for seeing roughly how the answer moves.
- full:  screen every layout, simulate every layout that passes the screen on the search weeks, then recheck
  the closest contenders on many fresh weeks in every kind of week. Minutes. The same method as the
  published results, precise enough to separate close calls.

Every layout sees the same random weeks (common random numbers). Work is split into chunks of weeks so
progress can be reported as it happens.

Writes nothing unless run from the command line, which saves the default-settings runs for the app:
    python -m engine.live            # quick and full
    python -m engine.live --mode quick
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from typing import Callable

import pandas as pd

from engine.config import REPO_ROOT, load_assumptions
from engine.randomness import draw_weeks
from engine.search import screen, summarize, year_round_table
from engine.simulation import get_design, load_market, simulate_week

_M = _A = None
# A simulated week costs about this many screened design-days on the same machine. Measured on default settings:
# quick runs 4.8–5.6 on 1–2 cores; full runs about 10.5 (they include the slow, overloaded layouts the quick run skips).
WEEK_PER_SCREEN_DAY = {"quick": 6.0, "full": 11.0}
STARTUP_SECONDS = 3.0
CHUNK_WEEKS = 20              # weeks per unit of work: small enough for smooth progress updates

Progress = Callable[[float, str, dict], None]


def settings_hash(a: dict) -> str:
    """Fingerprint of the model inputs. App wording and slider ranges (app_controls) are left out."""
    model = {k: v for k, v in a.items() if k != "app_controls"}
    return hashlib.sha1(json.dumps(model, sort_keys=True, default=str).encode()).hexdigest()[:16]


def available_cpus() -> int:
    """CPUs this process may actually use (container limits included), not the host's total."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


# ---------------------------------------------------------------- workers

def _init(market: str, a: dict) -> None:
    global _M, _A
    _M, _A = load_market(market), a


def _job(job: tuple) -> tuple:
    """One layout, one kind of week, one slice of weeks from one random stream."""
    name, scenario, stream, start, n = job
    d, sc = get_design(name, _A), _A["scenarios"][scenario]
    weeks = draw_weeks(n, len(_M.site_rank), len(_A["schedule"]["linehaul_waves"]), _A, stream, start=start)
    rows = [{**simulate_week(_M, d, w, _A, sc), "week": start + i, "hubs": d.label(), "n_hubs": len(d.hubs)}
            for i, w in enumerate(weeks)]
    return name, scenario, stream, rows


def _chunks(designs: list[str], scenarios: list[str], stream: str, n_weeks: int) -> list[tuple]:
    return [(d, sc, stream, s, min(CHUNK_WEEKS, n_weeks - s))
            for sc in scenarios for d in designs for s in range(0, n_weeks, CHUNK_WEEKS)]


def _simulate(pool, jobs: list[tuple], done: dict, total: int, label: str, progress: Progress | None) -> dict:
    """Run jobs; returns {(stream, scenario): [rows]}. `done` carries counters across phases."""
    out: dict = {}
    futures = [pool.submit(_job, j) for j in jobs]
    for f in as_completed(futures):
        name, sc, stream, rows = f.result()
        out.setdefault((stream, sc), []).extend(rows)
        done["weeks"] += len(rows)
        done["packages"] += sum(r["packages_due"] for r in rows)
        if progress:
            progress(done["weeks"] / total, label, dict(done, total=total))
    return out


# ---------------------------------------------------------------- screen, estimates

def scenarios_in_play(a: dict) -> list[str]:
    return [s for s in a["scenarios"] if a["weeks_per_year"].get(s, 0) > 0] or ["normal"]


def screen_all(a: dict, market: str = "msp") -> dict:
    """Average-day screen of every design in every scenario in play. Takes well under a second per
    scenario; its duration also calibrates the run-time estimate on whatever machine runs it."""
    m = load_market(market)
    t0 = time.perf_counter()
    screens = {sc: screen(m, a, sc) for sc in scenarios_in_play(a)}
    return {"screens": screens, "seconds": time.perf_counter() - t0}


def candidates(screens: dict, top_n: int) -> list[str]:
    keep = []
    for s in screens.values():
        for d in s[s["kept"]].nsmallest(top_n, "screen_cost")["design"]:
            if d not in keep:
                keep.append(d)
    return keep


def quick_estimate(a: dict, screens: dict) -> dict:
    """Instant answer from the average-day screen: no randomness, so a rough guide only. Cheapest design
    (weighted by weeks per year) among those that pass the screen in every kind of week."""
    weights = {sc: a["weeks_per_year"].get(sc, 0) for sc in screens}
    total = sum(weights.values()) or 1
    names = next(iter(screens.values()))["design"]
    rows = []
    for d in names:
        r = {sc: s.set_index("design").loc[d] for sc, s in screens.items()}
        rows.append({"design": d, "passes": sum(bool(x["kept"]) for x in r.values()),
                     "year_cost": sum(weights[sc] * x["screen_cost"] for sc, x in r.items()) / total})
    t = pd.DataFrame(rows).sort_values(["passes", "year_cost"], ascending=[False, True])
    best = t.iloc[0]
    return {"design": best["design"], "year_cost": float(best["year_cost"]),
            "passes_all": int(best["passes"]) == len(screens)}


def planned_weeks(screen_info: dict, mode: str, ctl: dict) -> int:
    """Layout-weeks a run will simulate (the full run's recheck size is estimated from the screen)."""
    screens = screen_info["screens"]
    if mode == "quick":
        q = ctl["quick"]
        return len(candidates(screens, q["top_designs_per_scenario"])) * len(screens) * q["weeks"]
    f = ctl["full"]
    search = sum(int(s["kept"].sum()) for s in screens.values()) * f["search_weeks"]
    recheck = len(candidates(screens, f["recheck_top_designs"])) * len(screens) * f["recheck_weeks"]
    return search + recheck


def estimate_seconds(screen_info: dict, mode: str, ctl: dict, workers: int | None = None) -> float:
    """Expected run time. A simulated week costs about WEEK_PER_SCREEN_DAY x one screened design-day on the
    same machine; the screen just measured that."""
    screens = screen_info["screens"]
    per_day = screen_info["seconds"] / max(sum(len(s) for s in screens.values()), 1)
    workers = workers or available_cpus()
    return planned_weeks(screen_info, mode, ctl) * WEEK_PER_SCREEN_DAY[mode] * per_day / workers + STARTUP_SECONDS


# ---------------------------------------------------------------- runs

def run(a: dict, mode: str = "quick", market: str = "msp", workers: int | None = None,
        progress: Progress | None = None, screen_info: dict | None = None) -> dict:
    t0 = time.perf_counter()
    ctl = a["app_controls"]["run"]
    screen_info = screen_info or screen_all(a, market)
    screens = screen_info["screens"]
    scenarios = list(screens)
    workers = workers or available_cpus()
    done = {"weeks": 0, "packages": 0}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(market, a)) as pool:
        if mode == "quick":
            q = ctl["quick"]
            designs = candidates(screens, q["top_designs_per_scenario"])
            jobs = _chunks(designs, scenarios, "final", q["weeks"])
            total = len(designs) * len(scenarios) * q["weeks"]
            rows = _simulate(pool, jobs, done, total, f"Simulating {len(designs)} layouts", progress)
            frames = {sc: summarize(pd.DataFrame(rows[("final", sc)]), a, sc) for sc in scenarios}
            weeks_note = {"weeks": q["weeks"]}
        else:
            f = ctl["full"]
            kept = {sc: s.loc[s["kept"], "design"].tolist() for sc, s in screens.items()}
            search_jobs = [j for sc in scenarios for j in _chunks(kept[sc], [sc], "search", f["search_weeks"])]
            n_search = sum(len(v) for v in kept.values()) * f["search_weeks"]
            guess = len(candidates(screens, f["recheck_top_designs"])) * len(scenarios) * f["recheck_weeks"]
            rows = _simulate(pool, search_jobs, done, n_search + guess,
                             f"Searching: every layout that passed the screen, {f['search_weeks']} weeks each",
                             progress)
            search = {sc: summarize(pd.DataFrame(rows[("search", sc)]), a, sc) for sc in scenarios}
            designs = []
            for sc in scenarios:
                for d in search[sc].index[: f["recheck_top_designs"]]:
                    if d not in designs:
                        designs.append(d)
            n_recheck = len(designs) * len(scenarios) * f["recheck_weeks"]
            rows = _simulate(pool, _chunks(designs, scenarios, "final", f["recheck_weeks"]), done,
                             n_search + n_recheck,
                             f"Rechecking the {len(designs)} closest layouts on {f['recheck_weeks']:,} fresh weeks",
                             progress)
            recheck = {sc: summarize(pd.DataFrame(rows[("final", sc)]), a, sc) for sc in scenarios}
            frames = {}
            for sc in scenarios:   # rechecked layouts use their fresh-week numbers; the rest keep search numbers
                fr = recheck[sc].combine_first(search[sc])[recheck[sc].columns]
                fr["feasible"] = fr["feasible"].astype(bool)
                frames[sc] = pd.concat([fr[fr["feasible"]].sort_values("cost_p50"),
                                        fr[~fr["feasible"]].sort_values(["weeks_meeting_service", "cost_p50"],
                                                                        ascending=[False, True])])
            weeks_note = {"search_weeks": f["search_weeks"], "recheck_weeks": f["recheck_weeks"],
                          "rechecked": designs}
    return {"mode": mode, "frames": frames, "year": year_round_table(frames, a), "screens": screens,
            "designs": designs, "scenarios": scenarios, "seconds": round(time.perf_counter() - t0, 1),
            "hash": settings_hash(a), "estimate": quick_estimate(a, screens), "weeks_simulated": done["weeks"],
            "packages_simulated": done["packages"], **weeks_note}


# ---------------------------------------------------------------- saved default runs

def _baseline_path(market: str):
    return REPO_ROOT / "scenarios" / market / "live_baseline.json"


def _to_json(r: dict) -> dict:
    out = {k: v for k, v in r.items() if k not in ("frames", "year", "screens")}
    out["frames"] = {k: v.to_json(orient="split") for k, v in r["frames"].items()}
    out["year"] = r["year"].to_json(orient="split")
    out["screens"] = {k: v.to_json(orient="split") for k, v in r["screens"].items()}
    return out


def save_baselines(runs: list[dict], market: str) -> None:
    p = _baseline_path(market)
    saved = json.loads(p.read_text()) if p.exists() else {}
    saved = {k: v for k, v in saved.items() if k in ("quick", "full")}
    saved.update({r["mode"]: _to_json(r) for r in runs})
    p.write_text(json.dumps(saved))


def load_baseline(market: str, mode: str, a: dict) -> dict | None:
    p = _baseline_path(market)
    if not p.exists():
        return None
    r = json.loads(p.read_text()).get(mode)
    if not r or r.get("hash") != settings_hash(a):
        return None

    def read(s):
        df = pd.read_json(StringIO(s), orient="split")
        if not isinstance(df.index, pd.RangeIndex):
            df.index.name = "design"
        return df
    return {**r, "frames": {k: read(v) for k, v in r["frames"].items()}, "year": read(r["year"]),
            "screens": {k: read(v) for k, v in r["screens"].items()}}


def share_sensitivity(a: dict, shares: list[float], market: str = "msp", progress: Progress | None = None) -> list:
    """Full run at several network shares; one summary row per share (recommended network, cost, packages a day)."""
    from engine.simulation import load_market
    m = load_market(market)
    households = float(m.tracts["households"].sum())
    out = []
    for share in shares:
        b = copy.deepcopy(a)
        b["demand"]["network_share"] = share
        r = (load_baseline(market, "full", b) if share == a["demand"]["network_share"] else None) or \
            run(b, "full", market, progress=progress)
        y = r["year"]
        ok = y[y["scenarios_meeting_targets"] == len(r["scenarios"])]
        pick = (ok if len(ok) else y).iloc[0]
        out.append({"share": share, "packages_per_day": households * b["demand"]["packages_per_household_per_day"] * share,
                    "design": str((ok if len(ok) else y).index[0]), "hubs": pick["hubs"],
                    "year_cost": float(pick["year_cost_per_package"]), "meets_all": bool(len(ok)),
                    "runner_up": str(y.index[1]) if len(y) > 1 else None,
                    "runner_up_hubs": y.iloc[1]["hubs"] if len(y) > 1 else None,
                    "runner_up_cost": float(y.iloc[1]["year_cost_per_package"]) if len(y) > 1 else None,
                    "hash": settings_hash(a)})
    return out


def default_settings() -> dict:
    return copy.deepcopy(load_assumptions())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", default="msp")
    p.add_argument("--mode", choices=["quick", "full", "both"], default="both")
    p.add_argument("--sensitivity", nargs="*", type=float, metavar="SHARE",
                   help="also run the full model at these network shares (e.g. 0.02 0.03 0.05)")
    args = p.parse_args()
    a = default_settings()
    runs = []
    for mode in (["quick", "full"] if args.mode == "both" else [args.mode]):
        last = [0.0]

        def show(frac, text, stats):
            if frac - last[0] >= 0.1 or frac >= 1:
                last[0] = frac
                print(f"  {frac:5.0%}  {text}  ({stats['weeks']:,} weeks, {stats['packages']:,} packages)", flush=True)
        r = run(a, mode, args.market, progress=show)
        y = r["year"]
        print(f"{mode}: {r['seconds']}s · {r['weeks_simulated']:,} weeks · {r['packages_simulated']:,} packages · "
              f"pick {y.index[0]} ${y.iloc[0]['year_cost_per_package']:.3f} · "
              f"meets {int(y.iloc[0]['scenarios_meeting_targets'])}/{len(r['scenarios'])}")
        runs.append(r)
    save_baselines(runs, args.market)
    if args.sensitivity:
        rows = share_sensitivity(a, args.sensitivity, args.market)
        (REPO_ROOT / "scenarios" / args.market / "share_sensitivity.json").write_text(json.dumps(rows, indent=2))
        for r in rows:
            print(f"share {r['share']:.0%}: {r['packages_per_day']:,.0f}/day · pick {r['design']} ${r['year_cost']:.3f}")


if __name__ == "__main__":
    main()
