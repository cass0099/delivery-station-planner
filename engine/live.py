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
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from io import StringIO
from typing import Callable

import numpy as np
import pandas as pd

from engine.config import REPO_ROOT, load_assumptions
from engine.randomness import draw_weeks
from engine.search import diagnose, screen, summarize, year_round_table
from engine.simulation import design_tiers, get_design, load_market, simulate_week

_M = _A = None
# A simulated week costs about this many screened design-days on the same machine. Measured on default settings:
# quick runs 4.8–5.6 on 1–2 cores; full runs about 10.5 (they include the slow, overloaded layouts the quick run skips).
WEEK_PER_SCREEN_DAY = {"quick": 6.0, "full": 11.0}
STARTUP_SECONDS = 3.0
# Real runs take longer than (planned weeks x one timed week / workers): many layouts run at once on shared cores,
# and the layouts simulated are on average heavier than the one timed. Measured on default and extreme settings:
# quick runs 1.2-1.8x, full runs about 2.4x.
WEEK_OVERHEAD = {"quick": 1.5, "full": 2.5}
CHUNK_WEEKS = 20              # weeks per unit of work: small enough for smooth progress updates

Progress = Callable[[float, str, dict], None]


class RunStopped(Exception):
    """Raised inside a run when its stop check says so (the visitor pressed Stop, or the time limit passed)."""


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
    return max(1, min(n, workers_memory_allows()))


WORKER_MB = 340          # one simulation worker: its own copy of the market (~210 MB) plus Streamlit imports (forkserver)
APP_RESERVE_MB = 500     # the Streamlit app itself and its cached data


def available_memory_mb() -> float | None:
    """Memory this process may still use: the container's limit if there is one, else what the system has free."""
    try:
        limit = open("/sys/fs/cgroup/memory.max").read().strip()
        if limit != "max":
            used = int(open("/sys/fs/cgroup/memory.current").read().strip())
            return (int(limit) - used) / 2**20
    except (OSError, ValueError):
        pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


def workers_memory_allows() -> int:
    """How many simulation workers fit in memory, so a small host (e.g. Streamlit Community Cloud) never runs out."""
    mb = available_memory_mb()
    if mb is None:
        return 64
    return max(1, int((mb - APP_RESERVE_MB) // WORKER_MB))


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
    """Run jobs; returns {(stream, scenario): [rows]}. `done` carries counters across phases. Checks the run's stop
    test (done["stop"]) at least once a second."""
    out: dict = {}
    pending = {pool.submit(_job, j) for j in jobs}
    while pending:
        ready, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
        if done.get("stop") and done["stop"]():
            raise RunStopped()
        for f in ready:
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


def screen_all(a: dict, market: str = "msp", tier: int | str = 0) -> dict:
    """Average-day screen of one group of layouts (0 = the standard list, "larger" = every larger network at once) in
    every scenario in play. Takes about a second; its duration also calibrates the run-time estimate on whatever
    machine runs it."""
    m = load_market(market)
    tiers = design_tiers(a, len(m.site_rank))
    designs = [d for t in tiers[1:] for d in t] if tier == "larger" else tiers[tier]
    t0 = time.perf_counter()
    screens = {sc: screen(m, a, sc, designs) for sc in scenarios_in_play(a)}
    out = {"screens": screens, "seconds": time.perf_counter() - t0, "tier": tier}
    if tier == 0:
        out["week_seconds"] = time_one_week(m, a, screens)
    return out


def time_one_week(m, a: dict, screens: dict) -> float:
    """Seconds to simulate one real week with these settings on this machine: the cheapest layout that passed the
    screen, one week in each kind of week, averaged. Extreme settings (late trucks, no-shows, huge volume) make weeks
    much slower than the average-day screen suggests, so run times are estimated from this."""
    took = []
    weeks = draw_weeks(1, len(m.site_rank), len(a["schedule"]["linehaul_waves"]), a, "search")
    for sc, s in screens.items():
        kept = s[s["kept"]] if s["kept"].any() else s
        d = get_design(kept.nsmallest(1, "screen_cost")["design"].iloc[0], a)
        t = time.perf_counter()
        simulate_week(m, d, weeks[0], a, a["scenarios"][sc])
        took.append(time.perf_counter() - t)
    return float(np.mean(took))


def n_tiers(a: dict, market: str = "msp") -> int:
    return len(design_tiers(a, len(load_market(market).site_rank)))


def candidates(screens: dict, top_n: int) -> list[str]:
    """Per kind of week: the cheapest few layouts that passed the screen, plus the cheapest few that also meet the
    OTD target on the average day, so the run always includes layouts likely to meet OTD."""
    keep = []
    for s in screens.values():
        k = s[s["kept"]]
        hit = k[k["meets_target"]] if "meets_target" in k else k.iloc[:0]
        for d in list(k.nsmallest(top_n, "screen_cost")["design"]) + list(hit.nsmallest(top_n, "screen_cost")["design"]):
            if d not in keep:
                keep.append(d)
    return keep


def larger_candidates(screens: dict, top_n: int) -> list[str]:
    """Among the larger networks, per kind of week: the cheapest few that meet the OTD target on the average day, and
    the few with the best average-day OTD (so the closest are simulated even if none meets the target)."""
    keep = []
    for s in screens.values():
        hit = s[s["meets_target"]].nsmallest(top_n, "screen_cost")["design"]
        best = s.sort_values(["screen_on_time", "screen_cost"], ascending=[False, True]).head(top_n)["design"]
        for d in list(hit) + list(best):
            if d not in keep:
                keep.append(d)
    return keep


def merge_screens(old: dict | None, new: dict) -> dict:
    return {sc: pd.concat([old[sc], s], ignore_index=True) if old else s for sc, s in new.items()}


def _estimate_table(a: dict, screens: dict) -> pd.DataFrame:
    weights = {sc: a["weeks_per_year"].get(sc, 0) for sc in screens}
    total = sum(weights.values()) or 1
    rows = []
    for d in next(iter(screens.values()))["design"]:
        r = {sc: s.set_index("design").loc[d] for sc, s in screens.items()}
        hit = [bool(x.get("meets_target", x["kept"])) for x in r.values()]
        rows.append({"design": d, "passes": sum(hit), "on_time": min(x["screen_on_time"] for x in r.values()),
                     "year_cost": sum(weights[sc] * x["screen_cost"] for sc, x in r.items()) / total})
    return pd.DataFrame(rows).sort_values(["passes", "year_cost"], ascending=[False, True])


def quick_estimate(a: dict, screens: dict, market: str = "msp") -> dict:
    """Instant answer from the average-day screen: no randomness, so a rough guide only. The cheapest layout
    (weighted by weeks per year) that meets the OTD target on an average day in every kind of week. If none in the
    standard list does, the larger networks are screened too (still well under a second each)."""
    tier, t = 0, _estimate_table(a, screens)
    while not (t["passes"] == len(screens)).any() and tier + 1 < n_tiers(a, market):
        tier += 1
        screens = merge_screens(screens, screen_all(a, market, tier)["screens"])
        t = _estimate_table(a, screens)
    ok = t[t["passes"] == len(screens)]
    best = ok.iloc[0] if len(ok) else t.sort_values("on_time", ascending=False).iloc[0]
    return {"design": best["design"], "year_cost": float(best["year_cost"]), "passes_all": bool(len(ok)),
            "tiers_screened": tier + 1}


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
    """Expected run time, if no larger networks are needed: the planned weeks times the measured time of one real
    week on this machine, shared across the workers. Falls back to the average-day screen's timing."""
    screens = screen_info["screens"]
    workers = workers or available_cpus()
    if screen_info.get("week_seconds"):
        return planned_weeks(screen_info, mode, ctl) * screen_info["week_seconds"] * WEEK_OVERHEAD[mode] / workers \
            + STARTUP_SECONDS
    per_day = screen_info["seconds"] / max(sum(len(s) for s in screens.values()), 1)
    return planned_weeks(screen_info, mode, ctl) * WEEK_PER_SCREEN_DAY[mode] * per_day / workers + STARTUP_SECONDS


# ---------------------------------------------------------------- runs

def _frames(rows: dict, a: dict, scenarios: list[str], mode: str) -> dict:
    """Summaries per kind of week from every week simulated so far. In a full run, rechecked layouts use their
    fresh-week numbers and the rest keep their search numbers."""
    if mode == "quick":
        return {sc: summarize(pd.DataFrame(rows[("final", sc)]), a, sc) for sc in scenarios}
    frames = {}
    for sc in scenarios:
        search = summarize(pd.DataFrame(rows[("search", sc)]), a, sc)
        if rows.get(("final", sc)):
            recheck = summarize(pd.DataFrame(rows[("final", sc)]), a, sc)
            fr = recheck.combine_first(search)[recheck.columns]
        else:
            fr = search
        fr["feasible"] = fr["feasible"].astype(bool)
        frames[sc] = pd.concat([fr[fr["feasible"]].sort_values("cost_p50"),
                                fr[~fr["feasible"]].sort_values(["weeks_meeting_service", "cost_p50"],
                                                                ascending=[False, True])])
    return frames


def _merge(into: dict, new: dict) -> None:
    for k, v in new.items():
        into.setdefault(k, []).extend(v)


def _batch(pool, a: dict, mode: str, ctl: dict, scenarios: list[str], designs_by_sc: dict, rows: dict, done: dict,
           progress: Progress | None, lead: str) -> list[str]:
    """Simulate one batch of layouts. Quick: every layout on the quick run's weeks. Full: search every layout on the
    search weeks, then recheck the closest few per kind of week on fresh weeks. Returns the layouts simulated on the
    final weeks."""
    if mode == "quick":
        q = ctl["quick"]
        new = list(dict.fromkeys(d for v in designs_by_sc.values() for d in v))
        _merge(rows, _simulate(pool, _chunks(new, scenarios, "final", q["weeks"]), done,
                               done["weeks"] + len(new) * len(scenarios) * q["weeks"],
                               f"{lead}Simulating {len(new)} layouts", progress))
        return new
    f = ctl["full"]
    jobs = [j for sc in scenarios for j in _chunks(designs_by_sc[sc], [sc], "search", f["search_weeks"])]
    n_search = sum(len(v) for v in designs_by_sc.values()) * f["search_weeks"]
    guess = min(f["recheck_top_designs"] * len(scenarios), len(set().union(*designs_by_sc.values()))) \
        * len(scenarios) * f["recheck_weeks"]
    _merge(rows, _simulate(pool, jobs, done, done["weeks"] + n_search + guess,
                           f"{lead}Searching: every layout that passed the screen, {f['search_weeks']} weeks each",
                           progress))
    search = {sc: summarize(pd.DataFrame(rows[("search", sc)]), a, sc) for sc in scenarios}
    new = []
    for sc in scenarios:
        for d in [x for x in search[sc].index if x in designs_by_sc[sc]][: f["recheck_top_designs"]]:
            if d not in new:
                new.append(d)
    n_recheck = len(new) * len(scenarios) * f["recheck_weeks"]
    _merge(rows, _simulate(pool, _chunks(new, scenarios, "final", f["recheck_weeks"]), done,
                           done["weeks"] + n_recheck,
                           f"{lead}Rechecking the {len(new)} closest layouts on {f['recheck_weeks']:,} fresh weeks",
                           progress))
    return new


def run(a: dict, mode: str = "quick", market: str = "msp", workers: int | None = None,
        progress: Progress | None = None, screen_info: dict | None = None,
        should_stop: Callable[[], bool] | None = None) -> dict:
    """Simulate the standard list of layouts. If none meets every OTD target in every kind of week, screen every
    larger network (design_grid.larger_networks) on an average day and simulate the most promising few in one more
    batch on the same random weeks. One extra batch keeps a hard case to roughly twice the usual time."""
    t0 = time.perf_counter()
    ctl = a["app_controls"]["run"]
    screen_info = screen_info or screen_all(a, market)
    screens = screen_info["screens"]
    scenarios = list(screens)
    workers = workers or available_cpus()
    tiers = n_tiers(a, market)
    done = {"weeks": 0, "packages": 0, "stop": should_stop}
    rows: dict = {}
    top = ctl["quick"]["top_designs_per_scenario"]
    # forkserver: workers start from a clean process, not a copy of this multi-threaded one (the web server, other
    # runs), which avoids rare fork deadlocks. It is the default on Python 3.14, which the hosted app uses.
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("forkserver"),
                               initializer=_init, initargs=(market, a))
    try:
        if mode == "quick":
            first = {sc: candidates(screens, top) for sc in scenarios}
        else:
            first = {sc: s.loc[s["kept"], "design"].tolist() for sc, s in screens.items()}
        designs = _batch(pool, a, mode, ctl, scenarios, first, rows, done, progress, "")
        frames = _frames(rows, a, scenarios, mode)
        year = year_round_table(frames, a)
        larger_tried = False
        if not (len(year) and bool(year["meets_all"].iloc[0])) and tiers > 1:
            if progress:
                progress(1.0, "No layout met every target. Screening larger networks", dict(done, total=done["weeks"]))
            big = screen_all(a, market, "larger")["screens"]
            screens = merge_screens(screens, big)
            pick = larger_candidates(big, top)
            designs += [d for d in _batch(pool, a, mode, ctl, scenarios, {sc: pick for sc in scenarios}, rows, done,
                                          progress, "No layout met every target, so the run is now trying larger "
                                                    "networks. ") if d not in designs]
            frames = _frames(rows, a, scenarios, mode)
            year = year_round_table(frames, a)
            larger_tried = True
    except BaseException:
        # Stopped or failed: drop the queued chunks and end the workers now, so the server is free at once.
        procs = list(getattr(pool, "_processes", {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            if p.is_alive():
                p.terminate()
        raise
    else:
        pool.shutdown(wait=True)
    done.pop("stop", None)
    weeks_note = ({"weeks": ctl["quick"]["weeks"]} if mode == "quick" else
                  {"search_weeks": ctl["full"]["search_weeks"], "recheck_weeks": ctl["full"]["recheck_weeks"],
                   "rechecked": designs})
    screened = [d for s in [next(iter(screens.values()))] for d in s["design"]]
    return {"mode": mode, "frames": frames, "year": year, "screens": screens,
            "designs": designs, "scenarios": scenarios, "seconds": round(time.perf_counter() - t0, 1),
            "hash": settings_hash(a), "estimate": quick_estimate(a, screen_info["screens"], market),
            "weeks_simulated": done["weeks"], "packages_simulated": done["packages"],
            "tiers_tried": tiers if larger_tried else 1, "tiers_available": tiers,
            "diagnosis": diagnose(frames, year, a),
            "max_stations": max(len(get_design(d, a).hubs) for d in screened), **weeks_note}


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
        ok = y[y["meets_all"]]
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
