"""Simulated delivery days and weeks for one market and one hub design.

A day: overnight trucks arrive at each hub -> the hub sorts at its rate (volume above the
staffed level is sorted slower, at overtime cost) -> at the sort cut-off, sorted packages are
built into routes -> routes are offered to drivers, with surge pay stepping up until each is
claimed (only extreme weather leaves routes unclaimed) -> routes load in waves through the
hub's loading lanes and go out. No-show routes are re-offered to a backup driver.
Packages that miss the cut-off or sit on routes nobody runs carry over to the next day, late.

Usage:
  python -m engine.simulation --market msp --design "2x medium" --day Mon
  python -m engine.simulation --market msp --all-designs --day Mon [--scenario holiday_peak]
"""
from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd

from engine.checks import print_checks, run_checks
from engine.config import load_assumptions, prepared_dir
from engine.demand import DAYS, packages_per_household
from engine.randomness import DayDraws, average_day
from engine.routing import Route, build_routes, load_minutes, packages_per_stop, tract_profiles

LAST_WAVE_MISS_HOURS = 2.0   # a package that misses the last truck arrives this much later


# ---------------------------------------------------------------- inputs

def clock(s: str) -> float:
    """'D-1 22:00' -> 22.0, 'D 06:30' -> 30.5 (hours since midnight the day before delivery)."""
    day, hm = s.split()
    h, m = hm.split(":")
    return {"D-1": 0, "D": 24}[day] + int(h) + int(m) / 60


def clock_label(t: float) -> str:
    if not np.isfinite(t):
        return "-"
    day = "D-1" if t < 24 else ("D" if t < 48 else "D+1")
    t = t % 24
    return f"{day} {int(t):02d}:{int(round(t % 1 * 60)) % 60:02d}"


@dataclass
class Market:
    key: str
    name: str
    tracts: pd.DataFrame          # one row per tract
    site_rank: np.ndarray         # candidate sites, in rank order
    site_xy: np.ndarray           # projected coordinates (m)
    tract_xy: np.ndarray
    drive_min: np.ndarray         # sites x tracts, free-flow minutes
    drive_km: np.ndarray
    profiles: dict


def load_market(market: str) -> Market:
    a = load_assumptions()
    crs = a["markets"][market]["projected_crs"]
    d = prepared_dir(market)
    tracts = gpd.read_file(d / "tract_demand.gpkg")
    sites = gpd.read_file(d / "hub_candidates.gpkg").sort_values("rank")
    dt = pd.read_parquet(d / "drive_times.parquet")
    tp = tracts.to_crs(crs).representative_point()
    sp = sites.to_crs(crs).representative_point()

    def matrix(col: str) -> np.ndarray:
        wide = dt.pivot(index="site_rank", columns="GEOID", values=col)
        return wide.loc[sites["rank"], tracts["GEOID"]].to_numpy()

    return Market(
        key=market,
        name=a["markets"][market]["name"],
        tracts=pd.DataFrame(tracts.drop(columns="geometry")),
        site_rank=sites["rank"].to_numpy(),
        site_xy=np.column_stack([sp.x, sp.y]),
        tract_xy=np.column_stack([tp.x, tp.y]),
        drive_min=matrix("drive_min"),
        drive_km=matrix("drive_km"),
        profiles=tract_profiles(tracts["households"].to_numpy(), tracts["households_per_km2"].to_numpy(),
                                tracts["area_km2"].to_numpy(), a["routing"]),
    )


@dataclass
class Design:
    name: str
    hubs: list[dict]              # [{"site": rank, "size": "small"|"medium"|"large"}]

    def label(self) -> str:
        return " + ".join(f"#{h['site']} {h['size']}" for h in self.hubs)


def design_grid(a: dict | None = None) -> list[Design]:
    """The hand-picked design list: k hubs at the top-k sites, one size, plus big-first-hub variants."""
    g = (a or load_assumptions())["design_grid"]
    sizes, out = g["sizes"], []
    for k in g["hub_counts"]:
        for i, size in enumerate(sizes):
            out.append(Design(f"{k}x {size}", [{"site": s + 1, "size": size} for s in range(k)]))
            if g["big_first_hub_variants"] and k >= 2 and i + 1 < len(sizes):
                hubs = [{"site": 1, "size": sizes[i + 1]}] + [{"site": s + 1, "size": size} for s in range(1, k)]
                out.append(Design(f"{k}x {size}, big #1", hubs))
    return out


def get_design(name: str, a: dict | None = None) -> Design:
    for d in design_grid(a):
        if d.name == name:
            return d
    raise KeyError(f"Unknown design '{name}'. Known: {[d.name for d in design_grid(a)]}")


def assign_tracts(m: Market, design: Design, closed: int | None = None) -> np.ndarray:
    """Each tract goes to its nearest open hub (by drive time). Returns hub number per tract."""
    sites = [int(np.flatnonzero(m.site_rank == h["site"])[0]) for h in design.hubs]
    times = m.drive_min[sites].copy()
    if closed is not None and len(sites) > 1:
        times[closed] = np.inf
    return np.argmin(times, axis=0)


def busiest_hub(m: Market, design: Design) -> int:
    hh = np.bincount(assign_tracts(m, design), weights=m.tracts["households"].to_numpy(), minlength=len(design.hubs))
    return int(hh.argmax())


def generate_packages(m: Market, day: str, rng: np.random.Generator, a: dict, volume_multiplier: float) -> pd.DataFrame:
    """One day of new packages: tract, volume and linehaul wave, in truck unload order.
    Depends only on the random draws, never on the design (common random numbers)."""
    lam = m.tracts["households"].to_numpy() * packages_per_household(a["demand"])
    lam = lam * a["demand"]["day_of_week_multipliers"][day] * volume_multiplier
    tract = np.repeat(np.arange(len(lam)), rng.poisson(lam))
    sizes = list(a["demand"]["package_sizes"].values())
    vol = rng.choice([s["volume_m3"] for s in sizes], size=len(tract), p=[s["share"] for s in sizes])
    waves = a["schedule"]["linehaul_waves"]
    wave = rng.choice(len(waves), size=len(tract), p=[w["share"] for w in waves])
    order = rng.permutation(len(tract))
    return pd.DataFrame({"tract": tract[order], "volume_m3": vol[order], "wave": wave[order]})


# ---------------------------------------------------------------- drivers

CUFT_PER_M3 = 35.3147     # unit conversion, not an assumption


def vehicle_cost_per_km(drv: dict, rt: dict) -> float:
    """Running cost per km for the vehicle size in use: the sedan rate, rising in a straight line to the van rate."""
    cuft = rt["car_cargo_m3"] * CUFT_PER_M3
    ref = drv.get("vehicle_reference_cuft")
    if not ref or "vehicle_cost_per_km_van" not in drv:
        return drv["vehicle_cost_per_km"]
    f = min(max((cuft - ref["sedan"]) / (ref["van"] - ref["sedan"]), 0.0), 1.0)
    return drv["vehicle_cost_per_km"] + f * (drv["vehicle_cost_per_km_van"] - drv["vehicle_cost_per_km"])


def vehicle_availability(drv: dict, rt: dict) -> float:
    """Share of Dashers whose vehicle has at least this much cargo space (1.0 for sedan-sized or smaller)."""
    pts = drv.get("vehicle_availability")
    if not pts:
        return 1.0
    xs, ys = zip(*pts)
    return float(np.interp(rt["car_cargo_m3"] * CUFT_PER_M3, xs, ys, left=ys[0], right=ys[-1]))


def price_route(r: Route, rt: dict, drv: dict, commute_min: float | None = None,
                commute_km: float | None = None, floor_per_hour: float | None = None) -> dict:
    """commute_min/km: the Dasher's unpaid drive to this hub (defaults to the fixed values in drivers).
    floor_per_hour: if set, base pay is raised so the route pays at least this much per hour after car costs
    (counting the unpaid commute). The raise is part of base pay, so it counts in the network's cost."""
    commute_min = drv["commute_minutes"] if commute_min is None else commute_min
    commute_km = drv["commute_km"] if commute_km is None else commute_km
    tot = r.totals(rt)
    base = drv["pay_base_per_route"] + drv["pay_per_stop"] * r.stops + drv["pay_per_km"] * tot["km"]
    driver_hours = tot["minutes"] / 60 + commute_min / 60
    car_cost = vehicle_cost_per_km(drv, rt) * (tot["km"] + commute_km)
    topup = 0.0
    if floor_per_hour is not None:
        topup = max(0.0, floor_per_hour * driver_hours + car_cost - base)
        base += topup
    return {
        "hub": r.hub, "pieces": r.pieces, "packages": r.packages, "stops": r.stops, "volume_m3": r.volume_m3,
        "minutes": tot["minutes"], "km": tot["km"], "base_pay": base, "pay": base, "surge": 1.0,
        "wait_hours": 0.0, "driver_hours": driver_hours, "car_cost": car_cost, "floor_topup": topup,
        "status": "unclaimed", "delay": 0.0,
    }


@lru_cache(maxsize=256)
def surge_levels(start: float, cap: float, step: float) -> tuple:
    levels = np.arange(start, cap - 1e-9, step)
    return tuple(np.r_[levels, cap] if len(levels) == 0 or levels[-1] < cap - 1e-9 else levels)


def claim_routes(rows: list[dict], draws: DayDraws, drv: dict, unc: dict, rng: np.random.Generator,
                 availability: float = 1.0) -> None:
    """Each route is offered at base pay, then at +surge_step, +2 steps ... up to the surge cap,
    until a driver claims it. The chance of a claim at each offer rises with net pay per hour.
    Extreme weather cuts the claim chance and caps surge, so some routes stay unclaimed.
    No-shows are re-offered one surge step higher; the backup driver leaves later."""
    # Fewer Dashers can take routes that need a big vehicle (availability), and fewer still in a storm.
    supply = (unc["extreme_supply_factor"] if draws.extreme else 1.0) * availability
    cap = unc["extreme_surge_max"] if draws.extreme else drv["surge_max"]

    def p(r, mult):
        net_hr = (r["base_pay"] * mult - r["car_cost"]) / r["driver_hours"]
        return supply / (1 + np.exp(-(net_hr - drv["claim_midpoint_net_per_hour"]) / drv["claim_curve_width"]))

    def offer(r, levels) -> bool:
        for mult in levels:
            if rng.random() < p(r, mult):
                r["surge"], r["pay"] = mult, r["base_pay"] * mult
                return True
        return False

    for r in rows:
        if offer(r, surge_levels(1.0, cap, drv["surge_step"])):
            r["status"] = "delivered"
    for r in rows:
        if r["status"] == "delivered" and rng.random() < draws.no_show_rate:
            rescue = surge_levels(min(r["surge"] + drv["surge_step"], cap), cap, drv["surge_step"])
            if offer(r, rescue):
                r["status"], r["delay"] = "rescued", drv["no_show_rescue_delay_hours"]
            else:
                r["status"] = "no_show"


def hub_commute(m: Market, site: int, served: np.ndarray, rt: dict, drv: dict) -> tuple[float, float]:
    """A Dasher's one-way drive to the hub. With commute_from_service_area, Dashers live where the hub's
    households live: the household-weighted drive from its service area, at morning-rush speed."""
    if not drv.get("commute_from_service_area", False) or not served.any():
        return drv["commute_minutes"], drv["commute_km"]
    w = m.tracts["households"].to_numpy()[served]
    if w.sum() <= 0:
        return drv["commute_minutes"], drv["commute_km"]
    minutes = float(np.average(m.drive_min[site, served], weights=w)) * rt["stem_congestion_factor"]
    km = float(np.average(m.drive_km[site, served], weights=w))
    return minutes, km


# ---------------------------------------------------------------- the day

def simulate_day(m: Market, design: Design, day: str, a: dict, draws: DayDraws, scenario: dict,
                 carry_in: pd.DataFrame | None = None, outage_hub: int | None = None, detail: bool = True) -> dict:
    """One day. carry_in: late packages from earlier days (tract, volume_m3, sorted)."""
    rt = {**a["routing"], "time_factor": draws.time_factor}
    drv, hub_cfg, sched = a["drivers"], a["hubs"], a["schedule"]
    pkg_rng = np.random.default_rng([*draws.seed, 0])
    ops_rng = np.random.default_rng([*draws.seed, 1])

    new = generate_packages(m, day, pkg_rng, a, scenario["volume_multiplier"] * draws.volume_factor)
    sites = [int(np.flatnonzero(m.site_rank == h["site"])[0]) for h in design.hubs]
    tract_hub = assign_tracts(m, design, closed=outage_hub)
    closed_all = outage_hub is not None and len(design.hubs) == 1   # nowhere to reroute

    # Arrival time of each new package: its wave, plus that truck's delay, or the next truck if it missed its own.
    wave_t = np.array([clock(w["arrive"]) for w in sched["linehaul_waves"]])
    hub_of_new = tract_hub[new["tract"].to_numpy()]
    site_of_new = np.array(sites)[hub_of_new]
    wave = new["wave"].to_numpy()
    missed = pkg_rng.random(len(new)) < draws.handoff_share
    w2 = np.where(missed, np.minimum(wave + 1, len(wave_t) - 1), wave)
    arrive = wave_t[w2] + draws.truck_delay_h[site_of_new, w2]
    arrive = arrive + np.where(missed & (wave == len(wave_t) - 1), LAST_WAVE_MISS_HOURS, 0.0)

    carry = carry_in if carry_in is not None and len(carry_in) else pd.DataFrame(
        {"tract": pd.Series(dtype=int), "volume_m3": pd.Series(dtype=float), "sorted": pd.Series(dtype=bool)})
    n_new, n_carry = len(new), len(carry)
    tract = np.concatenate([new["tract"].to_numpy(), carry["tract"].to_numpy().astype(int)])
    volume = np.concatenate([new["volume_m3"].to_numpy(), carry["volume_m3"].to_numpy()])
    is_carry = np.r_[np.zeros(n_new, bool), np.ones(n_carry, bool)]
    pkg_hub = tract_hub[tract]
    sorted_at = np.full(len(tract), np.inf)
    carry_sorted = carry["sorted"].to_numpy().astype(bool)
    sorted_at[n_new:][carry_sorted] = -1.0                      # already sorted yesterday
    arrive_all = np.concatenate([arrive, np.full(n_carry, wave_t.min())])
    needs_sort = np.isinf(sorted_at)

    cutoff, start, promise = (clock(sched[k]) for k in ["sort_cutoff", "route_start", "promise_end"])
    window = cutoff - wave_t.min()
    sorted_count = np.zeros(len(design.hubs), dtype=int)       # running count per hub (for overtime)
    overtime_sorted = np.zeros(len(design.hubs), dtype=int)
    routes: list[dict] = []
    horizon = 48.0                                              # hours from D-1 00:00; nothing starts after

    # Sort. Each hub works through truckloads first come, first served: a load starts when it arrives or
    # when the previous load is done, whichever is later. Volume above the staffed level is sorted
    # slower (overtime/temp labor). The order of events is fixed, so it is computed directly.
    flex = a.get("peak_flex", {})
    boost = flex.get("sort_capacity_boost", 0.0) if (scenario.get("peak") and flex.get("enabled")) else 0.0
    for h, hub in enumerate(design.hubs):
        if closed_all or h == outage_hub:
            continue                                            # closed: nothing gets sorted here today
        planned = hub_cfg["sizes"][hub["size"]]["sort_rate_per_hour"]
        # Peak flex: temporary labor and longer shifts raise the sort rate; the extra volume is above the staffed
        # level, so it is paid at the overtime rate.
        rate = planned * (1 + boost) * draws.sort_factor[sites[h]]
        slow = rate * hub_cfg["overtime_productivity"]
        staffed = hub_cfg["staffed_share_of_capacity"] * planned * window
        mine = np.flatnonzero((pkg_hub == h) & needs_sort)
        now = 0.0
        for t in np.unique(arrive_all[mine]):
            now = max(now, t)
            if now >= horizon:
                break
            idx = mine[arrive_all[mine] == t]
            rank = sorted_count[h] + np.arange(len(idx))
            over = rank >= staffed
            step = np.where(over, 1 / slow, 1 / rate)
            sorted_at[idx] = now + np.cumsum(step)
            sorted_count[h] += len(idx)
            overtime_sorted[h] += int(over.sum())
            now = now + step.sum()

    # Dispatch at the cut-off: packages sorted by then are built into routes, priced and offered.
    for h in range(len(design.hubs)):
        ready = np.flatnonzero((pkg_hub == h) & (sorted_at <= cutoff))
        counts = np.bincount(tract[ready], minlength=len(m.tracts))
        vols = np.bincount(tract[ready], weights=volume[ready], minlength=len(m.tracts))
        prof = {**m.profiles, "packages_per_stop": packages_per_stop(
            counts, m.profiles["stop_points"], rt["items_per_delivery"])}
        dx, dy = (m.tract_xy - m.site_xy[sites[h]]).T
        built = build_routes(h, np.flatnonzero(tract_hub == h), counts, vols, prof,
                             np.arctan2(dy, dx), m.drive_min[sites[h]], m.drive_km[sites[h]], rt)
        # Loading lanes: Dashers get staggered pickup slots, longest routes first. Each lane loads one route at a
        # time, taking that route's loading time; a route goes to the lane that frees up first. Later slots leave
        # (and finish) later; Dashers arrive at their slot, so they don't wait.
        lanes = hub_cfg["sizes"][design.hubs[h]["size"]]["loading_lanes"]
        c_min, c_km = hub_commute(m, sites[h], tract_hub == h, rt, drv)
        floor = a["objective"]["driver_net_per_hour_floor"] if drv.get("raise_base_to_floor") else None
        priced = sorted((price_route(r, rt, drv, c_min, c_km, floor) for r in built), key=lambda r: -r["minutes"])
        lane_free = [0.0] * lanes
        heapq.heapify(lane_free)
        for r in priced:
            start_h = heapq.heappop(lane_free)
            r["wait_hours"] = start_h
            heapq.heappush(lane_free, start_h + load_minutes(r["packages"], rt) / 60)
        routes.extend(priced)
    claim_routes(routes, draws, drv, a["uncertainty"], ops_rng, vehicle_availability(drv, rt))

    # Drive: a claimed route leaves at its pickup slot (later if a backup driver replaced a no-show).
    for r in routes:
        if r["status"] in ("delivered", "rescued"):
            r["finish"] = cutoff + (start + r["wait_hours"] + r["delay"] - cutoff + r["minutes"] / 60)
        else:
            r["finish"] = np.nan

    return summarize_day(m, design, day, a, routes, tract, volume, is_carry, pkg_hub, sorted_at,
                         wave_t, cutoff, promise, sites, draws, outage_hub, closed_all, overtime_sorted, detail,
                         scenario)


# ---------------------------------------------------------------- results

def hub_fixed_cost_per_day(size: dict, hub_cfg: dict) -> float:
    """Rent, fixed site crew, operating costs, and fit-out and opening costs spread over the lease term."""
    days = hub_cfg["amortization_years"] * 365
    sqft = size["square_feet"]
    fitout = size.get("fitout_per_sqft", hub_cfg["fitout_per_sqft"])
    return (sqft * (hub_cfg["lease_per_sqft_per_year"] + hub_cfg.get("nnn_per_sqft_per_year", 0.0)) / 365
            + size["fixed_staff_hours_per_day"] * hub_cfg["fixed_staff_wage_per_hour"]
            + sqft * hub_cfg["operating_per_sqft_per_year"] / 365
            + (sqft * fitout + hub_cfg["site_opening_cost"]) / days)


def loading_cost(routes: int, overtime_share: float, hub_cfg: dict, packages: int = 0) -> float:
    """Loader labor: a fixed part per route plus time per package; on overtime in the same share as the hub's
    sort volume."""
    premium = 1 + (hub_cfg["overtime_wage_multiplier"] - 1) * overtime_share
    minutes = routes * hub_cfg["loader_minutes_per_route"] + packages * hub_cfg.get("loader_seconds_per_package", 0) / 60
    return minutes / 60 * hub_cfg["sort_wage_per_hour"] * premium


ROUTE_COLUMNS = ["hub", "pieces", "packages", "stops", "volume_m3", "minutes", "km", "base_pay", "pay", "surge",
                 "wait_hours", "driver_hours", "car_cost", "floor_topup", "status", "delay", "finish"]


def summarize_day(m, design, day, a, routes, tract, volume, is_carry, pkg_hub, sorted_at,
                  wave_t, cutoff, promise, sites, draws, outage_hub, closed_all, overtime_sorted,
                  detail: bool = True, scenario: dict | None = None) -> dict:
    """Day metrics. detail=False skips the per-route and per-hub tables (used by the weekly runs)."""
    hub_cfg = a["hubs"]
    n_tracts = len(m.tracts)
    col = lambda k, dt=float: np.array([r[k] for r in routes], dtype=dt)   # noqa: E731
    status = np.array([r["status"] for r in routes], dtype=object)
    r_hub = col("hub", int)
    ran = (status == "delivered") | (status == "rescued")
    on_time = ran & (col("finish") <= promise)

    # Packages on time per tract; carried-over packages load first, so new ones on time = on time - carry.
    on_time_t = np.zeros(n_tracts)
    unrun_t = np.zeros(n_tracts)
    for r, ok, did_run in zip(routes, on_time.tolist(), ran.tolist()):
        if ok or not did_run:
            target = on_time_t if ok else unrun_t
            for t_, n in r["pieces"]:
                target[t_] += n
    ready = sorted_at <= cutoff
    carry_ready_t = np.bincount(tract[ready & is_carry], minlength=n_tracts)
    new_on_time = int(np.maximum(on_time_t - carry_ready_t, 0).sum())

    # Carry over: packages that missed the cut-off, plus packages on routes nobody ran. The hub keeps
    # sorting after the cut-off, so most late arrivals carry over already sorted (no second sort).
    unsorted = ~ready
    missed_cutoff_total = int(unsorted.sum())
    # Partner-carrier overflow: packages not sorted by the cut-off go to a partner, up to a share of the day's
    # packages, newest-due first (carried-over packages are already late).
    partner_cfg = a.get("partner", {})
    to_partner = np.zeros(len(tract), dtype=bool)
    n_new_today = int((~is_carry).sum())
    if partner_cfg.get("enabled") and unsorted.any():
        cap = int(partner_cfg["max_share_of_day"] * n_new_today)
        cand = np.flatnonzero(unsorted)
        cand = cand[np.argsort(is_carry[cand], kind="stable")]
        to_partner[cand[:cap]] = True
    partner_new = int((to_partner & ~is_carry).sum())
    partner_on_time = int(round(partner_new * partner_cfg.get("on_time_rate", 0.0)))
    new_on_time = min(new_on_time + partner_on_time, n_new_today)
    unsorted = unsorted & ~to_partner
    vol_t = np.bincount(tract, weights=volume, minlength=n_tracts) / np.maximum(np.bincount(tract, minlength=n_tracts), 1)
    unrun_tracts = np.repeat(np.arange(n_tracts), unrun_t.astype(int))
    carry_out = pd.DataFrame({
        "tract": np.r_[tract[unsorted], unrun_tracts],
        "volume_m3": np.r_[volume[unsorted], vol_t[unrun_tracts]],
        "sorted": np.r_[np.isfinite(sorted_at[unsorted]), np.ones(len(unrun_tracts), bool)],
    })

    wait = col("wait_hours")
    hubs = []
    for h, hub in enumerate(design.hubs):
        size = hub_cfg["sizes"][hub["size"]]
        mine = pkg_hub == h
        n_mine = int(mine.sum())
        closed = closed_all or h == outage_hub
        cap = 0 if closed else size["sort_rate_per_hour"] * draws.sort_factor[sites[h]] * (cutoff - wave_t.min())
        n_routes = int((r_hub == h).sum())
        row = {
            "hub": f"#{hub['site']} {hub['size']}" + (" (closed)" if closed else ""),
            "packages": n_mine,
            "sort_utilization": n_mine / cap if cap else np.nan,
            "missed_cutoff": int((mine & ~ready).sum()),
            "overtime_sorted": int(overtime_sorted[h]),
            "routes": n_routes,
            "fixed_cost": hub_fixed_cost_per_day(size, hub_cfg),
            "loading_cost": loading_cost(n_routes, overtime_sorted[h] / max(n_mine, 1), hub_cfg,
                                         int(col("packages", int)[r_hub == h].sum())),
        }
        if detail:
            done_sort = mine & np.isfinite(sorted_at)
            row["sort_finish"] = clock_label(sorted_at[done_sort].max()) if done_sort.any() else "-"
            row["last_route_leaves"] = (clock_label(clock(a["schedule"]["route_start"]) + wait[r_hub == h].max())
                                        if n_routes else "-")
        hubs.append(row)

    n_new = int((~is_carry).sum())
    sorted_today = int((np.isfinite(sorted_at) & (sorted_at >= 0)).sum())
    ot = int(overtime_sorted.sum())
    # Sort labor by station: each size has its own productivity (bigger stations are more automated).
    sort_regular = sort_ot = 0.0
    sorted_now = np.isfinite(sorted_at) & (sorted_at >= 0)
    for h, hub in enumerate(design.hubs):
        prod = hub_cfg["sizes"][hub["size"]].get("sort_packages_per_labor_hour", hub_cfg["sort_packages_per_labor_hour"])
        per = hub_cfg["sort_wage_per_hour"] / prod
        n_h = int((sorted_now & (pkg_hub == h)).sum())
        sort_regular += (n_h - int(overtime_sorted[h])) * per
        sort_ot += int(overtime_sorted[h]) * per * hub_cfg["overtime_wage_multiplier"] / hub_cfg["overtime_productivity"]
    flex = a.get("peak_flex", {})
    peak_setup = 0.0
    if scenario is not None and scenario.get("peak") and flex.get("enabled"):
        open_stations = sum(1 for h in range(len(design.hubs)) if not (closed_all or h == outage_hub))
        peak_setup = open_stations * flex.get("setup_cost_per_station_week", 0.0) / 7
    base_pay, pay = col("base_pay")[ran], col("pay")[ran]
    minutes = col("minutes")
    cost = {
        "hub_fixed": float(np.sum([h_["fixed_cost"] for h_ in hubs])),
        "sort_labor": sort_regular,
        "sort_overtime": sort_ot,
        "peak_setup": peak_setup,
        "partner": int(to_partner.sum()) * partner_cfg.get("cost_per_package", 0.0) if partner_cfg.get("enabled") else 0.0,
        "loading_labor": float(np.sum([h_["loading_cost"] for h_ in hubs])),
        "driver_base": base_pay.sum() - col("floor_topup")[ran].sum(),
        "driver_floor_topup": col("floor_topup")[ran].sum(),
        "driver_surge": (pay - base_pay).sum(),
        "overhead": n_new * a["overhead_per_package"],
    }
    util = [h_["sort_utilization"] for h_ in hubs]
    metrics = {
        "design": design.name, "hubs": design.label(), "day": day,
        "packages_due": n_new, "carried_in": int(is_carry.sum()),
        "delivered": int(col("packages", int)[ran].sum()) + int(to_partner.sum()), "on_time": new_on_time,
        "partner_packages": int(to_partner.sum()),
        "on_time_rate": new_on_time / n_new if n_new else 1.0,
        "missed_sort_cutoff": missed_cutoff_total,
        "on_unrun_routes": int(unrun_t.sum()),
        "routes": len(routes), "routes_unclaimed": int((status == "unclaimed").sum()),
        "routes_no_show": int((status == "no_show").sum()), "routes_rescued": int((status == "rescued").sum()),
        "routes_surged": int((col("surge")[ran] > 1).sum()), "routes_run": int(ran.sum()),
        "overtime_sorted": ot, "extreme_weather": bool(draws.extreme),
        "km": col("km")[ran].sum(), "stops": col("stops")[ran].sum(), "wait_hours": wait[ran].sum(),
        "route_hours_p50": np.quantile(minutes, 0.5) / 60 if len(routes) else np.nan,
        "route_hours_p90": np.quantile(minutes, 0.9) / 60 if len(routes) else np.nan,
        "driver_pay": pay.sum(), "driver_hours": col("driver_hours")[ran].sum(),
        "car_cost": col("car_cost")[ran].sum(),
        "max_sort_utilization": np.nanmax(util) if not np.all(np.isnan(util)) else np.nan,
        **{f"cost_{k}": v for k, v in cost.items()}, "cost_total": sum(cost.values()),
    }
    out = {"metrics": metrics, "carry_out": carry_out}
    if detail:
        out["hubs"] = pd.DataFrame(hubs)
        out["routes"] = pd.DataFrame(routes, columns=ROUTE_COLUMNS)
    return out


def simulate_week(m: Market, design: Design, week: list[DayDraws], a: dict, scenario: dict) -> dict:
    """Seven days in a row; late packages carry over. Returns weekly metrics."""
    outage_day = scenario.get("outage_day")
    outage = busiest_hub(m, design) if outage_day else None
    carry, days = None, []
    for day, draws in zip(DAYS, week):
        res = simulate_day(m, design, day, a, draws, scenario, carry, outage if day == outage_day else None,
                           detail=False)
        carry = res["carry_out"]
        days.append(res["metrics"])
    d = pd.DataFrame(days)
    s = d.sum(numeric_only=True)
    n = max(s["delivered"], 1)
    return {
        "design": design.name,
        "packages_due": int(s["packages_due"]),
        "on_time_rate": s["on_time"] / s["packages_due"],
        "cost_per_package": s["cost_total"] / n,
        "driver_gross_per_hour": s["driver_pay"] / max(s["driver_hours"], 1e-9),
        "driver_net_per_hour": (s["driver_pay"] - s["car_cost"]) / max(s["driver_hours"], 1e-9),
        "routes_unclaimed_share": s["routes_unclaimed"] / max(s["routes"], 1),
        "routes_surged_share": s["routes_surged"] / max(s["routes_run"], 1),
        "km_per_package": s["km"] / n,
        "packages_per_stop": n / max(s["stops"], 1e-9),
        "pickup_slot_delay_minutes": s["wait_hours"] * 60 / max(s["routes_run"], 1),
        "missed_sort_cutoff": int(s["missed_sort_cutoff"]),
        "max_sort_utilization": d["max_sort_utilization"].max(),
        "overtime_sorted_share": s["overtime_sorted"] / max(s["packages_due"], 1),
        "extreme_weather_days": int(d["extreme_weather"].sum()),
        "worst_day_on_time": d["on_time_rate"].min(),
        "left_at_week_end": len(carry),
        **{f"cost_{k}": s[f"cost_{k}"] / n
           for k in ["hub_fixed", "sort_labor", "sort_overtime", "peak_setup", "loading_labor", "driver_base",
                     "driver_floor_topup", "driver_surge", "partner", "overhead"]},
        "partner_share": s["partner_packages"] / max(s["packages_due"], 1),
    }


# ---------------------------------------------------------------- CLI: one average day

def print_day(res: dict) -> None:
    m, hubs = res["metrics"], res["hubs"]
    per = lambda k: m[k] / max(m["delivered"], 1)
    print(f"\n=== {m['design']} ({m['hubs']}) · {m['day']} ===")
    print(f"Packages due {m['packages_due']:,} · delivered {m['delivered']:,} · on time {m['on_time_rate']:.1%}")
    print(f"  late because: missed sort cut-off {m['missed_sort_cutoff']:,} · on routes nobody ran {m['on_unrun_routes']:,}")
    print(f"Routes {m['routes']:,} · surged {m['routes_surged']} · unclaimed {m['routes_unclaimed']} · "
          f"no-shows {m['routes_no_show']} (rescued {m['routes_rescued']}) · "
          f"route hours p50 {m['route_hours_p50']:.1f}, p90 {m['route_hours_p90']:.1f}")
    if m["delivered"]:
        print(f"Density: {m['delivered'] / m['stops']:.2f} packages per stop · {m['km'] / m['delivered']:.2f} km per package · "
              f"avg pickup slot {m['wait_hours'] * 60 / m['routes_run']:.0f} min after first wave")
    if m["driver_hours"]:
        print(f"Driver pay per hour (incl. unpaid commute): gross ${m['driver_pay'] / m['driver_hours']:.2f}, "
              f"net of car costs ${(m['driver_pay'] - m['car_cost']) / m['driver_hours']:.2f}")
    print(f"Cost per delivered package ${per('cost_total'):.2f} = hub ${per('cost_hub_fixed'):.2f} "
          f"+ sort ${per('cost_sort_labor'):.2f} + sort overtime ${per('cost_sort_overtime'):.2f} "
          f"+ loading ${per('cost_loading_labor'):.2f} "
          f"+ driver ${per('cost_driver_base'):.2f} + surge ${per('cost_driver_surge'):.2f} "
          f"+ overhead ${per('cost_overhead'):.2f}")
    show = hubs.drop(columns=["fixed_cost", "loading_cost"]).copy()
    show["sort_utilization"] = show["sort_utilization"].map(lambda v: "-" if pd.isna(v) else f"{v:.0%}")
    print(show.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", default="msp")
    p.add_argument("--design", default="2x medium")
    p.add_argument("--all-designs", action="store_true")
    p.add_argument("--day", default="Mon", choices=DAYS)
    p.add_argument("--scenario", default="normal")
    args = p.parse_args()

    a = load_assumptions()
    m = load_market(args.market)
    scenario = a["scenarios"][args.scenario]
    designs = design_grid(a) if args.all_designs else [get_design(args.design, a)]
    avg = average_day(len(m.site_rank), len(a["schedule"]["linehaul_waves"]), a)
    results = []
    for d in designs:
        outage = busiest_hub(m, d) if scenario.get("outage_day") == args.day else None
        results.append(simulate_day(m, d, args.day, a, avg, scenario, outage_hub=outage))
    for res in results:
        print_day(res)
    best = results[0]["metrics"]
    if not best["delivered"]:
        return
    print_checks(run_checks({
        "cost_per_package": best["cost_total"] / best["delivered"],
        "driver_gross_per_hour": best["driver_pay"] / best["driver_hours"],
        "on_time_rate": best["on_time_rate"],
    }))


if __name__ == "__main__":
    main()
