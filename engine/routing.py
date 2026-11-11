"""Route building and the fast route-cost estimate.

Fast estimate (used inside the search):
  route minutes = hub load
                + drive from hub to the area served (road network, x congestion)
                + in-area driving: k * sqrt(stops x area served) at the local speed
                + service time per stop
Routes are built with a sweep: neighborhoods are taken in angle order around the hub
and packed into a route until it hits the time, package-count or car-volume limit.
Packages per stop are not fixed: they come from how many deliveries land on each stop point
that day (see packages_per_stop), so busy days and dense areas get more packages per stop.
Real-road routing with OR-Tools (final check only) comes later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import math

import numpy as np


def tract_profiles(households: np.ndarray, households_per_km2: np.ndarray, area_km2: np.ndarray,
                   cfg: dict) -> dict[str, np.ndarray]:
    dense = households_per_km2 >= cfg["dense_households_per_km2"]
    out = {"area_km2": area_km2, "dense": dense}
    for key in ["households_per_stop_point", "service_minutes_per_stop", "local_speed_kmh"]:
        out[key] = np.where(dense, cfg["profiles"]["dense"][key], cfg["profiles"]["suburban"][key])
    out["stop_points"] = np.maximum(households / out["households_per_stop_point"], 1.0)
    return out


def packages_per_stop(packages: np.ndarray, stop_points: np.ndarray, items_per_delivery: float) -> np.ndarray:
    """Deliveries land on stop points at random, so the number of distinct stops is
    points x (1 - exp(-deliveries / points)). More deliveries per point -> more packages per stop."""
    deliveries = packages / items_per_delivery
    stops = stop_points * -np.expm1(-deliveries / stop_points)
    return np.where(packages > 0, packages / np.maximum(stops, 1e-9), items_per_delivery)


@dataclass
class Route:
    hub: int
    pieces: list[tuple[int, int]] = field(default_factory=list)   # (tract index, packages)
    packages: int = 0
    volume_m3: float = 0.0
    stops: float = 0.0
    area_km2: float = 0.0
    service_min: float = 0.0
    local_hours_per_km_x_stops: float = 0.0   # stop-weighted 1/speed, for the in-area drive
    stem_min_x_stops: float = 0.0
    stem_km_x_stops: float = 0.0

    def totals(self, cfg: dict) -> dict:
        return route_totals(self.stops, self.area_km2, self.service_min, self.local_hours_per_km_x_stops,
                            self.stem_min_x_stops, self.stem_km_x_stops, cfg, self.packages)


def load_minutes(packages: float, cfg: dict) -> float:
    """Time to load a route at the hub: a fixed part plus time per package."""
    return cfg["hub_load_minutes"] + packages * cfg.get("hub_load_seconds_per_package", 0) / 60


def route_totals(stops, area, service_min, inv_speed_x_stops, stem_min_x_stops, stem_km_x_stops, cfg,
                 packages: float = 0) -> dict:
    if stops <= 0:
        return {"minutes": 0.0, "km": 0.0, "local_km": 0.0, "stem_km": 0.0}
    local_km = cfg["circuity_k"] * np.sqrt(stops * area)
    local_min = local_km * (inv_speed_x_stops / stops) * 60
    stem_min = stem_min_x_stops / stops * cfg["stem_congestion_factor"]
    stem_km = stem_km_x_stops / stops
    minutes = load_minutes(packages, cfg) + (stem_min + local_min) * cfg.get("time_factor", 1.0) + service_min
    # Time uses the formula distance (it matched real roads); kilometres add the real-road distance factor.
    local_road_km = local_km * cfg.get("local_distance_factor", 1.0)
    return {"minutes": minutes, "km": stem_km + local_road_km, "local_km": local_road_km, "stem_km": stem_km}


def build_routes(hub: int, tracts: np.ndarray, packages: np.ndarray, volume_m3: np.ndarray,
                 prof: dict[str, np.ndarray], angle: np.ndarray, stem_min: np.ndarray, stem_km: np.ndarray,
                 cfg: dict) -> list[Route]:
    """Pack the given tracts' packages into routes. Arrays are indexed by tract number;
    `tracts` lists the tracts this hub serves."""
    max_min = cfg["max_route_hours"] * 60
    max_pkgs = cfg["max_packages_per_route"]
    cargo = cfg["car_cargo_m3"]

    # Same arithmetic as route_totals, inlined: the fit check runs hundreds of thousands of times a week.
    k, load, cong, tf = cfg["circuity_k"], cfg["hub_load_minutes"], cfg["stem_congestion_factor"], cfg.get("time_factor", 1.0)
    load_per_pkg = cfg.get("hub_load_seconds_per_package", 0) / 60
    pps_all, svc_all, spd_all, area_all = (prof["packages_per_stop"], prof["service_minutes_per_stop"],
                                           prof["local_speed_kmh"], prof["area_km2"])

    routes: list[Route] = []
    cur = Route(hub)
    order = tracts[np.argsort(angle[tracts], kind="stable")]
    order = order[packages[order] > 0]
    cols = zip(order.tolist(), packages[order].tolist(), volume_m3[order].tolist(), pps_all[order].tolist(),
               svc_all[order].tolist(), spd_all[order].tolist(), area_all[order].tolist(),
               stem_min[order].tolist(), stem_km[order].tolist())
    for t, n_t, vol_t, pps, svc, spd, area_t, smin, skm in cols:
        left = int(n_t)
        n_t = float(n_t)
        vol_each = vol_t / n_t
        inv_spd = 1 / spd
        area_per_pkg = area_t / n_t

        def add(r: Route, m: int) -> tuple:
            s = m / pps
            return (r.stops + s, r.area_km2 + area_per_pkg * m, r.service_min + s * svc,
                    r.local_hours_per_km_x_stops + s * inv_spd, r.stem_min_x_stops + s * smin,
                    r.stem_km_x_stops + s * skm)

        def fits(r: Route, m: int) -> bool:
            if r.packages + m > max_pkgs or r.volume_m3 + m * vol_each > cargo + 1e-9:
                return False
            s = m / pps
            stops = r.stops + s
            if stops <= 0:
                return True
            local_km = k * math.sqrt(stops * (r.area_km2 + area_per_pkg * m))
            local_min = local_km * ((r.local_hours_per_km_x_stops + s * inv_spd) / stops) * 60
            stem = (r.stem_min_x_stops + s * smin) / stops * cong
            return (load + load_per_pkg * (r.packages + m) + (stem + local_min) * tf
                    + (r.service_min + s * svc) <= max_min)

        while left > 0:
            # Largest m in [0, left] that fits (fits is monotone in m). The package and cargo limits
            # cap the search up front; fits() still checks them exactly.
            hi = min(left, max_pkgs - cur.packages)
            if vol_each > 0:
                hi = min(hi, int((cargo + 1e-9 - cur.volume_m3) / vol_each) + 1)
            lo, hi = 0, max(hi, 0)
            if hi > 0 and fits(cur, hi):                        # common case: everything left fits
                lo = hi
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if fits(cur, mid) else (lo, mid - 1)
            m = lo
            if m == 0:
                if cur.packages == 0:   # even one package breaks a limit on an empty route: send it anyway
                    m = 1
                else:
                    routes.append(cur)
                    cur = Route(hub)
                    continue
            (cur.stops, cur.area_km2, cur.service_min, cur.local_hours_per_km_x_stops,
             cur.stem_min_x_stops, cur.stem_km_x_stops) = add(cur, m)
            cur.packages += m
            cur.volume_m3 += m * vol_each
            cur.pieces.append((int(t), m))
            left -= m
    if cur.packages:
        routes.append(cur)
    return routes
