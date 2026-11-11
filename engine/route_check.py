"""Check the fast route estimate against routes planned on real streets.

For one hub's area on one day:
1. Draw the day's packages for the hub's nearest service-area tracts (same random draw the simulation uses).
2. Fast estimate: pack them into routes with engine.routing.build_routes (square-root distance formula).
3. Real roads: place each tract's stops on real street intersections inside the tract, download the full drive
   network from OpenStreetMap, compute drive times and distances between every pair of stops, and let
   OR-Tools plan routes under the same limits (route hours, packages, cargo space).
4. Compare routes, kilometres and hours.

Routes are open: a Dasher starts at the hub and ends at the last stop (no return trip), as in the fast estimate.
Writes scenarios/<market>/route_check.json.

    python -m engine.route_check [--market msp]
"""
from __future__ import annotations

import argparse
import json
import time

import networkx as nx
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.ops import unary_union
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from engine.config import REPO_ROOT, load_assumptions, prepared_dir
from engine.data_prep import _configure_osmnx, _overpass
from engine.routing import build_routes, load_minutes, packages_per_stop
from engine.simulation import assign_tracts, generate_packages, get_design, load_market


def sample_area(m, design, hub: int, n_tracts: int) -> np.ndarray:
    """The hub's n nearest service-area tracts by drive time."""
    site = int(np.flatnonzero(m.site_rank == design.hubs[hub]["site"])[0])
    served = np.flatnonzero(assign_tracts(m, design) == hub)
    return served[np.argsort(m.drive_min[site, served])[:n_tracts]], site


def fast_estimate(m, a, hub, site, tracts, counts, vols) -> dict:
    rt = a["routing"]
    full_counts = np.zeros(len(m.tracts), dtype=int)
    full_vols = np.zeros(len(m.tracts))
    full_counts[tracts], full_vols[tracts] = counts, vols
    prof = {**m.profiles, "packages_per_stop": packages_per_stop(full_counts, m.profiles["stop_points"],
                                                                 rt["items_per_delivery"])}
    dx, dy = (m.tract_xy - m.site_xy[site]).T
    routes = build_routes(hub, tracts, full_counts, full_vols, prof, np.arctan2(dy, dx), m.drive_min[site],
                          m.drive_km[site], rt)
    tot = [r.totals(rt) for r in routes]
    return {"routes": len(routes), "km": float(sum(t["km"] for t in tot)),
            "hours": float(sum(t["minutes"] for t in tot) / 60),
            "stops": float(sum(r.stops for r in routes)), "packages": int(sum(r.packages for r in routes)),
            "pps": prof["packages_per_stop"]}


def place_stops(G, tract_geoms, tracts, counts, vols, pps, rng) -> list[dict]:
    """Stops on real intersections inside each tract; the tract's packages are split evenly across its stops."""
    nodes = ox.graph_to_gdfs(G, edges=False)[["geometry"]]
    stops = []
    for t, n, v in zip(tracts, counts, vols):
        if n == 0:
            continue
        k = max(1, int(round(n / pps[t])))
        inside = nodes[nodes.within(tract_geoms[t])].index.to_numpy()
        if len(inside) == 0:
            continue
        pick = rng.choice(inside, size=k, replace=len(inside) < k)
        share = np.diff(np.round(np.linspace(0, n, k + 1)).astype(int))
        for node, pk in zip(pick, share):
            if pk > 0:
                stops.append({"tract": int(t), "node": int(node), "packages": int(pk), "volume": v * pk / n})
    return stops


def nearest_node(G, lon: float, lat: float) -> int:
    """Closest street node to a point (equirectangular distance; fine at city scale, no extra dependency)."""
    ids = np.array(list(G.nodes))
    xy = np.array([(G.nodes[i]["x"], G.nodes[i]["y"]) for i in ids])
    dx = (xy[:, 0] - lon) * np.cos(np.radians(lat))
    return int(ids[np.argmin(dx ** 2 + (xy[:, 1] - lat) ** 2)])


def matrices(G, nodes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Drive minutes and km between every pair of nodes (hub first)."""
    n = len(nodes)
    tmin, km = np.full((n, n), np.inf), np.full((n, n), np.inf)
    for i, src in enumerate(nodes):
        tt = nx.single_source_dijkstra_path_length(G, src, weight="travel_time")
        ll = nx.single_source_dijkstra_path_length(G, src, weight="length")
        tmin[i] = [tt.get(d, np.inf) / 60 for d in nodes]
        km[i] = [ll.get(d, np.inf) / 1000 for d in nodes]
    return tmin, km


def solve(tmin, km, stops, service_min, rt, n_vehicles: int, seconds: int) -> dict:
    """OR-Tools capacitated routing with open routes (dummy end depot), minimising total drive time."""
    n = len(stops) + 1                       # 0 = hub
    end = n                                  # dummy end: free to reach from anywhere
    cong = rt["stem_congestion_factor"]
    big = 10 ** 7
    t = np.zeros((n + 1, n + 1))
    t[:n, :n] = np.where(np.isfinite(tmin), tmin * cong, big)
    t[:, end] = 0
    t[end, :] = big
    svc = np.r_[0, service_min, 0]
    pk = np.r_[0, [s["packages"] for s in stops], 0].astype(int)
    vol = np.r_[0, [s["volume"] for s in stops], 0]
    vol_units = np.round(vol * 10000).astype(int)          # 0.0001 cubic metres
    manager = pywrapcp.RoutingIndexManager(n + 1, n_vehicles, [0] * n_vehicles, [end] * n_vehicles)
    model = pywrapcp.RoutingModel(manager)
    secs = np.round((t + svc[:, None]) * 60).astype(int)    # seconds: drive from i, plus service at i
    transit = model.RegisterTransitCallback(lambda i, j: int(secs[manager.IndexToNode(i), manager.IndexToNode(j)]))
    model.SetArcCostEvaluatorOfAllVehicles(transit)
    # Route time limit: the longest route, less a full vehicle's loading time (a conservative allowance).
    limit_s = int((rt["max_route_hours"] * 60 - load_minutes(rt["max_packages_per_route"], rt)) * 60)
    model.AddDimension(transit, 0, max(limit_s, 600), True, "time")
    pk_cb = model.RegisterUnaryTransitCallback(lambda i: int(pk[manager.IndexToNode(i)]))
    model.AddDimensionWithVehicleCapacity(pk_cb, 0, [rt["max_packages_per_route"]] * n_vehicles, True, "packages")
    vol_cb = model.RegisterUnaryTransitCallback(lambda i: int(vol_units[manager.IndexToNode(i)]))
    model.AddDimensionWithVehicleCapacity(vol_cb, 0, [int(rt["car_cargo_m3"] * 10000)] * n_vehicles, True, "cargo")
    model.SetFixedCostOfAllVehicles(int(rt["hub_load_minutes"] * 60))     # each extra route costs a load
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = seconds
    sol = model.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("OR-Tools found no feasible routes; raise route_check.vehicle_slack.")
    routes = []
    for v in range(n_vehicles):
        i = model.Start(v)
        path = []
        while not model.IsEnd(i):
            path.append(manager.IndexToNode(i))
            i = sol.Value(model.NextVar(i))
        if len(path) > 1:
            routes.append(path)
    out_km = out_min = 0.0
    for path in routes:
        packages = int(pk[path].sum())
        drive = sum(t[a, b] for a, b in zip(path, path[1:]))
        out_km += sum(km[a, b] for a, b in zip(path, path[1:]))
        out_min += load_minutes(packages, rt) + drive + svc[path].sum()
    return {"routes": len(routes), "km": float(out_km), "hours": float(out_min / 60),
            "stops": int(sum(len(p) - 1 for p in routes)), "packages": int(sum(pk[p].sum() for p in routes))}


def check_hub(m, a, design, hub: int, cfg: dict) -> dict:
    t0 = time.perf_counter()
    rt = a["routing"]
    tracts, site = sample_area(m, design, hub, cfg["tracts_per_hub"])
    rng = np.random.default_rng([cfg["seed"], hub])
    day = generate_packages(m, cfg["day"], rng, a, 1.0)
    day = day[day["tract"].isin(tracts)]
    counts = day.groupby("tract").size().reindex(tracts, fill_value=0).to_numpy()
    vols = day.groupby("tract")["volume_m3"].sum().reindex(tracts, fill_value=0.0).to_numpy()

    fast = fast_estimate(m, a, hub, site, tracts, counts, vols)

    d = prepared_dir(m.key)
    tg = gpd.read_file(d / "tract_demand.gpkg").to_crs(4326)
    sites = gpd.read_file(d / "hub_candidates.gpkg").to_crs(4326).set_index("rank")
    hub_pt = sites.loc[design.hubs[hub]["site"], "geometry"].representative_point()
    geoms = dict(zip(range(len(tg)), tg.geometry))
    area = unary_union([geoms[t] for t in tracts] + [hub_pt.buffer(0.01)]).convex_hull.buffer(0.01)
    _configure_osmnx()
    G = _overpass(ox.graph_from_polygon, area, network_type="drive", retain_all=False)
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    G = ox.truncate.largest_component(G, strongly=True)

    stops = place_stops(G, geoms, tracts, counts, vols, fast["pps"], rng)
    hub_node = nearest_node(G, hub_pt.x, hub_pt.y)
    tmin, km = matrices(G, [hub_node] + [s["node"] for s in stops])
    prof = m.profiles
    service = np.array([prof["service_minutes_per_stop"][s["tract"]] for s in stops])
    n_vehicles = max(2, int(np.ceil(fast["routes"] * cfg["vehicle_slack"])))
    real = solve(tmin, km, stops, service, rt, n_vehicles, cfg["solver_seconds"])

    fast.pop("pps")
    return {"hub": hub, "site_rank": int(design.hubs[hub]["site"]), "tracts": int(len(tracts)),
            "packages": int(counts.sum()), "street_nodes": G.number_of_nodes(), "fast": fast, "real_roads": real,
            "km_ratio": real["km"] / fast["km"], "hours_ratio": real["hours"] / fast["hours"],
            "routes_ratio": real["routes"] / fast["routes"], "seconds": round(time.perf_counter() - t0, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", default="msp")
    args = p.parse_args()
    a = load_assumptions()
    cfg = a["route_check"]
    m = load_market(args.market)
    design = get_design(cfg["design"], a)
    out = []
    for hub in cfg["hubs"]:
        r = check_hub(m, a, design, hub, cfg)
        f, q = r["fast"], r["real_roads"]
        print(f"hub {hub} (site #{r['site_rank']}): {r['packages']} packages in {r['tracts']} tracts, "
              f"{r['street_nodes']:,} street nodes, {r['seconds']}s")
        print(f"  fast estimate: {f['routes']} routes, {f['km']:.0f} km, {f['hours']:.1f} h")
        print(f"  real roads:    {q['routes']} routes, {q['km']:.0f} km, {q['hours']:.1f} h  "
              f"(km x{r['km_ratio']:.2f}, hours x{r['hours_ratio']:.2f}, routes x{r['routes_ratio']:.2f})")
        out.append(r)
    (REPO_ROOT / "scenarios" / args.market / "route_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
