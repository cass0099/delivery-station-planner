"""Download and prepare public data for one market.

Steps (each cached, safe to rerun):
  tracts  Census tract shapes + ACS household counts
  roads   OpenStreetMap main road network with drive times
  stores  OpenStreetMap shops (possible local pickup points)
  sites   Industrial land from OpenStreetMap, shortlisted as candidate hubs

Usage:
  python -m engine.data_prep --market msp            # all steps
  python -m engine.data_prep --market msp --step sites
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import Polygon

from engine.config import cache_dir, load_assumptions, market_settings, prepared_dir

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
ACS_VARIABLES = {"B11001_001E": "households", "B01003_001E": "population"}
TRACT_URL = "https://www2.census.gov/geo/tiger/GENZ{year}/shp/cb_{year}_{state}_tract_500k.zip"
ACS_URL = "https://api.census.gov/data/{year}/acs/acs5"
ZCTA_TRACT_URL = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_tract20_natl.txt"


# ---------------------------------------------------------------- helpers

def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    log.info("Downloading %s", url)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def _hide_key(text: str) -> str:
    return re.sub(r"key=[^&\s]+", "key=***", text)


def _census_get(url: str, params: dict) -> list[list[str]]:
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise RuntimeError("Set CENSUS_API_KEY (free key: api.census.gov/data/key_signup.html).")
    try:
        r = requests.get(url, params={**params, "key": key}, timeout=60, allow_redirects=False)
    except requests.RequestException as e:  # never let the key reach a log or traceback
        raise RuntimeError(_hide_key(str(e))) from None
    if r.status_code != 200:
        raise RuntimeError(f"Census API returned HTTP {r.status_code}: {_hide_key(r.text[:200])}")
    return r.json()


OVERPASS_RETRY_WAITS_S = [30, 60, 120, 240, 480]  # backoff the public server asks for when busy


def _overpass(fn, *args, **kwargs):
    """Call an OSMnx download, retrying with growing waits when Overpass is busy.
    Chunks that already succeeded are cached, so each retry resumes."""
    for attempt, wait in enumerate([*OVERPASS_RETRY_WAITS_S, None], start=1):
        try:
            return fn(*args, **kwargs)
        except (ox._errors.ResponseStatusCodeError, requests.exceptions.RequestException, ValueError) as e:
            if wait is None:
                raise
            log.warning("Overpass busy or failed (attempt %d: %s). Retrying in %ds...",
                        attempt, str(e).splitlines()[0][:120], wait)
            time.sleep(wait)


def _configure_osmnx() -> None:
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir("osmnx"))
    ox.settings.requests_timeout = 600
    ox.settings.log_console = False


def market_boundary(market: str) -> Polygon:
    """Outline of the market (tract union, small buffer, lake holes filled), in WGS84."""
    tracts = gpd.read_file(prepared_dir(market) / "tracts.gpkg")
    crs = market_settings(market)["projected_crs"]
    shape = tracts.to_crs(crs).union_all().buffer(500)
    if shape.geom_type == "MultiPolygon":
        shape = max(shape.geoms, key=lambda g: g.area)
    shape = Polygon(shape.exterior)
    return gpd.GeoSeries([shape], crs=crs).to_crs(WGS84).iloc[0]


# ---------------------------------------------------------------- steps

def prepare_tracts(market: str) -> gpd.GeoDataFrame:
    cfg = market_settings(market)
    src = load_assumptions()["data_sources"]
    state = cfg["state_fips"]
    counties = list(cfg["counties"])

    zip_path = _download(
        TRACT_URL.format(year=src["tiger_year"], state=state),
        cache_dir("raw", "census") / f"cb_{src['tiger_year']}_{state}_tract_500k.zip",
    )
    shapes = gpd.read_file(f"zip://{zip_path}")
    shapes = shapes[shapes["COUNTYFP"].isin(counties)][["GEOID", "COUNTYFP", "geometry"]]

    acs_cache = cache_dir("raw", "census") / f"acs{src['acs_year']}_{state}_{market}.csv"
    if acs_cache.exists():
        acs = pd.read_csv(acs_cache, dtype={"GEOID": str})
    else:
        rows = []
        for county in counties:
            data = _census_get(
                ACS_URL.format(year=src["acs_year"]),
                {"get": ",".join(ACS_VARIABLES), "for": "tract:*", "in": f"state:{state} county:{county}"},
            )
            rows.append(pd.DataFrame(data[1:], columns=data[0]))
        acs = pd.concat(rows, ignore_index=True)
        acs["GEOID"] = acs["state"] + acs["county"] + acs["tract"]
        acs = acs.rename(columns=ACS_VARIABLES)[["GEOID", *ACS_VARIABLES.values()]]
        acs.to_csv(acs_cache, index=False)
    for col in ACS_VARIABLES.values():
        acs[col] = pd.to_numeric(acs[col]).clip(lower=0).astype(int)

    tracts = shapes.merge(acs, on="GEOID", how="left")
    missing = tracts["households"].isna().sum()
    if missing:
        log.warning("%d tract shapes have no ACS row; treated as 0 households", missing)
    tracts[list(ACS_VARIABLES.values())] = tracts[list(ACS_VARIABLES.values())].fillna(0).astype(int)
    tracts["county"] = tracts["COUNTYFP"].map(cfg["counties"])
    tracts["area_km2"] = tracts.to_crs(cfg["projected_crs"]).area / 1e6
    tracts["households_per_km2"] = tracts["households"] / tracts["area_km2"]
    tracts = tracts.drop(columns="COUNTYFP").to_crs(WGS84)

    tracts.to_file(prepared_dir(market) / "tracts.gpkg", driver="GPKG")
    log.info("Tracts: %d, households: %s", len(tracts), f"{tracts['households'].sum():,}")
    return tracts


def prepare_roads(market: str) -> Path:
    _configure_osmnx()
    out = cache_dir("prepared", market) / "roads_main.graphml"
    if out.exists():
        log.info("Roads already prepared: %s", out)
        return out
    types = "|".join(load_assumptions()["roads"]["highway_types"])
    log.info("Downloading main road network (this can take several minutes)...")
    G = _overpass(
        ox.graph_from_polygon,
        market_boundary(market), network_type="drive", custom_filter=f'["highway"~"^({types})$"]'
    )
    G = ox.routing.add_edge_speeds(G)       # OSM speed limits, gaps filled by road-type average
    G = ox.routing.add_edge_travel_times(G)
    ox.save_graphml(G, out)
    log.info("Roads: %d nodes, %d edges -> %s", len(G.nodes), len(G.edges), out)
    return out


def prepare_stores(market: str) -> gpd.GeoDataFrame:
    _configure_osmnx()
    crs = market_settings(market)["projected_crs"]
    shops = _overpass(
        ox.features_from_polygon,
        market_boundary(market), tags={"shop": load_assumptions()["stores"]["shop_types"]}
    ).reset_index()
    shops["geometry"] = shops.to_crs(crs).representative_point().to_crs(WGS84)
    keep = [c for c in ["id", "name", "shop", "brand", "geometry"] if c in shops.columns]
    stores = gpd.GeoDataFrame(shops[keep], geometry="geometry", crs=WGS84)
    stores.to_file(prepared_dir(market) / "stores.gpkg", driver="GPKG")
    log.info("Stores: %d", len(stores))
    return stores


def unsuitable_sites(raw: pd.DataFrame, rules: dict) -> pd.Series:
    """True for sites whose OSM tags or name mark them as utilities, waste or heavy industry."""
    bad = pd.Series(False, index=raw.index)
    for tag, values in rules["exclude_tags"].items():
        if tag not in raw.columns:
            continue
        bad |= raw[tag].notna() if values == "any" else raw[tag].isin(values)
    if "name" in raw.columns:
        pattern = "|".join(re.escape(k) for k in rules["exclude_name_keywords"])
        bad |= raw["name"].fillna("").str.contains(pattern, case=False, regex=True)
    return bad


def shortlist_sites(sites: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame, rules: dict, crs: str) -> gpd.GeoDataFrame:
    """Pick sites in rounds (maximal covering). Each round takes the site that reaches the
    most households not already reached by earlier picks, so the list spreads across the
    market. Ties (e.g. once everything is reached) go to the site reaching the most
    households overall. Sites closer than min_spacing_km to an earlier pick are skipped."""
    s = sites.to_crs(crs).representative_point()
    t = tracts.to_crs(crs).representative_point()
    sx, sy = s.x.to_numpy(), s.y.to_numpy()
    tx, ty = t.x.to_numpy(), t.y.to_numpy()
    dist_km = np.hypot(sx[:, None] - tx[None, :], sy[:, None] - ty[None, :]) / 1000
    hh = tracts["households"].to_numpy().astype(float)
    reaches = dist_km <= rules["coverage_radius_km"]          # sites x tracts

    total = reaches @ hh
    covered = np.zeros(len(tracts), dtype=bool)
    available = np.ones(len(sites), dtype=bool)
    min_m = rules["min_spacing_km"] * 1000
    picks, new_hh = [], []
    while len(picks) < rules["max_candidates"] and available.any():
        gain = reaches[:, ~covered] @ hh[~covered]
        order = np.lexsort((-total, -gain))                    # best gain, then best total
        best = next(i for i in order if available[i])
        picks.append(best)
        new_hh.append(gain[best])
        covered |= reaches[best]
        available &= np.hypot(sx - sx[best], sy - sy[best]) >= min_m

    short = sites.iloc[picks].copy()
    short.insert(0, "rank", range(1, len(short) + 1))
    short["new_households_reached"] = new_hh
    short["households_in_radius"] = total[picks]
    short["avg_km_to_household"] = (dist_km[picks] @ hh) / hh.sum()
    return short


def prepare_sites(market: str) -> gpd.GeoDataFrame:
    _configure_osmnx()
    rules = load_assumptions()["hub_candidates"]
    crs = market_settings(market)["projected_crs"]

    raw = _overpass(ox.features_from_polygon, market_boundary(market), tags={"landuse": rules["landuse_types"]})
    raw = raw[raw.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index()
    raw["area_m2"] = raw.to_crs(crs).area
    excluded = unsuitable_sites(raw, rules)
    log.info("Excluded as utility/waste/heavy industry: %d", excluded.sum())
    sites = raw[~excluded & (raw["area_m2"] >= rules["min_site_area_m2"])]
    keep = [c for c in ["id", "name", "landuse", "area_m2", "geometry"] if c in sites.columns]
    sites = gpd.GeoDataFrame(sites[keep], geometry="geometry", crs=WGS84).reset_index(drop=True)
    log.info("Industrial sites: %d total, %d at least %s m²", len(raw), len(sites), f"{rules['min_site_area_m2']:,}")

    tracts = gpd.read_file(prepared_dir(market) / "tracts.gpkg")
    short = shortlist_sites(sites, tracts, rules, crs)
    short.to_file(prepared_dir(market) / "hub_candidates.gpkg", driver="GPKG")
    log.info("Hub candidates shortlisted: %d", len(short))
    return short


def _nearest(node_xy: np.ndarray, pts_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of and distance (m) to the nearest road node for each point."""
    idx, dist = np.empty(len(pts_xy), dtype=int), np.empty(len(pts_xy))
    for i, (x, y) in enumerate(pts_xy):
        d = np.hypot(node_xy[:, 0] - x, node_xy[:, 1] - y)
        idx[i] = d.argmin()
        dist[i] = d[idx[i]]
    return idx, dist


def prepare_drive_times(market: str) -> pd.DataFrame:
    """Drive minutes and km from every candidate hub to every tract on the main road network.
    Each end adds a short straight-line hop to the nearest main road at snap_speed_kmh."""
    import networkx as nx

    crs = market_settings(market)["projected_crs"]
    snap_kmh = load_assumptions()["routing"]["snap_speed_kmh"]
    G = ox.load_graphml(cache_dir("prepared", market) / "roads_main.graphml")
    G = ox.truncate.largest_component(G, strongly=True)  # snap only to the connected main network
    G = ox.project_graph(G, to_crs=crs)
    nodes = list(G.nodes)
    node_xy = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in nodes])

    hubs = gpd.read_file(prepared_dir(market) / "hub_candidates.gpkg")
    tracts = gpd.read_file(prepared_dir(market) / "tracts.gpkg")
    hp = hubs.to_crs(crs).representative_point()
    tp = tracts.to_crs(crs).representative_point()
    h_idx, h_snap = _nearest(node_xy, np.column_stack([hp.x, hp.y]))
    t_idx, t_snap = _nearest(node_xy, np.column_stack([tp.x, tp.y]))

    rows = []
    for rank, hi, hs, hx, hy in zip(hubs["rank"], h_idx, h_snap, hp.x, hp.y):
        secs = nx.single_source_dijkstra_path_length(G, nodes[hi], weight="travel_time")
        meters = nx.single_source_dijkstra_path_length(G, nodes[hi], weight="length")
        for geoid, ti, ts, tx, ty in zip(tracts["GEOID"], t_idx, t_snap, tp.x, tp.y):
            n = nodes[ti]
            snap_m = hs + ts
            if n in secs:
                minutes = secs[n] / 60 + snap_m / 1000 / snap_kmh * 60
                km = meters[n] / 1000 + snap_m / 1000
            else:  # not connected on the main network: fall back to a straight line
                km = np.hypot(hx - tx, hy - ty) / 1000 * 1.3
                minutes = km / snap_kmh * 60
            rows.append((int(rank), geoid, minutes, km, n not in secs))
    out = pd.DataFrame(rows, columns=["site_rank", "GEOID", "drive_min", "drive_km", "fallback"])
    out.to_parquet(prepared_dir(market) / "drive_times.parquet", index=False)
    log.info("Drive times: %d pairs, %d straight-line fallbacks, median %.0f min",
             len(out), out["fallback"].sum(), out["drive_min"].median())
    return out


def tract_zips(rel: pd.DataFrame, geoids: pd.Series) -> pd.Series:
    """Each tract's ZIP code: the ZIP code tabulation area (ZCTA) that holds most of the tract's land.
    Tracts and ZIPs do not nest, so a ZIP here is the group of tracts that mostly fall inside it."""
    rel = rel[rel["GEOID_TRACT_20"].isin(set(geoids)) & rel["GEOID_ZCTA5_20"].notna()].copy()
    rel["land"] = pd.to_numeric(rel["AREALAND_PART"])
    best = rel.sort_values("land").groupby("GEOID_TRACT_20")["GEOID_ZCTA5_20"].last()
    return geoids.map(best)


def prepare_zips(market: str) -> gpd.GeoDataFrame:
    path = _download(ZCTA_TRACT_URL, cache_dir("raw", "census") / "tab20_zcta520_tract20_natl.txt")
    rel = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8-sig")
    tracts = gpd.read_file(prepared_dir(market) / "tracts.gpkg")
    tracts["zip"] = tract_zips(rel, tracts["GEOID"])
    missing = tracts["zip"].isna().sum()
    if missing:
        log.warning("%d tracts have no ZIP; each is treated as its own area", missing)
        tracts["zip"] = tracts["zip"].fillna("T" + tracts["GEOID"])
    tracts.to_file(prepared_dir(market) / "tracts.gpkg", driver="GPKG")
    log.info("ZIPs: %d for %d tracts", tracts["zip"].nunique(), len(tracts))
    return tracts


STEPS = {"tracts": prepare_tracts, "zips": prepare_zips, "roads": prepare_roads, "stores": prepare_stores, "sites": prepare_sites,
         "drive_times": prepare_drive_times}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", default="msp")
    p.add_argument("--step", choices=[*STEPS, "all"], default="all")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for name, fn in STEPS.items():
        if args.step in (name, "all"):
            fn(args.market)


if __name__ == "__main__":
    main()
