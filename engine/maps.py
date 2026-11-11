"""Phase 1 map: neighborhoods colored by daily demand, candidate hubs, stores and main roads.

Usage:
  python -m engine.maps --market msp     # writes data/prepared/<market>/phase1_map.html (+ .png)
"""
from __future__ import annotations

import argparse
import logging

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from folium.plugins import FastMarkerCluster

from engine.config import cache_dir, market_settings, prepared_dir

log = logging.getLogger(__name__)
MAJOR_ROADS = {"motorway", "trunk"}


def _simplified(gdf: gpd.GeoDataFrame, crs: str, tolerance_m: float) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["geometry"] = gdf.to_crs(crs).simplify(tolerance_m).to_crs(gdf.crs)
    return out


def _major_roads(market: str) -> gpd.GeoDataFrame | None:
    path = cache_dir("prepared", market) / "roads_main.graphml"
    if not path.exists():
        log.warning("Road network not prepared yet; map will skip roads")
        return None
    edges = ox.graph_to_gdfs(ox.load_graphml(path), nodes=False)
    hw = edges["highway"].apply(lambda v: v if isinstance(v, str) else v[0])
    return edges[hw.isin(MAJOR_ROADS)][["geometry"]]


def build_map(market: str) -> folium.Map:
    cfg = market_settings(market)
    crs = cfg["projected_crs"]
    d = prepared_dir(market)
    demand = gpd.read_file(d / "tract_demand.gpkg")
    hubs = gpd.read_file(d / "hub_candidates.gpkg")
    stores = gpd.read_file(d / "stores.gpkg")

    center = demand.to_crs(crs).union_all().centroid
    lat_lon = gpd.GeoSeries([center], crs=crs).to_crs("EPSG:4326").iloc[0]
    m = folium.Map(location=[lat_lon.y, lat_lon.x], zoom_start=10, tiles="OpenStreetMap")

    # Neighborhood demand density, with quantile color breaks because demand is skewed.
    tracts = _simplified(demand, crs, 25)
    vals = tracts["pkgs_per_km2"]
    breaks = sorted(set(np.round(np.quantile(vals, [0, .2, .4, .6, .8, .95, 1]), 1)))
    colors = cm.StepColormap(
        ["#fff5eb", "#fdd0a2", "#fdae6b", "#fd8d3c", "#e6550d", "#a63603"][: len(breaks) - 1],
        index=breaks, vmin=breaks[0], vmax=breaks[-1],
        caption="Packages per day per km² (average day)",
    )
    for col in ["pkgs_per_day", "pkgs_Mon", "pkgs_per_km2"]:
        tracts[col] = tracts[col].round(1)
    folium.GeoJson(
        tracts[["GEOID", "county", "households", "pkgs_per_day", "pkgs_Mon", "pkgs_per_km2", "geometry"]],
        name="Neighborhood demand",
        style_function=lambda f: {
            "fillColor": colors(f["properties"]["pkgs_per_km2"]),
            "color": "#888", "weight": 0.4, "fillOpacity": 0.7,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["GEOID", "county", "households", "pkgs_per_day", "pkgs_Mon", "pkgs_per_km2"],
            aliases=["Tract", "County", "Households", "Packages / avg day", "Packages / Monday", "Packages / day / km²"],
            localize=True,
        ),
    ).add_to(m)
    colors.add_to(m)

    roads = _major_roads(market)
    if roads is not None:
        folium.GeoJson(
            _simplified(roads, crs, 30), name="Freeways",
            style_function=lambda f: {"color": "#4a4a4a", "weight": 1.5, "opacity": 0.7},
        ).add_to(m)

    hub_layer = folium.FeatureGroup(name="Candidate hub sites (ranked)")
    folium.GeoJson(
        hubs[["geometry"]], style_function=lambda f: {"color": "#08519c", "weight": 1, "fillOpacity": 0.4},
    ).add_to(hub_layer)
    pts = hubs.to_crs(crs).representative_point().to_crs("EPSG:4326")
    for row, pt in zip(hubs.itertuples(), pts):
        name = row.name if isinstance(getattr(row, "name", None), str) else "Unnamed industrial site"
        folium.Marker(
            [pt.y, pt.x],
            icon=folium.DivIcon(html=(
                f'<div style="background:#08519c;color:#fff;border-radius:50%;width:22px;height:22px;'
                f'text-align:center;font:bold 11px/22px sans-serif;border:2px solid #fff">{row.rank}</div>'
            )),
            tooltip=(f"#{row.rank} {name}<br>{row.area_m2 / 10000:.1f} ha<br>"
                     f"{row.households_in_radius:,.0f} households within radius<br>"
                     f"avg {row.avg_km_to_household:.1f} km to a household"),
        ).add_to(hub_layer)
    hub_layer.add_to(m)

    store_layer = folium.FeatureGroup(name=f"Stores ({len(stores):,})", show=False)
    FastMarkerCluster([[g.y, g.x] for g in stores.geometry]).add_to(store_layer)
    store_layer.add_to(m)

    total = demand["pkgs_per_day"].sum()
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:#fff;padding:8px 12px;'
        f'border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.3);font:13px sans-serif">'
        f'<b>{cfg["name"]}: phase 1 data</b><br>{len(demand):,} neighborhoods · '
        f'{demand["households"].sum():,} households · {total:,.0f} packages on an average day<br>'
        f'{len(hubs)} candidate hub sites · {len(stores):,} stores</div>'
    ))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def save_static_png(market: str) -> None:
    """Static snapshot for the README and quick review."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    crs = market_settings(market)["projected_crs"]
    d = prepared_dir(market)
    demand = gpd.read_file(d / "tract_demand.gpkg").to_crs(crs)
    hubs = gpd.read_file(d / "hub_candidates.gpkg").to_crs(crs)
    fig, ax = plt.subplots(figsize=(9, 9))
    bins = pd.qcut(demand["pkgs_per_km2"], 6, duplicates="drop")
    demand["density_class"] = bins.cat.rename_categories(
        [f"{max(iv.left, 0):,.0f} – {iv.right:,.0f}" for iv in bins.cat.categories])
    demand.plot(column="density_class", categorical=True, cmap="Oranges", ax=ax,
                edgecolor="#999", linewidth=0.2, legend=True,
                legend_kwds={"title": "Packages/day/km²", "loc": "lower left", "fontsize": 8})
    roads = _major_roads(market)
    if roads is not None:
        roads.to_crs(crs).plot(ax=ax, color="#555", linewidth=0.6)
    pts = hubs.representative_point()
    ax.scatter(pts.x, pts.y, s=110, color="#08519c", edgecolor="white", zorder=3)
    for r, p in zip(hubs["rank"], pts):
        ax.annotate(str(r), (p.x, p.y), ha="center", va="center", fontsize=6, color="white", zorder=4)
    ax.set_title(f"{market_settings(market)['name']}: daily demand and candidate hubs")
    ax.set_axis_off()
    fig.savefig(d / "phase1_map.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", default="msp")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = prepared_dir(args.market) / "phase1_map.html"
    build_map(args.market).save(out)
    log.info("Map saved: %s (%.1f MB)", out, out.stat().st_size / 1e6)
    save_static_png(args.market)
    log.info("Snapshot saved: %s", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
