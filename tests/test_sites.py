import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from engine.config import load_assumptions
from engine.data_prep import shortlist_sites, unsuitable_sites

CRS = "EPSG:26915"


def test_utility_and_waste_sites_excluded():
    rules = load_assumptions()["hub_candidates"]
    raw = pd.DataFrame({
        "name": ["Golden Valley Business Center", "Water Treatment Plant", None, "Riverside Generating Plant", None],
        "man_made": [None, None, "wastewater_plant", None, None],
        "power": [None, None, None, None, "plant"],
    })
    assert unsuitable_sites(raw, rules).tolist() == [False, True, True, True, True]


def _pts(xy):
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in xy], crs=CRS)


def test_shortlist_spreads_out_and_respects_spacing():
    # Two clusters of households 60 km apart; sites near each, plus a duplicate near cluster A.
    tracts = _pts([(0, 0), (1000, 0), (60000, 0), (61000, 0)])
    tracts["households"] = [5000, 5000, 3000, 3000]
    sites = _pts([(500, 0), (900, 0), (60500, 0)])
    rules = {"coverage_radius_km": 12, "min_spacing_km": 5, "max_candidates": 3}
    short = shortlist_sites(sites, tracts, rules, CRS)

    # Site at 900 m is within 5 km of the first pick, so the second pick is the far cluster.
    assert short.index.tolist() == [0, 2]
    assert short["new_households_reached"].tolist() == [10000, 6000]
    xy = np.array([(g.x, g.y) for g in short.geometry])
    assert np.hypot(*(xy[0] - xy[1])) >= 5000
