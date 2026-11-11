import pandas as pd
import pytest

from engine.config import load_assumptions
from engine.demand import DAYS, check_day_multipliers, packages_per_household, tract_demand


@pytest.fixture
def cfg():
    return load_assumptions()["demand"]


@pytest.fixture
def tracts():
    return pd.DataFrame({"households": [2000, 500, 0], "area_km2": [2.0, 10.0, 1.0]})


def test_day_multipliers_in_assumptions_average_one(cfg):
    check_day_multipliers(cfg["day_of_week_multipliers"])


def test_bad_day_multipliers_rejected():
    with pytest.raises(ValueError):
        check_day_multipliers({d: 1.1 for d in DAYS})


def test_tract_demand_is_households_times_rate(cfg, tracts):
    out = tract_demand(tracts, cfg)
    expected = tracts["households"] * packages_per_household(cfg)
    assert out["pkgs_per_day"].tolist() == pytest.approx(expected.tolist())


def test_zero_households_means_zero_demand(cfg, tracts):
    out = tract_demand(tracts, cfg)
    assert (out.loc[2, [f"pkgs_{d}" for d in DAYS]] == 0).all()


def test_weekly_total_matches_seven_average_days(cfg, tracts):
    out = tract_demand(tracts, cfg)
    week = sum(out[f"pkgs_{d}"].sum() for d in DAYS)
    assert week == pytest.approx(7 * out["pkgs_per_day"].sum())
