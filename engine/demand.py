"""Synthetic package demand by neighborhood (Census tract).

  market packages/day = households x packages_per_household_per_day x network_share
  tract packages/day  = market packages/day x tract share of households
  day-of-week and holiday multipliers scale that average.

Usage:
  python -m engine.demand --market msp
"""
from __future__ import annotations

import argparse
import logging

import geopandas as gpd
import pandas as pd

from engine.checks import print_checks, run_checks
from engine.config import load_assumptions, prepared_dir

log = logging.getLogger(__name__)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def packages_per_household(cfg: dict) -> float:
    """Average packages per household per day that our network delivers."""
    return cfg["packages_per_household_per_day"] * cfg["network_share"]


def check_day_multipliers(mults: dict) -> None:
    if set(mults) != set(DAYS):
        raise ValueError(f"day_of_week_multipliers needs exactly {DAYS}")
    mean = sum(mults.values()) / len(mults)
    if abs(mean - 1.0) > 1e-6:
        raise ValueError(f"day_of_week_multipliers must average 1.0 (they average {mean:.4f})")


def tract_demand(tracts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add average and per-weekday package forecasts to each tract."""
    check_day_multipliers(cfg["day_of_week_multipliers"])
    out = tracts.copy()
    out["pkgs_per_day"] = out["households"] * packages_per_household(cfg)
    for day in DAYS:
        out[f"pkgs_{day}"] = out["pkgs_per_day"] * cfg["day_of_week_multipliers"][day]
    out["pkgs_per_km2"] = out["pkgs_per_day"] / out["area_km2"]
    return out


def summarize(demand: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    daily = demand["pkgs_per_day"].sum()
    rows = [("Average day", daily)]
    rows += [(day, demand[f"pkgs_{day}"].sum()) for day in DAYS]
    rows += [("Week", daily * 7), ("Holiday-peak average day", daily * cfg["holiday_peak_multiplier"])]
    return pd.DataFrame(rows, columns=["period", "packages"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", default="msp")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_assumptions()["demand"]
    tracts = gpd.read_file(prepared_dir(args.market) / "tracts.gpkg")
    demand = tract_demand(tracts, cfg)
    demand.to_file(prepared_dir(args.market) / "tract_demand.gpkg", driver="GPKG")

    print("\nMarket package forecast")
    print(summarize(demand, cfg).to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    by_county = demand.groupby("county")[["households", "pkgs_per_day"]].sum().sort_values("pkgs_per_day", ascending=False)
    print("\nBy county (average day)")
    print(by_county.to_string(float_format=lambda v: f"{v:,.0f}"))
    print(f"\nBusiest tract: {demand['pkgs_per_day'].max():.0f}/day; "
          f"median tract: {demand['pkgs_per_day'].median():.0f}/day; "
          f"tracts with zero demand: {(demand['pkgs_per_day'] == 0).sum()}")

    print_checks(run_checks({
        "msp_households": demand["households"].sum(),
        "mean_households_per_tract": demand["households"].mean(),
        "packages_per_household_per_day": cfg["packages_per_household_per_day"],
    }))


if __name__ == "__main__":
    main()
