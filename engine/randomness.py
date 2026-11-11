"""Random weeks.

Each week draws its own volume, truck delays, late shipper handoffs, driver no-shows,
drive-time slowdowns, rare extreme-weather days and hub sort speed. Draws are made per candidate site (not per
hub in a design), so every design sees exactly the same weeks: common random numbers.
Search and final check use separate streams, so the final check is on fresh weeks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STREAMS = {"search": 1, "final": 2}


@dataclass
class DayDraws:
    volume_factor: float          # actual vs forecast volume
    truck_delay_h: np.ndarray     # candidate sites x linehaul waves, hours late
    handoff_share: float          # share of each wave's packages that miss their truck
    sort_factor: np.ndarray       # per candidate site, sort speed vs plan
    time_factor: float            # route time multiplier (weather, traffic)
    no_show_rate: float
    extreme: bool                 # extreme weather: fewer drivers, slow roads, capped surge
    seed: tuple                   # for package and claim draws that day


def draw_weeks(n_weeks: int, n_sites: int, n_waves: int, a: dict, stream: str = "search",
               start: int = 0) -> list[list[DayDraws]]:
    """Weeks start .. start + n_weeks - 1 of a stream. Each week has its own seed, so any slice of weeks
    is identical to the same weeks drawn as part of a longer run."""
    u = a["uncertainty"]
    weeks = []
    for w in range(start, start + n_weeks):
        rng = np.random.default_rng([a["random_seed"], STREAMS[stream], w])
        week_volume = rng.lognormal(0, u["volume_week_sigma"])
        no_show = rng.uniform(*u["no_show_rate_range"])
        days = []
        for d in range(7):
            late = rng.random((n_sites, n_waves)) < u["truck_late_prob"]
            slow = rng.random() < u["slow_day_prob"]
            extreme = rng.random() < u["extreme_weather_prob"]
            slow_factor = u["extreme_time_factor"] if extreme else (u["slow_day_time_factor"] if slow else 1.0)
            days.append(DayDraws(
                volume_factor=week_volume * rng.lognormal(0, u["volume_day_sigma"]),
                truck_delay_h=np.where(late, rng.exponential(u["truck_late_mean_hours"], (n_sites, n_waves)), 0.0),
                handoff_share=rng.uniform(0, u["late_handoff_share_max"]),
                sort_factor=np.clip(rng.normal(1, u["sort_rate_sigma"], n_sites), 0.7, 1.15),
                time_factor=slow_factor * rng.lognormal(0, u["drive_time_sigma"]),
                no_show_rate=no_show,
                extreme=bool(extreme),
                seed=(a["random_seed"], STREAMS[stream], w, d),
            ))
        weeks.append(days)
    return weeks


def average_day(n_sites: int, n_waves: int, a: dict) -> DayDraws:
    """Everything at its expected value: used by the fast average-day screen."""
    return DayDraws(1.0, np.zeros((n_sites, n_waves)), a["uncertainty"]["late_handoff_share_max"] / 2,
                    np.ones(n_sites), 1.0, float(np.mean(a["uncertainty"]["no_show_rate_range"])), False,
                    (a["random_seed"], 0, 0, 0))
