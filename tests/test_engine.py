import numpy as np
import pytest

from engine.config import load_assumptions, prepared_dir
from engine.randomness import average_day, draw_weeks
from engine.routing import build_routes, packages_per_stop, tract_profiles
from engine.search import summarize
from engine.simulation import (Design, claim_routes, clock, design_grid, get_design, load_market, simulate_day,
                               simulate_week)

A = load_assumptions()
# The core mechanics tests run without peak flex and partner overflow, so they test the station on its own.
A_PLAIN = {**A, "peak_flex": {**A.get("peak_flex", {}), "enabled": False},
           "partner": {**A.get("partner", {}), "enabled": False}}
RT = A["routing"]
NORMAL = A["scenarios"]["normal"]


def test_clock():
    assert clock("D-1 22:00") == 22.0
    assert clock("D 06:30") == 30.5


def _toy(n_tracts=12, pkgs=40):
    rng = np.random.default_rng(0)
    density = np.where(np.arange(n_tracts) % 2, 3000.0, 500.0)
    area = rng.uniform(0.5, 20, n_tracts)
    prof = tract_profiles(density * area, density, area, RT)
    packages = np.full(n_tracts, pkgs)
    prof["packages_per_stop"] = packages_per_stop(packages, prof["stop_points"], RT["items_per_delivery"])
    angle = rng.uniform(-np.pi, np.pi, n_tracts)
    stem_min = rng.uniform(5, 60, n_tracts)
    return np.arange(n_tracts), packages, packages * 0.02, prof, angle, stem_min, stem_min * 1.2


def test_routes_conserve_packages_and_respect_limits():
    tracts, packages, volume, prof, angle, stem_min, stem_km = _toy()
    routes = build_routes(0, tracts, packages, volume, prof, angle, stem_min, stem_km, RT)
    assert sum(r.packages for r in routes) == packages.sum()
    for r in routes:
        assert r.packages <= RT["max_packages_per_route"]
        assert r.volume_m3 <= RT["car_cargo_m3"] + 1e-9
        assert r.totals(RT)["minutes"] <= RT["max_route_hours"] * 60 + 1e-6


def test_slower_roads_mean_more_routes():
    tracts, packages, volume, prof, angle, stem_min, stem_km = _toy(pkgs=200)
    fast = build_routes(0, tracts, packages, volume, prof, angle, stem_min, stem_km, RT)
    slow = build_routes(0, tracts, packages, volume, prof, angle, stem_min, stem_km, {**RT, "time_factor": 2.0})
    assert len(slow) >= len(fast)


def test_design_grid_is_hand_picked_and_unique():
    names = [d.name for d in design_grid(A)]
    assert len(names) == len(set(names))
    assert get_design("2x medium", A).hubs == [{"site": 1, "size": "medium"}, {"site": 2, "size": "medium"}]


def test_random_weeks_repeat_and_streams_differ():
    w1 = draw_weeks(3, 25, 4, A, "search")
    w2 = draw_weeks(3, 25, 4, A, "search")
    w3 = draw_weeks(3, 25, 4, A, "final")
    assert [d.volume_factor for d in w1[2]] == [d.volume_factor for d in w2[2]]
    assert w1[0][0].volume_factor != w3[0][0].volume_factor


needs_data = pytest.mark.skipif(not (prepared_dir("msp") / "drive_times.parquet").exists(),
                                reason="run engine.data_prep and engine.demand first")


@pytest.fixture(scope="module")
def market():
    return load_market("msp")


@pytest.fixture(scope="module")
def avg(market):
    return average_day(len(market.site_rank), len(A["schedule"]["linehaul_waves"]), A)


@needs_data
def test_day_is_repeatable_with_fixed_seed(market, avg):
    d = get_design("2x medium", A)
    m1 = simulate_day(market, d, "Mon", A, avg, NORMAL)["metrics"]
    m2 = simulate_day(market, d, "Mon", A, avg, NORMAL)["metrics"]
    assert m1 == m2


@needs_data
def test_every_package_is_accounted_for(market, avg):
    res = simulate_day(market, get_design("2x medium", A), "Tue", A, avg, NORMAL)
    m = res["metrics"]
    assert m["delivered"] + len(res["carry_out"]) == m["packages_due"] + m["carried_in"]


@needs_data
def test_designs_see_the_same_packages(market, avg):
    due = {simulate_day(market, d, "Wed", A, avg, NORMAL)["metrics"]["packages_due"] for d in design_grid(A)}
    assert len(due) == 1


@needs_data
def test_undersized_hub_misses_sort_cutoff(market, avg):
    peak = A["scenarios"]["holiday_peak"]
    m = simulate_day(market, Design("tiny", [{"site": 1, "size": "small"}]), "Mon", A_PLAIN, avg, peak)["metrics"]
    assert m["missed_sort_cutoff"] > 0
    assert m["on_time_rate"] < 0.9


@needs_data
def test_single_hub_outage_delivers_nothing_that_day(market, avg):
    res = simulate_day(market, get_design("1x large", A), "Wed", A_PLAIN, avg, NORMAL, outage_hub=0)
    assert res["metrics"]["delivered"] == 0
    assert len(res["carry_out"]) == res["metrics"]["packages_due"]


@needs_data
def test_late_packages_carry_into_the_next_day(market):
    week = draw_weeks(1, len(market.site_rank), 4, A)[0]
    w = simulate_week(market, get_design("1x large", A), week, A_PLAIN, {**NORMAL, "outage_day": "Wed"})
    assert w["on_time_rate"] < 0.9          # a whole day lost
    assert w["packages_due"] > 0


@needs_data
def test_summary_ranks_feasible_designs_first(market):
    import pandas as pd
    rows = [{"design": "a", "hubs": "x", "n_hubs": 1, "cost_per_package": 2.0, "on_time_rate": 0.90,
             "driver_net_per_hour": 25, "routes_unclaimed_share": 0.1, "max_sort_utilization": 0.5, "km_per_package": 0.6, "packages_per_stop": 1.2, "cost_driver_surge": 0.1, "cost_sort_overtime": 0.0, "routes_surged_share": 0.3, "pickup_slot_delay_minutes": 30},
            {"design": "b", "hubs": "y", "n_hubs": 2, "cost_per_package": 3.0, "on_time_rate": 0.99,
             "driver_net_per_hour": 25, "routes_unclaimed_share": 0.0, "max_sort_utilization": 0.5, "km_per_package": 0.6, "packages_per_stop": 1.2, "cost_driver_surge": 0.1, "cost_sort_overtime": 0.0, "routes_surged_share": 0.3, "pickup_slot_delay_minutes": 30}]
    s = summarize(pd.DataFrame(rows * 5), A)
    assert s.index[0] == "b" and s.loc["b", "feasible"] and not s.loc["a", "feasible"]


def test_when_nothing_is_feasible_cheapest_near_best_service_wins():
    import pandas as pd
    base = {"n_hubs": 1, "driver_net_per_hour": 25, "routes_unclaimed_share": 0.0, "max_sort_utilization": 0.5, "km_per_package": 0.6, "packages_per_stop": 1.2, "cost_driver_surge": 0.1, "cost_sort_overtime": 0.0, "routes_surged_share": 0.3, "pickup_slot_delay_minutes": 30}
    rows = []
    for design, cost, hits in [("pricey", 3.0, 8), ("cheap", 2.5, 7), ("poor", 2.0, 2)]:
        for i in range(10):
            rows.append({**base, "design": design, "hubs": design, "cost_per_package": cost,
                         "on_time_rate": 0.96 if i < hits else 0.90})
    s = summarize(pd.DataFrame(rows), {**A, "objective": {**A["objective"], "infeasible_service_tolerance": 0.10,
                                                         "on_time_target": 0.95}})
    assert not s["feasible"].any()
    assert s.index.tolist() == ["cheap", "pricey", "poor"]


def test_more_packages_per_stop_on_busier_days():
    points = np.array([100.0, 100.0])
    pps = packages_per_stop(np.array([20, 80]), points, 1.15)
    assert pps[1] > pps[0] >= 1.15


def _route(base_pay=60.0, hours=3.0, car_cost=10.0):
    return {"base_pay": base_pay, "pay": base_pay, "surge": 1.0, "car_cost": car_cost, "driver_hours": hours,
            "status": "unclaimed", "delay": 0.0}


def test_surge_clears_unattractive_routes():
    import dataclasses
    avg = average_day(25, 4, A)
    rows = [_route(base_pay=35.0) for _ in range(200)]         # poor pay per hour at base
    claim_routes(rows, dataclasses.replace(avg, no_show_rate=0.0), A["drivers"], A["uncertainty"],
                 np.random.default_rng(0))
    assert all(r["status"] == "delivered" for r in rows) or \
        sum(r["status"] == "unclaimed" for r in rows) <= 2
    assert np.mean([r["surge"] > 1 for r in rows]) > 0.5


def test_extreme_weather_leaves_some_routes_unclaimed():
    import dataclasses
    storm = dataclasses.replace(average_day(25, 4, A), extreme=True, no_show_rate=0.0)
    rows = [_route(base_pay=35.0) for _ in range(200)]
    claim_routes(rows, storm, A["drivers"], A["uncertainty"], np.random.default_rng(0))
    assert sum(r["status"] == "unclaimed" for r in rows) > 0
    assert max(r["surge"] for r in rows) <= A["uncertainty"]["extreme_surge_max"] + 1e-9


@needs_data
def test_peak_overloads_a_single_hub_sort(market, avg):
    peak = A["scenarios"]["holiday_peak"]
    m = simulate_day(market, get_design("1x medium", A), "Mon", A_PLAIN, avg, peak)["metrics"]
    assert m["overtime_sorted"] > 0 and m["missed_sort_cutoff"] > 1000
    assert m["cost_sort_overtime"] > 0


@needs_data
def test_peak_drives_fewer_km_per_package(market, avg):
    d = get_design("4x small", A)
    normal = simulate_day(market, d, "Mon", A, avg, NORMAL)["metrics"]
    peak = simulate_day(market, d, "Mon", A, avg, A["scenarios"]["holiday_peak"])["metrics"]
    assert peak["km"] / peak["delivered"] < normal["km"] / normal["delivered"]


@needs_data
def test_packages_sorted_after_cutoff_are_not_sorted_twice(market, avg):
    peak = A["scenarios"]["holiday_peak"]
    res = simulate_day(market, get_design("1x medium", A), "Mon", A_PLAIN, avg, peak)
    carry = res["carry_out"]
    assert len(carry) > 0 and carry["sorted"].all()       # open hub: late arrivals still got sorted


def test_hub_cost_includes_more_than_rent_and_staff():
    from engine.simulation import hub_fixed_cost_per_day, loading_cost
    h = A["hubs"]
    small = h["sizes"]["small"]
    rent_staff = small["square_feet"] * h["lease_per_sqft_per_year"] / 365 \
        + small["fixed_staff_hours_per_day"] * h["fixed_staff_wage_per_hour"]
    assert hub_fixed_cost_per_day(small, h) > rent_staff
    assert loading_cost(100, 0.5, h) > loading_cost(100, 0.0, h) > 0


def test_storm_weeks_are_left_out_of_the_floor_and_reported():
    import pandas as pd
    base = {"design": "a", "hubs": "x", "n_hubs": 1, "cost_per_package": 2.0, "driver_net_per_hour": 25,
            "routes_unclaimed_share": 0.0, "max_sort_utilization": 0.5, "km_per_package": 0.6,
            "packages_per_stop": 1.2, "cost_driver_surge": 0.1, "cost_sort_overtime": 0.0,
            "routes_surged_share": 0.3, "pickup_slot_delay_minutes": 30}
    rows = [{**base, "on_time_rate": 0.99, "extreme_weather_days": 0} for _ in range(18)] + \
           [{**base, "on_time_rate": 0.93, "extreme_weather_days": 1} for _ in range(2)]
    on = summarize(pd.DataFrame(rows), {**A, "objective": {**A["objective"], "exclude_extreme_weather_weeks": True}})
    off = summarize(pd.DataFrame(rows), {**A, "objective": {**A["objective"], "exclude_extreme_weather_weeks": False}})
    assert on.loc["a", "weeks_meeting_service"] == 1.0 and on.loc["a", "feasible"]
    assert off.loc["a", "weeks_meeting_service"] == 0.9 and not off.loc["a", "feasible"]
    assert on.loc["a", "storm_week_share"] == 0.1 and on.loc["a", "storm_on_time_p50"] == 0.93


def test_bigger_routes_take_longer_to_load_and_cost_more_loader_time():
    from engine.routing import load_minutes
    from engine.simulation import loading_cost
    assert load_minutes(178, RT) > load_minutes(46, RT) > RT["hub_load_minutes"]
    assert loading_cost(1, 0.0, A["hubs"], packages=178) > loading_cost(1, 0.0, A["hubs"], packages=46)


def test_vans_cost_more_to_run_than_sedans():
    from engine.simulation import vehicle_cost_per_km, CUFT_PER_M3
    sedan = {**RT, "car_cargo_m3": 30 / CUFT_PER_M3}
    van = {**RT, "car_cargo_m3": 250 / CUFT_PER_M3}
    assert vehicle_cost_per_km(A["drivers"], sedan) == pytest.approx(A["drivers"]["vehicle_cost_per_km"])
    assert vehicle_cost_per_km(A["drivers"], van) == pytest.approx(A["drivers"]["vehicle_cost_per_km_van"])


def test_fewer_dashers_have_big_vehicles():
    from engine.simulation import vehicle_availability, CUFT_PER_M3
    at = lambda cuft: vehicle_availability(A["drivers"], {**RT, "car_cargo_m3": cuft / CUFT_PER_M3})  # noqa: E731
    assert at(30) == 1.0 and at(20) == 1.0
    assert 1.0 > at(70) > at(150) > at(250) > 0


def test_routes_below_the_pay_floor_get_their_base_pay_raised():
    from engine.routing import Route
    from engine.simulation import price_route
    r = Route(hub=0, packages=40, stops=30.0, area_km2=5.0, service_min=60.0, local_hours_per_km_x_stops=30 / 25,
              stem_min_x_stops=30 * 30.0, stem_km_x_stops=30 * 25.0)
    drv = {**A["drivers"], "pay_base_per_route": 0.0, "pay_per_stop": 0.0, "pay_per_km": 0.0}
    floor = A["objective"]["driver_net_per_hour_floor"]
    raised = price_route(r, RT, drv, floor_per_hour=floor)
    assert raised["floor_topup"] > 0
    assert (raised["base_pay"] - raised["car_cost"]) / raised["driver_hours"] == pytest.approx(floor)
    assert price_route(r, RT, drv)["floor_topup"] == 0.0



@needs_data
def test_partner_carrier_takes_overflow_and_peak_flex_adds_capacity(market, avg):
    peak = A["scenarios"]["holiday_peak"]
    d = get_design("1x medium", A)
    plain = simulate_day(market, d, "Mon", A_PLAIN, avg, peak)["metrics"]
    flex = simulate_day(market, d, "Mon", {**A_PLAIN, "peak_flex": A["peak_flex"]}, avg, peak)["metrics"]
    both = simulate_day(market, d, "Mon", {**A, "partner": {**A["partner"], "enabled": True}}, avg, peak)["metrics"]
    assert flex["missed_sort_cutoff"] < plain["missed_sort_cutoff"]          # more sort capacity at peak
    assert flex["cost_peak_setup"] > 0 and plain["cost_peak_setup"] == 0
    assert both["partner_packages"] > 0 and both["cost_partner"] > 0           # overflow goes to the partner
    assert both["partner_packages"] <= A["partner"]["max_share_of_day"] * both["packages_due"] + 1
    assert both["on_time_rate"] > flex["on_time_rate"]


def test_bigger_stations_sort_more_packages_per_labor_hour():
    sizes = A["hubs"]["sizes"]
    assert (sizes["small"]["sort_packages_per_labor_hour"] < sizes["medium"]["sort_packages_per_labor_hour"]
            < sizes["large"]["sort_packages_per_labor_hour"])


def test_peak_weeks_use_their_own_targets():
    from engine.search import objective_for
    assert objective_for(A, "holiday_peak")["on_time_target"] == A["objective"]["peak_targets"]["on_time_target"]
    assert objective_for(A, "normal")["on_time_target"] == A["objective"]["on_time_target"]


def test_station_costs_include_fitout_by_size_and_building_charges():
    from engine.simulation import hub_fixed_cost_per_day
    h = A["hubs"]
    small = h["sizes"]["small"]
    without = {**h, "nnn_per_sqft_per_year": 0.0}
    assert hub_fixed_cost_per_day(small, h) > hub_fixed_cost_per_day(small, without)
    assert h["sizes"]["large"]["fitout_per_sqft"] > h["sizes"]["small"]["fitout_per_sqft"]


def _frame(rows):
    """Minimal per-scenario summary: design -> (cost, OTD typical, share of weeks meeting the floor, feasible)."""
    import pandas as pd
    return pd.DataFrame([{"design": d, "hubs": d, "cost_p50": c, "on_time_p50": ot, "weeks_meeting_service": w,
                          "feasible": f, "storm_on_time_p50": ot} for d, c, ot, w, f in rows]).set_index("design")


def test_year_round_pick_puts_otd_before_cost():
    from engine.search import year_round_table
    a = {**A, "weeks_per_year": {"normal": 1}}
    # Nothing meets the targets: the closer layout ranks first even though it costs more.
    t = year_round_table({"normal": _frame([("cheap", 2.0, 0.90, 0.40, False), ("close", 3.0, 0.975, 0.93, False)])}, a)
    assert t.index[0] == "close" and not t["meets_all"].iloc[0]
    # One layout meets the targets: it wins over a cheaper one that misses.
    t = year_round_table({"normal": _frame([("cheap", 2.0, 0.90, 0.40, False), ("ok", 3.0, 0.99, 0.99, True)])}, a)
    assert t.index[0] == "ok" and t["meets_all"].iloc[0]


def test_larger_networks_come_after_the_standard_list():
    from engine.simulation import design_tiers
    tiers = design_tiers(A, n_sites=25)
    assert [d.name for d in tiers[0]] == [d.name for d in design_grid(A)]
    assert min(len(d.hubs) for d in tiers[1]) > max(len(d.hubs) for d in tiers[0])
    assert all(len(d.hubs) <= 25 for t in tiers for d in t)
    assert len(get_design("8x medium", A).hubs) == 8
