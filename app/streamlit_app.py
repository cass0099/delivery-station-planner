"""Delivery Station Planner: change the assumptions, rerun the simulation, compare with the defaults.

Run locally:  streamlit run app/streamlit_app.py
Loads small prepared files only; live runs use the fast route estimate (no road routing).
"""
from __future__ import annotations

import copy
import json
import math
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import altair as alt
import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
import yaml
from branca.element import MacroElement, Template
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _reload_if_engine_changed() -> None:
    """After a deploy, Streamlit reruns the new app script but keeps the engine modules (and anything cached) from the
    old code in memory, so new app code can call old engine code. When any engine file or assumptions.yaml has changed
    since the last run in this process, drop the engine modules so they import fresh, and clear the caches."""
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "engine").glob("*.py")) + [root / "assumptions.yaml"]
    stamp = tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files)
    holder = sys.modules.get("_engine_stamp")
    if holder is None:
        import types
        holder = sys.modules["_engine_stamp"] = types.ModuleType("_engine_stamp")
    if getattr(holder, "stamp", stamp) != stamp:
        # Reload in place, in import order, so every module object (and anything holding it, such as a run already
        # in progress in another session) sees the new code rather than a copy that no longer matches sys.modules.
        import importlib
        order = ["config", "randomness", "demand", "routing", "simulation", "checks", "search", "live"]
        loaded = [n for n in sys.modules if n.startswith("engine.")]
        for name in [f"engine.{m}" for m in order if f"engine.{m}" in loaded] + \
                [n for n in loaded if n.split(".", 1)[1] not in order]:
            importlib.reload(sys.modules[name])
        st.cache_data.clear()
        st.cache_resource.clear()
    holder.stamp = stamp


_reload_if_engine_changed()
from engine.demand import DAYS
from engine.search import objective_for
from engine.live import estimate_seconds, load_baseline, planned_weeks, quick_estimate, run as live_run, screen_all, settings_hash
from engine.simulation import CUFT_PER_M3 as _CUFT, clock, get_design, vehicle_availability, vehicle_cost_per_km

ROOT = Path(__file__).resolve().parents[1]
MARKET = "msp"
FONT = "Figtree"
# Neutrals from DoorDash's public web styles; no DoorDash red and no DoorDash typeface.
INK, BODY, MUTED, LINE, PANEL = "#191919", "#434343", "#6b6b6b", "#e7e7e7", "#f6f6f6"
# Chart colors: DoorDash blue plus three hues checked for color-vision deficiency (adjacent and all pairs pass).
BLUE, CORAL, TEAL, GOLD = "#4969f5", "#c8664e", "#3accc5", "#dcab53"
ORANGE, GREEN = CORAL, TEAL
NAVY = "#233069"
# Red accents: #eb1700 for buttons and red text (4.5:1 with white), brighter #ff3008 for lines and handles.
RED, RED_BRIGHT, RED_DARK = "#eb1700", "#ff3008", "#c81400"
HUB_COLORS = [BLUE, CORAL, TEAL, GOLD]
WEEK_COLORS = {"normal": BLUE, "holiday_peak": CORAL, "hub_outage": TEAL}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
         10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen"}
SCEN = {"normal": "Normal week", "holiday_peak": "Holiday peak", "hub_outage": "Station outage"}
# Esri light gray canvas: clean, needs no API key (CARTO's basemaps now require one). Labels sit in a separate
# layer drawn above the shaded tracts so place names stay readable.
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
TILES, TILE_LABELS = ESRI.format("World_Light_Gray_Base"), ESRI.format("World_Light_Gray_Reference")
TILES_ATTR = "Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors"


class _PhoneStill(MacroElement):
    """On touch screens, one finger scrolls the page instead of dragging the map; the zoom buttons still work."""
    _template = Template("""{% macro script(this, kwargs) %}
    if (L.Browser.mobile) { {{ this._parent.get_name() }}.dragging.disable();
      if ({{ this._parent.get_name() }}.tap) { {{ this._parent.get_name() }}.tap.disable(); } }
    {% endmacro %}""")


def add_basemap(m: folium.Map) -> None:
    folium.TileLayer(TILES, attr=TILES_ATTR, max_zoom=16).add_to(m)
    m.add_child(_PhoneStill())


def add_labels(m: folium.Map) -> None:
    pane = folium.map.CustomPane("labels", z_index=650, pointer_events=False)
    m.add_child(pane)
    folium.TileLayer(TILE_LABELS, attr=TILES_ATTR, max_zoom=16, pane="labels", overlay=True).add_to(m)

st.set_page_config(page_title="Delivery Station Planner", page_icon="📦", layout="wide",
                   initial_sidebar_state="collapsed")

st.html(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&display=swap');

/* ---------- base ---------- */
html, body, .stApp {{ background: #fff; }}
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{
  font-family: '{FONT}', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
  font-variant-ligatures: no-common-ligatures; }}
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stSidebarCollapsedControl"],
[data-testid="stDecoration"], [data-testid="stToolbar"] {{ display: none !important; }}
.block-container {{ max-width: 1224px; margin: 0 auto; padding: 28px clamp(16px, 4vw, 40px) 80px; }}
[data-testid="stVerticalBlock"] {{ gap: 0.75rem; }}

/* ---------- header and step navigation ---------- */
.masthead {{ display: flex; align-items: center; gap: 14px; padding-bottom: 18px; }}
.masthead svg {{ flex: none; }}
.app-name {{ font-size: 20px; line-height: 1.2; font-weight: 800; color: {INK}; letter-spacing: -0.015em; }}
.app-sub {{ font-size: 15px; line-height: 1.45; color: {MUTED}; margin-top: 3px; }}
/* Groups never split and labels never clip: if the row is too narrow, a whole group moves to the next line. */
.st-key-nav {{ flex-wrap: wrap !important; overflow: visible; gap: 10px 12px !important;
  /* Frozen at the top of the window while the page scrolls underneath. */
  position: sticky; top: 0; z-index: 1000; background: #fff; padding: 10px 0 !important;
  box-shadow: 0 1px 0 {LINE}; }}
.st-key-nav_scroll, .st-key-nav > :has(> .st-key-nav_scroll) {{ display: contents !important; }}
.st-key-nav_about, .st-key-nav_assume, .st-key-nav_run {{ flex-wrap: nowrap !important; flex: 0 0 auto !important;
  gap: 2px !important; align-items: center; }}
.st-key-nav_assume {{ background: {PANEL}; border-radius: 999px; padding: 4px 6px 4px 16px !important; }}
.st-key-nav_about {{ padding-right: 4px; }}
.st-key-nav_run {{ padding-left: 4px; }}
.st-key-nav [data-testid="stElementContainer"], .st-key-nav .stButton {{ width: auto !important; flex: 0 0 auto; }}
.nav-group {{ font-size: 13px; font-weight: 700; color: {INK}; white-space: nowrap; padding-right: 10px; margin-right: 4px;
  border-right: 1px solid #dcdcdc; line-height: 1.6; display: inline-block; }}
/* Menu items are pills: grey on hover (clearly clickable); the current page is a solid black pill. */
.st-key-nav button {{ white-space: nowrap !important; font-size: 14px !important; font-weight: 600 !important;
  color: {BODY} !important; text-decoration: none !important; padding: 7px 12px !important;
  border-radius: 999px !important; cursor: pointer; transition: background .12s ease, color .12s ease !important; }}
.st-key-nav button:hover {{ background: #ececec !important; color: {INK} !important; text-decoration: none !important; }}
.st-key-nav_assume button:hover {{ background: #e2e2e2 !important; }}
.st-key-nav_reset button {{ color: {RED} !important; }}
.st-key-nav_reset button:disabled {{ color: #a6a6a6 !important; }}
.st-key-nav button:disabled {{ color: #c4c4c4 !important; background: none !important; cursor: default; }}
/* No grey-out fade while the next page renders, and no element fade-in: pages swap instantly. */
[data-stale="true"], .stale-element {{ opacity: 1 !important; transition: none !important; }}
[data-testid="stElementContainer"], [data-testid="stVerticalBlock"] {{ animation: none !important;
  transition: none !important; }}
.step-on {{ font-weight: 700; color: {INK}; border-bottom: 2px solid {INK}; padding-bottom: 4px; display: inline-block;
           line-height: 1.6; }}
.step-off {{ color: #b5b5b5; line-height: 1.6; display: inline-block; }}
hr.nav {{ border: 0; border-top: 1px solid {LINE}; margin: 4px 0 32px; }}
.st-key-nav + div hr.nav, hr.nav.under-menu {{ border-top-color: transparent; margin-top: 0; }}

/* ---------- tabs (Run page) ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 26px; border-bottom: 1px solid {LINE}; }}
[data-testid="stTabs"] [data-baseweb="tab"] {{ padding: 10px 2px; height: auto; background: none; }}
[data-testid="stTabs"] [data-baseweb="tab"] p {{ font-size: 16px !important; font-weight: 600; color: {MUTED}; }}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p {{ color: {INK}; font-weight: 700; }}
[data-testid="stTabs"] [data-baseweb="tab"]:hover p {{ color: {INK}; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: {RED_BRIGHT}; height: 3px; }}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none; }}
.tab-body {{ padding-top: 16px; }}
[class*="st-key-runrow_"] {{ gap: 18px !important; margin-top: 6px; }}
[class*="st-key-runrow_"] [data-testid="stElementContainer"], [class*="st-key-runrow_"] .stButton {{ width: auto !important;
  flex: 0 0 auto; }}

/* ---------- scorecard (Results page) ---------- */
.score-wrap {{ overflow-x: auto; margin-top: 8px; }}
table.score {{ border-collapse: collapse; width: 100%; font-size: 14.5px; min-width: 860px; }}
table.score th {{ text-align: left; font-size: 13px; font-weight: 600; color: {MUTED}; padding: 8px 12px;
  border-bottom: 1px solid {LINE}; white-space: nowrap; }}
table.score td {{ padding: 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; color: {BODY};
  line-height: 1.4; }}
table.score td.name {{ font-weight: 600; color: {INK}; white-space: nowrap; }}
table.score td.cost {{ white-space: nowrap; }}
table.score td.cost span {{ display: inline-block; width: 52px; font-variant-numeric: tabular-nums; color: {INK};
  font-weight: 600; }}
table.score td.ok {{ color: {INK}; font-weight: 700; font-size: 16px; }}
table.score td.no {{ font-size: 13px; color: {BODY}; }}
table.score td.no b {{ color: {RED}; font-weight: 600; }}
table.score td.no .x {{ color: {RED}; font-weight: 700; margin-right: 6px; }}
table.score td.na {{ color: {MUTED}; font-size: 13px; }}
table.score td.verdict {{ font-size: 13.5px; }}
table.score.floor {{ min-width: 560px; max-width: 820px; }}
table.score th .s {{ font-weight: 400; font-size: 12px; color: #9a9a9a; white-space: nowrap; }}
table.score.floor td {{ font-variant-numeric: tabular-nums; color: {INK}; }}
table.score.floor td.low {{ color: {RED}; font-weight: 700; }}
table.score.floor td .yes {{ color: {INK}; margin-left: 4px; }}
table.score.floor td .x {{ color: {RED}; margin-left: 4px; }}
table.score td.topup {{ color: {INK}; font-variant-numeric: tabular-nums; white-space: nowrap; }}
table.score tr.pick td {{ background: #eef2fe; }}
table.score tr.pick td.name, table.score tr.pick td.verdict b {{ color: {BLUE}; }}
table.score tr.out td.name, table.score tr.out td.cost span {{ color: {MUTED}; }}

/* ---------- buttons ---------- */
.st-key-pager {{ margin-top: 28px; padding-top: 22px; border-top: 1px solid {LINE}; gap: 12px !important;
  flex-wrap: wrap !important; }}
.st-key-pager [data-testid="stElementContainer"], .st-key-pager .stButton {{ width: auto !important; flex: 0 0 auto; }}
.stButton > button {{ border-radius: 999px; font-weight: 600; font-size: 15px; padding: 10px 22px; min-height: 44px;
  white-space: nowrap; box-shadow: none; transition: background .15s ease; }}
.stButton > button[kind="primary"] {{ background: {RED}; border: 1px solid {RED}; color: #fff; }}
.stButton > button[kind="primary"]:hover {{ background: {RED_DARK}; border-color: {RED_DARK}; color: #fff; }}
.stButton > button[kind="primary"]:focus-visible {{ outline: 2px solid {RED_BRIGHT}; outline-offset: 2px; }}
.stButton > button[kind="secondary"] {{ background: {PANEL}; border: 1px solid {PANEL}; color: {INK}; }}
.stButton > button[kind="secondary"]:hover {{ background: #ececec; border-color: #ececec; color: {INK}; }}
.stButton > button[kind="tertiary"] {{ padding: 0; min-height: 0; border: 0; background: none; color: {MUTED};
  font-weight: 500; font-size: 14px; }}
.stButton > button[kind="tertiary"]:hover {{ color: {RED}; text-decoration: underline; text-underline-offset: 3px; }}
.stButton > button p {{ font-size: inherit !important; font-weight: inherit !important; white-space: nowrap; }}

/* ---------- sliders, toggles, segmented controls ---------- */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{ font-size: 14.5px !important;
  font-weight: 600 !important; color: {INK} !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
[data-testid="stSlider"] {{ padding-bottom: 2px; }}
[data-testid="stSlider"] [role="slider"] {{ background: #fff !important; border: 2px solid {RED_BRIGHT} !important;
  width: 18px !important; height: 18px !important; box-shadow: 0 1px 3px rgba(0,0,0,.12) !important; }}
[data-testid="stSliderThumbValue"] {{ font-weight: 700 !important; color: {INK} !important; font-size: 14px !important;
  font-variant-numeric: tabular-nums; }}
[data-testid="stSliderTickBar"] {{ display: none; }}
[data-testid="stSegmentedControl"] button {{ border-radius: 999px !important; font-weight: 600; }}
.stProgress > div > div > div > div {{ background: {RED_BRIGHT} !important; }}
[data-testid="stSpinner"] {{ font-size: 15px; color: {BODY}; padding: 10px 0; }}
[data-testid="stSpinner"] i {{ border-top-color: {RED_BRIGHT} !important; }}

/* ---------- type ---------- */
.q {{ font-size: 32px; line-height: 1.15; font-weight: 800; color: {INK}; letter-spacing: -0.02em; margin: 0 0 12px; }}
.note {{ font-size: 16px; line-height: 1.6; color: {BODY}; margin: 0 0 10px; max-width: 62ch; }}
.default {{ font-size: 13px; line-height: 1.45; color: {MUTED}; margin: -6px 0 18px; }}
.default b {{ color: {RED}; font-weight: 600; }}
.default.off {{ color: #a8a8a8; }}
.band {{ margin: -14px 0 10px; }}
.band-track {{ position: relative; height: 6px; margin: 0 9px; border-radius: 3px; background: #f0f0f0; }}
.band-range {{ position: absolute; top: 0; height: 6px; border-radius: 3px; background: #b6c5f8; }}
.band-mark {{ position: absolute; top: -3px; width: 2px; height: 12px; margin-left: -1px; background: #233069; }}
.band-ends {{ display: flex; justify-content: space-between; font-size: 11.5px; color: #a8a8a8; margin: 3px 9px 0; }}
.band-note {{ font-size: 12.5px; color: {MUTED}; margin: 6px 0 0; }}
.tag {{ display: inline-block; font-size: 11.5px; font-weight: 700; letter-spacing: .02em; padding: 1px 8px;
  border-radius: 999px; margin-right: 7px; vertical-align: 1px; line-height: 1.6; }}
.tag.fixed {{ background: #ececec; color: {BODY}; }}
.tag.vary {{ background: #e4ebf7; color: #233069; }}
.chart-title {{ font-size: 16px; font-weight: 700; color: {INK}; margin: 30px 0 2px; }}
.chart-note {{ font-size: 14px; color: {MUTED}; margin: 0 0 8px; line-height: 1.5; max-width: 90ch; }}
.result {{ font-size: 36px; line-height: 1.12; font-weight: 800; color: {INK}; letter-spacing: -0.025em; margin: 4px 0 10px; }}
.result-sub {{ font-size: 16px; color: {BODY}; margin: 0 0 26px; line-height: 1.6; max-width: 90ch; }}
.stepno {{ font-size: 14px; font-weight: 700; color: {RED}; margin: 0 0 6px; }}
.accent {{ color: {RED}; }}
.ask {{ font-size: 18px; font-weight: 600; color: {INK}; margin: 0 0 8px; }}
.section {{ font-size: 20px; font-weight: 800; color: {INK}; margin: 8px 0 12px; letter-spacing: -0.01em; }}
.lead-in {{ font-size: 15px; line-height: 1.55; color: {BODY}; margin: 0 0 22px; padding: 14px 16px; background: {PANEL};
           border-radius: 12px; }}
.uses {{ font-size: 15px; line-height: 1.6; color: {BODY}; max-width: 62ch; margin: 18px 0 8px; padding: 14px 16px;
        background: {PANEL}; border-radius: 12px; }}
.uses b {{ color: {INK}; font-weight: 700; }}

/* ---------- tables and lists ---------- */
table.cmp {{ border-collapse: collapse; width: 100%; font-size: 15px; }}
table.cmp th {{ text-align: right; font-weight: 600; color: {MUTED}; font-size: 13px; padding: 8px 12px;
               border-bottom: 1px solid {LINE}; white-space: nowrap; }}
table.cmp th:first-child, table.cmp td:first-child {{ text-align: left; padding-left: 0; }}
table.cmp td {{ text-align: right; padding: 12px; border-bottom: 1px solid #f0f0f0; color: {INK};
               font-variant-numeric: tabular-nums; }}
table.cmp td:first-child {{ color: {BODY}; }}
table.cmp tr.lead td {{ font-weight: 700; }}
.keyrow {{ font-size: 14px; color: {BODY}; margin: 2px 0 6px; }}
.keyrow i {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 6px 0 0; vertical-align: -1px; }}
.keyrow span {{ margin-right: 18px; white-space: nowrap; }}
.intro p {{ font-size: 17px; line-height: 1.6; color: {BODY}; max-width: 68ch; margin: 0 0 16px; }}
.intro p.muted {{ font-size: 14px; color: {MUTED}; }}
.intro dl {{ display: grid; grid-template-columns: 148px 1fr; gap: 0; margin: 4px 0 18px; padding: 6px 22px;
            background: {PANEL}; border-radius: 16px; font-size: 15.5px; line-height: 1.55; }}
.intro dt, .intro dd {{ padding: 14px 0; border-top: 1px solid #e2e2e2; }}
.intro dt {{ padding-right: 20px; }}
.intro dt:first-of-type, .intro dd:first-of-type {{ border-top: 0; }}
.intro dt {{ font-weight: 700; color: {INK}; }}
.intro dd {{ margin: 0; color: {BODY}; }}
@media (max-width: 700px) {{ .intro dl {{ grid-template-columns: 1fr; }} .intro dd {{ border-top: 0; padding-top: 0; }}
  .q {{ font-size: 27px; }} .result {{ font-size: 28px; }} }}
.flow {{ position: relative; margin: 4px 0 0 6px; }}
.flow-step {{ position: relative; padding: 0 0 18px 26px; border-left: 2px solid {LINE}; }}
.flow-step:last-child {{ border-left-color: transparent; padding-bottom: 0; }}
.flow-step::before {{ content: ""; position: absolute; left: -7px; top: 3px; width: 12px; height: 12px;
  border-radius: 50%; background: #fff; border: 2px solid {RED_BRIGHT}; }}
.flow-step:first-child::before {{ background: {RED_BRIGHT}; }}
.flow-title {{ font-size: 15.5px; font-weight: 700; color: {INK}; line-height: 1.3; margin: 0 0 4px; }}
.flow-step p {{ font-size: 15px; line-height: 1.6; color: {BODY}; margin: 0; max-width: 62ch; }}
ol.how {{ margin: 0; padding-left: 20px; font-size: 15.5px; line-height: 1.6; color: {BODY}; max-width: 70ch; }}
ol.how li {{ margin: 0 0 12px; padding-left: 4px; }}
table.jd {{ border-collapse: collapse; width: 100%; font-size: 14.5px; }}
table.jd th {{ text-align: left; font-weight: 600; color: {MUTED}; font-size: 13px; padding: 8px 12px 8px 0;
              border-bottom: 1px solid {LINE}; }}
table.jd td {{ padding: 12px 12px 12px 0; border-bottom: 1px solid #f0f0f0; color: {BODY}; vertical-align: top;
              line-height: 1.5; }}
table.jd td:first-child {{ color: {INK}; font-weight: 600; width: 34%; padding-right: 32px; }}
.method h3 {{ font-size: 18px; font-weight: 700; color: {INK}; margin: 28px 0 8px; padding: 0; }}
.method h3:first-child {{ margin-top: 0; }}
.method p {{ font-size: 15.5px; line-height: 1.65; color: {BODY}; margin: 0 0 12px; max-width: 70ch; }}
.method .src {{ font-size: 13.5px; color: {MUTED}; }}

/* ---------- phones (desktop layout above is unchanged) ---------- */
@media (max-width: 640px) {{
  .block-container {{ padding: 14px 16px 64px !important; }}
  .masthead {{ gap: 10px; padding-bottom: 10px; }}
  .masthead svg {{ width: 28px; height: 28px; }}
  .app-name {{ font-size: 17px; }}
  .app-sub {{ font-size: 13px; line-height: 1.4; }}
  /* Menu: the explanation pages and assumption steps form one swipeable row with an arrow at its right edge;
     Run simulation, Results and Reset sit on their own row underneath. */
  /* A strict two-row grid: row 1 is the swipe row, row 2 the run row. Anything else Streamlit puts inside the menu
     is hidden, so nothing can open a gap between the rows. */
  .st-key-nav {{ display: grid !important; grid-template-columns: minmax(0, 1fr); grid-auto-rows: auto;
    row-gap: 4px !important; column-gap: 0 !important; align-content: start !important; align-items: start !important;
    justify-items: start; padding: 8px 0 6px !important; height: auto !important; min-height: 0 !important; }}
  /* Each row sits inside a wrapper element: size the wrappers to their row and nothing more. */
  .stApp .st-key-nav > :is(.st-key-nav_scroll, :has(.st-key-nav_scroll)) {{ justify-self: stretch; width: auto !important;
    min-width: 0 !important; max-width: none !important; }}
  .st-key-nav > :is(.st-key-nav_run, :has(.st-key-nav_run)) {{ justify-self: start; }}
  .st-key-nav > :has(.st-key-nav_scroll), .st-key-nav > :has(.st-key-nav_run) {{ display: block !important;
    height: auto !important; min-height: 0 !important; margin: 0 !important; padding: 0 !important; flex: none !important; }}
  .st-key-nav > :not(.st-key-nav_scroll):not(.st-key-nav_run):not(:has(.st-key-nav_scroll)):not(:has(.st-key-nav_run)) {{
    display: none !important; }}
  .st-key-nav_scroll {{ display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important;
    align-items: center !important; height: 40px !important; min-height: 0 !important; flex: 0 0 auto !important;
    overflow-x: auto; overflow-y: hidden; gap: 6px !important; width: auto !important; max-width: none !important;
    margin: 0 -16px; padding: 0 44px 0 16px !important; scrollbar-width: none; -webkit-overflow-scrolling: touch; }}
  .st-key-nav_scroll::-webkit-scrollbar {{ display: none; }}
  .st-key-nav::after {{ content: "›"; position: absolute; top: 8px; right: -16px; width: 52px; height: 40px;
    display: flex; align-items: center; justify-content: flex-end; padding-right: 14px; box-sizing: border-box;
    font-size: 26px; font-weight: 700; line-height: 1; color: {INK}; pointer-events: none;
    background: linear-gradient(to right, rgba(255,255,255,0), #fff 55%); }}
  .st-key-nav.at-end::after {{ content: none; }}
  .st-key-nav_run {{ padding: 0 !important; margin: 0 !important; height: 34px !important; min-height: 0 !important;
    align-items: center !important; }}
  .st-key-nav_run .stButton > button[kind="tertiary"] {{ padding: 6px 10px !important; }}
  /* Nothing in the menu may shrink: each group and item keeps its full width, and the row scrolls instead. */
  .st-key-nav_about, .st-key-nav_assume, .st-key-nav_run,
  .st-key-nav [data-testid="stElementContainer"], .st-key-nav .stButton, .st-key-nav button {{
    flex: 0 0 auto !important; width: max-content !important; min-width: max-content !important;
    max-width: none !important; }}
  .st-key-nav button, .st-key-nav button p {{ white-space: nowrap !important; }}
  .st-key-nav button {{ font-size: 13px !important; padding: 6px 10px !important; }}
  .st-key-nav_assume {{ padding: 3px 4px 3px 12px !important; }}
  .nav-group {{ font-size: 12px; padding-right: 8px; }}
  hr.nav {{ margin: 4px 0 20px; }}
  /* Type */
  .q {{ font-size: 24px; line-height: 1.2; }}
  .result {{ font-size: 25px; line-height: 1.18; }}
  .result-sub, .note {{ font-size: 15px; }}
  .intro p {{ font-size: 15.5px; }}
  .section {{ font-size: 18px; }}
  .chart-note {{ font-size: 13.5px; }}
  [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{ white-space: normal; overflow: visible; }}
  /* Buttons wrap instead of running off the screen; Back and Next stack, Next on top. */
  .stButton > button, .stButton > button p {{ white-space: normal; text-align: center; }}
  .st-key-pager {{ flex-direction: column-reverse !important; align-items: stretch !important; }}
  .st-key-pager [data-testid="stElementContainer"], .st-key-pager .stButton,
  .st-key-pager .stButton > button {{ width: 100% !important; }}
  .st-key-pager .stButton > button[kind="tertiary"] {{ padding: 8px 0; }}
  [class*="st-key-runrow_"] {{ flex-wrap: wrap !important; gap: 12px !important; }}
  [data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 16px; }}
  [data-testid="stTabs"] [data-baseweb="tab"] p {{ font-size: 15px !important; }}
  /* Planner table: question above its answer. */
  table.jd tr:first-child {{ display: none; }}
  table.jd, table.jd tbody, table.jd tr, table.jd td {{ display: block; width: auto !important; }}
  table.jd tr {{ padding: 12px 0; border-bottom: 1px solid #f0f0f0; }}
  table.jd td {{ border: 0; padding: 0 !important; }}
  table.jd td:first-child {{ margin-bottom: 4px; }}
  /* Results summary table */
  table.cmp {{ font-size: 14px; }}
  table.cmp td {{ padding: 10px 0 10px 10px; }}
  /* Scorecard and OTD floor table: each station layout becomes a card with labelled lines. */
  .score-wrap {{ overflow: visible; }}
  table.score, table.score.floor {{ min-width: 0 !important; max-width: none; display: block; }}
  table.score tbody {{ display: block; }}
  table.score tr:first-child {{ display: none; }}
  table.score tr {{ display: block; padding: 12px 12px; border-bottom: 1px solid {LINE}; }}
  table.score tr.pick {{ background: #eef2fe; border-radius: 12px; border-bottom: 0; margin: 4px 0; }}
  table.score tr.pick td {{ background: none; }}
  table.score td {{ display: grid; grid-template-columns: minmax(0, 44%) 1fr; gap: 12px; padding: 5px 0;
    border: 0; white-space: normal !important; font-size: 13.5px; }}
  table.score td::before {{ content: attr(data-label); color: {MUTED}; font-size: 12.5px; font-weight: 500; }}
  table.score td.name {{ display: block; font-size: 16px; padding: 0 0 6px; }}
  table.score td.name::before {{ content: none; }}
  table.score td.ok {{ font-size: 15px; }}
  table.score td.cost span {{ width: auto; }}
}}
</style>""")


# ---------------------------------------------------------------- data

@st.cache_data
def defaults() -> dict:
    return yaml.safe_load((ROOT / "assumptions.yaml").read_text())


@st.cache_data
def load_geo(market: str):
    d = ROOT / "data" / "prepared" / market
    tracts = gpd.read_file(d / "tract_demand.gpkg")
    tracts["geometry"] = tracts.to_crs(26915).simplify(40).to_crs(4326)
    tracts = tracts[["GEOID", "county", "households", "pkgs_per_day", "pkgs_per_km2", "geometry"]]
    sites = gpd.read_file(d / "hub_candidates.gpkg")
    pts = sites.to_crs(26915).representative_point().to_crs(4326)
    sites = pd.DataFrame({"rank": sites["rank"], "lat": pts.y, "lon": pts.x})
    drive = pd.read_parquet(d / "drive_times.parquet").pivot(index="site_rank", columns="GEOID", values="drive_min")
    return tracts, sites, drive


@st.cache_data
def share_sensitivity_rows(market: str, a: dict) -> list:
    """Saved full runs at several network shares (python -m engine.live --sensitivity ...), if they match the
    current default settings."""
    f = ROOT / "scenarios" / market / "share_sensitivity.json"
    if not f.exists():
        return []
    rows = json.loads(f.read_text())
    return rows if rows and rows[0].get("hash") == settings_hash(a) else []


@st.cache_resource
def run_store() -> dict:
    """Finished runs keyed by settings hash and weeks, shared by everyone on this server."""
    return {}


# ---------------------------------------------------------------- settings

A0 = defaults()
CTL = A0["app_controls"]
STEPS = CTL["steps"]
CONTROLS = {c["path"]: c for s in STEPS for c in s["controls"]}
PCT = {"pct", "pct1"}


def get_path(a: dict, path: str):
    if "*" in path:
        return 1.0                                 # a scale control starts at x1
    v = a
    for p in path.split("."):
        v = v[int(p)] if isinstance(v, list) else v[p]     # numeric parts index into lists
    return v


def set_path(a: dict, path: str, value) -> None:
    keys = path.split(".")
    if "*" in keys:
        i = keys.index("*")
        parent = a
        for k in keys[:i]:
            parent = parent[k]
        for child in parent.values():
            leaf = child
            for k in keys[i + 1:-1]:
                leaf = leaf[k]
            leaf[keys[-1]] = leaf[keys[-1]] * value
        return
    parent = a
    for k in keys[:-1]:
        parent = parent[int(k)] if isinstance(parent, list) else parent[k]
    if isinstance(parent, list):
        parent[int(keys[-1])] = value
    else:
        parent[keys[-1]] = value


# Display units: shown = (model - offset) * scale. Percent controls show 3 rather than 0.03, a 1.8 multiplier
# shows as 80% above normal, and a daily probability of 0.015 shows as 5.5 days a year.
CUFT_PER_M3 = 35.3147
KM_PER_MILE = 1.609344
DISPLAY = {"pct": (100, 0), "pct1": (100, 0), "pct_above": (100, 1), "per_year": (365, 0), "cuft": (CUFT_PER_M3, 0),
           "money_per_mile": (KM_PER_MILE, 0)}


def _show(c: dict, v):
    scale, off = DISPLAY.get(c.get("format"), (1, 0))
    return round((v - off) * scale, 6) if (scale, off) != (1, 0) else v


def _model(c: dict, v):
    scale, off = DISPLAY.get(c.get("format"), (1, 0))
    return v / scale + off if (scale, off) != (1, 0) else v


def to_widget(c: dict, v):
    """Model value -> what the widget shows."""
    if c.get("kind") == "range":
        return tuple(_show(c, x) for x in v)
    return _show(c, v)


def from_widget(c: dict, v):
    if c.get("kind") == "range":
        return [_model(c, x) for x in v]
    return _model(c, v)


def fmt(c: dict, v) -> str:
    """Model value -> readable text, with the control's unit when it has one ("1,500 packages per hour")."""
    text = _fmt(c, v)
    return f"{text} {c['unit']}" if c.get("unit") else text


def _fmt(c: dict, v) -> str:
    kind, f = c.get("kind"), c.get("format")
    if kind == "toggle":
        return "Yes" if v else "No"
    if kind == "clock":
        return clock_label(v)
    if kind == "range":
        return f"{_fmt({'format': f}, v[0])}–{_fmt({'format': f}, v[1])}"
    return {"pct": lambda: f"{v:.0%}", "pct1": lambda: f"{v:.1%}", "money": lambda: f"${v:,.2f}",
            "money0": lambda: f"${v:,.0f}", "num1": lambda: f"{v:.1f}", "num2": lambda: f"{v:.2f}",
            "int": lambda: f"{v:,.0f}", "x": lambda: f"×{v:.2f}".rstrip("0").rstrip("."),
            "pct_above": lambda: f"{v - 1:.0%}", "money_per_mile": lambda: f"${v * KM_PER_MILE:,.2f}", "cuft": lambda: f"{v * CUFT_PER_M3:,.0f} cubic feet", "per_year": lambda: f"{v * 365:.1f}".rstrip("0").rstrip(".") + " days",
            "hours": lambda: f"{v:g} hours"}.get(f, lambda: str(v))()


def clock_label(s: str) -> str:
    h = int(clock(s)) % 24
    m = int(round((clock(s) % 1) * 60))
    return f"{(h % 12) or 12}:{m:02d} {'a.m.' if h < 12 else 'p.m.'}"


SLIDER_FMT = {"pct": "%.0f%%", "pct1": "%.1f%%", "money": "$%.2f", "money0": "$%.0f", "num1": "%.1f",
              "num2": "%.2f", "int": "%d", "x": "×%.2f", "hours": "%.2f h", "pct_above": "%.0f%%"}

if "vals" not in st.session_state:
    st.session_state.vals = {p: copy.deepcopy(get_path(A0, p)) for p in CONTROLS}
if "page" not in st.session_state:
    st.session_state.page = 0
vals = st.session_state.vals


def changed() -> dict:
    return {p: v for p, v in vals.items() if v != get_path(A0, p)}


def settings() -> dict:
    a = copy.deepcopy(A0)
    for p, v in changed().items():
        set_path(a, p, v)
        if p == "schedule.sort_cutoff":             # drivers start the same time after the cut-off as by default
            gap = clock(A0["schedule"]["route_start"]) - clock(A0["schedule"]["sort_cutoff"])
            h = clock(v) + gap
            a["schedule"]["route_start"] = f"D {int(h) % 24:02d}:{int(round(h % 1 * 60)):02d}"
    return a


def _save(path: str):
    c = CONTROLS[path]
    vals[path] = from_widget(c, st.session_state[f"w:{path}"])


def hint(path: str, a: dict) -> str | None:
    """One sentence saying what the current value means in practice."""
    dem, h, dr, u = a["demand"], a["hubs"], a["drivers"], a["uncertainty"]
    hh = HOUSEHOLDS
    per_day = hh * dem["packages_per_household_per_day"] * dem["network_share"]
    monday = per_day * max(dem["day_of_week_multipliers"].values())
    small, large = h["sizes"]["small"]["square_feet"], h["sizes"]["large"]["square_feet"]
    if path == "demand.network_share":
        return (f"At these settings, the network carries about {round(per_day, -2):,.0f} packages on an average day "
                f"and about {round(monday, -2):,.0f} on a Monday.")
    if path == "scenarios.holiday_peak.volume_multiplier":
        peak = monday * a["scenarios"]["holiday_peak"]["volume_multiplier"]
        return f"A Monday during the holiday peak would bring about {round(peak, -2):,.0f} packages."
    if path == "hubs.lease_per_sqft_per_year":
        return (f"Rent comes to about ${small * h['lease_per_sqft_per_year'] / 1e3:,.0f} thousand a year for a small "
                f"station and ${large * h['lease_per_sqft_per_year'] / 1e6:,.2f} million a year for a large one.")
    if path.startswith("hubs.sizes.") and path.endswith(".fitout_per_sqft"):
        size = path.split(".")[2]
        sq = h["sizes"][size]["square_feet"]
        fit = h["sizes"][size]["fitout_per_sqft"]
        return (f"Fitting out a {size} station of {sq:,} sq ft costs about ${sq * fit / 1e6:,.2f} million, spread over "
                f"{h['amortization_years']} years. Bigger stations cost more per square foot because they hold more "
                "automation.")
    if path == "hubs.nnn_per_sqft_per_year":
        total = h["lease_per_sqft_per_year"] + h["nnn_per_sqft_per_year"]
        return (f"Broker rents are quoted before property taxes, building insurance and common-area charges. With this "
                f"added, a station's building costs ${total:.2f} per sq ft a year.")
    if path == "hubs.fitout_per_sqft":
        return (f"Fitting out a small station costs about ${small * h['fitout_per_sqft'] / 1e6:,.2f} million, spread over "
                f"{h['amortization_years']} years.")
    if path.endswith(".sort_packages_per_labor_hour") and path.startswith("hubs.sizes."):
        size = path.split(".")[2]
        prod = h["sizes"][size].get("sort_packages_per_labor_hour", h["sort_packages_per_labor_hour"])
        return (f"At this rate and a ${h['sort_wage_per_hour']:.2f} wage, sorting at a {size} station costs about "
                f"{100 * h['sort_wage_per_hour'] / prod:.0f}¢ per package. Bigger stations usually sort more per labor "
                "hour because they can justify more automation.")
    if path == "peak_flex.enabled":
        return ("When this is on, each station adds temporary staff and longer shifts in holiday peak weeks instead "
                "of the network needing extra stations all year.")
    if path == "peak_flex.sort_capacity_boost":
        return (f"A small station would sort about {h['sizes']['small']['sort_rate_per_hour'] * (1 + a['peak_flex']['sort_capacity_boost']):,.0f} "
                "packages an hour at peak. The extra volume is paid at the overtime rate.")
    if path == "partner.enabled":
        return ("When this is on, packages a station cannot sort by the cut-off go to a partner carrier instead of "
                "waiting a day. This helps most in peak weeks and when a station closes.")
    if path == "partner.max_share_of_day":
        return ("A partner can only take so much on short notice. Anything above this share still waits a day.")
    if path == "hubs.sort_packages_per_labor_hour":
        return (f"At this rate and wage, sort labor costs about "
                f"{100 * h['sort_wage_per_hour'] / h['sort_packages_per_labor_hour']:.0f}¢ per package.")
    if path.endswith(".sort_rate_per_hour"):
        size = path.split(".")[2]
        sched = a["schedule"]
        window = clock(sched["sort_cutoff"]) - clock(sched["linehaul_waves"][0]["arrive"])
        return (f"At this speed, a {size} station of {h['sizes'][size]['square_feet']:,} sq ft can sort about "
                f"{h['sizes'][size]['sort_rate_per_hour'] * window:,.0f} packages between the first truck and the "
                f"{clock_label(sched['sort_cutoff'])} cut-off.")
    if path == "drivers.claim_curve_width":
        w, mid = dr["claim_curve_width"], dr["claim_midpoint_net_per_hour"]
        up = 1 / (1 + math.exp(-1))
        return (f"A route paying ${mid + w:.2f} an hour after car costs is accepted {up:.0%} of the time, and one paying "
                f"${mid - w:.2f} is accepted {1 - up:.0%} of the time. A smaller value means Dashers react more sharply "
                "to pay. This is an assumption, because there is no public data on how Dashers respond to pay.")
    if path == "drivers.claim_midpoint_net_per_hour":
        return ("The center of the curve comes from public gig-pay data. Amazon Flex drivers earn a median of about "
                "$20.89 an hour before costs (Gridwise), and car costs take roughly $7 of that.")
    if path.startswith("drivers.vehicle_availability."):
        pts = dr["vehicle_availability"]
        i = int(path.split(".")[2])
        cuft = pts[i][0]
        return (f"This is the share of Dashers whose vehicle holds at least {cuft} cubic feet. Routes that need that "
                "much space can only be accepted by these Dashers, so a smaller share means more routes need raised "
                "pay or go unclaimed. This is an assumption, because DoorDash does not publish its Dashers' vehicles.")
    if path == "drivers.surge_max":
        return (f"A route with $40 of base pay could have its pay raised to as much as ${40 * dr['surge_max']:,.0f} "
                "before it goes unclaimed.")
    if path == "routing.car_cargo_m3":
        sizes = dem["package_sizes"].values()
        avg = sum(s["share"] * s["volume_m3"] for s in sizes) / sum(s["share"] for s in sizes)
        per_mile = vehicle_cost_per_km(dr, a["routing"]) * 1.609
        share = vehicle_availability(dr, a["routing"])
        return (f"That fits about {a['routing']['car_cargo_m3'] / avg:,.0f} average packages. A vehicle this size "
                f"costs about ${per_mile:.2f} a mile to run, and about {share:.0%} of Dashers have one. For comparison, "
                "a sedan's trunk and back seat hold about 30 cubic feet, a midsize SUV with the seats folded holds "
                "about 70, and a cargo van holds about 250 to 300.")
    if path == "objective.driver_net_per_hour_floor":
        if dr.get("raise_base_to_floor"):
            return ("Every route is paid at least this much per hour after car costs. When a route would pay less, "
                    "its base pay is raised, and the extra pay counts in that station layout's cost.")
        return ("Raising pay to the floor is switched off in step 3, so any station layout whose Dashers earn less "
                "than this in a typical week is ruled out.")
    if path == "uncertainty.truck_late_prob":
        n = len(a["schedule"]["linehaul_waves"]) * 7 * u["truck_late_prob"]
        return f"Each station sees about {n:.1f} late truck arrivals a week."
    if path == "uncertainty.volume_week_sigma":
        return (f"In about two weeks out of three, a week's volume lands within {u['volume_week_sigma']:.0%} of the "
                "forecast.")
    return None


def variability(path: str, a: dict) -> tuple[bool, str]:
    """Whether a setting is used exactly as set (fixed) or sets an average or odds that each simulated day or week
    draws around (varies), with one sentence saying how."""
    u = a["uncertainty"]
    varies = {
        "demand.network_share": "This sets the average. Each simulated day's packages vary at random around it, by "
                                f"about {u['volume_week_sigma']:.0%} from week to week and {u['volume_day_sigma']:.0%} "
                                "from day to day.",
        "demand.packages_per_household_per_day": "This sets the average. Each simulated day's packages vary at "
                                                 "random around it.",
        "hubs.sizes.small.sort_rate_per_hour": f"Each station's actual speed changes from day to day by about "
                                               f"±{u['sort_rate_sigma']:.0%}.",
        "hubs.sizes.medium.sort_rate_per_hour": f"Each station's actual speed changes from day to day by about "
                                                f"±{u['sort_rate_sigma']:.0%}.",
        "hubs.sizes.large.sort_rate_per_hour": f"Each station's actual speed changes from day to day by about "
                                               f"±{u['sort_rate_sigma']:.0%}.",
        "drivers.claim_midpoint_net_per_hour": "This sets the odds. Whether a Dasher accepts each offer is drawn at "
                                               "random.",
        "drivers.claim_curve_width": "This shapes the odds. Whether a Dasher accepts each offer is drawn at random.",
        "uncertainty.truck_late_prob": "This is a probability. Each truck on each simulated night is late or on "
                                       "time at random.",
        "uncertainty.volume_week_sigma": "This sets the spread. Each simulated week draws its own volume.",
        "uncertainty.no_show_rate_range": "Each simulated week draws its own no-show rate from inside this range.",
        "uncertainty.slow_day_prob": "This is an average. Each simulated day is slow or normal at random.",
        "uncertainty.extreme_weather_prob": "This is an average. Each simulated day has a storm or not at random, "
                                            "so some years have more storm days than others.",
    }
    if path in varies:
        return True, varies[path]
    fixed = {
        "scenarios.holiday_peak.volume_multiplier": "Every holiday peak week uses this, and daily volume still "
                                                    "varies around it.",
        "weeks_per_year.holiday_peak": "This weights the kinds of week in the yearly cost.",
        "weeks_per_year.hub_outage": "This weights the kinds of week in the yearly cost.",
        "partner.on_time_rate": "The model applies this share to the partner's packages every day.",
    }
    return False, fixed.get(path, "")


def poisson_band(mean: float) -> tuple[float, float]:
    """10th and 90th percentiles of a Poisson count (for storm and slow days in a year)."""
    from math import exp
    def q(target):
        k, p, cum = 0, exp(-mean), exp(-mean)
        while cum < target and k < 2000:
            k += 1; p *= mean / k; cum += p
        return k
    return float(q(0.10)), float(q(0.90))


def binom_band(n: int, p: float) -> tuple[int, int]:
    """10th and 90th percentiles of a binomial count."""
    from math import comb
    cum, lo, hi = 0.0, None, None
    for k in range(n + 1):
        cum += comb(n, k) * p ** k * (1 - p) ** (n - k)
        if lo is None and cum >= 0.10:
            lo = k
        if hi is None and cum >= 0.90:
            hi = k
    return lo or 0, hi if hi is not None else n


def strip_spec(c: dict, a: dict) -> dict | None:
    """What the range strip under a varying slider shows: a band (lo, hi) and a marker on a scale (smin, smax), all in
    the same units, plus a sentence. Most strips use the slider's own scale; a few use a scale that suits what varies."""
    path, u, dr = c["path"], a["uncertainty"], a["drivers"]
    if "min" not in c or not variability(path, a)[0]:
        return None                                    # fixed settings, switches and time pickers have no strip
    smin, smax = _show(c, c["min"]), _show(c, c["max"])
    v = get_path(a, path)
    unit = " " + c["unit"] if c.get("unit") else ""
    show = lambda x: fmt({"format": c.get("format")}, _model(c, x))   # noqa: E731
    if path in ("demand.network_share", "demand.packages_per_household_per_day"):
        sd = math.sqrt(u["volume_week_sigma"] ** 2 + u["volume_day_sigma"] ** 2)
        lo, hi = _show(c, v * math.exp(-1.2816 * sd)), _show(c, v * math.exp(1.2816 * sd))
        return dict(lo=lo, hi=hi, mark=_show(c, v), smin=smin, smax=smax,
                    note=f"In 8 of 10 simulated days, the value the model uses falls between {show(lo)} and {show(hi)}{unit}.")
    if path.endswith(".sort_rate_per_hour"):
        lo = _show(c, v * max(0.7, 1 - 1.2816 * u["sort_rate_sigma"]))
        hi = _show(c, v * min(1.15, 1 + 1.2816 * u["sort_rate_sigma"]))
        return dict(lo=lo, hi=hi, mark=_show(c, v), smin=smin, smax=smax,
                    note=f"In 8 of 10 simulated days, a station sorts between {show(lo)} and {show(hi)}{unit}.")
    if path in ("uncertainty.slow_day_prob", "uncertainty.extreme_weather_prob"):
        lo_d, hi_d = poisson_band(365 * v)
        noun = c["label"].replace(" per year", "").lower()
        return dict(lo=lo_d, hi=hi_d, mark=365 * v, smin=smin, smax=smax,
                    note=f"In 8 of 10 simulated years, there are between {lo_d:.0f} and {hi_d:.0f} {noun}.")
    if path == "uncertainty.truck_late_prob":
        n = len(a["schedule"]["linehaul_waves"]) * 7
        lo_k, hi_k = binom_band(n, v)
        return dict(lo=100 * lo_k / n, hi=100 * hi_k / n, mark=100 * v, smin=smin, smax=smax,
                    note=f"In 8 of 10 simulated weeks, between {lo_k} and {hi_k} of a station's {n} truck arrivals are "
                         "late.")
    if path == "uncertainty.volume_week_sigma":
        lo, hi = 100 * (math.exp(-1.2816 * v) - 1), 100 * (math.exp(1.2816 * v) - 1)
        return dict(lo=lo, hi=hi, mark=0.0, smin=-40.0, smax=40.0, scale=("40% below forecast", "40% above"),
                    note=f"In 8 of 10 simulated weeks, volume lands between {abs(lo):.0f}% below and {hi:.0f}% above the "
                         "forecast.")
    if path == "uncertainty.no_show_rate_range":
        a0, b0 = v
        lo, hi = 100 * (a0 + 0.1 * (b0 - a0)), 100 * (a0 + 0.9 * (b0 - a0))
        return dict(lo=lo, hi=hi, mark=None, smin=smin, smax=smax,
                    note=f"In 8 of 10 simulated weeks, the no-show rate falls between {lo:.1f}% and {hi:.1f}%.")
    if path in ("drivers.claim_midpoint_net_per_hour", "drivers.claim_curve_width"):
        mid, w = dr["claim_midpoint_net_per_hour"], dr["claim_curve_width"]
        lo, hi = mid - 2.1972 * w, mid + 2.1972 * w
        own = path == "drivers.claim_midpoint_net_per_hour"
        return dict(lo=lo, hi=hi, mark=mid, smin=smin if own else 0.0, smax=smax if own else 40.0,
                    scale=None if own else ("$0 an hour", "$40 an hour"),
                    note=f"Acceptance climbs from 10% to 90% as a route's pay goes from ${max(lo, 0):.2f} to ${hi:.2f} an "
                         "hour after car costs.")
    return None


def band_strip(c: dict, a: dict) -> str:
    """A thin strip under a varying slider: the shaded band is the range described in the sentence, and the marker is
    the setting. It uses the slider's scale unless the strip labels its own ends."""
    spec = strip_spec(c, a)
    if not spec:
        return ""
    smin, smax = spec["smin"], spec["smax"]
    span = max(smax - smin, 1e-9)
    pos = lambda x: max(0.0, min(100.0, (x - smin) / span * 100))   # noqa: E731
    mark = "" if spec.get("mark") is None else f'<i class="band-mark" style="left:{pos(spec["mark"]):.1f}%"></i>'
    ends = ""
    if spec.get("scale"):
        ends = (f'<div class="band-ends"><span>{spec["scale"][0]}</span><span>{spec["scale"][1]}</span></div>')
    return (f'<div class="band"><div class="band-track"><i class="band-range" style="left:{pos(spec["lo"]):.1f}%;'
            f'width:{max(pos(spec["hi"]) - pos(spec["lo"]), 0.8):.1f}%"></i>{mark}</div>{ends}'
            f'<div class="band-note">{spec["note"]}</div></div>')


def control(c: dict) -> None:
    path, key = c["path"], f"w:{c['path']}"
    if key not in st.session_state:
        st.session_state[key] = to_widget(c, vals[path])
    kind = c.get("kind", "slider")
    label = c["label"]
    # Settings that only matter when a switch is on are greyed out and locked while it is off.
    off = bool(c.get("requires")) and not vals.get(c["requires"], get_path(A0, c["requires"]))
    if kind == "toggle":
        st.toggle(label, key=key, on_change=_save, args=(path,))
    elif kind == "clock":
        st.select_slider(label, options=c["options"], format_func=clock_label, key=key, on_change=_save,
                         args=(path,))
    else:
        f = c.get("format")
        scale = DISPLAY.get(f, (1, 0))[0]
        lo, hi = _show(c, c["min"]), _show(c, c["max"])
        step = c.get("display_step", c.get("step", 1) * scale)
        num = int if f == "int" or (f in ("money0", "pct", "pct_above") and float(step).is_integer()
                                    and float(lo).is_integer()) else float
        if num is float:
            st.session_state[key] = tuple(float(x) for x in st.session_state[key]) \
                if isinstance(st.session_state[key], tuple) else float(st.session_state[key])
        shown_fmt = (("%.0f days" if step >= 1 else "%.1f days") if f == "per_year"
                     else "%.0f cu ft" if f == "cuft" else "$%.2f" if f == "money_per_mile" else SLIDER_FMT.get(f, "%g"))
        st.slider(label, min_value=num(lo), max_value=num(hi), step=num(step), key=key, on_change=_save,
                  args=(path,), format=shown_fmt, disabled=off)
    d0 = get_path(A0, path)
    if c.get("kind") == "toggle":
        base = f"This is {'on' if d0 else 'off'} by default."
    else:
        shown = fmt(c, d0)
        base = f"The default is {shown}" + ("" if shown.endswith(".") else ".")
    lead = f"<b>You changed this value.</b> {base}" if vals[path] != d0 else base
    cur = settings()
    random_, how = variability(path, cur)
    tag = '<span class="tag vary">Varies</span>' if random_ else '<span class="tag fixed">Fixed</span>'
    meaning = hint(path, cur)
    if off:
        switch = CONTROLS[c["requires"]]["label"].lower()
        st.html(f'<div class="default off">This applies only when "{switch[0].upper() + switch[1:]}" is switched on. '
                f'{lead}</div>')
        return
    st.html(band_strip(c, cur)
            + f'<div class="default">{tag}{how + " " if how else ""}{lead}{" " + meaning if meaning else ""}</div>')



def go(i: int):
    st.session_state.page = i


def reset_step(step: dict):
    for c in step["controls"]:
        vals[c["path"]] = copy.deepcopy(get_path(A0, c["path"]))
        st.session_state.pop(f"w:{c['path']}", None)


def reset_all():
    for step in STEPS:
        reset_step(step)


# ---------------------------------------------------------------- names and charts

def hub_list(label: str) -> list[tuple[int, str]]:
    return [(int(n), s) for n, s in re.findall(r"#(\d+) (\w+)", label)]


def plain(label: str) -> str:
    counts = Counter(s for _, s in hub_list(label))
    parts = [f"{counts[s]} {s}" for s in ("large", "medium", "small") if counts[s]]
    return " + ".join(parts) + (" station" if sum(counts.values()) == 1 else " stations")


def spoken(label: str) -> str:
    """'#1 medium + #2 small + #3 small' -> 'one medium station and two small stations' (for sentences)."""
    counts = Counter(s for _, s in hub_list(label))
    parts = [f"{WORDS.get(counts[s], counts[s])} {s} station{'s' if counts[s] > 1 else ''}"
             for s in ("large", "medium", "small") if counts[s]]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def precision(n_weeks: int, share: float = 0.95) -> str:
    """Half-width of a 90% confidence range on the share of weeks meeting the floor, rounded to half a point."""
    pts = 1.645 * math.sqrt(share * (1 - share) / max(n_weeks, 1)) * 100
    return f"{round(pts * 2) / 2:g}"


def money(v): return f"${v:,.2f}"


def _style(chart, height):
    return chart.properties(height=height).configure_view(stroke=None).configure_axis(labelLimit=0, 
        labelFont=FONT, titleFont=FONT, labelColor=BODY, titleColor=MUTED, labelFontSize=13, titleFontSize=12,
        titleFontWeight=400, domainColor="#c9ccd2", tickColor="#c9ccd2", gridColor="#eef0f2").configure_text(
        font=FONT).configure_legend(labelFont=FONT, titleFont=FONT, labelFontSize=13, titleFontSize=12)


def volume_preview(a: dict, households: float) -> alt.Chart:
    dem = a["demand"]
    base = households * dem["packages_per_household_per_day"] * dem["network_share"]
    peak = a["scenarios"]["holiday_peak"]["volume_multiplier"]
    rows = []
    for day in DAYS:
        n = base * dem["day_of_week_multipliers"][day]
        rows += [{"day": day, "week": "Normal week", "pkgs": n}, {"day": day, "week": "Holiday peak", "pkgs": n * peak}]
    d = pd.DataFrame(rows)
    return _style(alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("day:N", sort=DAYS, title=None, axis=alt.Axis(labelAngle=0, ticks=False, domain=False)),
        xOffset=alt.XOffset("week:N", sort=["Normal week", "Holiday peak"]),
        y=alt.Y("pkgs:Q", title=None, axis=alt.Axis(format=",.0f", tickCount=4, domain=False, ticks=False)),
        color=alt.Color("week:N", sort=["Normal week", "Holiday peak"], scale=alt.Scale(range=[INK, RED]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("day:N", title="Day"), alt.Tooltip("week:N", title="Week"),
                 alt.Tooltip("pkgs:Q", title="Packages", format=",.0f")]), 230)


def sort_numbers(a: dict, households: float) -> dict:
    """One station's overnight sort capacity by size, and the busiest normal and peak nights."""
    dem, h = a["demand"], a["hubs"]
    monday = households * dem["packages_per_household_per_day"] * dem["network_share"] * \
        max(dem["day_of_week_multipliers"].values())
    window = clock(a["schedule"]["sort_cutoff"]) - clock(a["schedule"]["linehaul_waves"][0]["arrive"])
    cap = {size: s["sort_rate_per_hour"] * window for size, s in h["sizes"].items()}
    return {"cap": cap, "normal": monday, "peak": monday * a["scenarios"]["holiday_peak"]["volume_multiplier"]}


def sort_preview(a: dict, households: float) -> alt.Chart:
    n = sort_numbers(a, households)
    d = pd.DataFrame([{"size": f"{k.capitalize()} station", "cap": v} for k, v in n["cap"].items()])
    d["label"] = d["cap"].map(lambda v: f"{v:,.0f}")
    top = max(d["cap"].max(), n["peak"]) * 1.12
    y = alt.Y("size:N", sort=None, title=None, axis=alt.Axis(ticks=False, domain=False, labelFontSize=14,
                                                            labelPadding=10, labelLimit=0, labelOverlap=False, minExtent=96))
    x = alt.X("cap:Q", title="Packages sorted between the first truck and the cut-off",
              scale=alt.Scale(domain=[0, top]), axis=alt.Axis(format=",.0f", tickCount=5, domain=False, ticks=False))
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=22, color="#d9d9d9").encode(
        y=y, x=x, tooltip=[alt.Tooltip("size:N", title=""), alt.Tooltip("cap:Q", title="Overnight capacity",
                                                                         format=",.0f")])
    values = alt.Chart(d).mark_text(align="right", dx=-10, fontSize=13, fontWeight=700, color=INK).encode(
        y=y, x="cap:Q", text="label:N")      # inside the end of each bar, clear of the volume lines
    lines = pd.DataFrame({"v": [n["normal"], n["peak"]], "who": ["normal", "peak"],
                          "t": [f"Busiest normal day: {n['normal']:,.0f}", f"Busiest peak day: {n['peak']:,.0f}"]})
    color = alt.Color("who:N", scale=alt.Scale(domain=["normal", "peak"], range=[INK, RED]), legend=None)
    rule = alt.Chart(lines).mark_rule(strokeWidth=2).encode(x="v:Q", color=color,
                                                            tooltip=[alt.Tooltip("t:N", title="")])
    # The two lines are labelled in a key above the chart (sort_key), so nothing can overlap inside it.
    return _style(bars + values + rule, alt.Step(46))


def sort_key(a: dict) -> str:
    n = sort_numbers(a, HOUSEHOLDS)
    item = lambda c, txt: (f'<span><i style="display:inline-block;width:18px;height:0;border-top:2px solid {c};'   # noqa: E731
                           f'vertical-align:middle;margin-right:7px"></i>{txt}</span>')
    return ('<div class="keyrow">' + item(INK, f"Busiest normal day: {n['normal']:,.0f} packages")
            + item(RED, f"Busiest peak day: {n['peak']:,.0f} packages") + "</div>")


def sort_note(a: dict) -> str:
    n = sort_numbers(a, HOUSEHOLDS)
    flex = a.get("peak_flex", {})
    boost = flex.get("sort_capacity_boost", 0.0) if flex.get("enabled") else 0.0
    need = {k: math.ceil(n["peak"] / (v * (1 + boost))) for k, v in n["cap"].items()}
    word = lambda k: f'{WORDS.get(need[k], need[k])} {k} station{"s" if need[k] > 1 else ""}'   # noqa: E731
    text = ("Each bar shows how many packages one station of that size can sort between the first truck and the "
            "cut-off at its normal speed. The black line marks the busiest day of a normal week, and the red line "
            "marks the busiest day of the holiday peak. ")
    if boost:
        text += (f"In peak weeks, temporary staff raise each station's sort speed by {boost:.0%}, so the busiest "
                 "peak night takes at least ")
    else:
        text += "Sorting the busiest peak night takes at least "
    return text + (f"{word('small')}, {word('medium')} or {word('large')}, before allowing for the extra driving "
                   "that comes with fewer stations.")


def claim_note(a: dict) -> str:
    dr = a["drivers"]
    return ("This chart shows the chance that a Dasher accepts a route when it is offered, based on what the route pays "
            "per hour after car costs, counting the unpaid drive to the station. The curve is an S-shaped formula with "
            f"two settings: half of Dashers accept at ${dr['claim_midpoint_net_per_hour']:.2f} an hour, and every "
            f"${dr['claim_curve_width']:.2f} above or below that moves acceptance a step up or down the curve. Both "
            "are stated assumptions, because DoorDash does not publish acceptance data. The center is anchored to "
            "public gig-pay figures, and you can change both above. Each offer in the simulation is a random draw "
            "from this curve. Routes that nobody accepts are offered again with their base pay raised, up to a set "
            "limit.")


def claim_preview(a: dict) -> alt.Chart:
    dr, obj = a["drivers"], a["objective"]
    xs = [x / 2 for x in range(0, 81)]
    d = pd.DataFrame({"net": xs, "p": [1 / (1 + math.exp(-(x - dr["claim_midpoint_net_per_hour"])
                                                         / dr["claim_curve_width"])) for x in xs]})
    line = alt.Chart(d).mark_line(color=BLUE, strokeWidth=2.5).encode(
        x=alt.X("net:Q", title="A route's pay per hour after car costs", axis=alt.Axis(format="$.0f", domain=False)),
        y=alt.Y("p:Q", title=None, axis=alt.Axis(format="%", tickCount=5, domain=False, ticks=False)),
        tooltip=[alt.Tooltip("net:Q", title="Net pay per hour", format="$.2f"),
                 alt.Tooltip("p:Q", title="Chance a Dasher takes it", format=".0%")])
    floor = pd.DataFrame({"x": [obj["driver_net_per_hour_floor"]],
                          "t": [f"Dasher pay floor: ${obj['driver_net_per_hour_floor']:.0f} an hour"]})
    rule = alt.Chart(floor).mark_rule(color=RED, strokeWidth=1.5, strokeDash=[4, 3]).encode(x="x:Q")
    # Label near the bottom, right of the line, where the curve is far above it.
    txt = alt.Chart(floor).mark_text(align="left", dx=6, baseline="bottom", y=alt.expr("height - 6"), fontSize=12.5,
                                     fontWeight=600, color=RED).encode(x="x:Q", text="t:N")
    return _style(line + rule + txt, 220)


def schedule_preview(a: dict) -> alt.Chart:
    s = a["schedule"]
    ev = [{"t": clock(w["arrive"]), "what": f"Truck arrival ({w['share']:.0%} of packages)", "row": "At the station"} for w in s["linehaul_waves"]]
    ev += [{"t": clock(s["sort_cutoff"]), "what": "Sort cut-off", "row": "At the station"},
           {"t": clock(s["route_start"]), "what": "Dashers pick up routes", "row": "On the road"},
           {"t": clock(s["promise_end"]), "what": "Delivery deadline", "row": "On the road"}]
    d = pd.DataFrame(ev)
    d["label"] = d["t"].map(lambda h: f"{(int(h) % 24 % 12) or 12}{'a' if int(h) % 24 < 12 else 'p'}")
    span = pd.DataFrame([{"row": "At the station", "a": d["t"].min(), "b": clock(s["sort_cutoff"]), "c": "Sorting"},
                         {"row": "On the road", "a": clock(s["route_start"]), "b": clock(s["promise_end"]), "c": "Delivering"}])
    x = alt.X("t:Q", title=None, scale=alt.Scale(domain=[18, 45]), axis=alt.Axis(
        values=list(range(18, 46, 3)), labelExpr="(datum.value % 24) % 12 == 0 ? (datum.value % 24 == 0 ? '12a' : '12p')"
        " : ((datum.value % 24) % 12) + ((datum.value % 24) < 12 ? 'a' : 'p')", domain=False, ticks=False))
    bars = alt.Chart(span).mark_bar(height=16, cornerRadius=3, opacity=0.25).encode(
        x=alt.X("a:Q", scale=alt.Scale(domain=[18, 45])), x2="b:Q",
        y=alt.Y("row:N", sort=["At the station", "On the road"], title=None, axis=alt.Axis(ticks=False, domain=False,
                                                                           labelFontSize=14, labelLimit=0, labelOverlap=False)),
        color=alt.Color("c:N", scale=alt.Scale(domain=["Sorting", "Delivering"], range=[BLUE, GREEN]), legend=None))
    d["key"] = d["what"].map(lambda w: "deadline" if w in ("Sort cut-off", "Delivery deadline") else "event")
    yrow = alt.Y("row:N", sort=["At the station", "On the road"])
    pts = alt.layer(*[alt.Chart(d[d["key"] == k]).mark_point(filled=True, size=90, opacity=1, color=c).encode(
        x=x, y=yrow, tooltip=[alt.Tooltip("what:N", title="")]) for k, c in (("event", INK), ("deadline", RED))])
    first = d.iloc[[0]].assign(what="Trucks start arriving")
    labels = pd.concat([first.assign(align="left"),
                        d[d["what"].isin(["Sort cut-off", "Dashers pick up routes"])].assign(align="center"),
                        d[d["what"] == "Delivery deadline"].assign(align="right")])
    txt = alt.layer(*[alt.Chart(labels[labels["align"] == al]).mark_text(
        align=al, dx={"left": -4, "center": 0, "right": 4}[al], dy=-17, fontSize=12.5, fontWeight=600, color=INK
    ).encode(x="t:Q", y=yrow, text="what:N") for al in ("left", "center", "right")])
    return _style(bars + pts + txt, 140)


def schedule_note(a: dict) -> str:
    s = a["schedule"]
    waves = s["linehaul_waves"]
    return (f"Trucks bring packages to each station between {clock_label(waves[0]['arrive'])} and "
            f"{clock_label(waves[-1]['arrive'])} each night. The station sorts them into routes until the "
            f"{clock_label(s['sort_cutoff'])} cut-off, and anything sorted later waits a day. Dashers pick up their "
            f"routes from {clock_label(s['route_start'])}, and a package counts toward OTD if it is delivered by "
            f"{clock_label(s['promise_end'])} on the day it was due. Moving the cut-off later gives the station more "
            "time to sort but "
            "leaves Dashers less time to deliver.")


def disruption_preview(a: dict) -> alt.Chart:
    u, w = a["uncertainty"], a["weeks_per_year"]
    d = pd.DataFrame([
        {"what": "Severe-weather days", "n": 365 * u["extreme_weather_prob"]},
        {"what": "Slow-traffic days", "n": 365 * u["slow_day_prob"]},
        {"what": "Holiday peak weeks", "n": w.get("holiday_peak", 0)},
        {"what": "Station outage weeks", "n": w.get("hub_outage", 0)}])
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=3, height=16, color="#c9d3fb").encode(
        y=alt.Y("what:N", sort=None, title=None, axis=alt.Axis(ticks=False, domain=False, labelFontSize=14,
                                                              labelPadding=10, labelLimit=0, labelOverlap=False, minExtent=150)),
        x=alt.X("n:Q", title="In a year", scale=alt.Scale(domain=[0, max(d["n"].max() * 1.15, 1)]),
                axis=alt.Axis(tickCount=5, domain=False, ticks=False)))
    txt = bars.mark_text(align="left", dx=6, fontSize=13, fontWeight=600, color=INK).encode(
        text=alt.Text("n:Q", format=".0f"))
    return _style(bars + txt, alt.Step(44))


def network_map(tracts, sites, drive, label: str) -> folium.Map:
    hubs = hub_list(label)
    nearest = drive.loc[[r for r, _ in hubs], tracts["GEOID"]].to_numpy().argmin(axis=0)
    share = pd.Series(tracts["pkgs_per_day"].to_numpy()).groupby(nearest).sum() / tracts["pkgs_per_day"].sum()
    t = tracts.copy()
    t["hub"] = nearest
    t["opacity"] = 0.15 + 0.5 * t["pkgs_per_km2"].rank(pct=True)
    t["served"] = [f"Station {i + 1}" for i in nearest]
    t["pkgs"] = t["pkgs_per_day"].round(0)
    m = folium.Map(location=[44.96, -93.25], zoom_start=9, tiles=None, scrollWheelZoom=False, control_scale=True)
    add_basemap(m)
    folium.GeoJson(
        t, style_function=lambda f: {"fillColor": HUB_COLORS[f["properties"]["hub"] % 4], "color": "#ffffff",
                                     "weight": 0.25, "fillOpacity": f["properties"]["opacity"]},
        tooltip=folium.GeoJsonTooltip(["served", "county", "pkgs"],
                                      aliases=["Served by", "County", "Packages, average day (default volume)"],
                                      localize=True)).add_to(m)
    loc = sites.set_index("rank")
    for i, (r, size) in enumerate(hubs):
        s = loc.loc[r]
        html = (f'<div style="display:flex;align-items:center;gap:6px;white-space:nowrap;'
                f'font:600 13px \'{FONT}\',sans-serif;color:{INK};text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff">'
                f'<span style="flex:none;width:14px;height:14px;border-radius:50%;background:{HUB_COLORS[i % 4]};'
                f'border:2.5px solid #fff;box-shadow:0 0 0 1px {INK}"></span>'
                f'Station {i + 1} · {size} · {share.get(i, 0):.0%}</div>')
        folium.Marker([s.lat, s.lon], icon=folium.DivIcon(html=html, icon_size=(190, 20), icon_anchor=(9, 10)),
                      tooltip=f"Station {i + 1}: {size}").add_to(m)
    add_labels(m)
    return m


PAY_POLICY_ON = [True]   # set per run on the Results page


def missed_targets(r, obj: dict) -> list[tuple[str, str]]:
    """Every target a station layout misses: (short name, a plain sentence with its actual number)."""
    names = []
    if r["on_time_p50"] < obj["on_time_typical_target"]:
        names.append("OTD target")
    if r["weeks_meeting_service"] < obj["on_time_week_share"]:
        names.append("OTD floor")
    if r["driver_net_p50"] < obj["driver_net_per_hour_floor"] and not PAY_POLICY_ON[0]:
        names.append("Dasher pay floor")
    return [(n, short_miss(n, r, obj)) for n in names]


def missed_target(r, obj: dict) -> str:
    names = [n for n, _ in missed_targets(r, obj)]
    return "Misses " + " and ".join(names) if names else "Misses a target"


def diagnosis_text(dg: dict | None, a: dict) -> str:
    """Why no network meets the targets, in plain sentences: the target missed by the most, the best any network
    reached, and the main reason packages are late."""
    if not dg:
        return ""
    kind = {"normal": "a normal week", "holiday_peak": "a holiday peak week",
            "hub_outage": "a station outage week"}.get(dg["scenario"], "a week")
    kinds = {"normal": "normal weeks", "holiday_peak": "holiday peak weeks",
             "hub_outage": "station outage weeks"}.get(dg["scenario"], "weeks")
    if dg["target"] == "OTD target":
        miss = (f"The hardest target is the OTD target in {kind}. The best any network reached was "
                f"{dg['best']:.1%} OTD in a typical week, against a target of {dg['need']:.1%}.")
    else:
        obj = objective_for(a, dg["scenario"])
        miss = (f"The hardest target is the OTD floor in {kinds}. The best any network reached was "
                f"{dg['best']:.0%} of weeks at {obj['on_time_target']:.0%} OTD or better, against a floor of "
                f"{dg['need']:.0%} of weeks.")
    late, sort_miss = dg["late_share"], dg["sort_miss_share"]
    why = {
        "late_trucks": (f"In the closest network, {late:.1%} of packages are late in a typical week, and "
                        f"{sort_miss:.1%} miss the sort cut-off even though the stations use only about "
                        f"{dg['sort_utilization_p90']:.0%} of their sort capacity in a busy week. Those packages arrive on late "
                        "trucks or miss their truck upstream, so adding stations cannot fix them. A later sort "
                        "cut-off, fewer late trucks or a lower target would."),
        "sort_capacity": (f"In the closest network, {late:.1%} of packages are late in a typical week, and "
                          f"{sort_miss:.1%} miss the sort cut-off because the stations run out of sort capacity. "
                          "Faster sorting, a later cut-off or more peak flex would help."),
        "dashers": (f"In the closest network, {dg['unclaimed']:.1%} of routes go unclaimed in a typical week. More "
                    "Dashers or higher base pay would help."),
        "routes": (f"In the closest network, {late:.1%} of packages are late in a typical week, mostly because routes "
                   "finish after the 8 p.m. delivery deadline. Shorter routes or more Dashers would help."),
    }[dg["cause"]]
    return f"{miss} {why}"


def scorecard(run: dict, a: dict, pick: str) -> str:
    """One table that brings the three kinds of week together: year cost, a tick or cross per kind of week,
    and a verdict."""
    obj, frames, y = a["objective"], run["frames"], run["year"]
    keys = [k for k in frames]
    rows = y.sort_values("year_cost_per_package")
    best = rows.loc[pick, "year_cost_per_package"] if pick in rows.index else rows["year_cost_per_package"].min()
    weights = {k: a["weeks_per_year"].get(k, 0) for k in keys}
    wsum = sum(weights.values()) or 1

    def topup(d):
        vals = [(weights[k], frames[k].loc[d].get("floor_topup_p50", 0) or 0) for k in keys if d in frames[k].index]
        return sum(w * v for w, v in vals) / wsum
    floor = obj["driver_net_per_hour_floor"]
    show_topup = PAY_POLICY_ON[0]
    head = ("<tr><th>Station layout</th><th>Cost per package over a year</th>"
            + (f"<th>Of which, raising pay to ${floor:.0f}/h</th>" if show_topup else "")
            + "".join(f"<th>{SCEN.get(k, k)}</th>" for k in keys) + "<th>Verdict</th></tr>")
    body = ""
    for d, r in rows.iterrows():
        cells, failed = "", []
        for k in keys:
            if d not in frames[k].index:
                cells += f'<td class="na" data-label="{SCEN.get(k, k)}"><div>Not scored</div></td>'
                failed.append(SCEN.get(k, k).lower())
                continue
            misses = missed_targets(frames[k].loc[d], objective_for(a, k))
            if not misses and bool(frames[k].loc[d, "feasible"]):
                cells += f'<td class="ok" data-label="{SCEN.get(k, k)}"><div>✓</div></td>'
            else:
                failed.append(SCEN.get(k, k).lower())
                kind = {"normal": "normal", "holiday_peak": "holiday peak", "hub_outage": "station outage"}.get(k, "")
                text = "<br>".join(f"<b>Misses the {n}.</b> {short_miss(n, frames[k].loc[d], objective_for(a, k), kind)}"
                                   for n, _ in misses) \
                    or "<b>Misses a target</b>"
                cells += f'<td class="no" data-label="{SCEN.get(k, k)}"><div><span class="x">✗</span>{text}</div></td>'
        cost = r["year_cost_per_package"]
        if d == pick:
            verdict, cls = ("<b>Recommended</b>" if not failed else "<b>Closest to the OTD targets</b>"), "pick"
        elif not failed:
            verdict, cls = f"Meets every target, {abs(cost - best) * 100:.1f}¢ more per package", ""
        else:
            names = [{"normal week": "normal", "holiday peak": "holiday peak", "station outage": "station outage"}
                     .get(f, f) for f in failed]
            listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            verdict, cls = f"Ruled out in {listed} weeks", "out"
        body += (f'<tr class="{cls}"><td class="name">{plain(r["hubs"])}</td>'
                 f'<td class="cost" data-label="Cost per package over a year"><div><span>{money(cost)}</span></div></td>'
                 + (f'<td class="topup" data-label="Of which, raising pay to ${floor:.0f}/h"><div>{topup(d) * 100:.1f}¢'
                    '</div></td>' if show_topup else "")
                 + f'{cells}<td class="verdict" data-label="Verdict"><div>{verdict}</div></td></tr>')
    return f'<div class="score-wrap"><table class="score">{head}{body}</table></div>'


def floor_table(run: dict, a: dict) -> str:
    """Weeks (out of 1,000) that deliver at least the OTD floor, per station layout and kind of week."""
    obj, frames = a["objective"], run["frames"]
    keys = list(frames)
    rows = run["year"].sort_values("year_cost_per_package")
    names = {"normal": "Normal weeks", "holiday_peak": "Holiday peak weeks", "hub_outage": "Station outage weeks"}
    def goal(k):
        o = objective_for(a, k)
        return f'<div class="s">Goal: {o["on_time_week_share"] * 1000:,.0f} weeks at {o["on_time_target"]:.0%} or better</div>'
    head = ("<tr><th>Station layout</th>" + "".join(f"<th>{names.get(k, SCEN.get(k, k))}{goal(k)}</th>" for k in keys)
            + "</tr>")
    body = ""
    for d, r in rows.iterrows():
        cells = ""
        for k in keys:
            if d not in frames[k].index:
                cells += f'<td class="na" data-label="{names.get(k, SCEN.get(k, k))}"><div>Not scored</div></td>'
                continue
            share = frames[k].loc[d, "weeks_meeting_service"]
            ok = share >= objective_for(a, k)["on_time_week_share"]
            o = objective_for(a, k)
            label = (f'{names.get(k, SCEN.get(k, k))} (goal: {o["on_time_week_share"] * 1000:,.0f} at '
                     f'{o["on_time_target"]:.0%})')
            cells += (f'<td class="{"" if ok else "low"}" data-label="{label}"><div>{share * 1000:,.0f} '
                      f'<span class="{"yes" if ok else "x"}">{"✓" if ok else "✗"}</span></div></td>')
        body += f'<tr><td class="name">{plain(r["hubs"])}</td>{cells}</tr>'
    return f'<div class="score-wrap"><table class="score floor">{head}{body}</table></div>'


def short_miss(name: str, r, obj: dict, kind: str = "") -> str:
    """One plain sentence (or two) saying what a station layout missed, in a named kind of week."""
    weeks = f"{kind} weeks" if kind else "weeks"
    if name == "OTD target":
        return (f"OTD is {r['on_time_p50']:.1%} in a typical {kind + ' ' if kind else ''}week, below the "
                f"{obj['on_time_typical_target']:.0%} target.")
    if name == "OTD floor":
        share = r["weeks_meeting_service"]
        reach = f"reach {obj['on_time_target']:.0%} OTD"
        lead = (f"None of the {weeks} {reach}." if share < 0.0005 else f"Only {share:.1%} of {weeks} {reach}.")
        return f"{lead} The floor requires {obj['on_time_week_share']:.0%} of weeks."
    return (f"Dashers earn ${r['driver_net_p50']:.2f} an hour after car costs, below the "
            f"${obj['driver_net_per_hour_floor']:.0f} floor.")


def cost_chart(d: pd.DataFrame, pick: str, obj: dict, pick_word: str = "recommended") -> alt.Chart:
    d = d.rename_axis("design").reset_index()
    d["name"] = d["hubs"].map(plain)
    d["status"] = ["pick" if x == pick else ("meets" if f else "misses") for x, f in zip(d["design"], d["feasible"])]
    d["label"] = [money(r["cost_p50"]) + (f"   {pick_word}" if s == "pick" else
                                          f"   {missed_target(r, obj)}" if s == "misses" else "")
                  for (_, r), s in zip(d.iterrows(), d["status"])]
    d["why"] = [" ".join(x for _, x in missed_targets(r, obj)) or "Meets every target." for _, r in d.iterrows()]
    order = d.sort_values("cost_p50")["name"].tolist()
    lo, hi = d["cost_p50"].min(), d["cost_p90"].max()
    x = alt.X("cost_p50:Q", title=None, scale=alt.Scale(domain=[math.floor(lo * 10) / 10,
                                                               math.ceil(hi * 10) / 10 + 0.4 * max(hi - lo, 0.5)]),
              axis=alt.Axis(format="$.2f", tickCount=5, domain=False, ticks=False, orient="top"))
    y = alt.Y("name:N", sort=order, title=None,
              axis=alt.Axis(domain=False, ticks=False, labelPadding=10, labelLimit=0, labelOverlap=False, labelFontSize=14))
    status = ["pick", "meets", "misses"]
    color = alt.Color("status:N", scale=alt.Scale(domain=status, range=[BLUE, INK, "#d6d6d6"]), legend=None)
    tip = [alt.Tooltip("name:N", title="Station layout"), alt.Tooltip("cost_p50:Q", title="Typical week", format="$.2f"),
           alt.Tooltip("cost_p90:Q", title="Bad week", format="$.2f"),
           alt.Tooltip("on_time_p50:Q", title="OTD, typical week", format=".1%"),
           alt.Tooltip("weeks_meeting_service:Q", title="Weeks meeting the OTD floor", format=".0%")]
    base = alt.Chart(d).encode(y=y)
    line = base.mark_rule(strokeWidth=2).encode(x=x, x2="cost_p90:Q", color=color, tooltip=tip)
    dot = base.mark_circle(size=130, opacity=1, stroke="#fff", strokeWidth=1.5).encode(x=x, color=color, tooltip=tip)
    # Text gets fixed colors per status (a shared color scale would give it the dots' pale grey).
    text = alt.layer(*[alt.Chart(d[d["status"] == s]).mark_text(align="left", dx=10, fontSize=13,
                                                                fontWeight=600 if s == "pick" else 400, color=c)
                       .encode(y=y, x="cost_p90:Q", text="label:N")
                       for s, c in (("pick", BLUE), ("meets", BODY), ("misses", MUTED)) if (d["status"] == s).any()])
    return _style(line + dot + text, alt.Step(38))


def reliability_chart(frames: dict, designs: list[str], goal: float) -> alt.Chart:
    keys = list(frames)
    rows = [{"name": plain(frames[k].loc[x, "hubs"]), "week": SCEN.get(k, k), "share": frames[k].loc[x, "weeks_meeting_service"]}
            for k in keys for x in designs if x in frames[k].index]
    d = pd.DataFrame(rows)
    order = [plain(frames[keys[0]].loc[x, "hubs"]) for x in designs]
    y = alt.Y("name:N", sort=order, title=None,
              axis=alt.Axis(domain=False, ticks=False, labelPadding=10, labelLimit=0, labelOverlap=False, labelFontSize=14))
    x = alt.X("share:Q", title=None, scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%", tickCount=5, domain=False, ticks=False, orient="top"))
    span = alt.Chart(d).mark_rule(color="#d9dbe0", strokeWidth=1.5).encode(y=y, x="min(share):Q", x2="max(share):Q")
    # Each kind of week gets its own lane inside the row, so dots near 100% never hide each other.
    dots = alt.Chart(d).mark_circle(size=90, opacity=1, stroke="#fff", strokeWidth=1.5).encode(
        y=y, x=x, yOffset=alt.YOffset("week:N", sort=[SCEN.get(k, k) for k in keys], scale=alt.Scale(paddingOuter=0.6)),
        color=alt.Color("week:N", scale=alt.Scale(domain=[SCEN.get(k, k) for k in keys],
                                                            range=[WEEK_COLORS.get(k, MUTED) for k in keys]), legend=None),
        tooltip=[alt.Tooltip("name:N", title="Station layout"), alt.Tooltip("week:N", title="Kind of week"),
                 alt.Tooltip("share:Q", title="Weeks meeting the OTD floor", format=".0%")])
    g = pd.DataFrame({"g": [goal], "t": [f"goal {goal:.0%}"]})
    rule = alt.Chart(g).mark_rule(color=INK, strokeDash=[3, 3]).encode(x="g:Q")
    rule_t = alt.Chart(g).mark_text(align="right", dx=-6, baseline="top", dy=4, fontSize=12, color=INK, y=0).encode(
        x="g:Q", text="t:N")
    return _style(rule + dots + rule_t, alt.Step(52))


def week_key(keys) -> str:
    return '<div class="keyrow">' + "".join(
        f'<span><i style="background:{WEEK_COLORS.get(k, MUTED)}"></i>{SCEN.get(k, k)}</span>' for k in keys) + "</div>"


BLUES = ["#e4ebf7", "#b6c5f8", "#8099f6", "#4969f5", "#233069"]   # DoorDash blue ramp, light to dark


def demand_map(tracts, sites) -> folium.Map:
    t = tracts.copy()
    t["bin"] = pd.qcut(t["pkgs_per_km2"], 5, labels=False, duplicates="drop")
    t["pkgs"] = t["pkgs_per_day"].round(0)
    m = folium.Map(location=[44.96, -93.25], zoom_start=9, tiles=None, scrollWheelZoom=False, control_scale=True)
    add_basemap(m)
    folium.GeoJson(
        t, style_function=lambda f: {"fillColor": BLUES[int(f["properties"]["bin"])], "color": "#ffffff",
                                     "weight": 0.3, "fillOpacity": 0.75},
        tooltip=folium.GeoJsonTooltip(["county", "households", "pkgs"],
                                      aliases=["County", "Households", "Packages on an average day"],
                                      localize=True)).add_to(m)
    for _, s in sites.iterrows():
        folium.CircleMarker([s.lat, s.lon], radius=5, color="#ffffff", weight=1.5, fill=True, fill_color=INK,
                            fill_opacity=1, tooltip=f"Candidate station site, ranked {s['rank']}").add_to(m)
    add_labels(m)
    return m


def method_page() -> None:
    dem, hc, rt, h, dr = A0["demand"], A0["hub_candidates"], A0["routing"], A0["hubs"], A0["drivers"]
    pph, share = dem["packages_per_household_per_day"], dem["network_share"]
    per_day = households * pph * share
    days = dem["day_of_week_multipliers"]
    busiest, quietest = max(days, key=days.get), min(days, key=days.get)
    sizes = dem["package_sizes"]
    waves = A0["schedule"]["linehaul_waves"]
    dense = rt["dense_households_per_km2"]
    dense_share = (tracts["households"] / tracts.to_crs(26915).area * 1e6 >= dense).mean()
    cl = sorted(tracts["county"].unique())
    counties = ", ".join(cl[:-1]) + " and " + cl[-1]
    day_name = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday",
                "Sat": "Saturday", "Sun": "Sunday"}
    st.html('<div class="q">Core assumptions</div>'
            '<div class="note" style="max-width:80ch">The simulation rests on a handful of core assumptions about '
            'where people live, how many packages they receive, where stations could go and how long it takes to drive '
            'between places. This page explains how each one was built and where the numbers come from. Real public '
            'data is used wherever it exists, and every other number is a stated assumption that you can change in '
            'the steps that follow.</div>')
    st.write("")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.html(
            '<div class="method">'
            '<h3>Mapping where people live</h3>'
            f'<p>The study area is the seven-county Twin Cities region, which covers {counties} counties. It is '
            f'divided into {len(tracts)} Census tracts, which are small neighborhoods of a few thousand people each. '
            f'Household counts for every tract come from the US Census Bureau\'s American Community Survey for '
            f'2019 to 2023, and tract boundaries come from the Census Bureau\'s 2023 map files. Together the tracts '
            f'hold {households / 1e6:.2f} million households.</p>'
            '<h3>Estimating how many packages people receive</h3>'
            f'<p>About 22.4 billion parcels were shipped in the United States in 2024, according to the Pitney Bowes '
            f'Parcel Shipping Index. Dividing that by roughly 132 million US households and 365 days gives about '
            f'{pph} parcels per household per day across all carriers. The model assumes the network carries '
            f'{share:.0%} of those parcels. With {households / 1e6:.2f} million households, that comes to about '
            f'{round(per_day, -2):,.0f} packages on an average day.</p>'
            f'<p>Volume also changes through the week. {day_name[busiest]} is the busiest day at {days[busiest]:.2f} times '
            f'the average, because weekend orders arrive then, and {day_name[quietest]} is the quietest at '
            f'{days[quietest]:.2f} times the average. During the holiday peak, every day carries '
            f'{A0["scenarios"]["holiday_peak"]["volume_multiplier"]} times its normal volume.</p>'
            '<h3>Spreading packages across neighborhoods and days</h3>'
            '<p>Each tract\'s expected packages are proportional to its households. On every simulated day, the '
            'model draws the actual number of packages for each tract at random around that expected value, so no '
            'two days are exactly alike. Each package is also given a size and a truck arrival time. '
            f'{sizes["small"]["share"]:.0%} of packages are small, {sizes["medium"]["share"]:.0%} are medium and '
            f'{sizes["large"]["share"]:.0%} are large, and the size decides how much space a package takes in a '
            f'Dasher\'s car. Packages reach the stations in {WORDS.get(len(waves), len(waves))} overnight truck '
            f'arrivals. The first trucks arrive at {clock_label(waves[0]["arrive"])}, and the last ones arrive at '
            f'{clock_label(waves[-1]["arrive"])}</p>'
            '</div>')
    with right:
        st_folium(demand_map(tracts, sites), height=520, use_container_width=True, returned_objects=[])
        st.html('<div class="legend" style="font-size:13.5px;color:#353a43;display:flex;flex-wrap:wrap;gap:14px">'
                '<span>Packages per square mile:</span>'
                + "".join(f'<span><i class="sw" style="display:inline-block;width:11px;height:11px;border-radius:3px;'
                          f'background:{c};margin-right:6px"></i>{lab}</span>'
                          for c, lab in [(BLUES[0], "lowest fifth"), (BLUES[2], "middle fifth"),
                                         (BLUES[4], "highest fifth")])
                + f'<span><i style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{INK};'
                  'margin-right:6px"></i>candidate station site</span></div>'
                '<div class="chart-note" style="margin-top:6px">Each shaded area is a Census tract. Darker tracts '
                'receive more packages per square mile on an average day.</div>')

    st.write("")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.html(
            '<div class="method">'
            '<h3>Choosing candidate station sites</h3>'
            f'<p>Possible station sites come from industrial land mapped in OpenStreetMap. A site must cover at least '
            f'{hc["min_site_area_m2"] / 4046.86:.0f} acres, which is enough room for a small sort building, loading '
            'docks and parking for Dashers. Sites used for utilities, waste, scrap or heavy industry are removed.</p>'
            f'<p>From the remaining lots, the model picks a shortlist of {hc["max_candidates"]} sites. It picks them '
            f'one at a time, and each pick is the site that reaches the most households within '
            f'{hc["coverage_radius_km"] / 1.609344:.1f} miles that earlier picks have not already reached. No two sites '
            f'on the shortlist are closer than {hc["min_spacing_km"] / 1.609344:.1f} miles. This spreads the sites '
            'across the metro. '
            'A station layout with three stations uses the three highest-ranked sites, a station layout with four stations uses the top '
            'four, and so on.</p>'
            '<h3>Measuring drive times</h3>'
            '<p>Drive times come from the main road network in OpenStreetMap. For every pair of candidate site and '
            'tract, the model finds the fastest route along those roads and records its time and distance. Routes '
            f'leave during the morning rush, so the model multiplies these free-flow times by '
            f'{rt["stem_congestion_factor"]}.</p>'
            '</div>')
    with right:
        st.html(
            '<div class="method">'
            '<h3>Estimating delivery routes</h3>'
            '<p>The model does not plan every route turn by turn, because that would be too slow to test many '
            'station layouts across many weeks. Instead, it uses a standard estimate from logistics research. The estimate '
            'assumes that the distance a Dasher covers inside a neighborhood grows with the square root of the '
            'number of stops multiplied by the area served. The drive from the station to the neighborhood is then added '
            'to that distance.</p>'
            '<p>To test the estimate, routes for two stations were also planned on real streets with OR-Tools, a route '
            'optimization library, using the same limits. On real roads the routes needed the same number of '
            'Dashers and took within 0 to 7 percent of the estimated time. The formula understated distance inside '
            'neighborhoods, because real routes follow faster main roads that are longer than a direct path. The '
            f'model now multiplies that distance by {rt["local_distance_factor"]:g}, calibrated from the same test, '
            'which brings estimated miles within about 10 percent of the real routes.</p>'
            f'<p>Packages are grouped into stops before routes are built. In dense tracts, which have at least '
            f'{dense * 2.58999:,.0f} households per square mile and make up {dense_share:.0%} of tracts, several apartments '
            'share one building entrance, so one stop can serve many homes. In other tracts, each house is its own '
            f'stop. A route can last at most {rt["max_route_hours"]:g} hours, carry at most '
            f'{rt["max_packages_per_route"]} packages, and hold {rt["car_cargo_m3"] * _CUFT:.0f} cubic feet of '
            'cargo, which is about the usable space in a sedan. Loading a route at the station takes '
            f'{rt["hub_load_minutes"]:g} minutes plus {rt["hub_load_seconds_per_package"]:g} seconds per package.</p>'
            '<p>You can give Dashers bigger vehicles in step 3, but bigger vehicles cost more to run, and fewer '
            'Dashers have them. Running costs rise from the IRS mileage rate for a sedan to an assumed rate for a '
            'cargo van. The share of Dashers with a vehicle of at least a given size is also a stated assumption, '
            'falling from all Dashers for a sedan to a few percent for a cargo van.</p>'
            '<h3>Costs</h3>'
            f'<p>Warehouse rent is ${h["lease_per_sqft_per_year"]:.2f} per square foot per year, based on Twin Cities '
            'industrial market reports from Colliers, Transwestern and CBRE for 2025 and 2026. Those rents are quoted '
            f'before property taxes, insurance and common-area charges, so the model adds '
            f'${h["nnn_per_sqft_per_year"]:.2f} per square foot a year for them. Sort and loading '
            f'labor is paid ${h["sort_wage_per_hour"]:.2f} an hour, based on Bureau of Labor Statistics wages for '
            'material movers in the Minneapolis area. Dashers\' car costs use the IRS 2026 standard mileage rate. '
            f'Fit-out, site opening costs and overhead are stated assumptions, and you can change them in step 2. Sort labor productivity rises with station size, from {h["sizes"]["small"].get("sort_packages_per_labor_hour", h["sort_packages_per_labor_hour"])} packages per labor hour at a small station to {h["sizes"]["large"].get("sort_packages_per_labor_hour", h["sort_packages_per_labor_hour"])} at a large one, because bigger stations can justify more automation. This is also a stated assumption.</p>'
            '</div>')
    nav_buttons("Next: simulation methodology →")


@st.cache_data
def example_weeks(market: str) -> tuple[str, pd.DataFrame] | None:
    """Weekly results for one station layout from the saved 300-week final check (normal weeks)."""
    f = ROOT / "scenarios" / market / "normal" / "final_weeks.parquet"
    if not f.exists():
        return None
    w = pd.read_parquet(f, columns=["design", "hubs", "on_time_rate", "cost_per_package", "extreme_weather_days"])
    design = "3x small" if "3x small" in set(w["design"]) else w["design"].iloc[0]
    return design, w[w["design"] == design].reset_index(drop=True)


def otd_histogram(w: pd.DataFrame, floor: float) -> alt.Chart:
    d = w.assign(kind=lambda x: x["extreme_weather_days"].gt(0).map({True: "Week with a storm day",
                                                                       False: "Week without a storm"}))
    d["otd"] = (d["on_time_rate"] * 100).clip(lower=90.01)    # weeks below 90% fold into the first bar
    bars = alt.Chart(d).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("otd:Q", bin=alt.Bin(step=0.5, extent=[90, 100]), title="Weekly OTD (%)",
                axis=alt.Axis(domain=False, ticks=False, format=".0f")),
        y=alt.Y("count():Q", title="Weeks", axis=alt.Axis(domain=False, ticks=False, tickCount=4)),
        color=alt.Color("kind:N", sort=["Week without a storm", "Week with a storm day"],
                        scale=alt.Scale(range=[BLUE, GOLD]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("kind:N", title=""), alt.Tooltip("count():Q", title="Weeks")])
    g = pd.DataFrame({"x": [floor * 100], "t": [f"OTD floor {floor:.0%}"]})
    rule = alt.Chart(g).mark_rule(color=INK, strokeDash=[3, 3]).encode(x="x:Q")
    txt = alt.Chart(g).mark_text(align="right", dx=-5, y=8, fontSize=12, color=INK).encode(x="x:Q", text="t:N")
    return _style(bars + rule + txt, 230)


def cost_histogram(w: pd.DataFrame) -> alt.Chart:
    p50, p90 = w["cost_per_package"].quantile([0.5, 0.9])
    tallest = pd.cut(w["cost_per_package"], 30).value_counts().max()
    headroom = math.ceil(tallest * 1.3)          # empty band above the bars for the two labels
    bars = alt.Chart(w).mark_bar(color="#c9d3fb", cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("cost_per_package:Q", bin=alt.Bin(maxbins=30), title="Weekly cost per package",
                axis=alt.Axis(format="$.2f", domain=False, ticks=False)),
        y=alt.Y("count():Q", title="Weeks", axis=alt.Axis(domain=False, ticks=False, tickCount=4),
                scale=alt.Scale(domainMax=headroom)),
        tooltip=[alt.Tooltip("count():Q", title="Weeks")])
    marks = pd.DataFrame({"x": [p50, p90], "t": [f"Typical week {money(p50)}", f"Bad week {money(p90)}"]})
    rule = alt.Chart(marks).mark_rule(color=INK, strokeWidth=1.5).encode(x="x:Q")
    # Typical-week label reads leftward from its line, bad-week label rightward, so they never collide.
    left = alt.Chart(marks.iloc[[0]]).mark_text(align="right", dx=-6, y=8, fontSize=12.5, fontWeight=600,
                                                color=INK).encode(x="x:Q", text="t:N")
    right = alt.Chart(marks.iloc[[1]]).mark_text(align="left", dx=6, y=8, fontSize=12.5, fontWeight=600,
                                                 color=INK).encode(x="x:Q", text="t:N")
    return _style(bars + rule + left + right, 230)


def methodology_page() -> None:
    obj, unc, drv, h = A0["objective"], A0["uncertainty"], A0["drivers"], A0["hubs"]
    sch, wpy, run = A0["schedule"], A0["weeks_per_year"], CTL["run"]
    refs = A0["reference_checks"]
    lanes = h["sizes"]["small"]["loading_lanes"]
    st.html('<div class="q">Simulation methodology</div>'
            '<div class="note" style="max-width:80ch">This page explains how the simulation turns the assumptions '
            'into a recommendation. It covers what happens on one simulated day, how days add up to weeks, how the '
            'model keeps the comparison between station layouts fair, how it picks a winner, and how much weight the '
            'results can bear.</div>')
    st.write("")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.html(
            '<div class="method">'
            '<h3>What kind of simulation this is</h3>'
            '<p>This is a Monte Carlo simulation that runs one day at a time. The model follows every package '
            'individually from the truck that brings it to the station until a Dasher delivers it or it rolls over to '
            'the next day. Anything uncertain, such as volume, late trucks or Dasher no-shows, is drawn at random '
            'for each simulated week. Running many weeks shows the full range of outcomes that a station layout can '
            'produce, not just its average.</p>'
            '<h3>What happens on one simulated day</h3>'
            f'<p>Trucks reach the stations in waves between {clock_label(sch["linehaul_waves"][0]["arrive"])} and '
            f'{clock_label(sch["linehaul_waves"][-1]["arrive"])} each night. Each truck has a '
            f'{unc["truck_late_prob"]:.0%} '
            'chance of arriving late, and a small share of packages miss their truck and come on the next one.</p>'
            '<p>Each station sorts packages in the order they arrive, at its sort speed. Once a station has sorted its '
            f'normal staffed volume, which is {h["staffed_share_of_capacity"]:.0%} of its overnight capacity, extra '
            'packages are sorted with overtime labor, which is slower and costs more. Packages sorted after the '
            f'{clock_label(sch["sort_cutoff"])} cut-off miss that day\'s routes.</p>'
            f'<p>In holiday peak weeks, each station can add temporary staff and longer shifts, which raises its '
            f'sort rate by {A0["peak_flex"]["sort_capacity_boost"]:.0%}. The extra volume is paid at the overtime '
            'rate, and each open station pays a weekly setup cost. Packages that are still not sorted at the '
            'cut-off wait a day. In step 5 you can also switch on a partner carrier, which takes those packages '
            f'instead, up to {A0["partner"]["max_share_of_day"]:.0%} of the day\'s packages at '
            f'${A0["partner"]["cost_per_package"]:.2f} each, and delivers {A0["partner"]["on_time_rate"]:.0%} of '
            'them on time. It is off by default.</p>'
            '<p>At the cut-off, the model groups each station\'s sorted packages into routes. It sweeps around the station '
            'by direction and adds neighborhoods to a route until the route reaches its time, package or cargo '
            'limit. Each station loads routes through a fixed number of loading lanes, so Dashers receive staggered '
            f'pickup times. A small station has {lanes} lanes. Loading a route takes a few minutes plus a few seconds per '
            'package, so bigger routes take longer to load and push later routes back.</p>'
                        '<p>Before a route is offered, the model checks what it would pay per hour after car costs, counting the '
            'Dasher\'s unpaid commute. If that is below the Dasher pay floor, the route\'s base pay is raised to the '
            'floor, and the raise counts as a cost of the network. Station layouts with long drives therefore pay '
            'more to keep Dashers whole rather than being ruled out.</p>'
'<p>Each route is then offered to Dashers. The chance that a Dasher accepts it follows an S-shaped '
            'curve based on the route\'s pay per hour after car costs. If routes need a vehicle bigger than a sedan, '
            'the chance is scaled down by the share of Dashers who have one. The hours include the Dasher\'s unpaid '
            'drive from home to the station. The model assumes that Dashers live where households live and work at '
            'their nearest station, so each station\'s commute is the average drive from the neighborhoods it serves. A '
            'network with fewer stations therefore has longer commutes and lower pay per hour, so more routes need a pay '
            'increase before someone accepts them. If nobody accepts, the route is offered again '
            f'with pay raised by {drv["surge_step"]:.0%} of the base pay each time, up to '
            f'{"twice" if drv["surge_max"] == 2 else str(drv["surge_max"]) + " times"} the base pay. On '
            'storm days, fewer Dashers are available and pay can only rise to a lower limit, so some routes may go out '
            'unclaimed. A Dasher who accepts a route sometimes fails to show up. In that case, the route is offered '
            f'again and leaves {drv["no_show_rescue_delay_hours"]:g} hours later.</p>'
            f'<p>A package counts toward OTD if its route finishes by {clock_label(sch["promise_end"])} on the day it '
            'was due. Packages '
            'that missed the cut-off or sat on an unclaimed route carry over to the next day, and they are loaded '
            'first the following morning.</p>'
            '<h3>From days to weeks</h3>'
            '<p>The model runs seven days in a row, carrying late packages from one day into the next. For each '
            'week it records the OTD of the packages that were due, the cost per delivered package, Dasher pay per '
            'hour after car costs, miles driven per package and several other measures.</p>'
            '</div>')
    with right:
        st.html(
            '<div class="method">'
            '<h3>Keeping the comparison fair</h3>'
            '<p>Every station layout faces exactly the same simulated weeks. The same packages appear on the same days, '
            'the same trucks run late and the same Dashers fail to show up. Because the station layouts share their luck, '
            'any difference in results comes from the station layout itself. The full search also checks its top station layouts '
            'on a second, fresh set of weeks, so a station layout cannot win only because it happened to suit the first '
            'set.</p>'
            '<h3>Typical week and bad week</h3>'
            '<p>Results are reported as ranges rather than averages. The typical week is the median week, so half '
            'of the simulated weeks were better and half were worse. The bad week is the worst week in ten. For cost '
            'that is the 90th percentile, and for OTD and Dasher pay it is the 10th percentile.</p>'
            '</div>')
        ex = example_weeks(MARKET)
        if ex:
            design, w = ex
            name = spoken(w["hubs"].iloc[0])
            st.html(f'<div class="chart-title">Weekly OTD for {name} in {len(w)} normal weeks</div>'
                    '<div class="chart-note">Each bar counts the weeks that reached a given OTD. Most weeks sit well '
                    'above the floor, and most of the weeks below it had a storm day.</div>')
            st.altair_chart(otd_histogram(w, obj["on_time_target"]), width="stretch")
            st.html(f'<div class="chart-title">Weekly cost per package for {name}</div>'
                    '<div class="chart-note">The typical week is the median. The bad week is the 90th percentile, '
                    'which means only one week in ten cost more.</div>')
            st.altair_chart(cost_histogram(w), width="stretch")

    st.write("")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.html(
            '<div class="method">'
            '<h3>How a station layout wins</h3>'
            '<p>The model first runs every station layout through one average day with no randomness. It drops any station layout '
            f'whose OTD falls below {obj["screen_min_on_time"]:.0%} on that day or whose cost is more than '
            f'{obj["screen_max_cost_vs_best"] - 1:.0%} above the cheapest station layout.</p>'
            '<p>When you reach the Run simulation page in the menu, you can choose between two kinds of run. A quick run '
            f'simulates only the {WORDS.get(run["quick"]["top_designs_per_scenario"])} cheapest remaining station layouts '
            f'from each kind of week, for {run["quick"]["weeks"]} weeks each, and finishes in seconds. A full run '
            f'simulates every remaining station layout for {run["full"]["search_weeks"]} weeks and then rechecks the '
            f'closest ones on {run["full"]["recheck_weeks"]:,} fresh weeks, which takes several minutes.</p>'
            f'<p>A station layout meets the targets in a kind of week if its typical week reaches '
            f'{obj["on_time_typical_target"]:.0%} OTD, at least {obj["on_time_week_share"]:.0%} of its weeks reach '
            f'{obj["on_time_target"]:.0%} OTD. Holiday peak weeks use lower targets: '
            f'{obj["peak_targets"]["on_time_typical_target"]:.0%} OTD in a typical peak week, and at least '
            f'{obj["peak_targets"]["on_time_target"]:.0%} OTD in {obj["peak_targets"]["on_time_week_share"]:.0%} of '
            'peak weeks, because carriers plan for lower performance at peak. Dasher pay is not a pass or fail test. Every route is paid at least '
            f'${obj["driver_net_per_hour_floor"]:.0f} an hour after car costs, and the cost of raising pay to that '
            'level is part of each station layout\'s cost per package. Weeks with a storm day are left out of the '
            'weekly OTD test, in '
            'the same way that carriers exclude severe weather from their service guarantees, but they are still '
            'counted in cost.</p>'
            f'<p>The yearly cost of a station layout weights its typical-week cost by {wpy["normal"]} normal weeks, '
            f'{wpy["holiday_peak"]} peak weeks and {wpy["hub_outage"]} outage weeks. The recommendation is the '
            'cheapest station layout that meets the targets in every kind of week. When two station layouts are within '
            f'{obj["year_round_cost_tie"]:.0%} of each other on yearly cost, the model treats them as tied and picks '
            'the one that holds OTD better in its hardest kind of week. If no station layout meets every target, the model '
            'reports the closest one and says so.</p>'
            '</div>')
    with right:
        st.html(
            '<div class="method">'
            '<h3>How much to trust the numbers</h3>'
            f'<p>The share of weeks that meets the OTD floor is an estimate, and its accuracy depends on how many '
            'weeks the model simulates. That is why the app offers the two kinds of run described under How a station '
            'layout wins. A quick '
            f'run uses {run["quick"]["weeks"]} weeks, so its estimate can be off by about '
            f'{precision(run["quick"]["weeks"])} percentage points. It is good for seeing roughly how the answer moves '
            f'when you change an assumption. A full run rechecks the closest station layouts on '
            f'{run["full"]["recheck_weeks"]:,} weeks, so its estimate is accurate to about '
            f'{precision(run["full"]["recheck_weeks"])} percentage points, which is enough to separate most close '
            'calls.</p>'
            '<p>The model\'s outputs were compared with public figures. Its cost per package falls at the low end of '
            f'published estimates of ${refs["cost_per_package"]["low"]:.0f} to '
            f'${refs["cost_per_package"]["high"]:.0f}, which is expected because the model leaves out long-haul '
            'trucking and upstream sorting. Its typical OTD sits within the range reported for national carriers, '
            'and Dasher pay before car costs sits within the range that gig delivery platforms advertise.</p>'
            '<p>The main simplification is the route estimate. Routes are estimated with a formula rather than '
            'planned on real streets. A check against routes planned on real roads with OR-Tools found the same '
            'number of routes and similar hours. Distance was calibrated from that check, but it rests on two stations '
            'and one day. Demand is also scaled '
            'from household counts rather than real orders, and the station sites come from a fixed shortlist.</p>'
            '</div>')
    nav_buttons("Next: set the volume assumptions →")


# ---------------------------------------------------------------- runs

def get_screen(a: dict) -> dict:
    """Average-day screen for these settings (well under a second); cached per settings."""
    key = ("screen", settings_hash(a))
    store = run_store()
    if key not in store:
        store[key] = screen_all(a, MARKET)
    return store[key]


def get_estimate(a: dict, si: dict) -> dict:
    """Instant estimate for these settings; cached, since it may screen larger networks too."""
    key = ("estimate", settings_hash(a))
    store = run_store()
    if key not in store:
        store[key] = quick_estimate(a, si["screens"], MARKET)
    return store[key]


def saved_run(a: dict, mode: str) -> dict | None:
    """A finished run for these settings: from this server's memory, or the saved default runs."""
    key = (settings_hash(a), mode)
    store = run_store()
    if key not in store:
        saved = load_baseline(MARKET, mode, a)
        if saved:
            store[key] = saved
    return store.get(key)


def job_key(a: dict, mode: str) -> tuple:
    return ("job", settings_hash(a), mode)


def start_job(a: dict, mode: str) -> tuple:
    """Run in a background thread so a long run survives page changes. One run at a time per server: a second
    run waits for the first, so two visitors never split the same CPUs."""
    store = run_store()
    key = job_key(a, mode)
    job = store.get(key)
    if job and job["state"] in ("queued", "running"):
        return key
    lock = store.setdefault("cpu_lock", threading.Lock())
    screen_user, screen_default = get_screen(a), get_screen(A0)
    need_default = saved_run(A0, mode) is None and settings_hash(a) != settings_hash(A0)
    job = {"state": "queued", "mode": mode, "frac": 0.0, "text": "Waiting for another run on this server to finish",
           "stats": {"weeks": 0, "packages": 0, "total": planned_weeks(screen_user, mode, CTL["run"])},
           "started": time.time(), "result": None, "error": None, "a": a}
    store[key] = job

    def work():
        with lock:
            job.update(state="running", started=time.time(), text="Checking every station layout on an average day")
            try:
                def report(prefix=""):
                    return lambda f, text, stats: job.update(frac=f, text=prefix + text, stats=stats)
                r = live_run(a, mode, MARKET, progress=report(), screen_info=screen_user)
                store[(settings_hash(a), mode)] = r
                if need_default:
                    job.update(frac=0.0, text="Running the default settings for comparison")
                    store[(settings_hash(A0), mode)] = live_run(
                        A0, mode, MARKET, progress=report("Default settings for comparison. "),
                        screen_info=screen_default)
                job.update(state="done", result=r, frac=1.0)
            except Exception as err:      # shown to the visitor; the server keeps running
                job.update(state="failed", error=repr(err))
    threading.Thread(target=work, daemon=True).start()
    return key


def show_result(a: dict, mode: str, run: dict) -> None:
    st.session_state.result = {"run": run, "base": saved_run(A0, mode) or run, "a": a, "changes": changed(),
                               "mode": mode}
    st.session_state.page = RESULTS


def elapsed_label(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m} minute{'s' if m != 1 else ''} {sec} seconds" if m else f"{sec} seconds"


def big_number(n: int) -> str:
    return f"{n / 1e9:.2f} billion" if n >= 1e9 else f"{n / 1e6:.1f} million" if n >= 1e6 else f"{n:,}"


@st.fragment(run_every=1.0)
def progress_panel(key: tuple) -> None:
    job = run_store().get(key)
    if not job:
        return
    stats = job["stats"]
    if job["state"] in ("queued", "running"):
        st.progress(min(job["frac"], 1.0), text=job["text"])
        if job["state"] == "running":
            st.html(f'<div class="note" style="max-width:none">So far the model has simulated '
                    f'{stats["weeks"]:,} of about {stats["total"]:,} weeks, counting each station layout separately, '
                    'and covered '
                    f'{big_number(stats["packages"])} packages. Time elapsed: '
                    f'{elapsed_label(time.time() - job["started"])}. You can leave this page and come back, and the '
                    'run will keep going.</div>')
    elif job["state"] == "done":
        st.session_state.pop("active_job", None)
        show_result(job["a"], job["mode"], job["result"])
        st.rerun(scope="app")
    else:
        st.error(f"The run stopped with an error: {job['error']}")
        st.session_state.pop("active_job", None)


def seconds_label(s: float) -> str:
    if s < 10:
        return "a few seconds"
    if s < 55:
        return f"about {int(math.ceil(s / 5) * 5)} seconds"
    m = s / 60
    return "about a minute" if m < 1.5 else f"about {math.ceil(m)} minutes"


def summary(run: dict, a: dict) -> dict:
    y = run["year"]
    pick = y.index[0]
    p = y.iloc[0]
    n = run["frames"].get("normal", next(iter(run["frames"].values()))).loc[pick]
    return {"pick": pick, "name": plain(p["hubs"]), "said": spoken(p["hubs"]), "hubs": p["hubs"], "year_cost": p["year_cost_per_package"],
            "cost": n["cost_p50"], "cost_bad": n["cost_p90"], "on_time": n["on_time_p50"], "on_time_bad": n["on_time_p10"],
            "worst_share": p["worst_weeks_meeting_service"], "driver": n["driver_net_p50"],
            "km": n["km_per_package_p50"],
            "meets": int(p["scenarios_meeting_targets"]), "of": len(run["frames"])}


# ---------------------------------------------------------------- page

tracts, sites, drive = load_geo(MARKET)
households = tracts["households"].sum()
HOUSEHOLDS = households
short = ["overview", "core assumptions", "simulation methodology"] + [s["name"].lower() for s in STEPS] + ["run", "results"]
FIRST_STEP = 3
names = ["Overview", "Core assumptions", "Methodology"] + [f"{i}. {s['name']}" for i, s in enumerate(STEPS, 1)] + ["Run simulation", "Results"]
RESULTS = len(names) - 1
page = st.session_state.page
has_result = "result" in st.session_state

MARK = ('<svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true">'
        f'<rect width="34" height="34" rx="9" fill="{RED}"/>'
        '<path d="M10 23 L17 11 L24 23 Z" fill="none" stroke="#fff" stroke-width="2" stroke-linejoin="round"/>'
        '<circle cx="17" cy="11" r="3" fill="#fff"/><circle cx="10" cy="23" r="3" fill="#fff"/>'
        '<circle cx="24" cy="23" r="3" fill="#fff"/></svg>')
st.html(f'<div class="masthead">{MARK}<div><div class="app-name">Delivery Station Planner</div>'
        '<div class="app-sub">Simulate next-day DoorDash Parcel delivery in Minneapolis–St. Paul and find the '
        'cheapest network of delivery stations that still delivers on time.</div></div></div>'
        # Phones only: scroll the swipeable menu so the current page's item is in view (desktop menus wrap instead).
        f"""<script>(function(){{
  if (!window.matchMedia('(max-width: 640px)').matches) return;
  var tries = 0;
  (function center(){{
    var nav = document.querySelector('.st-key-nav'), row = document.querySelector('.st-key-nav_scroll');
    if (!nav || !row) {{ if (tries++ < 20) setTimeout(center, 100); return; }}
    var b = row.querySelector('.st-key-nav{page} button');
    if (b) {{
      var r = b.getBoundingClientRect(), n = row.getBoundingClientRect();
      row.scrollLeft += (r.left - n.left) - (n.width - r.width) / 2;
    }}
    var mark = function() {{
      nav.classList.toggle('at-end', row.scrollWidth > row.clientWidth + 4 &&
                                     row.scrollLeft + row.clientWidth >= row.scrollWidth - 4); }};
    if (!row.dataset.watch) {{ row.dataset.watch = '1'; row.addEventListener('scroll', mark, {{passive: true}}); }}
    nav.classList.remove('at-end');
    setTimeout(mark, 600);
  }})();
}})();</script>""", unsafe_allow_javascript=True)
def nav_item(i: int) -> None:
    """Every item is the same kind of element (a button), so clicking never changes the row's layout.
    The current page is styled through its key; Results stays disabled until there is a run."""
    st.button(names[i], key=f"nav{i}", type="tertiary", on_click=go, args=(i,),
              disabled=(i == RESULTS and not has_result))


# Style the current page's menu item (bold, underlined) without changing its size.
st.html(f"""<style>
.st-key-nav{page} button, .st-key-nav{page} button:hover {{ background: {INK} !important; color: #fff !important;
  font-weight: 700 !important; text-decoration: none !important; cursor: default; }}
</style>""")


# Three groups: the explanation pages, the assumption steps (their own shaded section), and running the model.
NAV_GROUPS = [("about", range(0, FIRST_STEP), None),
              ("assume", range(FIRST_STEP, FIRST_STEP + len(STEPS)), "Assumptions"),
              ("run", range(FIRST_STEP + len(STEPS), len(names)), None)]
def nav_group(g: str, idxs, title: str | None) -> None:
    with st.container(key=f"nav_{g}", horizontal=True, vertical_alignment="center", width="content"):
        if title:
            st.html(f'<span class="nav-group">{title}</span>')
        for i in idxs:
            nav_item(i)
        if g == "run":
            st.button("↺ Reset to defaults", key="nav_reset", type="tertiary", on_click=reset_all,
                      disabled=not changed(), help="Set every assumption back to its default value.")


# On desktop the scroll wrapper is invisible (display: contents), so all three groups share one wrapping row.
# On phones the explanation pages and assumption steps become one swipeable row, with the run group on its own row.
with st.container(key="nav", horizontal=True, vertical_alignment="center"):
    with st.container(key="nav_scroll", horizontal=True, vertical_alignment="center"):
        for g, idxs, title in NAV_GROUPS[:2]:
            nav_group(g, idxs, title)
    nav_group(*NAV_GROUPS[2])
st.html('<hr class="nav under-menu">')

a = settings()


def nav_buttons(next_label: str | None = None):
    """Back on the left, Next on the right, in one row that sizes each button to its label."""
    st.write("")
    with st.container(key="pager", horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        if page > 0:
            st.button(f"← Back: {short[page - 1]}", on_click=go, args=(page - 1,), key="pager_back")
        else:
            st.html("<span></span>")
        if next_label:
            st.button(next_label, on_click=go, args=(page + 1,), type="primary", key="pager_next")


if page == 0:
    obj0 = a["objective"]
    n_sites = len(sites)
    n_layouts = len(get_screen(A0)["screens"]["normal"])
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.html(
            '<div class="intro">'
            '<div class="q" style="font-size:34px">Using a basic simulation to find the lowest-cost delivery station network '
            'for DoorDash Parcel in Minneapolis–St. Paul</div>'
            '<p>In a parcel network like DoorDash Parcel, packages are sorted overnight at local delivery stations. '
            'A delivery station receives overnight trucks, sorts packages into routes, and is where Dashers pick up '
            'their routes in the morning. This project asks how many delivery stations the network in '
            'Minneapolis–St. Paul should run and how large each one should be. The rest of the app calls them '
            'stations for short.</p>'
            f'<p>The model simulates next-day delivery. Packages reach the stations overnight, and each one must be '
            f'delivered by {clock_label(A0["schedule"]["promise_end"])} the following day to count as on time.</p>'
            '<p>Each extra station adds rent and staff costs, but it also shortens two drives for Dashers. One is the '
            'drive from home to the station, and the other is the drive from the station to the first delivery. Dashers are '
            'not delivering during either drive, so shorter drives raise their pay per hour, and fewer routes need '
            'their base pay raised before someone accepts them. Extra stations also give the '
            'network a backup if another station closes. A network with fewer stations costs less in rent, but it can fall '
            'behind during the holiday peak or when a station shuts down for a day.</p>'
            f'<p>The model tests {n_layouts} station layouts. A station layout is one possible network: how many stations it '
            'has, how large each one is, and which shortlisted sites they occupy. The model runs every station layout '
            'through the same set of simulated '
            'weeks, including normal weeks, holiday peak weeks and weeks when a station closes. It then recommends the '
            'cheapest station layout that still meets the on-time delivery (OTD) targets.</p>'
            f'<p>Before you run the simulation, you will set its assumptions in {WORDS[len(STEPS)]} steps. These '
            'steps cover package volume, station costs, Dasher pay, service targets and disruptions. Every assumption '
            'already has a default value based on public data or a stated estimate. You can run the simulation '
            'with the defaults, or you can change any of them first.</p></div>')
        st.write("")
        c, _ = st.columns([1.3, 1])
        with c:
            st.button("Next: core assumptions →", type="primary", on_click=go, args=(1,), width="stretch")
            with st.container(horizontal=True, horizontal_alignment="center"):
                st.button("Skip to setting the assumptions", type="tertiary", on_click=go, args=(FIRST_STEP,))
    with right:
        st.html(
            '<div class="intro"><dl>'
            '<dt>Decision</dt><dd>The model decides how many stations to open and how large each one should be. The '
            'station sites come from a shortlist of industrial sites chosen to cover the metro area.</dd>'
            '<dt>Goal</dt><dd>The goal is the lowest cost per package over a full year.</dd>'
            f'<dt>Guardrails</dt><dd>A station layout must reach {obj0["on_time_typical_target"]:.0%} OTD in a typical '
            f'week. It must also reach at least {obj0["on_time_target"]:.0%} OTD in '
            f'{obj0["on_time_week_share"]:.0%} of weeks. Holiday peak weeks have their own, lower targets: '
            f'{obj0["peak_targets"]["on_time_typical_target"]:.0%} OTD in a typical peak week, and at least '
            f'{obj0["peak_targets"]["on_time_target"]:.0%} in {obj0["peak_targets"]["on_time_week_share"]:.0%} of peak '
            'weeks, in line with how carriers plan for peak.</dd>'
            f'<dt>Dasher pay</dt><dd>Every route pays at least ${obj0["driver_net_per_hour_floor"]:.0f} an hour after '
            'car costs. When a route would pay less, its base pay is raised, and the extra pay is counted in that '
            'station layout\'s cost.</dd>'
            '<dt>Tested against</dt><dd>Each station layout is tested in normal weeks, in holiday peak weeks and in weeks '
            'when the busiest station closes for a day.</dd>'
            '</dl>'
            '</div>')
        steps = [
            ("Set the assumptions", f"You set the assumptions in steps 1 to {len(STEPS)}. Each step already has "
             "default values filled in."),
            ("Screen every station layout", "The model runs every station layout through one average day. It drops any station layout that "
             "clearly misses the targets or costs far more than the others."),
            ("Simulate day by day", "The remaining station layouts are simulated through the same random weeks. Each night, "
             "trucks bring packages to the stations, and the stations sort them until the cut-off time. Each morning, the "
             "sorted packages are grouped into routes and offered to Dashers. If no Dasher accepts a route, its pay "
             "goes up until someone does. Packages that are not delivered on time carry over to the next day."),
            ("Score each station layout", "The model reports cost per package, OTD and Dasher pay for a typical week and for "
             "a bad week. The recommended station layout is the cheapest one that meets every target in every kind of "
             "week."),
            ("Compare with the defaults", "The results page compares your settings with the defaults, so you can see "
             "which changes moved the answer."),
        ]
        st.html('<div class="section" style="margin-top:28px">How the simulation works</div><div class="flow">'
                + "".join(f'<div class="flow-step"><div class="flow-title">{h}</div><p>{b}</p></div>'
                          for h, b in steps) + '</div>')

    st.html('<hr class="nav" style="margin:40px 0 28px">')
    rows = [
        ("Where should packages be sorted, and how much capacity is needed?",
         "The model compares station counts and sizes, and it accounts for sort speed, staffing and overtime."),
        ("What delivery promise and cut-off times should the network set?",
         "You can change the sort cut-off and the OTD targets in step 4."),
        ("How much capacity is needed for peak season and disruptions?",
         "The model tests holiday peak weeks and station outage weeks, along with storms and late trucks."),
        ("Can routes be attractive to Dashers and still profitable?",
         "The model sets route pay, raises it above the base rate when a route goes unaccepted, and enforces a minimum hourly pay "
         "for Dashers."),
        ("How do cost, speed, productivity, Dasher experience and reliability vary from week to week?",
         "The model reports cost per package, OTD, miles driven per package and Dasher pay as "
         "ranges, showing both a typical week and a bad week."),
    ]
    st.html('<div class="section">What a network planner can use it for</div>'
            '<table class="jd"><tr><th>Planning question</th><th>How the model handles it</th></tr>'
            + "".join(f"<tr><td>{q}</td><td>{h}</td></tr>" for q, h in rows) + "</table>")

elif page == 1:
    method_page()

elif page == 2:
    methodology_page()

elif page < FIRST_STEP + len(STEPS):
    step = STEPS[page - FIRST_STEP]
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.html(f'<div class="stepno">Step {page - FIRST_STEP + 1} of {len(STEPS)}</div>'
                f'<div class="q">{step["title"]}</div>'
                f'<div class="ask">{step["question"]}</div><div class="note">{step["note"]}</div>'
                f'<div class="uses"><b>How the simulation uses this.</b> {step.get("uses", "")}</div>')
        n_changed = sum(1 for c in step["controls"] if vals[c["path"]] != get_path(A0, c["path"]))
        if n_changed:
            st.button("Reset this step to the defaults", type="tertiary",
                      on_click=reset_step, args=(step,))
    with right:
        st.html('<div class="lead-in">The sliders below start at the default values. You can keep the defaults '
                'or move any slider to set your own assumption. Any value you change is marked.<br><br>'
                '<span class="tag fixed">Fixed</span>settings are used exactly as set in every simulated week. '
                '<span class="tag vary">Varies</span>settings set an average or the odds, and each simulated day '
                'or week draws its own value around them. That randomness is what lets the model show typical '
                'weeks and bad weeks rather than a single answer.</div>')
        for c in step["controls"]:
            control(c)
    previews = {
        "volume": ("Packages per day", "This chart shows the number of packages on each day of the week at the "
                   "current settings.", lambda: volume_preview(a, households)),
        "hubs": ("How much one station can sort overnight", sort_note(a), lambda: sort_preview(a, households)),
        "drivers": ("How likely a Dasher is to accept a route", claim_note(a), lambda: claim_preview(a)),
        "service": ("When packages arrive, get sorted and are delivered", schedule_note(a),
                    lambda: schedule_preview(a)),
        "disruptions": ("What a year looks like", "This chart shows how many of each event to expect in a year at "
                        "the current settings.", lambda: disruption_preview(a)),
    }
    if step["key"] in previews:
        title, note, fn = previews[step["key"]]
        st.html(f'<div class="chart-title">{title}</div><div class="chart-note">{note}</div>'
                + (sort_key(a) if step["key"] == "hubs" else ""))
        st.altair_chart(fn(), width="stretch")
    nav_buttons(f"Next: {short[page + 1]} →" if page < FIRST_STEP + len(STEPS) - 1 else "Next: run the simulation →")

elif page == RESULTS - 1:
    ch = changed()
    rc = CTL["run"]
    st.html('<div class="q">Run the simulation</div>'
            '<div class="note" style="max-width:80ch">Choose how to run the model with the assumptions you set. The '
            'instant estimate is a rough first answer. The quick run and the full run are live simulations that follow '
            'every package day by day, so a run with new settings takes real computing time. If your settings match a '
            'run that has already finished on this server, the app shows that result and lets you run it again from '
            'scratch.</div>')
    if ch:
        rows = "".join(f"<tr><td>{CONTROLS[p]['label']}</td><td>{fmt(CONTROLS[p], get_path(A0, p))}</td>"
                       f"<td><b>{fmt(CONTROLS[p], v)}</b></td></tr>" for p, v in ch.items())
        st.html('<div style="max-width:760px"><table class="cmp"><tr><th>Your changes</th><th>Default</th>'
                '<th>Yours</th></tr>' + rows + "</table></div>")
    else:
        st.html('<div class="note" style="max-width:none">You have not changed any assumptions, so the run will use '
                'the default settings. You can go back to any step to change them.</div>')
    st.write("")
    # New settings need a fresh average-day screen of every layout before the run times can be shown. The page above is
    # already on screen; the spinner shows while the screen runs (a few seconds on the hosted server).
    if ("screen", settings_hash(a)) in run_store():
        si = get_screen(a)
        est = get_estimate(a, si)
    else:
        with st.spinner("Checking every station layout on an average day with your settings. This takes a few "
                        "seconds."):
            si = get_screen(a)
            est = get_estimate(a, si)

    q, f = rc["quick"], rc["full"]
    copy_ = {
        "quick": ("Quick run",
                  f'The quick run simulates the {WORDS.get(q["top_designs_per_scenario"])} cheapest station layouts from '
                  f'each kind of week for {q["weeks"]} weeks each. It is useful for seeing roughly how the answer moves '
                  'when you change an assumption. With this few weeks, the share of weeks that meets the OTD floor can '
                  f'be off by about {precision(q["weeks"])} percentage points, so station layouts that are close on cost '
                  'may swap places in the full run.'),
        "full": ("Full run",
                 f'The full run simulates every station layout that passes the screen for {f["search_weeks"]} weeks in '
                 f'each kind of week. It then rechecks the closest station layouts on {f["recheck_weeks"]:,} fresh weeks '
                 'in every kind of week. Its estimate of the share of weeks that meets the OTD floor is accurate to '
                 f'about {precision(f["recheck_weeks"])} percentage points, which is enough to separate most close '
                 'calls. This is the same method that produced the recommendation shown with the default settings.'),
    }
    tab_est, tab_quick, tab_full = st.tabs(["Instant estimate", "Quick run", "Full run"])
    with tab_est:
        lead = ('On an average day without disruptions, the cheapest station layout that meets the OTD targets is '
                if est.get("passes_all", True) else
                'On an average day without disruptions, no station layout the model tried meets the OTD targets. The '
                'closest is ')
        st.html('<div class="tab-body"><div class="note" style="max-width:75ch">' + lead +
                f'<b>{spoken(get_design(est["design"], a).label())}</b>, at about {money(est["year_cost"])} per '
                'package.</div><div class="note" style="max-width:75ch">This estimate takes under a second because it '
                'skips the random weeks. Late trucks, Dasher no-shows and swings in volume often mean the network '
                'needs more capacity than an average day suggests, so use the quick run or the full run to test '
                'it.</div></div>')
    for tab, mode in ((tab_quick, "quick"), (tab_full, "full")):
        title, text = copy_[mode]
        weeks = planned_weeks(si, mode, rc)
        secs = estimate_seconds(si, mode, rc)
        done = saved_run(a, mode)
        with tab:
            st.html(f'<div class="tab-body"><div class="note" style="max-width:75ch">{text}</div>'
                    f'<div class="note" style="max-width:75ch">This run simulates about {weeks:,} weeks, counting each '
                    f'station layout separately, and should take {seconds_label(secs)} on this server. If no layout of up to '
                    f'{WORDS.get(max(a["design_grid"]["hub_counts"]))} stations meets every target, the run also tries '
                    'larger networks and takes longer.</div></div>')
            with st.container(horizontal=True, vertical_alignment="center", key=f"runrow_{mode}"):
                if done is not None:
                    st.button(f"See the {title.lower()} result", type="primary", key=f"see_{mode}",
                              on_click=show_result, args=(a, mode, done))
                    if st.button(f"Run the {title.lower()} again from scratch", type="tertiary", key=f"again_{mode}"):
                        run_store().pop((settings_hash(a), mode), None)
                        st.session_state.active_job = start_job(a, mode)
                        st.rerun()
                else:
                    if st.button(f"Start the {title.lower()} ({seconds_label(secs)})", type="primary",
                                 key=f"start_{mode}"):
                        st.session_state.active_job = start_job(a, mode)
                        st.rerun()
    if st.session_state.get("active_job"):
        st.write("")
        progress_panel(st.session_state.active_job)
    nav_buttons(None)

else:
    R = st.session_state.get("result")
    if not R:
        st.session_state.page = RESULTS - 1
        st.rerun()
    run, base, ra = R["run"], R["base"], R["a"]
    yours, dflt = summary(run, ra), summary(base, A0)
    if run["year"].empty:
        st.html('<div class="result">The model could not score any station layout in every kind of week.</div>')
        st.stop()
    feasible = yours["meets"] == yours["of"]
    who = "With your settings" if R["changes"] else "With the default settings"
    head = (f"{who}, the model recommends <span class=\"accent\">{yours['said']}</span>." if feasible else
            f"{who}, no station layout meets every target. The closest is <span class=\"accent\">{yours['said']}</span>.")
    st.html(f'<div class="result">{head}</div>')
    if R["changes"]:
        same = yours["pick"] == dflt["pick"]
        delta = (yours["year_cost"] - dflt["year_cost"]) * 100
        more = "more" if delta > 0 else "less"
        sub = (f"This is the same network the model recommends with the default settings. Over a year, it costs "
               f"{abs(delta):.0f}¢ {more} per package than it does with the defaults." if same else
               f"With the default settings, the model recommends {dflt['said']}. Your changes alter the "
               f"recommendation and make the yearly cost {abs(delta):.0f}¢ per package {'higher' if delta > 0 else 'lower'}.")
    else:
        sub = ("You can change any assumption in the earlier steps and run the simulation again to see how the "
               "recommendation changes.")
    std_max = max(ra["design_grid"]["hub_counts"])
    if feasible and run.get("tiers_tried", 1) > 1:
        sub += (f" No network of up to {WORDS.get(std_max, std_max)} stations met every target, so the model also "
                f"tried larger networks, up to {run.get('max_stations')} stations.")
    if not feasible:
        sub += ((f" This station layout meets every target in {yours['meets']} of {yours['of']} kinds of week."
                 if yours["meets"] else " This station layout misses at least one target in every kind of week.")
                + f" The model tried networks of up to {run.get('max_stations', std_max)} stations. "
                + diagnosis_text(run.get("diagnosis"), ra))
    est = run.get("estimate")
    if est:
        est_name = spoken(get_design(est["design"], ra).label())
        sub += (" The instant estimate from an average day recommended the same network." if est_name == yours["said"]
                else f" The instant estimate from an average day recommended {est_name}, but the random weeks "
                     "changed the answer.")
    if R.get("mode") == "full":
        sub += (f" These results come from the full run, which simulated {run.get('weeks_simulated', 0):,} "
                f"weeks across all station layouts, covering {big_number(run.get('packages_simulated', 0))} packages in "
                f"{elapsed_label(run.get('seconds', 0))}. The closest station layouts were rechecked on "
                f"{run.get('recheck_weeks', 0):,} fresh weeks in every kind of week.")
    else:
        sub += (f" These results come from the quick run, which simulated {run.get('weeks_simulated', 0):,} "
                f"weeks across all station layouts, covering {big_number(run.get('packages_simulated', 0))} packages in "
                f"{elapsed_label(run.get('seconds', 0))}. Station layouts that are close on cost may swap places in the full run.")
    st.html(f'<div class="result-sub">{sub}</div>')

    goal = ra["objective"]

    def row(label, note, f, lead_row=False):
        return (f'<tr class="{"lead" if lead_row else ""}"><td>{label}<div class="s">{note}</div></td>'
                f'<td>{f(yours)}</td></tr>')

    def kinds(s):
        return f"All {s['of']} kinds of week" if s["meets"] == s["of"] else f"{s['meets']} of {s['of']} kinds of week"

    wpy = ra["weeks_per_year"]
    table = ('<table class="cmp"><tr><th></th><th>'
             + ("With your settings" if R["changes"] else "With the default settings") + '</th></tr>'
             + row("Recommended network" if feasible else "Closest network",
                   "This is the number and size of delivery stations to run." if feasible else
                   "No network meets every target. This one comes closest on OTD.",
                   lambda s: s["name"], True)
             + row("Cost per package, year average",
                   f"It weights {wpy.get('normal', 0)} normal, {wpy.get('holiday_peak', 0)} peak and "
                   f"{wpy.get('hub_outage', 0)} outage weeks.", lambda s: money(s["year_cost"]))
             + row("Cost per package in a typical normal week", "This is the middle week of the simulated normal weeks.",
                   lambda s: money(s["cost"]))
             + row("Cost per package in a bad normal week", "Only one normal week in ten costs more.",
                   lambda s: money(s["cost_bad"]))
             + row("OTD in a typical normal week", f"The target is {goal['on_time_typical_target']:.0%}.",
                   lambda s: f'{s["on_time"]:.1%}')
             + row("OTD in a bad normal week", "Only one normal week in ten does worse.",
                   lambda s: f'{s["on_time_bad"]:.1%}')
             + row("Weeks that reach the OTD floor",
                   f"Share of weeks at {goal['on_time_target']:.0%} OTD or better, in whichever kind of week is "
                   f"hardest for this network. The target is {goal['on_time_week_share']:.0%} of weeks.",
                   lambda s: f'{s["worst_share"]:.1%}')
             + row("Miles driven per package", "In a typical normal week. Lower means denser routes.",
                   lambda s: f'{s["km"] / 1.609344:.2f} miles')
             + row("Dasher pay per hour after car costs",
                   f"In a typical normal week. The floor is ${goal['driver_net_per_hour_floor']:.0f}.",
                   lambda s: money(s["driver"]))
             + row("Kinds of week where every target is met", "These are normal weeks, holiday peak weeks and "
                   "station outage weeks.", kinds)
             + "</table>")
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.html(table)
    with right:
        st_folium(network_map(tracts, sites, drive, yours["hubs"]), height=460, use_container_width=True,
                  returned_objects=[])

    frames = run["frames"]
    PAY_POLICY_ON[0] = bool(ra["drivers"].get("raise_base_to_floor"))
    shown = [d for d in run["year"].index if all(d in f.index for f in frames.values())]
    shown = sorted(shown, key=lambda d: frames[list(frames)[0]].loc[d, "cost_p50"])
    obj = ra["objective"]

    st.html('<div class="chart-title" style="margin-top:34px">'
            + ("Why this network wins" if feasible else "How close each network comes") + '</div>'
            '<div class="chart-note" style="max-width:95ch">The recommendation has to meet every target in all three '
            'kinds of week: normal weeks, holiday peak weeks and station outage weeks. This table lists every '
            'station layout that the model scored in all three, from cheapest to most expensive over a year. A tick '
            'means the layout meets every target in that kind of week. A cross names the target it misses. '
            + ('The recommended network is the cheapest one with a tick in every column, so any cheaper row above it '
               'is ruled out by a cross.' if feasible else
               'No layout has a tick in every column, so there is no recommendation. The layout marked as closest '
               'comes nearest to the OTD targets, whatever it costs.')
            + (f' Every station layout pays Dashers at least ${obj["driver_net_per_hour_floor"]:.0f} an hour after car '
               'costs. The column after the cost shows how much of each layout\'s cost per package goes to raising '
               'pay to that level. Station layouts with longer drives need bigger raises.'
               if ra["drivers"].get("raise_base_to_floor") else '')
            + '</div>' + scorecard(run, ra, yours["pick"]))

    sens = share_sensitivity_rows(MARKET, A0)
    if sens:
        rows_html = "".join(
            f'<tr class="{"pick" if abs(r["share"] - A0["demand"]["network_share"]) < 1e-9 else ""}">'
            f'<td class="name">{r["share"]:.0%}</td><td>{round(r["packages_per_day"], -2):,.0f}</td>'
            f'<td class="name">{plain(r["hubs"])}</td><td>{money(r["year_cost"])}</td></tr>' for r in sens)
        st.html('<div class="chart-title" style="margin-top:34px">How the answer changes with volume</div>'
                '<div class="chart-note" style="max-width:95ch">The network share is the hardest default to pin '
                'down, so the full simulation was also run at other shares with every other default unchanged. '
                'Each row shows the cheapest station layout that meets every target at that volume.</div>'
                '<div class="score-wrap"><table class="score floor"><tr><th>Network share</th>'
                '<th>Packages on an average day</th><th>Recommended network</th><th>Cost per package over a year</th>'
                '</tr>' + rows_html + '</table></div>')

    with st.expander("Cost per package in each kind of week"):
        st.html('<div class="chart-note" style="max-width:95ch">Each row is one station layout. The dot shows its '
                'cost per package in a typical week, and the line runs out to its cost in a bad week, which only one '
                'week in ten exceeds. Blue is the recommended network, black rows meet every target in the chosen '
                'kind of week, and grey rows miss a target in that week.</div>')
        k = st.segmented_control("Show costs for", list(frames), format_func=lambda x: SCEN.get(x, x),
                                 default=list(frames)[0], key="res_week") or list(frames)[0]
        st.altair_chart(cost_chart(frames[k], yours["pick"], objective_for(ra, k),
                                  "recommended" if feasible else "closest to the targets"), width="stretch")
    with st.expander("How many weeks stay on time?"):
        st.html('<div class="chart-note">Out of every 1,000 simulated weeks, how many reach the on-time floor. '
                'Each kind of week has its own goal, shown under its heading.</div>' + floor_table(run, ra))
    st.write("")
    l, _ = st.columns([1, 3.6])
    l.button("← Change the assumptions", on_click=go, args=(FIRST_STEP,), width="stretch")
