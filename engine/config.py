"""Shared settings: the assumptions file and where data lives."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSUMPTIONS_PATH = REPO_ROOT / "assumptions.yaml"


@lru_cache(maxsize=1)
def load_assumptions(path: Path = ASSUMPTIONS_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def code_stamp() -> tuple:
    """Names, change times and sizes of the engine files and assumptions.yaml as they are on disk now."""
    files = sorted((REPO_ROOT / "engine").glob("*.py")) + [ASSUMPTIONS_PATH]
    return tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files)


# The engine files as they were when this module was loaded (or reloaded). The app reloads the engine when the files
# on disk no longer match, and run workers compare theirs with the app's so they never run older code.
LOADED_STAMP = code_stamp()


def market_settings(market: str) -> dict:
    markets = load_assumptions()["markets"]
    if market not in markets:
        raise KeyError(f"Unknown market '{market}'. Known: {', '.join(markets)}")
    return markets[market]


def cache_dir(*parts: str) -> Path:
    """Downloaded public data. Lives outside the repo, in $PARCEL_CACHE_DIR."""
    root = os.environ.get("PARCEL_CACHE_DIR")
    if not root:
        raise RuntimeError("Set PARCEL_CACHE_DIR to a folder for downloaded data.")
    path = Path(root).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepared_dir(market: str) -> Path:
    """Cleaned, per-market files used by the engine and the map."""
    path = REPO_ROOT / "data" / "prepared" / market
    path.mkdir(parents=True, exist_ok=True)
    return path
