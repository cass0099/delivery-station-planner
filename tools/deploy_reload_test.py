"""Deploy test: after the engine changes on disk, do new runs use the new code?

A long-lived process (like the Streamlit server) runs a pool of workers, then the engine changes on disk (a deploy),
the process reloads its engine (as the app does), and a second pool runs. The second pool's workers must see the
new code. On Python 3.14 under Streamlit, the forkserver preloads the app script, and with it the engine, so this
test preloads the engine by name to get the same effect on any Python version.

    python -m tools.deploy_reload_test

It appends one line to engine/demand.py for a moment and always restores the file.
"""
from __future__ import annotations

import importlib
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import engine.config  # noqa: F401  (loaded before the deploy, like the app)
import engine.demand  # noqa: F401
import engine.live
from engine.config import REPO_ROOT

DEMAND = REPO_ROOT / "engine" / "demand.py"
ORDER = ["config", "randomness", "demand", "routing", "simulation", "checks", "search", "live"]


def marker(_) -> str:
    import engine.demand as d
    return getattr(d, "DEPLOY_MARKER", "old code")


def run_pool() -> set:
    with ProcessPoolExecutor(2, mp_context=sys.modules["engine.live"]._pool_context()) as pool:
        return set(pool.map(marker, range(4)))


def main() -> None:
    multiprocessing.get_context("forkserver").set_forkserver_preload(["engine.config", "engine.demand", "engine.live"])
    first = run_pool()
    original = DEMAND.read_text()
    try:
        time.sleep(0.05)                                   # a later change time than the loaded files
        DEMAND.write_text(original + '\nDEPLOY_MARKER = "new code"\n')
        for name in ORDER:                                 # what the app does when the engine files change
            if f"engine.{name}" in sys.modules:
                importlib.reload(sys.modules[f"engine.{name}"])
        second = run_pool()
    finally:
        DEMAND.write_text(original)
    ok = first == {"old code"} and second == {"new code"}
    print(f"Before the deploy, workers ran: {sorted(first)}. After it, workers ran: {sorted(second)}.")
    print("PASS: runs after a deploy use the new engine code." if ok else "FAIL: runs after a deploy used old code.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
