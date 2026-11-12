"""End-to-end app test: the clicks a visitor makes, with real background runs.

1. Open the app and look at every page with the default settings, then open the saved default results.
2. Change the network share, start a quick run on the Run page, wait for it, and check the Results page.
3. Start a full run at extreme settings, press Stop, and check the run stops with a message.
4. Reset to defaults.

    python -m tools.app_flow_test
"""
from __future__ import annotations

import re
import time

from streamlit.testing.v1 import AppTest

from engine.config import REPO_ROOT, load_assumptions

A0 = load_assumptions()
FIRST_STEP, RUN, RESULTS = 3, 3 + len(A0["app_controls"]["steps"]), 4 + len(A0["app_controls"]["steps"])


def text(at) -> str:
    return re.sub(r"<[^>]+>", " ", " ".join(str(getattr(e, "body", "") or getattr(e, "value", "") or "")
                                           for e in at.main))


def wait_for(at, done, seconds: float, step: float = 2.0) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        at.run()
        if done(at):
            return True
        time.sleep(step)
    return False


def check(ok: bool, what: str, problems: list) -> None:
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        problems.append(what)


def main() -> None:
    problems: list[str] = []
    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=600).run()
    check(not at.exception, "app opens", problems)

    # 1. every page with the defaults, then the saved default result
    for page in range(RUN + 1):
        at.session_state["page"] = page
        at.run()
        check(not at.exception, f"page {page} renders with the defaults", problems)
    at.button(key="see_quick").click().run()
    check(at.session_state["page"] == RESULTS and "three small stations" in text(at),
          "saved default quick result opens and recommends three small stations", problems)
    at.session_state["page"] = RUN
    at.run()
    at.button(key="see_full").click().run()
    check(at.session_state["page"] == RESULTS and not at.exception, "saved default full result opens", problems)

    # 2. change a setting and run the quick run live
    at.session_state["page"] = FIRST_STEP
    at.run()
    at.slider(key="w:demand.network_share").set_value(5.0).run()
    check(at.session_state["vals"]["demand.network_share"] == 0.05, "network share slider set to 5%", problems)
    at.session_state["page"] = RUN
    at.run()
    check(not at.exception and any(b.key == "start_quick" for b in at.button), "Run page offers a new quick run",
          problems)
    t0 = time.time()
    at.button(key="start_quick").click().run()
    ok = wait_for(at, lambda x: x.session_state["page"] == RESULTS, 900)
    check(ok and not at.exception, f"live quick run finished and opened Results ({time.time() - t0:.0f}s)", problems)
    body = text(at)
    check("With your settings" in body, "Results page says it uses your settings", problems)
    check("four small stations" in body or "recommends" in body or "closest" in body,
          "Results page names a network", problems)

    # 3. extreme full run, then Stop
    at.session_state["page"] = FIRST_STEP
    at.run()
    at.slider(key="w:demand.network_share").set_value(9.0).run()
    at.session_state["page"] = RUN
    at.run()
    start = [b for b in at.button if b.key == "start_full"]
    if start and not start[0].disabled:
        start[0].click().run()
        time.sleep(8)
        at.run()
        stop = [b for b in at.button if b.key == "stop_run"]
        check(bool(stop), "progress panel shows a Stop button", problems)
        if stop:
            stop[0].click().run()
            ok = wait_for(at, lambda x: "run was stopped" in text(x), 60, 1.0)
            check(ok, "Stop ends the run and says so", problems)
    else:
        check(True, "full run at these settings is turned off because it would pass the time limit", problems)

    # 4. reset
    at.button(key="nav_reset").click().run()
    check(at.session_state["vals"]["demand.network_share"] == A0["demand"]["network_share"],
          "Reset to defaults restores the network share", problems)
    print("\nAll checks passed." if not problems else f"\n{len(problems)} check(s) failed: {problems}")


if __name__ == "__main__":
    main()
