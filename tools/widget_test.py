"""Move every control in the app to both ends of its widget and check the stored value is the right real unit.

    python -m tools.widget_test
"""
from __future__ import annotations

import math

from streamlit.testing.v1 import AppTest

from engine.config import REPO_ROOT, load_assumptions

A0 = load_assumptions()
STEPS = A0["app_controls"]["steps"]
FIRST_STEP = 3


def close(a, b, rel=0.02) -> bool:
    if isinstance(a, (list, tuple)):
        return all(close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(a, str):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-6)


def main() -> None:
    problems, checked = [], 0
    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=300).run()
    for si, step in enumerate(STEPS):
        at.session_state["page"] = FIRST_STEP + si
        at.run()
        for c in step["controls"]:
            path, kind, key = c["path"], c.get("kind", "slider"), f"w:{c['path']}"
            if c.get("requires"):                      # switch the dependency on first
                req = next(x for x in step["controls"] if x["path"] == c["requires"])
                if not at.session_state["vals"][req["path"]]:
                    at.toggle(key=f"w:{req['path']}").set_value(True).run()
            if kind == "toggle":
                w = at.toggle(key=key)
                before = at.session_state["vals"][path]
                w.set_value(not w.value).run()
                ok = at.session_state["vals"][path] == (not before)
                at.toggle(key=key).set_value(before).run()
                targets = [("flip", ok)]
            elif kind == "clock":
                targets = []
                for opt in (c["options"][0], c["options"][-1]):
                    at.select_slider(key=key).set_value(opt).run()
                    targets.append((opt, at.session_state["vals"][path] == opt))
            else:
                w = at.slider(key=key)
                targets = []
                for end, want in (("min", c["min"]), ("max", c["max"])):
                    v = w.min if end == "min" else w.max
                    w.set_value((w.min, w.max) if kind == "range" and end == "max" else
                                ((w.min, w.min) if kind == "range" else v)).run()
                    got = at.session_state["vals"][path]
                    expect = ([c["min"], c["max"]] if kind == "range" and end == "max" else
                              [c["min"], c["min"]] if kind == "range" else want)
                    targets.append((f"{end} -> {got}", close(got, expect)))
                    w = at.slider(key=key)
            checked += 1
            if at.exception:
                problems.append(f"{path}: {at.exception[0].message}")
            for label, ok in targets:
                if not ok:
                    problems.append(f"{path}: {label} (expected {c.get('min')}..{c.get('max')})")
        # leave the step at its defaults so later steps start clean
        at.session_state["page"] = FIRST_STEP + si
    print(f"Checked {checked} controls.")
    print("\n".join(problems) if problems else "Every control stores the expected value at both ends.")


if __name__ == "__main__":
    main()
