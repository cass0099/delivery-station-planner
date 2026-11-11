"""Dev tool: screenshot every page of the running app at desktop and phone widths, and flag text that
overflows its box or wraps where it shouldn't. Needs the dev container's Chromium (see .devcontainer/Dockerfile).

    streamlit run app/streamlit_app.py --server.headless true --server.port 8501   # in another terminal
    python tools/screenshots.py [--out shots] [--pages Overview "1 Volume"] [--widths 1440 390]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = ["Overview", "Core assumptions", "Methodology", "1. Volume", "2. Station costs", "3. Dasher pay",
         "4. Service targets", "5. Disruptions", "Run simulation", "Results"]

# Elements whose text should never spill out of its box or be cut off.
OVERFLOW_JS = """
() => {
  const out = [];
  for (const el of document.querySelectorAll('button, label, p, div, span, td, th, h1, h2, h3, h4, dt, dd')) {
    if (!el.offsetParent || el.children.length > 3) continue;
    const text = (el.innerText || '').trim();
    if (!text || text.length > 160) continue;
    const s = getComputedStyle(el);
    const clipped = el.scrollWidth > el.clientWidth + 1 && s.overflowX !== 'visible';
    const spills = el.scrollWidth > el.clientWidth + 1 && s.whiteSpace === 'nowrap';
    if (clipped || spills) out.push(text.slice(0, 80));
  }
  return [...new Set(out)].slice(0, 30);
}
"""


def chromium() -> str:
    for name in ("chromium", "chromium-browser", "google-chrome"):
        if path := shutil.which(name):
            return path
    raise SystemExit("No Chromium found. Rebuild the dev container (see .devcontainer/Dockerfile).")


def open_page(page, name: str) -> None:
    if name == "Overview":
        return
    if name == "Results":                       # results need a run: use the saved default quick run
        open_page(page, "Run simulation")
        page.get_by_role("button", name="See the quick run result").click()
    else:
        page.get_by_role("button", name=name, exact=True).first.click()
    page.wait_for_timeout(1500)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8501")
    p.add_argument("--out", default="shots")
    p.add_argument("--pages", nargs="*", default=PAGES)
    p.add_argument("--widths", nargs="*", type=int, default=[1440, 390])
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chromium(), args=["--no-sandbox"])
        for width in args.widths:
            for name in args.pages:
                page = browser.new_page(viewport={"width": width, "height": 900}, device_scale_factor=1)
                page.goto(args.url, wait_until="networkidle")
                page.wait_for_selector("text=Delivery Station Planner", timeout=30000)
                page.wait_for_timeout(1500)
                open_page(page, name)
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(2500)              # maps and charts finish drawing
                file = out / f"{width}_{PAGES.index(name):02d}_{name.replace(' ', '_')}.png"
                page.screenshot(path=str(file), full_page=True)
                issues = page.evaluate(OVERFLOW_JS)
                print(f"{file}" + (f"  · overflowing text: {issues}" if issues else ""))
                page.close()
        browser.close()


if __name__ == "__main__":
    main()
