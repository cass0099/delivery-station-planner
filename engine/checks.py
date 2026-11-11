"""Compare model outputs with the public reference ranges in assumptions.yaml."""
from __future__ import annotations

import pandas as pd

from engine.config import load_assumptions


def run_checks(values: dict[str, float]) -> pd.DataFrame:
    refs = load_assumptions()["reference_checks"]
    rows = []
    for name, value in values.items():
        ref = refs[name]
        ok = ref["low"] <= value <= ref["high"]
        rows.append({
            "check": ref["description"],
            "model": value,
            "reference": f"{ref['low']:,} – {ref['high']:,}",
            "result": "OK" if ok else "OUTSIDE RANGE",
            "source": ref["source"],
        })
    return pd.DataFrame(rows)


def print_checks(table: pd.DataFrame) -> None:
    print("\nSanity checks against public reference numbers")
    for r in table.itertuples():
        model = f"{r.model:,.2f}" if r.model < 10 else f"{r.model:,.0f}"
        print(f"  [{r.result}] {r.check}: model {model}, reference {r.reference}")
        print(f"         source: {r.source}")
