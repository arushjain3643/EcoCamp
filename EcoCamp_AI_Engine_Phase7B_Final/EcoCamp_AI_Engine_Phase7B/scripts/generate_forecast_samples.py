#!/usr/bin/env python3
"""Regenerate the Phase 6 forecasting sample requests in ``samples/forecasting/``.

Run from the project root::

    python scripts/generate_forecast_samples.py              # inputs only (stdlib, deterministic)
    python scripts/generate_forecast_samples.py --outputs    # also write *.response.json (needs the
                                                             # engine's dependencies: pydantic, numpy)

The data is SYNTHETIC (fixed seed, byte-identical on every run). Each ``forecast_*.json`` file is a
ready-to-POST body for ``forecast_payload`` / ``POST /api/v1/ai/forecast``::

    {"records": [ ... ], "options": {"horizon_days": 7}}

``--outputs`` runs the real engine on every request and stores the exact response next to it, so
the outputs in the repository are never hand-written.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "samples" / "forecasting"
START = datetime(2026, 6, 1, tzinfo=timezone.utc)  # a Monday
UNITS = {"electricity": "kWh", "water": "liters", "waste": "kg"}
# Weekday multipliers Mon..Sun (weekend dip) and typical daily level (synthetic).
WEEK = [1.00, 1.04, 1.02, 1.05, 0.98, 0.62, 0.55]
LEVEL = {
    ("Library", "electricity"): 420.0,
    ("Hostel A", "electricity"): 780.0,
    ("Hostel B", "water"): 38000.0,
    ("Canteen", "water"): 9500.0,
    ("Canteen", "waste"): 85.0,
    ("Admin Block", "waste"): 18.0,
}
DECIMALS = {"electricity": 2, "water": 0, "waste": 2}


def _iso(dt: datetime) -> str:
    if dt.utcoffset() == timedelta(0):
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.isoformat()


def daily(
    rng: random.Random, building: str, resource: str, days: int, *,
    noise: float = 0.04, skip: Sequence[int] = (), offset_hours: float = 0.0,
    start_day: int = 0,
) -> List[Dict[str, Any]]:
    level = LEVEL[(building, resource)]
    tz = timezone(timedelta(hours=offset_hours))
    out: List[Dict[str, Any]] = []
    for i in range(start_day, start_day + days):
        value = level * WEEK[i % 7] * max(0.0, 1.0 + rng.gauss(0.0, noise))  # always drawn: keeps the stream stable
        if i in set(skip):
            continue
        ts = (START + timedelta(days=i)).astimezone(tz) if offset_hours else START + timedelta(days=i)
        out.append({
            "timestamp": _iso(ts), "building": building, "resource_type": resource,
            "consumption_value": round(value, DECIMALS[resource]), "unit": UNITS[resource],
        })
    return out


def build() -> List[Tuple[str, str, Dict[str, Any]]]:
    """(file stem, description, request body) for every sample."""
    rng = random.Random(2026)
    files: List[Tuple[str, str, Dict[str, Any]]] = []

    recs = (
        daily(rng, "Library", "electricity", 56)
        + daily(rng, "Hostel B", "water", 56)
        + daily(rng, "Canteen", "waste", 56)
        + daily(rng, "Hostel A", "electricity", 56)
    )
    rng.shuffle(recs)  # input order must not matter
    files.append((
        "forecast_normal_7day",
        "Four series (4 buildings, 3 resources), 56 daily "
        "readings each, shuffled, 7-day horizon.",
        {"records": recs, "options": {"horizon_days": 7}},
    ))

    files.append((
        "forecast_next_day",
        "One series (Library electricity), 35 daily readings, next-day forecast (horizon_days = 1).",
        {"records": daily(rng, "Library", "electricity", 35), "options": {"horizon_days": 1}},
    ))

    irregular = daily(rng, "Canteen", "water", 49, skip=(9, 10, 23), offset_hours=5.5)
    dup = dict(irregular[4])  # an exact duplicate reading (same timestamp and value)
    irregular.append(dup)
    files.append((
        "forecast_missing_irregular",
        "Canteen water, 49 days with 3 missing days (never imputed), +05:30 timestamps and one exact "
        "duplicate reading (counted once). Expect data_quality_status 'suspect' notes, not a refusal.",
        {"records": irregular, "options": {"horizon_days": 7}},
    ))

    files.append((
        "forecast_insufficient_history",
        "Admin Block waste with only 5 days, plus Hostel B water with 3 days: both must return "
        "forecast_available = false / insufficient_data with no numbers.",
        {"records": daily(rng, "Admin Block", "waste", 5) + daily(rng, "Hostel B", "water", 3),
         "options": {"horizon_days": 7}},
    ))

    files.append((
        "forecast_long_history",
        "Library electricity, 150 daily readings with weekly pattern and 8% noise: long enough for the "
        "optional gradient-boosting model to be backtested against the baseline. Which model wins is "
        "decided by the backtest, not by this file.",
        {"records": daily(rng, "Library", "electricity", 150, noise=0.08),
         "options": {"horizon_days": 7}},
    ))
    return files


def write_inputs() -> List[Dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for stem, description, body in build():
        (OUT_DIR / f"{stem}.json").write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
        manifest.append({
            "file": f"{stem}.json", "records": len(body["records"]),
            "horizon_days": body["options"]["horizon_days"], "description": description,
        })
    (OUT_DIR / "manifest.json").write_text(
        json.dumps({"synthetic": True, "generator": "scripts/generate_forecast_samples.py",
                    "files": manifest}, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_outputs() -> None:
    sys.path.insert(0, str(ROOT))
    from ai_engine import forecast_payload  # needs pydantic + numpy (scikit-learn optional)

    for stem, _, body in build():
        outcome = forecast_payload(body)
        (OUT_DIR / f"{stem}.response.json").write_text(
            json.dumps({"http_status": outcome.http_status, "body": outcome.body}, indent=1) + "\n",
            encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--outputs", action="store_true", help="also run the engine and store responses")
    args = parser.parse_args(argv)
    manifest = write_inputs()
    print(f"wrote {len(manifest)} request files to {OUT_DIR.relative_to(ROOT)}")
    if args.outputs:
        write_outputs()
        print("wrote *.response.json (engine output)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
