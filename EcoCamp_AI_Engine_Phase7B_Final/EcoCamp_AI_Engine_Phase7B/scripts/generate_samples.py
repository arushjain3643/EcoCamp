#!/usr/bin/env python3
"""Regenerate the sample datasets in ``samples/`` (deterministic, stdlib only).

Run from the project root::

    python scripts/generate_samples.py

The data is SYNTHETIC. Magnitudes are plausible for a mid-sized campus but are
not measurements from any real campus. A fixed random seed makes the output
reproducible, so regenerating produces byte-identical files.

Each scenario file is a ready-to-POST request body::

    {"records": [ {timestamp, building, resource_type, consumption_value, unit}, ... ]}
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

OUT_DIR = Path(__file__).resolve().parent.parent / "samples"
START = datetime(2026, 8, 1, tzinfo=timezone.utc)  # a Saturday

UNITS = {"electricity": "kWh", "water": "liters", "waste": "kg"}
# Typical daily consumption per building (synthetic).
BASE = {
    "electricity": {
        "Library": 420.0, "Hostel A": 780.0, "Admin Block": 310.0, "Science Block": 560.0,
        "Lecture Hall Complex": 640.0, "Sports Arena": 250.0, "Hostel C": 720.0,
        "Workshop Block": 480.0, "Gymnasium": 190.0, "Guest House": 95.0,
        "Innovation Hub": 330.0,
    },
    "water": {
        "Hostel B": 38000.0, "Science Block": 6200.0, "Canteen": 9500.0,
        "Lecture Hall Complex": 7200.0, "Sports Arena": 11000.0, "Hostel C": 41000.0,
        "Workshop Block": 5400.0, "Auditorium": 4300.0, "New Sports Complex": 8000.0,
    },
    "waste": {
        "Canteen": 85.0, "Hostel A": 60.0, "Admin Block": 18.0,
        "Lecture Hall Complex": 22.0, "Sports Arena": 14.0, "Hostel C": 55.0,
        "Workshop Block": 30.0, "Innovation Hub": 16.0,
    },
}
NOISE = {"electricity": 0.03, "water": 0.04, "waste": 0.06}  # gaussian sigma (fraction)
WEEKEND = {"electricity": 0.90, "water": 0.92, "waste": 0.90}  # mild weekend dip
DECIMALS = {"electricity": 2, "water": 0, "waste": 2}


def _ts(day: int, offset_hours: float = 0.0) -> str:
    """ISO-8601 timestamp for day ``day`` (0-based). UTC 'Z' or a fixed offset."""
    if offset_hours == 0.0:
        return (START + timedelta(days=day)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tz = timezone(timedelta(hours=offset_hours))
    local_midnight = datetime(START.year, START.month, START.day, tzinfo=tz)
    return (local_midnight + timedelta(days=day)).isoformat()


def series(
    rng: random.Random,
    building: str,
    resource: str,
    days: int,
    *,
    multipliers: Optional[Dict[int, float]] = None,
    override: Optional[Dict[int, float]] = None,
    offset_hours: float = 0.0,
    unit: Optional[str] = None,
    resource_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Daily readings with noise and a weekend dip. Day numbers are 1-based.

    ``multipliers`` scales the normal value on given days; ``override`` sets an
    exact value (e.g. 0 for a meter outage).
    """
    multipliers = multipliers or {}
    override = override or {}
    out: List[Dict[str, Any]] = []
    for d in range(days):
        day_no = d + 1
        value = BASE[resource][building] * rng.gauss(1.0, NOISE[resource])
        if (START + timedelta(days=d)).weekday() >= 5:
            value *= WEEKEND[resource]
        value *= multipliers.get(day_no, 1.0)
        if day_no in override:
            value = override[day_no]
        value = round(max(0.0, value), DECIMALS[resource])
        if DECIMALS[resource] == 0:
            value = float(int(value))
        out.append({
            "timestamp": _ts(d, offset_hours),
            "building": building,
            "resource_type": resource_label or resource,
            "consumption_value": value,
            "unit": unit or UNITS[resource],
        })
    return out


def build() -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(20260801)
    files: Dict[str, List[Dict[str, Any]]] = {}

    # 1. Normal consumption: 4 buildings x 3 resources x 30 days, no anomalies.
    recs: List[Dict[str, Any]] = []
    for b in ("Lecture Hall Complex", "Sports Arena", "Hostel C", "Workshop Block"):
        for res in ("electricity", "water", "waste"):
            recs += series(rng, b, res, 30)
    files["normal_consumption"] = recs

    # 2. Unusual electricity usage.
    recs = []
    # Library: sustained after-hours load on days 31-33 (about 2.3-2.6x normal).
    recs += series(rng, "Library", "electricity", 35, multipliers={31: 2.3, 32: 2.6, 33: 2.4})
    recs += series(rng, "Hostel A", "electricity", 35)  # normal control
    # Admin Block: a single-day spike on day 34 (about 1.9x).
    recs += series(rng, "Admin Block", "electricity", 35, multipliers={34: 1.9})
    files["electricity_unusual_usage"] = recs

    # 3. Water wastage: Hostel B runs ~1.9x normal for 6 days (e.g. a running
    # fixture or overnight flow). Timestamps carry a +05:30 offset.
    recs = []
    recs += series(
        rng, "Hostel B", "water", 40, offset_hours=5.5,
        multipliers={31: 1.85, 32: 1.9, 33: 1.95, 34: 1.9, 35: 2.0, 36: 1.9},
    )
    recs += series(rng, "Science Block", "water", 40, offset_hours=5.5)  # control
    recs += series(rng, "Canteen", "water", 40, offset_hours=5.5)  # control
    files["water_wastage"] = recs

    # 4. Waste anomalies.
    recs = []
    recs += series(rng, "Canteen", "waste", 40, multipliers={31: 2.4, 32: 2.7, 33: 2.5})
    recs += series(rng, "Hostel A", "waste", 40, multipliers={38: 2.1})
    recs += series(rng, "Admin Block", "waste", 40)  # control
    files["waste_anomalies"] = recs

    # 5. Insufficient history: every building/resource has fewer than the
    # default minimum_history (7) earlier readings, so no baseline can be built.
    recs = []
    recs += series(rng, "Innovation Hub", "electricity", 5, multipliers={5: 3.0})
    recs += series(rng, "Innovation Hub", "waste", 6)
    recs += series(rng, "New Sports Complex", "water", 3)
    files["insufficient_history"] = recs

    # 6. Edge cases: meter outage, unusually low use, spelling normalization.
    recs = []
    recs += series(rng, "Auditorium", "water", 21, override={21: 0.0})  # zero reading
    recs += series(rng, "Gymnasium", "electricity", 21, multipliers={21: 0.30})  # low
    recs += series(  # unit/resource spelling variants that the engine normalizes
        rng, "Guest House", "electricity", 21, unit="kwh", resource_label="Electricity"
    )
    files["edge_cases"] = recs
    return files


def _dump(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = build()
    for name, recs in files.items():
        _dump(OUT_DIR / f"{name}.json", {"records": recs})

    # Mixed campus: every scenario above in one request, shuffled, to show that
    # input order is preserved and that groups do not interfere. Building and
    # resource combinations are disjoint across scenario files by design.
    combined = [r for recs in files.values() for r in recs]
    random.Random(7).shuffle(combined)
    _dump(OUT_DIR / "mixed_campus.json", {"records": combined})

    # Invalid records: used to demonstrate the 422 error body (not analyzable).
    invalid = [
        {"timestamp": "2026-08-01T00:00:00Z", "building": "Library", "resource_type": "electricity", "consumption_value": 410.0, "unit": "kWh"},
        {"timestamp": "2026-08-02T00:00:00Z", "building": "Library", "resource_type": "electricity", "consumption_value": -5.0, "unit": "kWh"},
        {"timestamp": "2026-08-03T00:00:00Z", "building": "Library", "resource_type": "gas", "consumption_value": 12.0, "unit": "m3"},
        {"timestamp": "2026-08-04T00:00:00Z", "building": "   ", "resource_type": "water", "consumption_value": 900.0, "unit": "liters"},
        {"timestamp": "2026-08-05T00:00:00Z", "building": "Library", "resource_type": "electricity", "consumption_value": 3.1, "unit": "MWh"},
        {"timestamp": 1785542400, "building": "Library", "resource_type": "electricity", "consumption_value": 400.0, "unit": "kWh"},
    ]
    _dump(OUT_DIR / "invalid_records.json", {"records": invalid})
    print(f"Wrote {len(files) + 2} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
