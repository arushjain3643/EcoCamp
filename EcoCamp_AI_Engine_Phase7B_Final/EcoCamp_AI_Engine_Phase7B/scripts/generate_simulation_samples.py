#!/usr/bin/env python3
"""Regenerate the Phase 7A what-if simulation samples in ``samples/simulation/``.

Run from the project root::

    python scripts/generate_simulation_samples.py              # request bodies only (stdlib, deterministic)
    python scripts/generate_simulation_samples.py --outputs    # also write *.response.json (needs the
                                                               # engine's dependencies: pydantic, numpy)

The data is SYNTHETIC and fully deterministic (no random numbers). Each ``sim_*.json`` file is a
ready-to-POST body for ``simulate_payload`` / ``POST /api/v1/ai/simulate``::

    {"records": [ ... ], "scenario": {...}}

``--outputs`` runs the real engine on every request and stores the exact response body next to it
(``<name>.response.json``), so the outputs in the repository are never hand-written. They are produced
with the default ``AnalysisConfig`` (no emission factor, so CO2e is ``null``). One extra file,
``sim_building_electricity_15pct.co2e_example.response.json``, uses an ILLUSTRATIVE electricity
factor of 0.5 kg CO2e/kWh to show the CO2e fields; it is a placeholder, not a real grid factor.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "samples" / "simulation"
START = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)  # a Monday
UNITS = {"electricity": "kWh", "water": "liters", "waste": "kg"}
WEEK = [1.00, 1.04, 1.02, 1.05, 0.98, 0.62, 0.55]  # Mon..Sun multipliers (weekend dip)
DECIMALS = {"electricity": 2, "water": 0, "waste": 2}
ILLUSTRATIVE_FACTOR = 0.5  # kg CO2e per kWh: a placeholder for the documentation example only


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def daily(building: str, resource: str, level: float, days: int, skip: Sequence[int] = ()) -> List[Dict[str, Any]]:
    """One reading per day: ``level`` x weekday multiplier, rounded; days in ``skip`` are missing."""
    out = []
    for i in range(days):
        if i in skip:
            continue
        value = round(level * WEEK[i % 7], DECIMALS[resource])
        out.append({
            "timestamp": _iso(START + timedelta(days=i)), "building": building,
            "resource_type": resource, "consumption_value": value, "unit": UNITS[resource],
        })
    return out


def build_requests() -> Dict[str, Dict[str, Any]]:
    return {
        "sim_building_electricity_15pct": {
            "records": daily("BLOCK_A", "electricity", 420.0, 45),
            "scenario": {"resource_type": "electricity", "building_code": "BLOCK_A",
                         "reduction_percentage": 15, "baseline_period_days": 30},
        },
        "sim_campus_water_10pct": {
            "records": (daily("HOSTEL_A", "water", 38000.0, 30) + daily("CANTEEN", "water", 9500.0, 30)
                        + daily("LIBRARY", "water", 2100.0, 30) + daily("LIBRARY", "electricity", 420.0, 30)),
            "scenario": {"resource_type": "water", "reduction_percentage": 10},
        },
        "sim_waste_explicit_baseline": {
            "scenario": {"resource_type": "waste", "building_code": "CANTEEN", "reduction_percentage": 25,
                         "baseline_consumption": {"value": 2550, "unit": "kg", "period_days": 30}},
        },
        "sim_partial_coverage": {
            "records": daily("BLOCK_A", "electricity", 420.0, 30, skip=(3, 7, 11, 15, 19)),
            "scenario": {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 15},
        },
        "sim_unknown_building": {
            "records": daily("BLOCK_A", "electricity", 420.0, 30),
            "scenario": {"resource_type": "electricity", "building_code": "BLOCK_Z", "reduction_percentage": 15},
        },
        "sim_zero_baseline": {
            "records": daily("LAB_ANNEX", "electricity", 0.0, 30),
            "scenario": {"resource_type": "electricity", "building_code": "LAB_ANNEX", "reduction_percentage": 20},
        },
        "sim_invalid_percentage": {
            "records": daily("BLOCK_A", "electricity", 420.0, 30),
            "scenario": {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 120},
        },
    }


def build_compare_requests() -> Dict[str, Dict[str, Any]]:
    return {
        "sim_compare_multiple_buildings": {
            "records": (
                daily("BLOCK_A", "electricity", 100.0, 30)
                + daily("BLOCK_B", "electricity", 200.0, 30)
                + daily("BLOCK_C", "electricity", 300.0, 30)
            ),
            "scenarios": [
                {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 10},
                {"resource_type": "electricity", "building_code": "BLOCK_B", "reduction_percentage": 15},
                {"resource_type": "electricity", "building_code": "BLOCK_C", "reduction_percentage": 20},
            ],
        },
        "sim_compare_overlapping_scenarios": {
            "records": (
                daily("BLOCK_A", "electricity", 100.0, 30)
                + daily("BLOCK_B", "electricity", 100.0, 30)
            ),
            "scenarios": [
                {"scenario_id": "campus-10", "resource_type": "electricity", "reduction_percentage": 10},
                {"scenario_id": "block-a-15", "resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 15},
            ],
        },
    }


def build_forecast_requests() -> Dict[str, Dict[str, Any]]:
    return {
        "sim_forecast_baseline_electricity": {
            "records": daily("BLOCK_A", "electricity", 420.0, 35),
            "scenario": {
                "resource_type": "electricity",
                "building_code": "BLOCK_A",
                "reduction_percentage": 15,
                "baseline_source": "forecast",
                "forecast_horizon_days": 7,
            },
        },
    }


DESCRIPTIONS = {
    "sim_building_electricity_15pct": "The spec example: BLOCK_A electricity, 15 % cut, 30-day baseline (45 days of records, only the latest 30 used).",
    "sim_campus_water_10pct": "Campus-wide water, 10 % cut; electricity records are present and ignored; per-building breakdown.",
    "sim_waste_explicit_baseline": "Waste with an explicit baseline (value, unit and period supplied) and no records.",
    "sim_partial_coverage": "25 of 30 baseline days have readings: calculated but data_quality_status = suspect, no gap filling.",
    "sim_unknown_building": "Building not in the records: HTTP 200 with calculation_status = not_calculated.",
    "sim_zero_baseline": "All-zero baseline: savings 0, data_quality_status = suspect with a warning.",
    "sim_invalid_percentage": "Reduction of 120 %: rejected with HTTP 422 and error.code = invalid_scenario.",
    "sim_compare_multiple_buildings": "Phase 7B comparison: 3 buildings with different reduction targets and rankings.",
    "sim_compare_overlapping_scenarios": "Phase 7B comparison: campus-wide vs building-level overlapping targets deduplicated deterministically.",
    "sim_forecast_baseline_electricity": "Phase 7B forecast-based: 7-day projected baseline generated via Phase 6 forecasting engine.",
}


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--outputs", action="store_true", help="also write *.response.json using the real engine")
    args = parser.parse_args(list(argv) if argv else None)

    requests = build_requests()
    compare_requests = build_compare_requests()
    forecast_requests = build_forecast_requests()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    comparison_manifest: List[Dict[str, Any]] = []
    extra: List[str] = []

    if args.outputs:
        sys.path.insert(0, str(ROOT))
        from ai_engine import AnalysisConfig, compare_payload, simulate_payload

    # 1. Standard single simulation requests (Phase 7A)
    for name, body in requests.items():
        _dump(OUT_DIR / f"{name}.json", body)
        entry: Dict[str, Any] = {
            "file": f"{name}.json", "description": DESCRIPTIONS[name], "records": len(body.get("records", [])),
        }
        if args.outputs:
            outcome = simulate_payload(body, AnalysisConfig())
            _dump(OUT_DIR / f"{name}.response.json", outcome.body)
            entry["response_file"] = f"{name}.response.json"
            entry["http_status"] = outcome.http_status
            entry["calculation_status"] = (outcome.body.get("result") or {}).get("calculation_status")
        manifest.append(entry)

    # 2. Forecast-based single simulation requests (Phase 7B)
    for name, body in forecast_requests.items():
        _dump(OUT_DIR / f"{name}.json", body)
        extra.append(f"{name}.json")
        if args.outputs:
            f_outcome = simulate_payload(body, AnalysisConfig())
            _dump(OUT_DIR / f"{name}.response.json", f_outcome.body)
            extra.append(f"{name}.response.json")

    # 3. Scenario comparison requests (Phase 7B)
    for name, body in compare_requests.items():
        _dump(OUT_DIR / f"{name}.json", body)
        c_entry: Dict[str, Any] = {
            "file": f"{name}.json", "description": DESCRIPTIONS[name], "records": len(body.get("records", [])),
        }
        if args.outputs:
            c_outcome = compare_payload(body, AnalysisConfig())
            _dump(OUT_DIR / f"{name}.response.json", c_outcome.body)
            c_entry["response_file"] = f"{name}.response.json"
            c_entry["http_status"] = c_outcome.http_status
            c_entry["scenarios_count"] = len(body.get("scenarios", []))
        comparison_manifest.append(c_entry)
        extra.append(f"{name}.json")
        if args.outputs:
            extra.append(f"{name}.response.json")

    if args.outputs:
        cfg = AnalysisConfig(
            electricity_emission_factor_kg_co2e_per_kwh=ILLUSTRATIVE_FACTOR,
            electricity_emission_factor_source="illustrative placeholder for documentation, not a real factor",
        )
        name = "sim_building_electricity_15pct.co2e_example.response.json"
        _dump(OUT_DIR / name, simulate_payload(requests["sim_building_electricity_15pct"], cfg).body)
        extra.append(name)

    _dump(OUT_DIR / "manifest.json", {
        "note": "Synthetic data. Generated by scripts/generate_simulation_samples.py.",
        "files": manifest,
        "comparison_files": comparison_manifest,
        "extra_files": sorted(set(extra)),
    })
    total_reqs = len(requests) + len(compare_requests) + len(forecast_requests)
    print(f"wrote {total_reqs} request(s)" + (f" and responses" if args.outputs else "") + f" to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
