"""Phase 7B test suite: advanced scenario comparison & forecast-based scenarios.

Covers:
- Multiple scenarios across buildings and resources
- Overlapping scenarios and deterministic double-counting prevention policy
- Forecast-based simulations (sufficient and insufficient history)
- Uncertainty propagation (mathematically justified bounds, no fabrication)
- Invalid inputs and error envelopes
- JSON serialization
- Compatibility and non-interference with existing AI functions
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ai_engine import (
    SIMULATION_CONTRACT_VERSION,
    AnalysisConfig,
    ConsumptionRecord,
    ForecastConfig,
    RecordValidationError,
    ResourceType,
    ScenarioValidationError,
    analyze_consumption,
    compare_payload,
    compare_scenarios,
    forecast_consumption,
    simulate_payload,
    simulate_scenario,
)

ROOT = Path(__file__).resolve().parents[2]
START = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)


def _rec(
    building: str = "BLOCK_A",
    resource: str = "electricity",
    ts: datetime = START,
    val: float = 100.0,
    unit: str = "kWh",
) -> Dict[str, Any]:
    return {
        "timestamp": ts.isoformat(),
        "building": building,
        "resource_type": resource,
        "consumption_value": val,
        "unit": unit,
    }


def _daily_series(
    building: str = "BLOCK_A",
    resource: str = "electricity",
    days: int = 35,
    val: float = 100.0,
    unit: str = "kWh",
    start: datetime = START,
) -> List[Dict[str, Any]]:
    return [
        _rec(building, resource, start + timedelta(days=i), val, unit)
        for i in range(days)
    ]


# --------------------------------------------------------------------------- #
# 1. Multiple scenarios across buildings and resources
# --------------------------------------------------------------------------- #
def test_compare_multiple_scenarios_different_buildings():
    records = (
        _daily_series("BLOCK_A", "electricity", days=30, val=100.0)
        + _daily_series("BLOCK_B", "electricity", days=30, val=200.0)
        + _daily_series("BLOCK_C", "electricity", days=30, val=300.0)
    )
    scenarios = [
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 10},
        {"resource_type": "electricity", "building_code": "BLOCK_B", "reduction_percentage": 15},
        {"resource_type": "electricity", "building_code": "BLOCK_C", "reduction_percentage": 20},
    ]
    res = compare_scenarios(records, scenarios)

    assert res["scenarios_count"] == 3
    assert len(res["scenarios"]) == 3
    assert all(s["calculation_status"] == "calculated" for s in res["scenarios"])

    # No overlaps between different buildings
    assert res["portfolio"]["overlapping_scenarios_detected"] is False
    assert len(res["portfolio"]["included_scenario_ids"]) == 3
    assert len(res["portfolio"]["excluded_scenario_ids"]) == 0

    # Electricity totals
    e_tot = res["resource_totals"]["electricity"]
    assert e_tot["scenarios_count"] == 3
    assert e_tot["calculated_scenarios_count"] == 3
    assert e_tot["included_in_portfolio_count"] == 3
    assert e_tot["excluded_from_portfolio_count"] == 0

    # BLOCK_A: 3000 * 0.10 = 300, BLOCK_B: 6000 * 0.15 = 900, BLOCK_C: 9000 * 0.20 = 1800 -> 3000
    assert e_tot["total_estimated_savings"] == pytest.approx(3000.0)
    assert e_tot["portfolio_savings"] == pytest.approx(3000.0)
    assert e_tot["unadjusted_gross_savings"] == pytest.approx(3000.0)
    assert e_tot["total_baseline"] == pytest.approx(18000.0)

    # Rankings
    ranks = res["rankings"]["by_resource"]["electricity"]
    assert len(ranks) == 3
    assert ranks[0]["building_code"] == "BLOCK_C"
    assert ranks[0]["rank"] == 1
    assert ranks[1]["building_code"] == "BLOCK_B"
    assert ranks[1]["rank"] == 2
    assert ranks[2]["building_code"] == "BLOCK_A"
    assert ranks[2]["rank"] == 3


def test_compare_multiple_scenarios_different_resources():
    records = (
        _daily_series("BLOCK_A", "electricity", days=30, val=100.0, unit="kWh")
        + _daily_series("BLOCK_A", "water", days=30, val=500.0, unit="liters")
        + _daily_series("BLOCK_A", "waste", days=30, val=20.0, unit="kg")
    )
    scenarios = [
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 10},
        {"resource_type": "water", "building_code": "BLOCK_A", "reduction_percentage": 20},
        {"resource_type": "waste", "building_code": "BLOCK_A", "reduction_percentage": 25},
    ]
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.5)
    res = compare_scenarios(records, scenarios, cfg)

    assert res["scenarios_count"] == 3
    # Native units preserved per resource
    assert res["resource_totals"]["electricity"]["unit"] == "kWh"
    assert res["resource_totals"]["electricity"]["portfolio_savings"] == pytest.approx(300.0)

    assert res["resource_totals"]["water"]["unit"] == "liters"
    assert res["resource_totals"]["water"]["portfolio_savings"] == pytest.approx(3000.0)

    assert res["resource_totals"]["waste"]["unit"] == "kg"
    assert res["resource_totals"]["waste"]["portfolio_savings"] == pytest.approx(150.0)

    # CO2e: electricity had factor 0.5 -> 300 * 0.5 = 150 kg CO2e
    assert res["resource_totals"]["electricity"]["total_estimated_co2e_savings_kg"] == pytest.approx(150.0)
    assert res["resource_totals"]["water"]["total_estimated_co2e_savings_kg"] is None
    assert res["resource_totals"]["waste"]["total_estimated_co2e_savings_kg"] is None
    assert res["total_co2e_savings_kg"] == pytest.approx(150.0)
    assert res["co2e_supported"] is True


def test_compare_different_percentage_reductions():
    records = _daily_series("BLOCK_A", "electricity", days=30, val=100.0)
    scenarios = [
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 5},
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 15},
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 25},
    ]
    res = compare_scenarios(records, scenarios)
    ranks = res["rankings"]["by_percentage"]
    assert [r["reduction_percentage"] for r in ranks] == [25.0, 15.0, 5.0]


# --------------------------------------------------------------------------- #
# 2. Overlapping scenarios and double-counting prevention policy
# --------------------------------------------------------------------------- #
def test_overlapping_scenarios_same_building_deduplication():
    records = _daily_series("BLOCK_A", "electricity", days=30, val=100.0)
    scenarios = [
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 10, "scenario_id": "opt-10"},
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 20, "scenario_id": "opt-20"},
    ]
    res = compare_scenarios(records, scenarios)

    assert res["portfolio"]["overlapping_scenarios_detected"] is True
    assert res["portfolio"]["included_scenario_ids"] == ["opt-20"]
    assert res["portfolio"]["excluded_scenario_ids"] == ["opt-10"]

    # Deduplicated portfolio savings equals opt-20 (600 kWh), not 300 + 600 = 900 kWh!
    e_tot = res["resource_totals"]["electricity"]
    assert e_tot["portfolio_savings"] == pytest.approx(600.0)
    assert e_tot["total_estimated_savings"] == pytest.approx(600.0)
    assert e_tot["unadjusted_gross_savings"] == pytest.approx(900.0)
    assert e_tot["overlapping_detected"] is True
    assert e_tot["included_in_portfolio_count"] == 1
    assert e_tot["excluded_from_portfolio_count"] == 1

    # Overlap group recorded
    assert len(res["portfolio"]["overlap_groups"]) == 1
    grp = res["portfolio"]["overlap_groups"][0]
    assert grp["conflict_type"] == "competing_building_scenarios"
    assert grp["selected_scenario_id"] == "opt-20"
    assert grp["excluded_scenario_ids"] == ["opt-10"]

    # Warnings reflect overlap
    assert any("Overlapping scenarios detected" in w for w in res["warnings"])


def test_overlapping_scenarios_campus_vs_building():
    # Campus total: BLOCK_A (100/day * 30 = 3000) + BLOCK_B (100/day * 30 = 3000) = 6000
    records = _daily_series("BLOCK_A", "electricity", days=30, val=100.0) + _daily_series("BLOCK_B", "electricity", days=30, val=100.0)
    # Scenario 1: Campus cut 10% -> 600 kWh savings
    # Scenario 2: BLOCK_A cut 15% -> 450 kWh savings
    scenarios = [
        {"resource_type": "electricity", "building_code": None, "reduction_percentage": 10, "scenario_id": "campus-10"},
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 15, "scenario_id": "block-a-15"},
    ]
    res = compare_scenarios(records, scenarios)

    assert res["portfolio"]["overlapping_scenarios_detected"] is True
    # Campus savings (600) > Block A savings (450), so campus scenario is chosen
    assert res["portfolio"]["included_scenario_ids"] == ["campus-10"]
    assert res["portfolio"]["excluded_scenario_ids"] == ["block-a-15"]
    assert res["resource_totals"]["electricity"]["portfolio_savings"] == pytest.approx(600.0)
    assert res["resource_totals"]["electricity"]["unadjusted_gross_savings"] == pytest.approx(1050.0)


def test_overlapping_scenarios_buildings_beat_campus():
    # Campus total: BLOCK_A (100/day * 30 = 3000) + BLOCK_B (100/day * 30 = 3000) = 6000
    records = _daily_series("BLOCK_A", "electricity", days=30, val=100.0) + _daily_series("BLOCK_B", "electricity", days=30, val=100.0)
    # Scenario 1: Campus cut 5% -> 300 kWh savings
    # Scenario 2: BLOCK_A cut 20% -> 600 kWh savings
    # Scenario 3: BLOCK_B cut 20% -> 600 kWh savings (combined buildings = 1200 kWh)
    scenarios = [
        {"resource_type": "electricity", "building_code": None, "reduction_percentage": 5, "scenario_id": "campus-5"},
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 20, "scenario_id": "block-a-20"},
        {"resource_type": "electricity", "building_code": "BLOCK_B", "reduction_percentage": 20, "scenario_id": "block-b-20"},
    ]
    res = compare_scenarios(records, scenarios)

    assert res["portfolio"]["overlapping_scenarios_detected"] is True
    # Combined buildings (1200) > campus (300), so building scenarios are chosen
    assert sorted(res["portfolio"]["included_scenario_ids"]) == ["block-a-20", "block-b-20"]
    assert res["portfolio"]["excluded_scenario_ids"] == ["campus-5"]
    assert res["resource_totals"]["electricity"]["portfolio_savings"] == pytest.approx(1200.0)


# --------------------------------------------------------------------------- #
# 3. Forecast-based scenarios
# --------------------------------------------------------------------------- #
def test_forecast_based_simulation_sufficient_history():
    records = _daily_series("BLOCK_A", "electricity", days=35, val=100.0)
    scenario = {
        "resource_type": "electricity",
        "building_code": "BLOCK_A",
        "reduction_percentage": 10,
        "baseline_source": "forecast",
        "forecast_horizon_days": 7,
    }
    result = simulate_scenario(records, scenario)

    assert result["calculation_status"] == "calculated"
    assert result["simulation_type"] == "forecast"
    assert result["baseline"]["source"] == "forecast"
    assert result["baseline"]["period_days"] == 7
    # 7 days of 100 kWh -> 700 kWh predicted baseline
    assert result["baseline"]["value"] == pytest.approx(700.0, abs=1.0)
    assert result["estimated_savings"]["value"] == pytest.approx(70.0, abs=0.5)
    assert result["projected_consumption"]["value"] == pytest.approx(630.0, abs=0.5)
    assert result["forecast_details"]["horizon_days"] == 7
    assert result["data_quality_status"] in {"ok", "suspect"}
    assert any("Phase 6 forecasting engine" in a for a in result["assumptions"])


def test_forecast_based_simulation_insufficient_history():
    # Only 5 days of data; Phase 6 requires minimum_history_days (14 days by default)
    records = _daily_series("BLOCK_A", "electricity", days=5, val=100.0)
    scenario = {
        "resource_type": "electricity",
        "building_code": "BLOCK_A",
        "reduction_percentage": 10,
        "baseline_source": "forecast",
    }
    result = simulate_scenario(records, scenario)

    assert result["calculation_status"] == "not_calculated"
    assert result["not_calculated_reason"] == "insufficient_forecast_data"
    assert result["data_quality_status"] == "insufficient_data"
    assert result["baseline"]["value"] is None
    assert result["estimated_savings"]["value"] is None
    assert result["projected_consumption"]["value"] is None
    assert any("Usable history" in w or "Insufficient" in w for w in result["warnings"])


def test_forecast_based_simulation_campus_wide():
    records = (
        _daily_series("BLOCK_A", "electricity", days=35, val=100.0)
        + _daily_series("BLOCK_B", "electricity", days=35, val=200.0)
    )
    scenario = {
        "resource_type": "electricity",
        "building_code": None,
        "reduction_percentage": 15,
        "baseline_source": "forecast",
        "forecast_horizon_days": 7,
    }
    result = simulate_scenario(records, scenario)

    assert result["calculation_status"] == "calculated"
    assert result["scope"]["level"] == "campus"
    assert result["scope"]["buildings_included"] == ["BLOCK_A", "BLOCK_B"]
    # 7 * (100 + 200) = 2100
    assert result["baseline"]["value"] == pytest.approx(2100.0, abs=2.0)
    assert result["estimated_savings"]["value"] == pytest.approx(315.0, abs=1.0)
    assert len(result["building_breakdown"]) == 2


# --------------------------------------------------------------------------- #
# 4. Uncertainty handling & no fabrication
# --------------------------------------------------------------------------- #
def test_forecast_uncertainty_propagation_when_justified():
    # 60 days of consistent data provides sufficient backtest residuals for empirical intervals
    records = _daily_series("BLOCK_A", "electricity", days=60, val=100.0)
    scenario = {
        "resource_type": "electricity",
        "building_code": "BLOCK_A",
        "reduction_percentage": 20,
        "baseline_source": "forecast",
        "forecast_horizon_days": 7,
    }
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.5)
    result = simulate_scenario(records, scenario, cfg)

    assert result["calculation_status"] == "calculated"
    # Check if backtest residuals supported uncertainty intervals
    if "uncertainty" in result and result["uncertainty"] is not None:
        u = result["uncertainty"]
        assert u["interval_level"] == 0.8
        assert "method" in u
        b_low = u["baseline"]["lower_bound"]
        b_up = u["baseline"]["upper_bound"]
        assert b_low <= result["baseline"]["value"] <= b_up

        # Savings bounds scale proportionally: savings = baseline * 20%
        s_low = u["estimated_savings"]["lower_bound"]
        s_up = u["estimated_savings"]["upper_bound"]
        assert s_low == pytest.approx(b_low * 0.20, abs=1e-4)
        assert s_up == pytest.approx(b_up * 0.20, abs=1e-4)

        # Projected bounds: projected = baseline * 80%
        p_low = u["projected_consumption"]["lower_bound"]
        p_up = u["projected_consumption"]["upper_bound"]
        assert p_low == pytest.approx(b_low * 0.80, abs=1e-4)
        assert p_up == pytest.approx(b_up * 0.80, abs=1e-4)

        # CO2e bounds: co2e = savings * 0.5
        c_low = u["estimated_co2e_savings_kg"]["lower_bound"]
        c_up = u["estimated_co2e_savings_kg"]["upper_bound"]
        assert c_low == pytest.approx(s_low * 0.5, abs=1e-4)
        assert c_up == pytest.approx(s_up * 0.5, abs=1e-4)


def test_no_fabrication_of_intervals_or_emissions_factors():
    # If no emission factor is configured: CO2e is strictly None
    records = _daily_series("BLOCK_A", "electricity", days=35, val=100.0)
    scenario = {
        "resource_type": "electricity",
        "building_code": "BLOCK_A",
        "reduction_percentage": 10,
        "baseline_source": "forecast",
    }
    result = simulate_scenario(records, scenario, config=AnalysisConfig())
    assert result["estimated_co2e_savings_kg"] is None
    assert result["emission_factor"] is None
    if "uncertainty" in result and result["uncertainty"]:
        assert result["uncertainty"]["estimated_co2e_savings_kg"] is None


# --------------------------------------------------------------------------- #
# 5. Invalid inputs
# --------------------------------------------------------------------------- #
def test_invalid_scenarios_input_types():
    records = _daily_series("BLOCK_A", "electricity", days=10)
    with pytest.raises(TypeError, match="scenarios must be a sequence"):
        compare_scenarios(records, None)  # type: ignore

    with pytest.raises(TypeError, match="scenarios must be a sequence"):
        compare_scenarios(records, "not_a_list")  # type: ignore

    with pytest.raises(TypeError, match="scenarios must be a sequence"):
        compare_scenarios(records, {"a": "dict"})  # type: ignore

    with pytest.raises(TypeError, match="must be a mapping"):
        compare_scenarios(records, ["not_a_dict"])  # type: ignore


def test_invalid_scenario_values_raise_validation_error():
    records = _daily_series("BLOCK_A", "electricity", days=10)
    with pytest.raises(ScenarioValidationError) as exc_info:
        compare_scenarios(records, [
            {"resource_type": "electricity", "reduction_percentage": 150},
            {"resource_type": "unknown_resource", "reduction_percentage": 10},
        ])
    errors = exc_info.value.errors
    assert any(e["loc"] == [0, "reduction_percentage"] for e in errors)
    assert any(e["loc"] == [1, "resource_type"] for e in errors)


def test_conflicting_baseline_source_and_consumption():
    records = _daily_series("BLOCK_A", "electricity", days=10)
    with pytest.raises(ScenarioValidationError):
        simulate_scenario(records, {
            "resource_type": "electricity",
            "reduction_percentage": 10,
            "baseline_source": "forecast",
            "baseline_consumption": {"value": 500.0, "unit": "kWh", "period_days": 30},
        })


# --------------------------------------------------------------------------- #
# 6. JSON serialization
# --------------------------------------------------------------------------- #
def test_json_serialization_roundtrip():
    records = _daily_series("BLOCK_A", "electricity", days=30, val=100.0)
    scenarios = [
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 10},
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 20},
    ]
    res = compare_scenarios(records, scenarios)
    text = json.dumps(res, allow_nan=False)
    decoded = json.loads(text)
    assert decoded["simulation_contract_version"] == SIMULATION_CONTRACT_VERSION
    assert decoded["scenarios_count"] == 2
    assert decoded == res


# --------------------------------------------------------------------------- #
# 7. Compatibility with existing AI functions
# --------------------------------------------------------------------------- #
def test_compatibility_with_existing_ai_functions():
    records = _daily_series("BLOCK_A", "electricity", days=30, val=100.0)
    # 1. Existing analyze_consumption
    analyze_before = analyze_consumption(records)

    # 2. Existing forecast_consumption
    forecast_before = forecast_consumption(records, horizon_days=7)

    # 3. Run compare_scenarios
    compare_scenarios(records, [
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 15}
    ])

    # 4. Verify analyze_consumption and forecast_consumption behave identically afterwards
    analyze_after = analyze_consumption(records)
    forecast_after = forecast_consumption(records, horizon_days=7)

    assert len(analyze_before) == len(analyze_after)
    assert [r.severity for r in analyze_before] == [r.severity for r in analyze_after]
    assert len(forecast_before) == len(forecast_after)
    assert forecast_before[0].predicted_total == forecast_after[0].predicted_total


# --------------------------------------------------------------------------- #
# 8. compare_payload contract envelope
# --------------------------------------------------------------------------- #
def test_compare_payload_success_envelope():
    records = _daily_series("BLOCK_A", "electricity", days=30, val=100.0)
    scenarios = [
        {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 10},
    ]
    outcome = compare_payload({"records": records, "scenarios": scenarios})
    assert outcome.ok is True
    assert outcome.http_status == 200
    assert outcome.body["status"] == "ok"
    assert outcome.body["contract_version"] == "1.0"
    assert outcome.body["simulation_contract_version"] == SIMULATION_CONTRACT_VERSION
    assert outcome.body["result"]["scenarios_count"] == 1


def test_compare_payload_error_envelopes():
    # 1. Non-dict payload -> 422
    out = compare_payload("not_a_dict")
    assert out.http_status == 422 and out.body["error"]["code"] == "invalid_request"

    # 2. Missing scenarios -> 422
    out = compare_payload({"records": []})
    assert out.http_status == 422 and out.body["error"]["code"] == "invalid_request"

    # 3. Invalid scenario -> 422 invalid_scenario
    out = compare_payload({"records": [], "scenarios": [{"resource_type": "gas", "reduction_percentage": 10}]})
    assert out.http_status == 422 and out.body["error"]["code"] == "invalid_scenario"

    # 4. Too many records -> 413
    out = compare_payload({"records": [{"dummy": 1}] * 5, "scenarios": []}, max_records=2)
    assert out.http_status == 413 and out.body["error"]["code"] == "too_many_records"


# --------------------------------------------------------------------------- #
# 9. Phase 7B sample payloads match the engine
# --------------------------------------------------------------------------- #
def test_simulation_samples_comparison_files_match_engine():
    sim_dir = ROOT / "samples" / "simulation"
    manifest = json.loads((sim_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest.get("comparison_files", []):
        req = json.loads((sim_dir / entry["file"]).read_text(encoding="utf-8"))
        stored = json.loads((sim_dir / entry["response_file"]).read_text(encoding="utf-8"))
        outcome = compare_payload(req, AnalysisConfig())
        assert outcome.http_status == entry["http_status"], entry["file"]
        assert outcome.body == stored, entry["file"]


def test_simulation_samples_forecast_file_matches_engine():
    sim_dir = ROOT / "samples" / "simulation"
    req = json.loads((sim_dir / "sim_forecast_baseline_electricity.json").read_text(encoding="utf-8"))
    stored = json.loads((sim_dir / "sim_forecast_baseline_electricity.response.json").read_text(encoding="utf-8"))
    outcome = simulate_payload(req, AnalysisConfig())
    assert outcome.http_status == 200
    assert outcome.body == stored
