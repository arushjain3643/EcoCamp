"""Phase 7A: samples, documentation, and proof that earlier public behavior is unchanged."""
import inspect
import json
from pathlib import Path

import pytest

from ai_engine import (
    SIMULATION_CONTRACT_VERSION,
    AnalysisConfig,
    __version__,
    simulate_payload,
    simulate_scenario,
)

ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = ROOT / "samples" / "simulation"


def _load(name):
    return json.loads((SIM_DIR / name).read_text(encoding="utf-8"))


def _manifest():
    return _load("manifest.json")


# --------------------------------------------------------------------------- #
# Samples are real engine output (regenerate with scripts/generate_simulation_samples.py --outputs)
# --------------------------------------------------------------------------- #
def test_manifest_lists_existing_files_with_matching_record_counts():
    manifest = _manifest()
    assert len(manifest["files"]) == 7
    for entry in manifest["files"]:
        request = _load(entry["file"])
        assert set(request) <= {"records", "scenario"} and "scenario" in request
        assert len(request.get("records", [])) == entry["records"]
        assert (SIM_DIR / entry["response_file"]).is_file()
    for name in manifest["extra_files"]:
        assert (SIM_DIR / name).is_file()


def test_stored_responses_match_what_the_engine_returns_today():
    for entry in _manifest()["files"]:
        outcome = simulate_payload(_load(entry["file"]), AnalysisConfig())
        assert outcome.http_status == entry["http_status"], entry["file"]
        assert outcome.body == _load(entry["response_file"]), entry["file"]


def test_co2e_example_response_matches_the_engine_with_the_illustrative_factor():
    cfg = AnalysisConfig(
        electricity_emission_factor_kg_co2e_per_kwh=0.5,
        electricity_emission_factor_source="illustrative placeholder for documentation, not a real factor",
    )
    request = _load("sim_building_electricity_15pct.json")
    stored = _load("sim_building_electricity_15pct.co2e_example.response.json")
    assert simulate_payload(request, cfg).body == stored
    assert stored["result"]["estimated_co2e_savings_kg"] == pytest.approx(
        stored["result"]["estimated_savings"]["value"] * 0.5
    )


def test_sample_outcomes_cover_every_documented_situation():
    results = {e["file"]: e for e in _manifest()["files"]}
    assert results["sim_unknown_building.json"]["calculation_status"] == "not_calculated"
    assert results["sim_invalid_percentage.json"]["http_status"] == 422
    ok = {e["calculation_status"] for e in results.values() if e["http_status"] == 200}
    assert ok == {"calculated", "not_calculated"}
    assert _load("sim_partial_coverage.response.json")["result"]["data_quality_status"] == "suspect"
    assert _load("sim_zero_baseline.response.json")["result"]["estimated_savings"]["value"] == 0.0


# --------------------------------------------------------------------------- #
# Earlier public API is unchanged
# --------------------------------------------------------------------------- #
def test_existing_public_function_signatures_are_unchanged():
    from ai_engine import analyze_consumption, forecast_consumption

    assert list(inspect.signature(analyze_consumption).parameters) == ["records", "config"]
    assert inspect.signature(analyze_consumption).parameters["config"].default is None
    forecast = inspect.signature(forecast_consumption).parameters
    assert list(forecast) == ["records", "horizon_days", "config"]
    assert forecast["horizon_days"].default == 7 and forecast["config"].default is None


def test_every_earlier_public_name_is_still_exported():
    import ai_engine

    earlier = [
        "analyze_consumption", "analyze_payload", "forecast_consumption", "forecast_payload",
        "aggregate_impact", "generate_recommendations", "engine_info", "json_schemas",
        "serialize_result", "serialize_results", "serialize_forecast", "serialize_forecasts",
        "AnalysisConfig", "ForecastConfig", "LLMConfig", "ConsumptionRecord", "AnalysisResult",
        "ForecastResult", "RecordValidationError", "ResourceType", "STANDARD_UNITS",
        "CONTRACT_VERSION", "FORECAST_CONTRACT_VERSION",
    ]
    for name in earlier:
        assert name in ai_engine.__all__ and hasattr(ai_engine, name), name
    for name in (
        "simulate_scenario",
        "simulate_payload",
        "compare_scenarios",
        "compare_payload",
        "ScenarioValidationError",
        "SIMULATION_CONTRACT_VERSION",
    ):
        assert name in ai_engine.__all__ and hasattr(ai_engine, name), name


def test_analyze_and_forecast_still_work_on_simulation_sample_data():
    from ai_engine import analyze_consumption, analyze_payload, forecast_consumption, serialize_results

    records = _load("sim_building_electricity_15pct.json")["records"]
    before = serialize_results(analyze_consumption(records))
    simulate_scenario(records, _load("sim_building_electricity_15pct.json")["scenario"])
    assert serialize_results(analyze_consumption(records)) == before  # simulator has no side effects
    assert len(before) == len(records)
    assert analyze_payload({"records": records}).http_status == 200
    forecasts = forecast_consumption(records, horizon_days=3)
    assert len(forecasts) == 1


def test_simulator_accepts_validated_record_objects():
    from ai_engine import ConsumptionRecord

    request = _load("sim_building_electricity_15pct.json")
    objects = [ConsumptionRecord.model_validate(r) for r in request["records"]]
    assert simulate_scenario(objects, request["scenario"]) == simulate_scenario(request["records"], request["scenario"])


def test_engine_and_contract_versions():
    assert SIMULATION_CONTRACT_VERSION == "1.0"
    body = simulate_payload(_load("sim_building_electricity_15pct.json")).body
    assert body["engine_version"] == __version__ == body["result"]["engine_version"]


# --------------------------------------------------------------------------- #
# Documentation and delivery
# --------------------------------------------------------------------------- #
def test_json_contract_documents_the_simulation_contract():
    doc = (ROOT / "docs" / "JSON_CONTRACT.md").read_text(encoding="utf-8")
    assert f'"simulation_contract_version": "{SIMULATION_CONTRACT_VERSION}"' in doc
    for needle in ("/api/v1/ai/simulate", "simulate_payload", "simulate_scenario", "invalid_scenario",
                   "not_calculated", "hypothetical", "baseline_consumption", "reduction_percentage"):
        assert needle in doc, needle


def test_documented_example_matches_the_real_sample_output():
    doc = (ROOT / "docs" / "JSON_CONTRACT.md").read_text(encoding="utf-8")
    result = _load("sim_building_electricity_15pct.response.json")["result"]
    assert f'"scenario_id": "{result["scenario_id"]}"' in doc
    for value in (result["baseline"]["value"], result["estimated_savings"]["value"],
                  result["projected_consumption"]["value"]):
        assert f'"value": {value}' in doc, value
    assert result["estimated_savings"]["value"] == pytest.approx(result["baseline"]["value"] * 0.15, abs=1e-6)


def test_readmes_mention_phase_7a_and_7b():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "simulate_scenario" in readme and "compare_scenarios" in readme
    assert "Phase 7B" in readme
    samples = (ROOT / "samples" / "README.md").read_text(encoding="utf-8")
    assert "samples/simulation" in samples or "simulation/" in samples


def test_phase_7b_files_are_present_and_stop_at_7b():
    for rel in ("ai_engine/scenario_simulator.py", "scripts/generate_simulation_samples.py",
                "samples/simulation/manifest.json", "docs/PHASE7B_VERIFICATION.md"):
        assert (ROOT / rel).is_file(), rel
    import ai_engine

    # Phase 7B exports are present
    assert hasattr(ai_engine, "compare_scenarios") and "compare_scenarios" in ai_engine.__all__
    assert hasattr(ai_engine, "compare_payload") and "compare_payload" in ai_engine.__all__

    # Scope guard: STOP after Phase 7B. Do not implement Phase 8.
    assert not [n for n in dir(ai_engine) if "phase8" in n.lower() or "phase_8" in n.lower()]
