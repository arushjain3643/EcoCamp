"""Phase 5: the FastAPI integration example (skipped if fastapi/httpx are not installed)."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by fastapi.testclient

from fastapi.testclient import TestClient  # noqa: E402

from ai_engine import AnalysisConfig, LLMConfig  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[2] / "examples" / "fastapi_integration" / "app.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("ecocamp_fastapi_example", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example(monkeypatch):
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ECOCAMP_LLM_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    try:  # a developer's local .env must not leak into the tests
        import dotenv

        monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    except ImportError:
        pass
    return _load_example()


@pytest.fixture
def client(example):
    return TestClient(example.create_app(AnalysisConfig()))


def test_health_reports_rules_mode_and_no_secrets(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["recommendation_mode"] == "rules"
    assert body["llm"]["enabled"] is False
    assert "api_key" not in json.dumps(body).replace("api_key_configured", "")


def test_analyze_mixed_campus_sample(client, load_sample):
    response = client.post("/api/v1/ai/analyze", json={"records": load_sample("mixed_campus")})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and len(body["results"]) == 782
    assert body["summary"]["anomalies_flagged"] == 16
    flagged = [r for r in body["results"] if r["severity"] != "none"]
    assert all(r["recommendation"]["llm_status"] == "disabled" for r in flagged)


def test_analyze_invalid_records_returns_contract_error(client, load_sample):
    response = client.post("/api/v1/ai/analyze", json={"records": load_sample("invalid_records")})
    assert response.status_code == 422
    error = response.json()["error"]
    assert response.json()["status"] == "error" and error["code"] == "invalid_records"
    assert [d["index"] for d in error["details"]] == [1, 2, 3, 4, 5]


def test_malformed_json_returns_contract_error(client):
    response = client.post(
        "/api/v1/ai/analyze", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_json"


def test_record_limit_is_enforced(example, load_sample):
    small = TestClient(example.create_app(AnalysisConfig(), max_records=10))
    response = small.post("/api/v1/ai/analyze", json={"records": load_sample("normal_consumption")})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "too_many_records"


def test_schema_endpoint(client):
    schemas = client.get("/api/v1/ai/schema").json()
    assert set(schemas) == {"record", "result", "impact_summary"}


def test_llm_enabled_without_key_still_answers(example, monkeypatch, load_sample):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = AnalysisConfig(llm=LLMConfig(enabled=True, provider="gemini", model="any-model-name"))
    client = TestClient(example.create_app(cfg))
    assert client.get("/health").json()["llm"]["api_key_configured"] is False
    response = client.post("/api/v1/ai/analyze", json={"records": load_sample("waste_anomalies")})
    assert response.status_code == 200
    counts = response.json()["summary"]["llm_status_counts"]
    assert counts["missing_api_key"] == 4 and counts["used"] == 0


def test_build_engine_config_from_environment(example):
    cfg = example.build_engine_config({})
    assert cfg.llm.enabled is False and cfg.electricity_emission_factor_kg_co2e_per_kwh is None
    cfg = example.build_engine_config({
        "ECOCAMP_LLM_ENABLED": "true", "ECOCAMP_LLM_PROVIDER": "openai",
        "ECOCAMP_LLM_MODEL": "any-model-name", "ECOCAMP_LLM_TIMEOUT_SECONDS": "12",
        "ECOCAMP_REDUCTION_FRACTION": "0.3",
        "ECOCAMP_ELECTRICITY_EF_KG_CO2E_PER_KWH": "0.5",
        "ECOCAMP_ELECTRICITY_EF_SOURCE": "test fixture",
    })
    assert cfg.llm.enabled and cfg.llm.provider.value == "openai" and cfg.llm.timeout_seconds == 12
    assert cfg.achievable_reduction_fraction == 0.3
    assert cfg.electricity_emission_factor_kg_co2e_per_kwh == 0.5
    with pytest.raises(ValueError):  # enabled without a model fails fast at start-up
        example.build_engine_config({"ECOCAMP_LLM_ENABLED": "true"})


# --------------------------------------------------------------------------- #
# Phase 6: forecast route
# --------------------------------------------------------------------------- #
def _forecast_sample(name):
    path = APP_PATH.parents[1].parent / "samples" / "forecasting" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_forecast_route_returns_contract_response(client):
    response = client.post("/api/v1/ai/forecast", json=_forecast_sample("forecast_next_day"))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["summary"]["forecasts_produced"] == 1
    assert len(body["forecasts"][0]["predicted_values"]) == 1


def test_forecast_route_insufficient_history_is_a_200_without_numbers(client):
    response = client.post("/api/v1/ai/forecast", json=_forecast_sample("forecast_insufficient_history"))
    assert response.status_code == 200
    assert response.json()["summary"]["forecasts_produced"] == 0


def test_forecast_route_errors_use_the_contract_envelope(client):
    bad_json = client.post("/api/v1/ai/forecast", content=b"{not json",
                           headers={"Content-Type": "application/json"})
    assert bad_json.status_code == 422 and bad_json.json()["error"]["code"] == "invalid_json"
    bad_horizon = client.post("/api/v1/ai/forecast", json={
        "records": _forecast_sample("forecast_next_day")["records"], "options": {"horizon_days": 0}})
    assert bad_horizon.status_code == 422 and bad_horizon.json()["error"]["code"] == "invalid_options"


def test_analyze_route_still_works_next_to_forecast_route(client, load_sample):
    response = client.post("/api/v1/ai/analyze", json={"records": load_sample("water_wastage")})
    assert response.status_code == 200 and response.json()["summary"]["anomalies_flagged"] == 6


# --------------------------------------------------------------------------- #
# Phase 7A: simulate route
# --------------------------------------------------------------------------- #
def _simulation_sample(name):
    path = APP_PATH.parents[1].parent / "samples" / "simulation" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_simulate_route_returns_contract_response(client):
    response = client.post("/api/v1/ai/simulate", json=_simulation_sample("sim_building_electricity_15pct"))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["result"]["calculation_status"] == "calculated"
    assert body["result"]["estimated_savings"]["unit"] == "kWh"
    assert body["result"]["projection_type"] == "hypothetical"


def test_simulate_route_not_calculated_is_http_200(client):
    response = client.post("/api/v1/ai/simulate", json=_simulation_sample("sim_unknown_building"))
    assert response.status_code == 200
    assert response.json()["result"]["calculation_status"] == "not_calculated"


def test_simulate_route_errors_use_the_contract_envelope(client):
    bad_json = client.post("/api/v1/ai/simulate", content=b"{not json",
                           headers={"Content-Type": "application/json"})
    assert bad_json.status_code == 422 and bad_json.json()["error"]["code"] == "invalid_json"
    bad_scenario = client.post("/api/v1/ai/simulate", json=_simulation_sample("sim_invalid_percentage"))
    assert bad_scenario.status_code == 422 and bad_scenario.json()["error"]["code"] == "invalid_scenario"


def test_simulate_route_uses_the_engine_emission_factor(example):
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.5)
    client = TestClient(example.create_app(cfg))
    body = client.post("/api/v1/ai/simulate", json=_simulation_sample("sim_building_electricity_15pct")).json()
    result = body["result"]
    assert result["estimated_co2e_savings_kg"] == pytest.approx(result["estimated_savings"]["value"] * 0.5, abs=1e-5)


# --------------------------------------------------------------------------- #
# Phase 7B: /api/simulate and /api/simulate/compare routes
# --------------------------------------------------------------------------- #
def test_direct_api_simulate_route(client):
    response = client.post("/api/simulate", json=_simulation_sample("sim_building_electricity_15pct"))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["result"]["calculation_status"] == "calculated"
    assert body["result"]["estimated_savings"]["unit"] == "kWh"


def test_simulate_compare_route_multiple_scenarios(client):
    records = _simulation_sample("sim_building_electricity_15pct")["records"]
    payload = {
        "records": records,
        "scenarios": [
            {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 10},
            {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 20},
        ],
    }
    response = client.post("/api/simulate/compare", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    res = body["result"]
    assert res["scenarios_count"] == 2
    assert res["portfolio"]["overlapping_scenarios_detected"] is True
    assert len(res["portfolio"]["included_scenario_ids"]) == 1
    # 20% scenario selected
    assert res["resource_totals"]["electricity"]["portfolio_savings"] > 0


def test_simulate_compare_route_errors(client):
    bad_json = client.post("/api/simulate/compare", content=b"{not json",
                           headers={"Content-Type": "application/json"})
    assert bad_json.status_code == 422 and bad_json.json()["error"]["code"] == "invalid_json"

    bad_scenarios = client.post("/api/simulate/compare", json={"scenarios": "not_a_list"})
    assert bad_scenarios.status_code == 422 and bad_scenarios.json()["error"]["code"] == "invalid_request"

    bad_scenario_item = client.post("/api/simulate/compare", json={
        "scenarios": [{"resource_type": "electricity", "reduction_percentage": 150}]
    })
    assert bad_scenario_item.status_code == 422 and bad_scenario_item.json()["error"]["code"] == "invalid_scenario"

