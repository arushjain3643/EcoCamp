"""Phase 6: forecasting integration (JSON contract, sample requests, backend-style usage).

Complements ``test_forecasting_engine.py`` (algorithms). Nothing here needs network access.
"""
import json
from pathlib import Path

import pytest

from ai_engine import (
    FORECAST_CONTRACT_VERSION,
    ForecastConfig,
    analyze_consumption,
    analyze_payload,
    forecast_consumption,
    forecast_payload,
    serialize_forecasts,
)

FORECAST_SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "forecasting"
ROOT = Path(__file__).resolve().parents[2]


def _request(stem):
    return json.loads((FORECAST_SAMPLES / f"{stem}.json").read_text(encoding="utf-8"))


def _ok(stem, config=None):
    outcome = forecast_payload(_request(stem), config)
    assert outcome.ok and outcome.http_status == 200, outcome.body
    json.dumps(outcome.body, allow_nan=False)  # strictly JSON-serializable: no NaN/Infinity, no objects
    return outcome.body


def test_forecast_sample_manifest_matches_files():
    manifest = json.loads((FORECAST_SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    names = {e["file"] for e in manifest["files"]}
    assert names == {
        "forecast_normal_7day.json", "forecast_next_day.json", "forecast_missing_irregular.json",
        "forecast_insufficient_history.json", "forecast_long_history.json",
    }
    for entry in manifest["files"]:
        body = json.loads((FORECAST_SAMPLES / entry["file"]).read_text(encoding="utf-8"))
        assert len(body["records"]) == entry["records"]
        assert body["options"]["horizon_days"] == entry["horizon_days"]


def test_normal_sample_forecasts_every_series_seven_days():
    body = _ok("forecast_normal_7day")
    assert body["status"] == "ok" and body["forecast_contract_version"] == FORECAST_CONTRACT_VERSION
    assert body["summary"]["series_count"] == 4 and body["summary"]["forecasts_produced"] == 4
    units = {"electricity": "kWh", "water": "liters", "waste": "kg"}
    keys = [(f["building"], f["resource_type"]) for f in body["forecasts"]]
    assert keys == sorted(keys) and len(set(keys)) == 4  # one forecast per series, stable order
    for f in body["forecasts"]:
        assert f["forecast_available"] and f["unit"] == units[f["resource_type"]]
        assert len(f["predicted_values"]) == len(f["lower_bound"]) == len(f["upper_bound"]) == 7
        assert len(f["forecast_dates"]) == 7
        assert all(lo <= p <= hi for lo, p, hi in zip(f["lower_bound"], f["predicted_values"], f["upper_bound"]))
        assert f["evaluation"]["backtest_performed"] and f["evaluation"]["metrics"]


def test_next_day_sample():
    body = _ok("forecast_next_day")
    (f,) = body["forecasts"]
    assert f["forecast_available"] and f["forecast_horizon_days"] == 1
    assert len(f["predicted_values"]) == 1 and f["predicted_total"] == f["predicted_values"][0]


def test_missing_and_irregular_sample_is_forecast_without_imputation():
    (f,) = _ok("forecast_missing_irregular")["forecasts"]
    assert f["forecast_available"]
    assert f["history"]["missing_days"] == 3 and f["history"]["observed_days"] == 46
    assert f["data_quality_status"] == "suspect"
    notes = " ".join(f["confidence_notes"])
    assert "not imputed" in notes and "duplicate" in notes


def test_insufficient_history_sample_returns_no_numbers():
    body = _ok("forecast_insufficient_history")
    assert body["summary"]["series_count"] == 2 and body["summary"]["forecasts_produced"] == 0
    for f in body["forecasts"]:
        assert f["forecast_available"] is False and f["data_quality_status"] == "insufficient_data"
        assert f["predicted_values"] == [] and f["lower_bound"] == [] and f["upper_bound"] == []
        assert f["predicted_total"] is None and f["model_used"] is None
        assert f["confidence_notes"]


def test_long_history_sample_evaluates_advanced_model_against_baseline():
    pytest.importorskip("sklearn")
    (f,) = _ok("forecast_long_history")["forecasts"]
    assert f["forecast_available"]
    ev = f["evaluation"]
    assert ev["advanced_model_considered"] is True
    methods = {m["method"]: m for m in ev["metrics"]}
    assert set(methods) == {"seasonal_naive", "hist_gradient_boosting"}
    # the choice must follow the reported evidence (10% default improvement threshold)
    justified = methods["hist_gradient_boosting"]["mae"] <= 0.9 * methods["seasonal_naive"]["mae"]
    assert (f["model_used"] == "hist_gradient_boosting") == justified


def test_forecast_payload_matches_forecast_consumption():
    request = _request("forecast_next_day")
    direct = serialize_forecasts(forecast_consumption(request["records"], 1))
    assert forecast_payload(request).body["forecasts"] == direct


@pytest.mark.parametrize("options", [{"horizon_days": 0}, {"horizon_days": 31}, {"horizon_days": True},
                                     {"horizon_days": "7"}, {"horizon_days": 7.5}])
def test_invalid_horizon_is_a_contract_error(options):
    outcome = forecast_payload({"records": _request("forecast_next_day")["records"], "options": options})
    assert not outcome.ok and outcome.http_status == 422
    assert outcome.body["error"]["code"] == "invalid_options"


def test_unknown_option_and_bad_bodies_are_contract_errors():
    records = _request("forecast_next_day")["records"]
    assert forecast_payload({"records": records, "options": {"nope": 1}}).body["error"]["code"] == "invalid_options"
    assert forecast_payload({"records": records, "extra": 1}).body["error"]["code"] == "invalid_request"
    assert forecast_payload({"records": []}).body["error"]["code"] == "invalid_request"
    assert forecast_payload([]).http_status == 422


def test_invalid_records_and_record_limit():
    outcome = forecast_payload({"records": [{"timestamp": "not a date"}]})
    assert outcome.http_status == 422 and outcome.body["error"]["code"] == "invalid_records"
    outcome = forecast_payload({"records": _request("forecast_next_day")["records"]}, max_records=3)
    assert outcome.http_status == 413 and outcome.body["error"]["code"] == "too_many_records"


def test_horizon_limit_follows_forecast_config():
    request = {"records": _request("forecast_next_day")["records"], "options": {"horizon_days": 14}}
    assert forecast_payload(request, ForecastConfig(max_horizon_days=10)).http_status == 422
    assert forecast_payload(request, ForecastConfig(max_horizon_days=14)).http_status == 200


def test_analyze_consumption_and_payload_are_unchanged_by_phase_6():
    records = _request("forecast_normal_7day")["records"]
    results = analyze_consumption(records)
    assert len(results) == len(records)
    outcome = analyze_payload({"records": records})
    assert outcome.http_status == 200 and len(outcome.body["results"]) == len(records)


def test_input_order_does_not_change_forecasts():
    records = _request("forecast_normal_7day")["records"]
    a = serialize_forecasts(forecast_consumption(records, 7))
    b = serialize_forecasts(forecast_consumption(list(reversed(records)), 7))
    assert a == b


def test_documentation_mentions_forecasting_contract():
    doc = (ROOT / "docs" / "JSON_CONTRACT.md").read_text(encoding="utf-8")
    assert f'"forecast_contract_version": "{FORECAST_CONTRACT_VERSION}"' in doc
    assert "/api/v1/ai/forecast" in doc and "forecast_payload" in doc
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "forecast_consumption" in readme and "Phase 6" in readme
