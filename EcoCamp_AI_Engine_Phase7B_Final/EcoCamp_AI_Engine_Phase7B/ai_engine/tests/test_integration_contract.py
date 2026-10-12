"""Phase 5: JSON contract, serialization and the ``analyze_payload`` facade."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from ai_engine import (
    CONTRACT_VERSION,
    AnalysisConfig,
    AnalysisResult,
    ImpactSummary,
    LLMConfig,
    WasteEmissionFactor,
    analyze_consumption,
    analyze_payload,
    engine_info,
    json_schemas,
    serialize_result,
    serialize_results,
    to_json_string,
)
from ai_engine.contract import DEFAULT_MAX_RECORDS, TUNABLE_OPTIONS

DAY = timedelta(days=1)


def _series(building="Library", resource="electricity", unit="kWh", n=10, start=None,
            value=100.0, last=None, tz=timezone.utc):
    start = start or datetime(2026, 3, 1, tzinfo=tz)
    out = []
    for i in range(n):
        v = last if (last is not None and i == n - 1) else value + (i % 3)
        out.append({
            "timestamp": (start + i * DAY).isoformat(),
            "building": building,
            "resource_type": resource,
            "consumption_value": v,
            "unit": unit,
        })
    return out


FACTORED = AnalysisConfig(
    achievable_reduction_fraction=0.5,
    electricity_emission_factor_kg_co2e_per_kwh=0.4,  # illustrative test value
    electricity_emission_factor_source="test fixture, not a real source",
    water_emission_factor_kg_co2e_per_liter=0.0003,  # illustrative test value
    water_emission_factor_source="test fixture, not a real source",
    waste_emission_factor=WasteEmissionFactor(
        value=0.2, unit="kg_co2e_per_kg", source="test fixture, not a real source"
    ),
)


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def test_every_result_serializes_to_strict_json():
    records = (
        _series("Library", "electricity", "kWh", last=400.0)
        + _series("Hostel", "water", "liters", value=5000.0, last=11000.0)
        + _series("Canteen", "waste", "kg", value=80.0, last=0.0)  # zero -> suspect
        + _series("New Block", "electricity", "kWh", n=3)  # insufficient history
    )
    results = analyze_consumption(records, FACTORED)
    payload = serialize_results(results)
    text = json.dumps(payload, allow_nan=False)  # raises on datetime/NaN/object
    assert isinstance(json.loads(text), list) and len(payload) == len(records)
    assert to_json_string(payload) == json.dumps(payload, ensure_ascii=False, allow_nan=False)


def test_serialized_types_timestamps_enums_and_nulls():
    results = analyze_consumption(_series(last=400.0), FACTORED)
    flagged = serialize_result(results[-1])
    assert flagged["timestamp"] == "2026-03-10T00:00:00Z"  # tz-aware UTC -> 'Z'
    assert flagged["resource_type"] == "electricity"
    assert flagged["severity"] in {"low", "medium", "high", "critical"}
    assert flagged["data_quality_status"] == "ok"
    assert flagged["calculation_status"] == "calculated"
    assert set(flagged["estimated_savings"]) == {"value", "unit"}
    assert flagged["estimated_savings"]["unit"] == "kWh"
    assert flagged["estimated_savings"]["value"] > 0
    assert isinstance(flagged["recommendation"], dict)
    rec = flagged["recommendation"]
    assert rec["priority"] in {"low", "medium", "high", "urgent"}
    assert rec["anomaly_type"] == "above_baseline"
    assert rec["source"] == "rules" and rec["llm_status"] == "disabled"
    assert rec["llm_explanation"] is None and rec["llm_suggestions"] == []

    normal = serialize_result(results[0])  # first record: insufficient history
    # Optional fields stay present, as null, so clients can rely on the keys.
    for key in ("baseline_value", "deviation_percentage", "recommended_action",
                "estimated_savings", "savings_unit", "estimated_co2e_reduction_kg",
                "recommendation"):
        assert key in normal and normal[key] is None
    assert normal["data_quality_status"] == "insufficient_data"
    assert normal["calculation_status"] == "insufficient_baseline"


def test_result_keys_are_stable():
    expected = {
        "timestamp", "building", "resource_type", "unit", "observed_value", "baseline_value",
        "deviation_percentage", "anomaly_score", "severity", "explanation",
        "recommended_action", "estimated_savings", "calculation_assumptions",
        "data_quality_status", "savings_unit", "estimated_co2e_reduction_kg",
        "calculation_status", "recommendation",
    }
    out = serialize_result(analyze_consumption(_series())[0])
    assert set(out) == expected


@pytest.mark.parametrize(
    "tz, suffix",
    [(timezone.utc, "Z"), (timezone(timedelta(hours=5, minutes=30)), "+05:30")],
)
def test_aware_timestamps_keep_their_offset(tz, suffix):
    out = serialize_results(analyze_consumption(_series(tz=tz)))
    assert all(r["timestamp"].endswith(suffix) for r in out)


def test_naive_timestamps_are_serialized_without_an_offset():
    records = _series()
    for r in records:
        r["timestamp"] = r["timestamp"].replace("+00:00", "")  # naive ISO string
    out = serialize_results(analyze_consumption(records))
    assert out[0]["timestamp"] == "2026-03-01T00:00:00"


def test_serialized_result_round_trips_through_the_model():
    results = analyze_consumption(_series(last=400.0), FACTORED)
    for r in results:
        data = json.loads(json.dumps(serialize_result(r)))
        assert AnalysisResult.model_validate(data) == r


def test_co2e_fields_are_null_without_factor_and_numeric_with_factor():
    records = _series(last=400.0)
    no_factor = serialize_result(analyze_consumption(records)[-1])
    assert no_factor["estimated_co2e_reduction_kg"] is None
    assert no_factor["calculation_status"] == "calculated_without_co2e"
    with_factor = serialize_result(analyze_consumption(records, FACTORED)[-1])
    assert with_factor["estimated_co2e_reduction_kg"] == pytest.approx(
        with_factor["estimated_savings"]["value"] * 0.4, rel=1e-3
    )


# --------------------------------------------------------------------------- #
# analyze_payload: success path
# --------------------------------------------------------------------------- #
def test_payload_success_envelope():
    outcome = analyze_payload({"records": _series(last=400.0)})
    assert outcome.ok and outcome.http_status == 200
    body = outcome.body
    assert body["status"] == "ok" and body["contract_version"] == CONTRACT_VERSION
    assert set(body) == {"contract_version", "status", "engine_version", "summary",
                         "results", "impact_summary"}
    assert body["impact_summary"] is None
    summary = body["summary"]
    assert summary["records_analyzed"] == 10 == len(body["results"])
    assert summary["anomalies_flagged"] == 1
    assert summary["recommendations_generated"] == 1
    assert set(summary["severity_counts"]) == {"none", "low", "medium", "high", "critical"}
    assert sum(summary["severity_counts"].values()) == 10
    assert set(summary["data_quality_counts"]) == {"ok", "insufficient_data", "suspect"}
    assert set(summary["llm_status_counts"]) == {
        "disabled", "used", "missing_api_key", "sdk_unavailable", "timeout",
        "provider_error", "invalid_response", "call_limit_reached",
    }
    assert summary["llm_status_counts"]["disabled"] == 1
    json.dumps(body, allow_nan=False)


def test_payload_matches_direct_call():
    records = _series(last=400.0)
    direct = serialize_results(analyze_consumption(records))
    assert analyze_payload({"records": records}).body["results"] == direct


def test_payload_impact_summary_option():
    records = _series(last=400.0) + _series("Hostel", "water", "liters", value=5000.0, last=11000.0)
    outcome = analyze_payload(
        {"records": records, "options": {"include_impact_summary": True, "impact_by_building": True}},
        FACTORED,
    )
    impact = outcome.body["impact_summary"]
    ImpactSummary.model_validate(impact)  # the output is a valid ImpactSummary
    assert impact["records_counted"] == 2
    assert {t["building"] for t in impact["totals"]} == {"Library", "Hostel"}
    assert {t["savings_unit"] for t in impact["totals"]} == {"kWh", "liters"}
    json.dumps(outcome.body, allow_nan=False)


def test_payload_config_overrides_change_detection():
    records = _series(last=130.0)  # about +28% -> below the default 50% threshold
    default = analyze_payload({"records": records}).body["summary"]["anomalies_flagged"]
    lowered = analyze_payload(
        {"records": records, "options": {"config": {"deviation_threshold_percentage": 10}}}
    ).body["summary"]["anomalies_flagged"]
    assert default == 0 and lowered == 1


def test_payload_override_keeps_server_side_factors():
    outcome = analyze_payload(
        {"records": _series(last=400.0), "options": {"config": {"achievable_reduction_fraction": 0.25}}},
        FACTORED,
    )
    last = outcome.body["results"][-1]
    assert last["calculation_status"] == "calculated"  # factor from server config survived
    assert last["estimated_co2e_reduction_kg"] is not None


# --------------------------------------------------------------------------- #
# analyze_payload: client errors
# --------------------------------------------------------------------------- #
def _assert_error(outcome, status, code):
    assert not outcome.ok and outcome.http_status == status
    body = outcome.body
    assert body["status"] == "error" and body["contract_version"] == CONTRACT_VERSION
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and isinstance(body["error"]["details"], list)
    json.dumps(body, allow_nan=False)


@pytest.mark.parametrize("payload", [None, [], "records", 42, [{"building": "x"}]])
def test_non_object_body(payload):
    _assert_error(analyze_payload(payload), 422, "invalid_request")


@pytest.mark.parametrize("records", [None, [], {}, "x", 5])
def test_records_must_be_non_empty_array(records):
    _assert_error(analyze_payload({"records": records}), 422, "invalid_request")


def test_missing_records_and_unknown_top_level_field():
    _assert_error(analyze_payload({}), 422, "invalid_request")
    outcome = analyze_payload({"records": _series(), "extra": 1})
    _assert_error(outcome, 422, "invalid_request")
    assert outcome.body["error"]["details"][0]["errors"][0]["loc"] == ["extra"]


def test_too_many_records():
    outcome = analyze_payload({"records": _series(n=5)}, max_records=4)
    _assert_error(outcome, 413, "too_many_records")
    assert DEFAULT_MAX_RECORDS >= 1000


def test_invalid_records_report_every_bad_index():
    records = _series(n=4)
    records[1]["consumption_value"] = -3
    records[3]["unit"] = "MWh"
    records.append("not an object")
    outcome = analyze_payload({"records": records})
    _assert_error(outcome, 422, "invalid_records")
    details = outcome.body["error"]["details"]
    assert [d["index"] for d in details] == [1, 3, 4]
    for d in details:
        for e in d["errors"]:
            assert isinstance(e["loc"], list) and isinstance(e["msg"], str)
    assert details[0]["errors"][0]["loc"] == ["consumption_value"]


def test_invalid_sample_file_yields_422(load_sample):
    outcome = analyze_payload({"records": load_sample("invalid_records")})
    _assert_error(outcome, 422, "invalid_records")
    assert [d["index"] for d in outcome.body["error"]["details"]] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "options",
    [
        "yes",
        {"unknown": True},
        {"include_impact_summary": "true"},
        {"impact_by_building": 1},
        {"config": []},
        {"config": {"anomaly_method": "isolation_forest"}},  # not adjustable per request
        {"config": {"electricity_emission_factor_kg_co2e_per_kwh": 0.4}},
        {"config": {"llm": {"enabled": True}}},
        {"config": {"baseline_window": 2}},  # fails AnalysisConfig validation
        {"config": {"baseline_window": 5, "minimum_history": 9}},  # window < history
        {"config": {"achievable_reduction_fraction": 1.5}},
    ],
)
def test_invalid_options(options):
    _assert_error(analyze_payload({"records": _series(), "options": options}), 422, "invalid_options")


def test_tunable_options_are_a_safe_subset():
    banned = {"llm", "anomaly_method", "random_seed"}
    assert banned.isdisjoint(TUNABLE_OPTIONS)
    assert not any("emission" in name for name in TUNABLE_OPTIONS)


def test_wrong_config_type_is_a_programmer_error():
    with pytest.raises(TypeError):
        analyze_payload({"records": _series()}, {"baseline_window": 5})  # type: ignore[arg-type]


def test_unexpected_failure_becomes_generic_500(monkeypatch):
    import ai_engine.contract as contract

    def boom(records, cfg):
        raise RuntimeError("secret-internal-detail")

    monkeypatch.setattr(contract, "analyze_consumption", boom)
    outcome = analyze_payload({"records": _series()})
    _assert_error(outcome, 500, "analysis_failed")
    assert "secret-internal-detail" not in json.dumps(outcome.body)


# --------------------------------------------------------------------------- #
# engine_info and schemas
# --------------------------------------------------------------------------- #
def test_engine_info_default_has_no_llm_and_no_key_lookup(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "should-never-appear")
    info = engine_info()
    assert info["recommendation_mode"] == "rules"
    assert info["llm"]["enabled"] is False and info["llm"]["api_key_configured"] is None
    assert info["supported_resources"] == {"electricity": "kWh", "water": "liters", "waste": "kg"}
    assert "should-never-appear" not in json.dumps(info)


def test_engine_info_reports_key_presence_not_value(monkeypatch):
    cfg = AnalysisConfig(llm=LLMConfig(enabled=True, provider="openai", model="some-model"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert engine_info(cfg)["llm"]["api_key_configured"] is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    info = engine_info(cfg)
    assert info["llm"]["api_key_configured"] is True
    assert info["recommendation_mode"] == "rules_llm_with_fallback"
    assert "sk-not-a-real-key" not in json.dumps(info)


def test_json_schemas_are_json_serializable():
    schemas = json_schemas()
    assert set(schemas) == {"record", "result", "impact_summary"}
    json.dumps(schemas)
    assert "timestamp" in schemas["record"]["properties"]
