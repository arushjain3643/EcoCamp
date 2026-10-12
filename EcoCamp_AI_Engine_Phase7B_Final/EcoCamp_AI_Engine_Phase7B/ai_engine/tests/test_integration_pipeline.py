"""Phase 5: end-to-end smoke tests of the whole pipeline on the sample datasets.

The pipeline is: validation -> anomaly detection -> impact calculation ->
recommendation generation -> structured output. These tests also prove the
engine works with no LLM API key and no network access.
"""
import json
import socket
from datetime import datetime

import pytest

import ai_engine.llm_client as llm_client_mod
import ai_engine.recommendation_engine as rec_mod
from ai_engine import (
    AnalysisConfig,
    LLMConfig,
    LLMStatus,
    RecommendationSource,
    Severity,
    analyze_consumption,
    analyze_payload,
    serialize_results,
)
from ai_engine.llm_client import LLMTimeoutError

SCENARIOS = [
    "normal_consumption", "electricity_unusual_usage", "water_wastage",
    "waste_anomalies", "insufficient_history", "edge_cases",
]


@pytest.fixture(autouse=True)
def offline_and_keyless(monkeypatch):
    """Every test here runs with no API keys, no SDK client and no network."""
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ECOCAMP_LLM_ENABLED",
                 "ECOCAMP_LLM_PROVIDER", "ECOCAMP_LLM_MODEL", "ECOCAMP_LLM_API_KEY_ENV"):
        monkeypatch.delenv(name, raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted during an offline test")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _flagged(results):
    return [r for r in results if r.severity is not Severity.NONE]


def _by(results, building, resource):
    return [r for r in results if r.building == building and r.resource_type.value == resource]


# --------------------------------------------------------------------------- #
# Scenario datasets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", SCENARIOS + ["mixed_campus"])
def test_sample_runs_through_full_pipeline_and_serializes(name, load_sample):
    records = load_sample(name)
    results = analyze_consumption(records)
    assert len(results) == len(records)
    json.dumps(serialize_results(results), allow_nan=False)
    for rec, res in zip(records, results):  # same order, same identity
        assert res.building == rec["building"].strip()
        assert res.observed_value == rec["consumption_value"]
        assert res.timestamp == _ts(rec["timestamp"])


def test_normal_consumption_raises_no_alerts(load_sample):
    records = load_sample("normal_consumption")
    results = analyze_consumption(records)
    assert len(records) == 360
    assert _flagged(results) == []
    assert all(r.recommendation is None and r.recommended_action is None for r in results)
    # 12 building/resource series; the first 7 readings of each have no baseline yet.
    insufficient = [r for r in results if r.data_quality_status.value == "insufficient_data"]
    assert len(insufficient) == 12 * 7
    assert sum(1 for r in results if r.data_quality_status.value == "ok") == 360 - 12 * 7


def test_unusual_electricity_usage_is_flagged(load_sample):
    results = analyze_consumption(load_sample("electricity_unusual_usage"))
    library = _flagged(_by(results, "Library", "electricity"))
    assert len(library) == 3
    assert len(_flagged(_by(results, "Admin Block", "electricity"))) == 1
    assert _flagged(_by(results, "Hostel A", "electricity")) == []  # control series
    for r in library:
        assert r.deviation_percentage > 100
        assert r.estimated_savings is not None and r.estimated_savings.unit == "kWh"
        assert r.estimated_savings.value == pytest.approx(
            (r.observed_value - r.baseline_value) * 0.5, rel=1e-4
        )
        assert r.estimated_co2e_reduction_kg is None  # no emission factor configured
        assert r.calculation_status.value == "calculated_without_co2e"
        rec = r.recommendation
        assert rec is not None and rec.anomaly_type.value == "above_baseline"
        assert r.recommended_action == rec.recommended_actions[0]
        assert any("lighting" in a.lower() or "hvac" in a.lower() for a in rec.recommended_actions)


def test_water_wastage_is_flagged_and_does_not_claim_a_leak(load_sample):
    results = analyze_consumption(load_sample("water_wastage"))
    hostel = _flagged(_by(results, "Hostel B", "water"))
    assert len(hostel) == 6  # the six elevated days
    assert _flagged(_by(results, "Science Block", "water")) == []
    assert _flagged(_by(results, "Canteen", "water")) == []
    for r in hostel:
        assert r.unit == "liters" and r.estimated_savings.unit == "liters"
        assert r.timestamp.utcoffset().total_seconds() == 5.5 * 3600  # offset preserved
        rec = r.recommendation
        assert "A reading above baseline cannot confirm a leak." in rec.caveats
        assert "leak" in " ".join(rec.recommended_actions).lower()


def test_waste_anomalies_are_flagged(load_sample):
    results = analyze_consumption(load_sample("waste_anomalies"))
    assert len(_flagged(_by(results, "Canteen", "waste"))) == 3
    assert len(_flagged(_by(results, "Hostel A", "waste"))) == 1
    assert _flagged(_by(results, "Admin Block", "waste")) == []
    for r in _flagged(results):
        assert r.unit == "kg" and r.estimated_savings.unit == "kg"
        assert r.recommendation.resource_type.value == "waste"


def test_insufficient_history_gives_no_verdict(load_sample):
    records = load_sample("insufficient_history")
    results = analyze_consumption(records)
    assert len(results) == 14
    for r in results:
        assert r.data_quality_status.value == "insufficient_data"
        assert r.severity is Severity.NONE and r.anomaly_score == 0
        assert r.baseline_value is None and r.deviation_percentage is None
        assert r.estimated_savings is None and r.recommendation is None
        assert r.calculation_status.value == "insufficient_baseline"
    # Even the 3x spike on the 5th Innovation Hub reading is not judged.
    assert max(_by(results, "Innovation Hub", "electricity"), key=lambda r: r.observed_value).severity is Severity.NONE


def test_history_boundary_is_exactly_minimum_history(load_sample):
    records = load_sample("normal_consumption")
    one = [r for r in records if r["building"] == "Hostel C" and r["resource_type"] == "water"][:8]
    results = analyze_consumption(one)
    assert [r.data_quality_status.value for r in results] == ["insufficient_data"] * 7 + ["ok"]


def test_edge_cases_outage_low_use_and_normalization(load_sample):
    results = analyze_consumption(load_sample("edge_cases"))
    outage = _by(results, "Auditorium", "water")[-1]
    assert outage.observed_value == 0
    assert outage.data_quality_status.value == "suspect"
    assert outage.estimated_savings is None
    assert outage.recommendation.anomaly_type.value == "suspect_reading"
    assert outage.recommendation.priority.value in {"low", "medium"}

    low = _by(results, "Gymnasium", "electricity")[-1]
    assert low.deviation_percentage < -50
    assert low.estimated_savings is None and low.calculation_status.value == "not_applicable"
    assert low.recommendation.anomaly_type.value == "below_baseline"

    guest = _by(results, "Guest House", "electricity")  # input used "Electricity" / "kwh"
    assert len(guest) == 21 and all(r.unit == "kWh" for r in guest)
    assert _flagged(guest) == []


def test_mixed_file_matches_the_separate_scenario_runs(load_sample):
    """Order and neighbouring series must not change any result."""
    mixed_records = load_sample("mixed_campus")
    assert len(mixed_records) == sum(
        len(load_sample(n)) for n in SCENARIOS
    )
    mixed = analyze_consumption(mixed_records)

    def key(r):
        return (r.building, r.resource_type.value, r.timestamp)

    separate = {}
    for name in SCENARIOS:
        for r in analyze_consumption(load_sample(name)):
            assert key(r) not in separate  # scenarios use disjoint series
            separate[key(r)] = r.model_dump()
    assert len(separate) == len(mixed)
    for r in mixed:
        assert r.model_dump() == separate[key(r)]

    flagged_series = {(r.building, r.resource_type.value) for r in _flagged(mixed)}
    assert flagged_series == {
        ("Library", "electricity"), ("Admin Block", "electricity"),
        ("Hostel B", "water"), ("Canteen", "waste"), ("Hostel A", "waste"),
        ("Gymnasium", "electricity"), ("Auditorium", "water"),
    }
    assert len(_flagged(mixed)) == 16


def test_results_are_deterministic(load_sample):
    records = load_sample("mixed_campus")
    first = serialize_results(analyze_consumption(records))
    second = serialize_results(analyze_consumption(records))
    assert first == second


def test_payload_pipeline_on_samples(load_sample):
    outcome = analyze_payload(
        {"records": load_sample("mixed_campus"), "options": {"include_impact_summary": True}}
    )
    assert outcome.ok
    assert outcome.body["summary"]["anomalies_flagged"] == 16
    impact = outcome.body["impact_summary"]
    assert {t["resource_type"] for t in impact["totals"]} == {"electricity", "water", "waste"}
    assert impact["total_co2e_reduction_kg"] is None  # no factors configured -> no CO2e
    assert impact["co2e_complete"] is False
    json.dumps(outcome.body, allow_nan=False)


# --------------------------------------------------------------------------- #
# No API key / no LLM: deterministic recommendations
# --------------------------------------------------------------------------- #
def test_default_config_never_builds_an_llm_client(monkeypatch, load_sample):
    def forbidden(*args, **kwargs):
        raise AssertionError("an LLM client must not be created when the LLM is disabled")

    monkeypatch.setattr(rec_mod, "create_llm_client", forbidden)
    results = analyze_consumption(load_sample("mixed_campus"))
    flagged = _flagged(results)
    assert flagged and all(r.recommendation is not None for r in flagged)
    for r in flagged:
        rec = r.recommendation
        assert rec.source is RecommendationSource.RULES
        assert rec.llm_status is LLMStatus.DISABLED
        assert rec.llm_explanation is None and rec.llm_suggestions == [] and rec.llm_impact_summary is None
        assert rec.recommended_actions and rec.supporting_evidence and rec.explanation


def test_llm_enabled_but_key_missing_falls_back_to_identical_rules(monkeypatch, load_sample):
    monkeypatch.setattr(rec_mod, "create_llm_client", llm_client_mod.create_llm_client)
    records = load_sample("electricity_unusual_usage")
    baseline = analyze_consumption(records)  # LLM disabled
    for provider in ("gemini", "openai"):
        cfg = AnalysisConfig(llm=LLMConfig(enabled=True, provider=provider, model="any-model-name"))
        results = analyze_consumption(records, cfg)  # key variable is unset
        assert len(results) == len(baseline)
        for with_llm, without in zip(results, baseline):
            if without.recommendation is None:
                assert with_llm.recommendation is None
                continue
            rec = with_llm.recommendation
            assert rec.llm_status is LLMStatus.MISSING_API_KEY
            assert rec.source is RecommendationSource.RULES
            assert rec.llm_explanation is None and rec.llm_suggestions == []
            assert rec.model_dump(exclude={"llm_status"}) == without.recommendation.model_dump(exclude={"llm_status"})
            assert with_llm.recommended_action == without.recommended_action
            assert with_llm.estimated_savings == without.estimated_savings


def test_llm_timeout_falls_back_and_stops_after_one_call(monkeypatch, load_sample):
    class SlowClient:
        calls = 0

        def generate_json(self, system_prompt, user_prompt):
            SlowClient.calls += 1
            raise LLMTimeoutError("ReadTimeout")

    monkeypatch.setattr(rec_mod, "create_llm_client", lambda cfg, environ=None: SlowClient())
    records = load_sample("electricity_unusual_usage")
    plain = analyze_consumption(records)
    cfg = AnalysisConfig(llm=LLMConfig(enabled=True, model="any-model-name", timeout_seconds=2))
    results = analyze_consumption(records, cfg)
    flagged = _flagged(results)
    assert len(flagged) == 4 and SlowClient.calls == 1  # an outage costs one timeout, not four
    assert all(r.recommendation.llm_status is LLMStatus.TIMEOUT for r in flagged)
    for a, b in zip(results, plain):  # numbers and rules are untouched by the failure
        assert a.estimated_savings == b.estimated_savings
        assert a.severity == b.severity and a.anomaly_score == b.anomaly_score
        if b.recommendation is not None:
            assert a.recommendation.recommended_actions == b.recommendation.recommended_actions


def test_llm_provider_error_message_is_never_exposed(monkeypatch, load_sample):
    class LeakyClient:
        def generate_json(self, system_prompt, user_prompt):
            raise RuntimeError("401 invalid key sk-THIS-MUST-NOT-LEAK")

    monkeypatch.setattr(rec_mod, "create_llm_client", lambda cfg, environ=None: LeakyClient())
    cfg = AnalysisConfig(llm=LLMConfig(enabled=True, model="any-model-name"))
    outcome = analyze_payload({"records": load_sample("waste_anomalies")}, cfg)
    assert outcome.ok
    text = json.dumps(outcome.body)
    assert "THIS-MUST-NOT-LEAK" not in text
    assert outcome.body["summary"]["llm_status_counts"]["provider_error"] == 4
