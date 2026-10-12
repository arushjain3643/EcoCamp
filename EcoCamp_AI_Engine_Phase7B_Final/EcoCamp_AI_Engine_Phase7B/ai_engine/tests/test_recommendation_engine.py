"""Tests for Phase 4: deterministic recommendations and the optional LLM layer.

No test performs a real network call. LLM providers are replaced by fakes, and
by default the client factory is patched to fail loudly if anything tries to
build a real client.
"""
import inspect
import json
import logging
import re
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import ai_engine.llm_client as llm_client_mod
import ai_engine.recommendation_engine as rec_mod
from ai_engine import (
    AnalysisConfig,
    AnalysisResult,
    ConsumptionRecord,
    LLMConfig,
    LLMProvider,
    LLMStatus,
    Priority,
    Recommendation,
    RecommendationSource,
    Severity,
    analyze_consumption,
    generate_recommendations,
)
from ai_engine.anomaly_detector import detect_anomalies
from ai_engine.impact_calculator import apply_impact
from ai_engine.llm_client import (
    GeminiClient,
    LLMProviderError,
    LLMTimeoutError,
    LLMUnavailableError,
    OpenAIClient,
    create_llm_client,
)
from ai_engine.schemas import AnomalyType

BASE = datetime(2026, 1, 1)
NOISY = [100, 102, 98, 101, 99, 103, 97, 100, 102, 98]  # median 100, MAD 2
UNITS = {"electricity": "kWh", "water": "liters", "waste": "kg"}
PKG = Path(__file__).resolve().parents[1]


def rec(value, day=0, building="Library", resource="electricity"):
    return ConsumptionRecord(
        timestamp=BASE + timedelta(days=day), building=building,
        resource_type=resource, consumption_value=value, unit=UNITS[resource],
    )


def series(values, **kw):
    return [rec(v, day=i, **kw) for i, v in enumerate(values)]


def last(values, config=None, **kw):
    return analyze_consumption(series(values, **kw), config)[-1]


def llm_config(**kw):
    return AnalysisConfig(
        llm=LLMConfig(enabled=True, provider="openai", model="test-model", **kw)
    )


# --------------------------------------------------------------------------- #
# Fakes and fixtures
# --------------------------------------------------------------------------- #
class FakeClient:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def generate_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.exc is not None:
            raise self.exc
        return self.reply


def use_client(monkeypatch, client):
    monkeypatch.setattr(rec_mod, "create_llm_client", lambda cfg, environ=None: client)


GOOD_REPLY = json.dumps({
    "explanation": (
        "Library electricity use was 250.0 kWh, which is 150% above its usual "
        "100.0 kWh. The cause is not known yet."
    ),
    "recommended_actions": [
        "Walk through the building after hours and note what is left running.",
        "Ask facilities whether any events were held.",
    ],
    "impact_summary": "If part of the excess is avoidable, about 75.0 kWh could be saved.",
})


@pytest.fixture(autouse=True)
def isolate_llm(monkeypatch):
    """Offline by default: no env keys, and building a real client is an error."""
    for name in (
        "OPENAI_API_KEY", "GEMINI_API_KEY", "ECOCAMP_LLM_ENABLED", "ECOCAMP_LLM_PROVIDER",
        "ECOCAMP_LLM_MODEL", "ECOCAMP_LLM_API_KEY_ENV", "ECOCAMP_LLM_TIMEOUT_SECONDS",
        "ECOCAMP_LLM_MAX_RECOMMENDATIONS",
    ):
        monkeypatch.delenv(name, raising=False)

    def forbidden(cfg, environ=None):
        raise AssertionError("a real LLM client must not be created in tests")

    monkeypatch.setattr(rec_mod, "create_llm_client", forbidden)


# --------------------------------------------------------------------------- #
# Deterministic recommendations
# --------------------------------------------------------------------------- #
def test_electricity_recommendation():
    r = last(NOISY + [250])
    rc = r.recommendation
    assert isinstance(rc, Recommendation)
    assert (rc.resource_type.value, rc.building) == ("electricity", "Library")
    assert rc.severity is Severity.HIGH and rc.priority is Priority.HIGH
    assert rc.anomaly_type is AnomalyType.ABOVE_BASELINE
    evidence = " ".join(rc.supporting_evidence)
    assert "250.00 kWh" in evidence and "100.00 kWh" in evidence and "150.0% above" in evidence
    actions = " ".join(rc.recommended_actions).lower()
    assert "lighting" in actions and "hvac" in actions and "idle equipment" in actions
    assert "150.0% above" in rc.explanation
    assert "not been identified" in rc.explanation
    assert r.recommended_action == rc.recommended_actions[0]
    assert rc.source is RecommendationSource.RULES
    assert rc.llm_status is LLMStatus.DISABLED


def test_water_recommendation_does_not_claim_a_leak():
    r = last([v * 10 for v in NOISY] + [2500], resource="water")
    rc = r.recommendation
    assert rc.resource_type.value == "water"
    actions = " ".join(rc.recommended_actions).lower()
    assert "leak" in actions and "taps" in actions and "schedules" in actions
    blob = (rc.explanation + " " + " ".join(rc.recommended_actions + rc.caveats)).lower()
    assert "cannot confirm a leak" in blob
    for claim in ("leak has been", "leak is confirmed", "confirmed leak", "is leaking", "found a leak"):
        assert claim not in blob


def test_waste_recommendation():
    r = last(NOISY + [250], resource="waste")
    rc = r.recommendation
    assert rc.resource_type.value == "waste"
    actions = " ".join(rc.recommended_actions).lower()
    assert "segregation" in actions and "audit" in actions and "reduction" in actions


@pytest.mark.parametrize(
    "spike,severity,priority",
    [
        (160, Severity.LOW, Priority.LOW),
        (220, Severity.MEDIUM, Priority.MEDIUM),
        (250, Severity.HIGH, Priority.HIGH),
        (1000, Severity.CRITICAL, Priority.URGENT),
    ],
)
def test_priority_follows_severity(spike, severity, priority):
    rc = last(NOISY + [spike]).recommendation
    assert rc.severity is severity
    assert rc.priority is priority


def test_normal_reading_has_no_recommendation():
    r = last(NOISY + [101])
    assert r.severity is Severity.NONE
    assert r.recommendation is None
    assert r.recommended_action is None


def test_below_baseline_recommends_verification_not_savings():
    r = last(NOISY + [20])
    rc = r.recommendation
    assert rc.anomaly_type is AnomalyType.BELOW_BASELINE
    assert rc.estimated_savings is None
    assert "meter" in " ".join(rc.recommended_actions).lower()
    assert "below" in rc.explanation
    assert _rank(rc.priority) <= _rank(Priority.MEDIUM)


def test_zero_reading_is_treated_as_possible_data_problem():
    rc = last(NOISY + [0]).recommendation
    assert rc.anomaly_type is AnomalyType.SUSPECT_READING
    text = (rc.explanation + " ".join(rc.recommended_actions)).lower()
    assert "meter" in text and "missing data" in text
    assert rc.estimated_savings is None
    assert _rank(rc.priority) <= _rank(Priority.MEDIUM)


def _rank(p):
    return [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.URGENT].index(p)


def test_insufficient_history_has_no_recommendation():
    r = last([100, 100, 100, 500])
    assert r.baseline_value is None
    assert r.recommendation is None and r.recommended_action is None


def test_zero_baseline_cases_have_no_recommendation():
    assert last([0] * 10 + [50]).recommendation is None
    assert last([0] * 10 + [0]).recommendation is None


def test_multiple_buildings_and_resources_get_matching_recommendations():
    recs = (
        series(NOISY + [250], building="Library")
        + series([v * 10 for v in NOISY] + [2500], building="Hostel", resource="water")
        + series(NOISY + [250], building="Canteen", resource="waste")
        + series(NOISY + [101], building="Lab")
    )
    out = analyze_consumption(recs)
    flagged = {(r.building, r.resource_type.value): r.recommendation for r in out if r.recommendation}
    assert set(flagged) == {
        ("Library", "electricity"), ("Hostel", "water"), ("Canteen", "waste"),
    }
    for (building, resource), rc in flagged.items():
        assert (rc.building, rc.resource_type.value) == (building, resource)
    assert all(r.recommendation is None for r in out if r.building == "Lab")


# --------------------------------------------------------------------------- #
# No fabricated savings
# --------------------------------------------------------------------------- #
def test_recommendation_savings_are_copied_from_impact_calculation():
    r = last(NOISY + [250])
    assert r.estimated_savings.value == pytest.approx(75.0)
    assert r.recommendation.estimated_savings == r.estimated_savings
    assert r.recommendation.estimated_co2e_reduction_kg == r.estimated_co2e_reduction_kg
    assert "75.00 kWh" in " ".join(r.recommendation.supporting_evidence)


def test_no_savings_or_co2e_text_when_not_calculated():
    for values in (NOISY + [20], NOISY + [0]):
        rc = last(values).recommendation
        assert rc.estimated_savings is None and rc.estimated_co2e_reduction_kg is None
        assert "avoidable" not in " ".join(rc.supporting_evidence)
    # Savings exist but no emission factor: no CO2e wording anywhere.
    rc = last(NOISY + [250]).recommendation
    assert rc.estimated_co2e_reduction_kg is None
    assert "co2e" not in (" ".join(rc.supporting_evidence) + rc.explanation).lower()


def test_co2e_appears_only_when_a_factor_is_configured():
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.4)  # test value
    rc = last(NOISY + [250], cfg).recommendation
    assert rc.estimated_co2e_reduction_kg == pytest.approx(30.0)
    assert "30.00 kg" in " ".join(rc.supporting_evidence)


def test_recommendation_never_changes_detection_or_impact_fields():
    recs = (
        series(NOISY + [250, 20, 0]) + series([v * 10 for v in NOISY] + [2500], resource="water")
    )
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.4)
    expected = [apply_impact(r, cfg) for r in detect_anomalies(recs, cfg)]
    actual = analyze_consumption(recs, cfg)
    skip = {"recommendation", "recommended_action"}
    for e, a in zip(expected, actual):
        assert e.model_dump(exclude=skip) == a.model_dump(exclude=skip)


# --------------------------------------------------------------------------- #
# Interface compatibility
# --------------------------------------------------------------------------- #
def test_public_signature_and_order_preserved():
    assert list(inspect.signature(analyze_consumption).parameters) == ["records", "config"]
    recs = series(NOISY + [250]) + series(NOISY + [101], building="Lab")
    out = analyze_consumption(recs)
    assert [(r.building, r.timestamp) for r in out] == [(r.building, r.timestamp) for r in recs]
    assert analyze_consumption(recs) == analyze_consumption(recs, None) == analyze_consumption(recs, AnalysisConfig())


def test_results_with_recommendations_round_trip_json():
    r = last(NOISY + [250], AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.4))
    assert AnalysisResult.model_validate_json(r.model_dump_json()) == r
    d = r.model_dump(mode="json")["recommendation"]
    for key in (
        "resource_type", "building", "severity", "priority", "supporting_evidence",
        "recommended_actions", "estimated_savings", "explanation", "llm_status", "source",
    ):
        assert key in d


def test_generate_recommendations_does_not_mutate_inputs():
    detected = [apply_impact(r, AnalysisConfig()) for r in detect_anomalies(series(NOISY + [250]))]
    before = [r.model_dump() for r in detected]
    out = generate_recommendations(detected)
    assert [r.model_dump() for r in detected] == before
    assert out[-1].recommendation is not None and detected[-1].recommendation is None


# --------------------------------------------------------------------------- #
# LLM disabled by default
# --------------------------------------------------------------------------- #
def test_llm_disabled_by_default_never_builds_or_calls_a_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "irrelevant-test-value")
    monkeypatch.setenv("GEMINI_API_KEY", "irrelevant-test-value")
    assert AnalysisConfig().llm.enabled is False
    r = last(NOISY + [250])  # autouse fixture raises if a client is created
    assert r.recommendation.llm_status is LLMStatus.DISABLED
    assert r.recommendation.llm_explanation is None and r.recommendation.llm_suggestions == []


def test_injected_client_is_ignored_when_llm_disabled():
    client = FakeClient(reply=GOOD_REPLY)
    detected = [apply_impact(r, AnalysisConfig()) for r in detect_anomalies(series(NOISY + [250]))]
    out = generate_recommendations(detected, AnalysisConfig(), llm_client=client)
    assert client.calls == []
    assert out[-1].recommendation.llm_status is LLMStatus.DISABLED


# --------------------------------------------------------------------------- #
# LLM success path
# --------------------------------------------------------------------------- #
def test_valid_llm_reply_is_attached_without_changing_deterministic_fields(monkeypatch):
    baseline = last(NOISY + [250]).recommendation
    client = FakeClient(reply=GOOD_REPLY)
    use_client(monkeypatch, client)
    r = last(NOISY + [250], llm_config())
    rc = r.recommendation
    assert len(client.calls) == 1
    assert rc.llm_status is LLMStatus.USED and rc.source is RecommendationSource.RULES_LLM
    assert "cause is not known yet" in rc.llm_explanation
    assert len(rc.llm_suggestions) == 2
    assert "75.0 kWh" in rc.llm_impact_summary
    for field in (
        "supporting_evidence", "recommended_actions", "estimated_savings",
        "explanation", "priority", "severity", "caveats",
    ):
        assert getattr(rc, field) == getattr(baseline, field)
    assert r.estimated_savings.value == pytest.approx(75.0)


def test_llm_prompt_contains_only_structured_facts(monkeypatch):
    client = FakeClient(reply=GOOD_REPLY)
    use_client(monkeypatch, client)
    last(NOISY + [250], llm_config())
    system_prompt, user_prompt = client.calls[0]
    assert "never say a cause is confirmed" in system_prompt.lower()
    facts = json.loads(user_prompt.split("FACTS:\n", 1)[1])
    assert facts["observed_value"] == 250.0 and facts["baseline_value"] == 100.0
    assert facts["estimated_savings"] == {"value": 75.0, "unit": "kWh"}
    assert "2026" not in user_prompt  # no timestamps are sent


def test_fenced_json_reply_is_accepted(monkeypatch):
    use_client(monkeypatch, FakeClient(reply="```json\n" + GOOD_REPLY + "\n```"))
    assert last(NOISY + [250], llm_config()).recommendation.llm_status is LLMStatus.USED


def test_hedged_and_negated_wording_is_accepted(monkeypatch):
    reply = json.dumps({
        "explanation": "A leak has not been confirmed; it is worth checking.",
        "recommended_actions": ["Check for dripping taps."],
        "impact_summary": None,
    })
    use_client(monkeypatch, FakeClient(reply=reply))
    r = last([v * 10 for v in NOISY] + [2500], llm_config(), resource="water")
    assert r.recommendation.llm_status is LLMStatus.USED


def test_co2e_wording_allowed_only_with_a_co2e_estimate(monkeypatch):
    reply = json.dumps({
        "explanation": "Use was above normal.",
        "recommended_actions": ["Check lighting schedules."],
        "impact_summary": "This could avoid roughly 30.0 kg of CO2e.",
    })
    use_client(monkeypatch, FakeClient(reply=reply))
    with_factor = AnalysisConfig(
        electricity_emission_factor_kg_co2e_per_kwh=0.4,
        llm=LLMConfig(enabled=True, provider="openai", model="m"),
    )
    assert last(NOISY + [250], with_factor).recommendation.llm_status is LLMStatus.USED
    without = last(NOISY + [250], llm_config())
    assert without.recommendation.llm_status is LLMStatus.INVALID_RESPONSE


# --------------------------------------------------------------------------- #
# Invalid LLM replies -> deterministic fallback
# --------------------------------------------------------------------------- #
def _reply(**over):
    data = {
        "explanation": "Use was above normal.",
        "recommended_actions": ["Check lighting schedules."],
        "impact_summary": None,
    }
    data.update(over)
    return json.dumps(data)


INVALID_REPLIES = {
    "not json": "Sure! Here is my answer.",
    "empty": "",
    "json list": "[1, 2, 3]",
    "missing keys": json.dumps({"explanation": "x"}),
    "extra key": _reply(savings_kwh=5),
    "no actions": _reply(recommended_actions=[]),
    "blank action": _reply(recommended_actions=["  "]),
    "non-string action": _reply(recommended_actions=[7]),
    "too long": _reply(explanation="a" * 601),
    "invented number": _reply(explanation="You will save 500 kWh by acting."),
    "invented date": _reply(explanation="On 2026-01-11 use spiked."),
    "invented number in action": _reply(recommended_actions=["Switch off 40 lights."]),
    "invented number in summary": _reply(impact_summary="Savings of 999 kWh are likely."),
    "confirmed cause": _reply(explanation="The overnight heating is confirmed as the cause."),
    "certainty": _reply(explanation="This is definitely wasted energy."),
    "leak claim": _reply(explanation="Staff found a leak in the pipes."),
    "carbon without estimate": _reply(impact_summary="This lowers carbon emissions."),
}


@pytest.mark.parametrize("name", sorted(INVALID_REPLIES))
def test_invalid_llm_replies_fall_back_to_deterministic(monkeypatch, name):
    baseline = last(NOISY + [250]).recommendation
    use_client(monkeypatch, FakeClient(reply=INVALID_REPLIES[name]))
    r = last(NOISY + [250], llm_config())
    rc = r.recommendation
    assert rc.llm_status is LLMStatus.INVALID_RESPONSE, name
    assert rc.source is RecommendationSource.RULES
    assert rc.llm_explanation is None and rc.llm_suggestions == [] and rc.llm_impact_summary is None
    assert rc.recommended_actions == baseline.recommended_actions
    assert rc.estimated_savings == baseline.estimated_savings
    assert r.estimated_savings.value == pytest.approx(75.0)


def test_non_text_reply_is_invalid(monkeypatch):
    use_client(monkeypatch, FakeClient(reply=None))
    assert last(NOISY + [250], llm_config()).recommendation.llm_status is LLMStatus.INVALID_RESPONSE


def test_llm_cannot_invent_savings_where_none_were_calculated(monkeypatch):
    reply = _reply(impact_summary="Fixing this could save 30 kWh.")  # 30 is not in the facts
    use_client(monkeypatch, FakeClient(reply=reply))
    r = last(NOISY + [20], llm_config())  # below baseline: no savings
    assert r.estimated_savings is None
    assert r.recommendation.estimated_savings is None
    assert r.recommendation.llm_status is LLMStatus.INVALID_RESPONSE
    assert r.recommendation.llm_impact_summary is None


# --------------------------------------------------------------------------- #
# API key, timeout and provider failures
# --------------------------------------------------------------------------- #
def test_missing_api_key_falls_back_without_error(monkeypatch):
    monkeypatch.setattr(rec_mod, "create_llm_client", llm_client_mod.create_llm_client)
    baseline = last(NOISY + [250]).recommendation
    r = last(NOISY + [250], llm_config())
    rc = r.recommendation
    assert rc.llm_status is LLMStatus.MISSING_API_KEY
    assert rc.source is RecommendationSource.RULES
    assert rc.recommended_actions == baseline.recommended_actions
    assert r.recommended_action == baseline.recommended_actions[0]


def test_missing_key_for_custom_env_var_name(monkeypatch):
    monkeypatch.setattr(rec_mod, "create_llm_client", llm_client_mod.create_llm_client)
    monkeypatch.setenv("OPENAI_API_KEY", "set-but-wrong-variable")  # not the configured one
    cfg = llm_config(api_key_env_var="MY_CAMPUS_KEY")
    assert last(NOISY + [250], cfg).recommendation.llm_status is LLMStatus.MISSING_API_KEY


def test_sdk_not_installed_falls_back(monkeypatch):
    monkeypatch.setattr(rec_mod, "create_llm_client", llm_client_mod.create_llm_client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setitem(sys.modules, "openai", None)  # makes `import openai` fail
    r = last(NOISY + [250], llm_config())
    assert r.recommendation.llm_status is LLMStatus.SDK_UNAVAILABLE
    assert r.recommendation.recommended_actions


@pytest.mark.parametrize("exc", [LLMTimeoutError("APITimeoutError"), TimeoutError("slow")])
def test_api_timeout_falls_back(monkeypatch, exc):
    baseline = last(NOISY + [250]).recommendation
    use_client(monkeypatch, FakeClient(exc=exc))
    rc = last(NOISY + [250], llm_config()).recommendation
    assert rc.llm_status is LLMStatus.TIMEOUT
    assert rc.recommended_actions == baseline.recommended_actions


@pytest.mark.parametrize(
    "exc", [LLMProviderError("RateLimitError"), RuntimeError("boom"), ConnectionError("down")]
)
def test_provider_failure_falls_back(monkeypatch, exc):
    baseline = last(NOISY + [250]).recommendation
    use_client(monkeypatch, FakeClient(exc=exc))
    rc = last(NOISY + [250], llm_config()).recommendation
    assert rc.llm_status is LLMStatus.PROVIDER_ERROR
    assert rc.recommended_actions == baseline.recommended_actions


def test_provider_error_messages_are_never_exposed(monkeypatch, caplog):
    secret = "sk-test-secret-should-never-leak"
    use_client(monkeypatch, FakeClient(exc=RuntimeError(f"Incorrect API key provided: {secret}")))
    with caplog.at_level(logging.DEBUG):
        r = last(NOISY + [250], llm_config())
    assert secret not in r.model_dump_json()
    assert secret not in caplog.text


def test_client_setup_failure_falls_back(monkeypatch):
    def broken(cfg, environ=None):
        raise RuntimeError("cannot build client")

    monkeypatch.setattr(rec_mod, "create_llm_client", broken)
    rc = last(NOISY + [250], llm_config()).recommendation
    assert rc.llm_status is LLMStatus.PROVIDER_ERROR
    assert rc.recommended_actions


def test_outage_stops_further_calls_in_the_batch(monkeypatch):
    client = FakeClient(exc=LLMTimeoutError("APITimeoutError"))
    use_client(monkeypatch, client)
    recs = series(NOISY + [250], building="A") + series(NOISY + [250], building="B")
    out = analyze_consumption(recs, llm_config())
    assert len(client.calls) == 1
    flagged = [r.recommendation for r in out if r.recommendation]
    assert len(flagged) == 2
    assert all(rc.llm_status is LLMStatus.TIMEOUT for rc in flagged)


def test_invalid_reply_does_not_stop_later_calls(monkeypatch):
    client = FakeClient(reply="not json")
    use_client(monkeypatch, client)
    recs = series(NOISY + [250], building="A") + series(NOISY + [250], building="B")
    analyze_consumption(recs, llm_config())
    assert len(client.calls) == 2


def test_call_limit_prefers_highest_priority(monkeypatch):
    client = FakeClient(reply=_reply(explanation="Use was above normal."))
    use_client(monkeypatch, client)
    recs = series(NOISY + [180], building="A") + series(NOISY + [1000], building="B")
    out = analyze_consumption(recs, llm_config(max_llm_recommendations=1))
    a = next(r for r in out if r.building == "A" and r.recommendation).recommendation
    b = next(r for r in out if r.building == "B" and r.recommendation).recommendation
    assert len(client.calls) == 1
    assert b.priority is Priority.URGENT and b.llm_status is LLMStatus.USED
    assert a.llm_status is LLMStatus.CALL_LIMIT_REACHED and a.recommended_actions


def test_fallback_output_equals_llm_disabled_output(monkeypatch):
    disabled = analyze_consumption(series(NOISY + [250]))[-1].recommendation
    use_client(monkeypatch, FakeClient(exc=RuntimeError("x")))
    failed = analyze_consumption(series(NOISY + [250]), llm_config())[-1].recommendation
    for field in (
        "supporting_evidence", "recommended_actions", "estimated_savings",
        "explanation", "caveats", "priority", "severity", "anomaly_type", "source",
    ):
        assert getattr(failed, field) == getattr(disabled, field)


def test_normal_and_insufficient_records_never_trigger_llm_calls(monkeypatch):
    client = FakeClient(reply=GOOD_REPLY)
    use_client(monkeypatch, client)
    out = analyze_consumption(series(NOISY + [101]) + series([1, 2, 3], building="New"), llm_config())
    assert client.calls == []
    assert all(r.recommendation is None for r in out)


# --------------------------------------------------------------------------- #
# LLMConfig
# --------------------------------------------------------------------------- #
def test_llm_config_defaults_are_disabled():
    c = LLMConfig()
    assert c.enabled is False and c.model is None
    assert c.provider is LLMProvider.GEMINI
    assert c.resolved_api_key_env_var == "GEMINI_API_KEY"
    assert LLMConfig(provider="openai").resolved_api_key_env_var == "OPENAI_API_KEY"
    assert LLMConfig(api_key_env_var="MY_KEY").resolved_api_key_env_var == "MY_KEY"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": True},                                   # model required
        {"enabled": True, "model": "   "},
        {"provider": "claude"},                              # unsupported
        {"api_key_env_var": "sk-abc123-not-a-name"},         # looks like a key
        {"api_key_env_var": "has space"},
        {"api_key_env_var": ""},
        {"timeout_seconds": 0},
        {"timeout_seconds": -1},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": True},
        {"timeout_seconds": 1000},
        {"max_llm_recommendations": 0},
        {"max_llm_recommendations": 10_000},
        {"unknown_field": 1},
    ],
)
def test_invalid_llm_config_rejected(kwargs):
    with pytest.raises(ValidationError):
        LLMConfig(**kwargs)


def test_llm_config_from_env():
    env = {
        "ECOCAMP_LLM_ENABLED": "true", "ECOCAMP_LLM_PROVIDER": "OpenAI",
        "ECOCAMP_LLM_MODEL": "my-model", "ECOCAMP_LLM_API_KEY_ENV": "CAMPUS_KEY",
        "ECOCAMP_LLM_TIMEOUT_SECONDS": "7.5", "ECOCAMP_LLM_MAX_RECOMMENDATIONS": "3",
    }
    c = LLMConfig.from_env(env)
    assert (c.enabled, c.provider, c.model) == (True, LLMProvider.OPENAI, "my-model")
    assert (c.api_key_env_var, c.timeout_seconds, c.max_llm_recommendations) == ("CAMPUS_KEY", 7.5, 3)
    assert LLMConfig.from_env({}).enabled is False
    with pytest.raises(ValueError):
        LLMConfig.from_env({"ECOCAMP_LLM_ENABLED": "maybe"})
    with pytest.raises(ValidationError):
        LLMConfig.from_env({"ECOCAMP_LLM_ENABLED": "true"})  # no model


# --------------------------------------------------------------------------- #
# Provider adapters (official SDKs replaced by fakes)
# --------------------------------------------------------------------------- #
def test_create_llm_client_reads_key_from_environment_only():
    cfg = LLMConfig(enabled=True, provider="openai", model="m")
    assert isinstance(create_llm_client(cfg, {"OPENAI_API_KEY": "k"}), OpenAIClient)
    gem = LLMConfig(enabled=True, provider="gemini", model="m")
    assert isinstance(create_llm_client(gem, {"GEMINI_API_KEY": "k"}), GeminiClient)
    custom = LLMConfig(enabled=True, provider="gemini", model="m", api_key_env_var="X_KEY")
    assert isinstance(create_llm_client(custom, {"X_KEY": "k"}), GeminiClient)
    for env in ({}, {"OPENAI_API_KEY": ""}, {"OPENAI_API_KEY": "   "}):
        with pytest.raises(LLMUnavailableError) as info:
            create_llm_client(cfg, env)
        assert info.value.code == "missing_api_key"


def _fake_openai(monkeypatch, create):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            captured["create"] = kwargs
            return create(kwargs)

    mod = types.ModuleType("openai")
    mod.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    return captured


def test_openai_adapter_call_shape(monkeypatch):
    captured = _fake_openai(
        monkeypatch,
        lambda kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": 1}'))]),
    )
    out = OpenAIClient("some-model", "key-from-env", 12.5).generate_json("SYS", "USER")
    assert out == '{"ok": 1}'
    assert captured["init"]["api_key"] == "key-from-env" and captured["init"]["timeout"] == 12.5
    assert captured["create"]["model"] == "some-model"
    assert captured["create"]["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in captured["create"]["messages"]] == ["system", "user"]


def test_openai_adapter_maps_errors_and_drops_messages(monkeypatch):
    class APITimeoutError(Exception):
        pass

    class AuthenticationError(Exception):
        pass

    def timeout(_):
        raise APITimeoutError("Request timed out")

    def auth(_):
        raise AuthenticationError("Incorrect API key provided: sk-leaky")

    _fake_openai(monkeypatch, timeout)
    with pytest.raises(LLMTimeoutError):
        OpenAIClient("m", "k", 5).generate_json("s", "u")
    _fake_openai(monkeypatch, auth)
    with pytest.raises(LLMProviderError) as info:
        OpenAIClient("m", "k", 5).generate_json("s", "u")
    assert str(info.value) == "AuthenticationError" and "leaky" not in str(info.value)
    _fake_openai(monkeypatch, lambda kw: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]))
    with pytest.raises(LLMProviderError):
        OpenAIClient("m", "k", 5).generate_json("s", "u")


def test_adapters_report_missing_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(LLMUnavailableError) as info:
        OpenAIClient("m", "k", 5).generate_json("s", "u")
    assert info.value.code == "sdk_unavailable"
    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.genai", None)
    with pytest.raises(LLMUnavailableError) as info:
        GeminiClient("m", "k", 5).generate_json("s", "u")
    assert info.value.code == "sdk_unavailable"


def _fake_gemini(monkeypatch, generate):
    captured = {}

    class Options:
        def __init__(self, **kw):
            self.kw = kw

    class Client:
        def __init__(self, **kw):
            captured["init"] = kw
            self.models = SimpleNamespace(generate_content=self._gen)

        def _gen(self, **kw):
            captured["gen"] = kw
            return generate(kw)

    gtypes = types.ModuleType("google.genai.types")
    gtypes.HttpOptions = Options
    gtypes.GenerateContentConfig = Options
    genai = types.ModuleType("google.genai")
    genai.Client, genai.types = Client, gtypes
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    return captured


def test_gemini_adapter_call_shape(monkeypatch):
    captured = _fake_gemini(monkeypatch, lambda kw: SimpleNamespace(text='{"ok": 2}'))
    out = GeminiClient("gem-model", "key-from-env", 12.5).generate_json("SYS", "USER")
    assert out == '{"ok": 2}'
    assert captured["init"]["api_key"] == "key-from-env"
    assert captured["init"]["http_options"].kw["timeout"] == 12500
    assert captured["gen"]["model"] == "gem-model" and captured["gen"]["contents"] == "USER"
    cfg = captured["gen"]["config"].kw
    assert cfg["system_instruction"] == "SYS" and cfg["response_mime_type"] == "application/json"


def test_gemini_adapter_maps_errors(monkeypatch):
    class DeadlineTimeout(Exception):
        pass

    def timeout(_):
        raise DeadlineTimeout("deadline")

    def server(_):
        raise RuntimeError("500 secret detail")

    _fake_gemini(monkeypatch, timeout)
    with pytest.raises(LLMTimeoutError):
        GeminiClient("m", "k", 5).generate_json("s", "u")
    _fake_gemini(monkeypatch, server)
    with pytest.raises(LLMProviderError) as info:
        GeminiClient("m", "k", 5).generate_json("s", "u")
    assert "secret" not in str(info.value)


def test_end_to_end_with_fake_openai_sdk(monkeypatch):
    """Real factory + real adapter + fake SDK: key comes from the environment."""
    monkeypatch.setattr(rec_mod, "create_llm_client", llm_client_mod.create_llm_client)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-value")
    captured = _fake_openai(
        monkeypatch,
        lambda kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=GOOD_REPLY))]),
    )
    rc = last(NOISY + [250], llm_config(timeout_seconds=9)).recommendation
    assert rc.llm_status is LLMStatus.USED
    assert captured["init"]["api_key"] == "env-key-value" and captured["init"]["timeout"] == 9


# --------------------------------------------------------------------------- #
# Secrets hygiene
# --------------------------------------------------------------------------- #
def test_no_hardcoded_api_keys_in_package_sources():
    pattern = re.compile(r"sk-[A-Za-z0-9_\-]{20,}|AIza[0-9A-Za-z_\-]{20,}")
    for path in PKG.glob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name


def test_env_example_has_no_real_secrets():
    text = (PKG / ".env.example").read_text(encoding="utf-8")
    assert "ECOCAMP_LLM_ENABLED" in text
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "API_KEY" in line.split("=")[0]:
            assert line.split("=", 1)[1].strip() == "", line
    assert not re.search(r"sk-[A-Za-z0-9]{10,}|AIza[0-9A-Za-z_\-]{10,}", text)
