"""Schema validation tests for the EcoCamp AI engine (Phase 1)."""
import pytest
from pydantic import ValidationError

from ai_engine import (
    AnalysisConfig,
    AnalysisResult,
    ConsumptionRecord,
    DataQualityStatus,
    ResourceType,
    Severity,
)


def make(**overrides):
    data = {
        "timestamp": "2026-03-01T10:00:00",
        "building": "Library",
        "resource_type": "electricity",
        "consumption_value": 120.5,
        "unit": "kWh",
    }
    data.update(overrides)
    return ConsumptionRecord(**data)


# ---- valid inputs ---------------------------------------------------------- #
@pytest.mark.parametrize(
    "resource,unit",
    [("electricity", "kWh"), ("water", "liters"), ("waste", "kg")],
)
def test_valid_record_each_resource(resource, unit):
    r = make(resource_type=resource, unit=unit)
    assert r.resource_type == ResourceType(resource)
    assert r.unit == unit


def test_zero_consumption_allowed():
    assert make(consumption_value=0).consumption_value == 0


def test_timestamp_iso_string_and_datetime_accepted():
    from datetime import datetime

    assert make(timestamp="2026-03-01T10:00:00Z").timestamp.year == 2026
    assert make(timestamp=datetime(2026, 3, 1, 10)).timestamp.hour == 10


def test_normalization_of_case_and_whitespace():
    r = make(resource_type=" Water ", unit="Litres", building="  Hostel A ")
    assert r.resource_type == ResourceType.WATER
    assert r.unit == "liters"
    assert r.building == "Hostel A"
    assert make(unit="KWH").unit == "kWh"


# ---- negative / non-finite consumption ------------------------------------- #
def test_negative_consumption_rejected():
    with pytest.raises(ValidationError, match="negative"):
        make(consumption_value=-1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_consumption_rejected(bad):
    with pytest.raises(ValidationError):
        make(consumption_value=bad)


@pytest.mark.parametrize("bad", ["abc", None, True])
def test_non_numeric_consumption_rejected(bad):
    with pytest.raises(ValidationError):
        make(consumption_value=bad)


# ---- resource type --------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["gas", "", "solar", None, 5])
def test_unsupported_resource_type_rejected(bad):
    with pytest.raises(ValidationError):
        make(resource_type=bad)


# ---- units ----------------------------------------------------------------- #
@pytest.mark.parametrize(
    "resource,bad_unit",
    [
        ("electricity", "liters"),
        ("electricity", "MWh"),
        ("water", "kWh"),
        ("water", "gallons"),
        ("waste", "liters"),
        ("waste", "lbs"),
        ("waste", ""),
    ],
)
def test_incompatible_units_rejected(resource, bad_unit):
    with pytest.raises(ValidationError, match="incompatible unit"):
        make(resource_type=resource, unit=bad_unit)


# ---- timestamps ------------------------------------------------------------ #
@pytest.mark.parametrize(
    "bad", ["not-a-date", "2026-13-45", "", "   ", None, 1700000000, 12.5]
)
def test_malformed_timestamp_rejected(bad):
    with pytest.raises(ValidationError):
        make(timestamp=bad)


# ---- other structural rules ------------------------------------------------ #
def test_empty_building_rejected():
    with pytest.raises(ValidationError):
        make(building="   ")


def test_missing_field_rejected():
    with pytest.raises(ValidationError):
        ConsumptionRecord(timestamp="2026-03-01T10:00:00", building="X")


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        make(extra_field=1)


def test_record_is_immutable():
    with pytest.raises(ValidationError):
        make().consumption_value = 5


# ---- output schema --------------------------------------------------------- #
def valid_result(**overrides):
    data = {
        "timestamp": "2026-03-01T10:00:00",
        "building": "Library",
        "resource_type": "electricity",
        "unit": "kWh",
        "observed_value": 150.0,
        "baseline_value": 100.0,
        "deviation_percentage": 50.0,
        "anomaly_score": 0.8,
        "severity": "high",
        "explanation": "test",
        "recommended_action": "test",
        "estimated_savings": {"value": 10.0, "unit": "kWh"},
        "calculation_assumptions": ["test assumption"],
        "data_quality_status": "ok",
    }
    data.update(overrides)
    return AnalysisResult(**data)


def test_valid_analysis_result():
    r = valid_result()
    assert r.severity is Severity.HIGH
    assert r.data_quality_status is DataQualityStatus.OK
    assert r.estimated_savings.unit == "kWh"


def test_analysis_result_optional_fields():
    r = valid_result(
        baseline_value=None,
        deviation_percentage=None,
        recommended_action=None,
        estimated_savings=None,
        data_quality_status="insufficient_data",
    )
    assert r.baseline_value is None and r.estimated_savings is None


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_anomaly_score_bounds(score):
    with pytest.raises(ValidationError):
        valid_result(anomaly_score=score)


def test_invalid_severity_rejected():
    with pytest.raises(ValidationError):
        valid_result(severity="catastrophic")


def test_negative_savings_rejected():
    with pytest.raises(ValidationError):
        valid_result(estimated_savings={"value": -1, "unit": "kWh"})


def test_result_json_roundtrip():
    r = valid_result()
    assert AnalysisResult.model_validate_json(r.model_dump_json()) == r


# ---- config ---------------------------------------------------------------- #
def test_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        AnalysisConfig(threshold=3)
