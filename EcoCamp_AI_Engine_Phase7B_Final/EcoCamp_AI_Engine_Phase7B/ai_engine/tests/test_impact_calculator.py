"""Tests for Phase 3 environmental impact calculation.

Emission factors used below are ARBITRARY TEST VALUES chosen to make the
arithmetic easy to check. They are not real emission factors.
"""
import math
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from ai_engine import (
    AnalysisConfig,
    AnalysisResult,
    CalculationStatus,
    ConsumptionRecord,
    DataQualityStatus,
    ImpactSummary,
    ResourceType,
    Severity,
    WasteEmissionFactor,
    aggregate_impact,
    analyze_consumption,
)
from ai_engine.anomaly_detector import detect_anomalies
from ai_engine.impact_calculator import apply_impact, estimate_impact

BASE = datetime(2026, 1, 1)
# Median 100, MAD 2. A final reading of 250 is +150% and flagged HIGH.
NOISY = [100, 102, 98, 101, 99, 103, 97, 100, 102, 98]
UNITS = {"electricity": "kWh", "water": "liters", "waste": "kg"}

# Test-only factors.
EF = 0.4        # kg CO2e per kWh
WF = 0.0003     # kg CO2e per liter
WASTE_F = 0.2   # kg CO2e per kg

FULL_CFG = AnalysisConfig(
    achievable_reduction_fraction=0.5,
    electricity_emission_factor_kg_co2e_per_kwh=EF,
    electricity_emission_factor_source="test value",
    water_emission_factor_kg_co2e_per_liter=WF,
    water_emission_factor_source="test value",
    waste_emission_factor=WasteEmissionFactor(
        value=WASTE_F, unit="kg_co2e_per_kg", source="test value"
    ),
)


def rec(value, day=0, building="Library", resource="electricity"):
    return ConsumptionRecord(
        timestamp=BASE + timedelta(days=day),
        building=building,
        resource_type=resource,
        consumption_value=value,
        unit=UNITS[resource],
    )


def series(values, **kw):
    return [rec(v, day=i, **kw) for i, v in enumerate(values)]


def last(values, config=None, **kw):
    return analyze_consumption(series(values, **kw), config)[-1]


# ---- electricity ------------------------------------------------------------ #
def test_electricity_savings_and_co2e():
    r = last(NOISY + [250], FULL_CFG)
    assert r.baseline_value == 100.0
    assert r.calculation_status is CalculationStatus.CALCULATED
    assert r.estimated_savings.value == pytest.approx(75.0)  # (250-100)*0.5
    assert r.estimated_savings.unit == "kWh"
    assert r.savings_unit == "kWh"
    assert r.estimated_co2e_reduction_kg == pytest.approx(75.0 * EF)


def test_electricity_without_factor_has_savings_but_no_co2e():
    r = last(NOISY + [250])  # default config: all factors None
    assert r.calculation_status is CalculationStatus.CALCULATED_WITHOUT_CO2E
    assert r.estimated_savings.value == pytest.approx(75.0)
    assert r.estimated_co2e_reduction_kg is None
    assert any("No electricity emission factor" in a for a in r.calculation_assumptions)


def test_zero_emission_factor_gives_zero_co2e_not_none():
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.0)
    r = last(NOISY + [250], cfg)
    assert r.estimated_co2e_reduction_kg == 0.0
    assert r.calculation_status is CalculationStatus.CALCULATED


# ---- water ------------------------------------------------------------------ #
def test_water_savings_in_liters_and_documented_co2e():
    r = last([v * 10 for v in NOISY] + [2500], FULL_CFG, resource="water")
    assert r.baseline_value == 1000.0
    assert r.estimated_savings.value == pytest.approx(750.0)
    assert r.estimated_savings.unit == "liters"
    assert r.savings_unit == "liters"
    assert r.estimated_co2e_reduction_kg == pytest.approx(750.0 * WF)
    assert any("source: test value" in a for a in r.calculation_assumptions)


def test_water_without_conversion_factor_has_no_co2e():
    r = last([v * 10 for v in NOISY] + [2500], resource="water")
    assert r.calculation_status is CalculationStatus.CALCULATED_WITHOUT_CO2E
    assert r.estimated_savings.value == pytest.approx(750.0)
    assert r.estimated_co2e_reduction_kg is None


# ---- waste ------------------------------------------------------------------ #
def test_waste_savings_in_kg_without_carbon_claim_by_default():
    r = last(NOISY + [250], resource="waste")
    assert r.estimated_savings.value == pytest.approx(75.0)
    assert r.estimated_savings.unit == "kg"
    assert r.savings_unit == "kg"
    assert r.estimated_co2e_reduction_kg is None
    assert r.calculation_status is CalculationStatus.CALCULATED_WITHOUT_CO2E
    assert any("no carbon-equivalence" in a for a in r.calculation_assumptions)


def test_waste_co2e_only_with_explicit_factor():
    r = last(NOISY + [250], FULL_CFG, resource="waste")
    assert r.estimated_co2e_reduction_kg == pytest.approx(75.0 * WASTE_F)
    assert r.calculation_status is CalculationStatus.CALCULATED


def test_electricity_factor_is_not_applied_to_waste_or_water():
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=EF)
    assert last(NOISY + [250], cfg, resource="waste").estimated_co2e_reduction_kg is None
    assert last(NOISY + [250], cfg, resource="water").estimated_co2e_reduction_kg is None


# ---- normal / below baseline ------------------------------------------------ #
def test_normal_reading_has_no_savings():
    r = last(NOISY + [101], FULL_CFG)
    assert r.severity is Severity.NONE
    assert r.estimated_savings is None
    assert r.savings_unit is None
    assert r.estimated_co2e_reduction_kg is None
    assert r.calculation_status is CalculationStatus.NOT_APPLICABLE


def test_slightly_above_baseline_but_not_anomalous_has_no_savings():
    r = last(NOISY + [110], FULL_CFG)  # +10%, below the 50% threshold
    assert r.severity is Severity.NONE
    assert r.estimated_savings is None
    assert r.calculation_status is CalculationStatus.NOT_APPLICABLE


def test_below_baseline_anomaly_has_no_savings_and_never_negative():
    r = last(NOISY + [20], FULL_CFG)
    assert r.severity is not Severity.NONE  # unusually low is still flagged
    assert r.deviation_percentage < 0
    assert r.estimated_savings is None
    assert r.estimated_co2e_reduction_kg is None
    assert r.calculation_status is CalculationStatus.NOT_APPLICABLE


def test_zero_reading_is_suspect_and_has_no_savings():
    r = last(NOISY + [0], FULL_CFG)
    assert r.data_quality_status is DataQualityStatus.SUSPECT
    assert r.estimated_savings is None
    assert r.calculation_status is CalculationStatus.NOT_APPLICABLE


# ---- zero / insufficient baseline ------------------------------------------- #
def test_insufficient_history_gives_no_savings():
    r = last([100, 100, 100, 500], FULL_CFG)  # only 3 earlier observations
    assert r.baseline_value is None
    assert r.data_quality_status is DataQualityStatus.INSUFFICIENT_DATA
    assert r.estimated_savings is None
    assert r.estimated_co2e_reduction_kg is None
    assert r.calculation_status is CalculationStatus.INSUFFICIENT_BASELINE


def test_zero_baseline_with_positive_observed_gives_no_savings():
    r = last([0] * 10 + [50], FULL_CFG)
    assert r.baseline_value == 0.0
    assert r.estimated_savings is None
    assert r.calculation_status is CalculationStatus.INSUFFICIENT_BASELINE


def test_zero_baseline_zero_observed_is_not_applicable():
    r = last([0] * 10 + [0], FULL_CFG)
    assert r.estimated_savings is None
    assert r.calculation_status is CalculationStatus.NOT_APPLICABLE


# ---- reduction fraction ----------------------------------------------------- #
@pytest.mark.parametrize("fraction,expected", [(0.0, 0.0), (0.2, 30.0), (1.0, 150.0)])
def test_reduction_fraction_scales_savings(fraction, expected):
    r = last(NOISY + [250], AnalysisConfig(achievable_reduction_fraction=fraction))
    assert r.estimated_savings.value == pytest.approx(expected)
    assert r.estimated_savings.value >= 0


def test_zero_fraction_is_calculated_with_zero_savings():
    r = last(NOISY + [250], AnalysisConfig(achievable_reduction_fraction=0.0))
    assert r.calculation_status is CalculationStatus.CALCULATED_WITHOUT_CO2E
    assert r.estimated_savings.value == 0.0


# ---- invalid configuration -------------------------------------------------- #
@pytest.mark.parametrize(
    "bad", [-0.1, 1.1, 2, float("nan"), float("inf"), float("-inf"), True, "abc"]
)
def test_invalid_reduction_fraction_rejected(bad):
    with pytest.raises(ValidationError):
        AnalysisConfig(achievable_reduction_fraction=bad)


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), True, "abc"])
def test_invalid_electricity_factor_rejected(bad):
    with pytest.raises(ValidationError):
        AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=bad)


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True])
def test_invalid_water_factor_rejected(bad):
    with pytest.raises(ValidationError):
        AnalysisConfig(
            water_emission_factor_kg_co2e_per_liter=bad,
            water_emission_factor_source="src",
        )


def test_water_factor_requires_documented_source():
    with pytest.raises(ValidationError):
        AnalysisConfig(water_emission_factor_kg_co2e_per_liter=0.001)
    with pytest.raises(ValidationError):
        AnalysisConfig(
            water_emission_factor_kg_co2e_per_liter=0.001,
            water_emission_factor_source="   ",
        )


def test_source_without_factor_rejected():
    with pytest.raises(ValidationError):
        AnalysisConfig(water_emission_factor_source="src")
    with pytest.raises(ValidationError):
        AnalysisConfig(electricity_emission_factor_source="src")


@pytest.mark.parametrize(
    "factor",
    [
        {"value": 0.2, "unit": "kg_co2e_per_tonne", "source": "s"},  # wrong unit
        {"value": 0.2, "unit": "kg_co2e_per_kg", "source": ""},      # no source
        {"value": -0.2, "unit": "kg_co2e_per_kg", "source": "s"},    # negative
        {"value": float("nan"), "unit": "kg_co2e_per_kg", "source": "s"},
        {"value": 0.2, "source": "s"},                               # no unit
        {"value": 0.2, "unit": "kg_co2e_per_kg", "source": "s", "extra": 1},
    ],
)
def test_invalid_waste_factor_rejected(factor):
    with pytest.raises(ValidationError):
        AnalysisConfig(waste_emission_factor=factor)


def test_waste_factor_accepts_dict_and_defaults_are_none():
    cfg = AnalysisConfig(
        waste_emission_factor={"value": 0.2, "unit": "kg_co2e_per_kg", "source": "s"}
    )
    assert cfg.waste_emission_factor.value == 0.2
    d = AnalysisConfig()
    assert d.electricity_emission_factor_kg_co2e_per_kwh is None
    assert d.water_emission_factor_kg_co2e_per_liter is None
    assert d.waste_emission_factor is None
    assert 0.0 <= d.achievable_reduction_fraction <= 1.0


def test_config_remains_frozen_and_rejects_unknown_fields():
    cfg = AnalysisConfig()
    with pytest.raises(ValidationError):
        cfg.achievable_reduction_fraction = 0.9
    with pytest.raises(ValidationError):
        AnalysisConfig(achievable_reduction_fraction_typo=0.5)


# ---- estimate_impact (pure formula) ----------------------------------------- #
def test_estimate_impact_formula_directly():
    est = estimate_impact(rec(250), 100.0, FULL_CFG)
    assert est.status is CalculationStatus.CALCULATED
    assert est.savings_value == pytest.approx(75.0)
    assert est.savings_unit == "kWh"
    assert est.co2e_reduction_kg == pytest.approx(30.0)


@pytest.mark.parametrize("baseline", [None, -5.0, float("nan")])
def test_estimate_impact_invalid_baseline(baseline):
    est = estimate_impact(rec(250), baseline, FULL_CFG)
    assert est.status is CalculationStatus.INSUFFICIENT_BASELINE
    assert est.savings_value is None and est.co2e_reduction_kg is None


@pytest.mark.parametrize("observed", [100, 99, 0])
def test_estimate_impact_never_negative(observed):
    est = estimate_impact(rec(observed), 100.0, FULL_CFG)
    assert est.status is CalculationStatus.NOT_APPLICABLE
    assert est.savings_value is None


# ---- multiple buildings and resources --------------------------------------- #
def multi_records():
    recs = []
    spec = [
        ("Library", "electricity", 1, 250),
        ("Library", "water", 10, 2500),
        ("Hostel", "electricity", 1, 250),
        ("Hostel", "waste", 1, 250),
        ("Lab", "electricity", 1, 101),  # normal
    ]
    for building, resource, scale, spike in spec:
        recs += [
            rec(v * scale, day=i, building=building, resource=resource)
            for i, v in enumerate(NOISY + [spike / scale])
        ]
    return recs


def test_multiple_buildings_and_resources_units_and_values():
    recs = multi_records()
    out = analyze_consumption(recs, FULL_CFG)
    assert len(out) == len(recs)
    for r_in, r_out in zip(recs, out):  # same order as input
        assert (r_out.building, r_out.resource_type, r_out.timestamp) == (
            r_in.building, r_in.resource_type, r_in.timestamp,
        )
        if r_out.estimated_savings is not None:
            assert r_out.estimated_savings.unit == UNITS[r_out.resource_type.value]
            assert r_out.savings_unit == r_out.estimated_savings.unit
            assert r_out.estimated_savings.value >= 0

    spikes = {
        (r.building, r.resource_type.value): r
        for r in out
        if r.estimated_savings is not None
    }
    assert set(spikes) == {
        ("Library", "electricity"), ("Library", "water"),
        ("Hostel", "electricity"), ("Hostel", "waste"),
    }
    assert spikes[("Library", "electricity")].estimated_savings.value == pytest.approx(75.0)
    assert spikes[("Library", "water")].estimated_savings.value == pytest.approx(750.0)
    assert spikes[("Hostel", "waste")].estimated_savings.value == pytest.approx(75.0)
    assert spikes[("Hostel", "waste")].estimated_co2e_reduction_kg == pytest.approx(15.0)


# ---- aggregation / no double counting --------------------------------------- #
def test_aggregate_sums_per_resource_without_mixing_units():
    out = analyze_consumption(multi_records(), FULL_CFG)
    summary = aggregate_impact(out)
    assert isinstance(summary, ImpactSummary)
    by_res = {t.resource_type.value: t for t in summary.totals}
    assert set(by_res) == {"electricity", "water", "waste"}
    assert by_res["electricity"].savings_value == pytest.approx(150.0)  # Library + Hostel
    assert by_res["electricity"].savings_unit == "kWh"
    assert by_res["electricity"].records_counted == 2
    assert by_res["water"].savings_value == pytest.approx(750.0)
    assert by_res["water"].savings_unit == "liters"
    assert by_res["waste"].savings_value == pytest.approx(75.0)
    assert by_res["waste"].savings_unit == "kg"
    expected_co2e = 150.0 * EF + 750.0 * WF + 75.0 * WASTE_F
    assert summary.total_co2e_reduction_kg == pytest.approx(expected_co2e)
    assert summary.co2e_complete is True
    assert summary.records_counted == 4
    assert summary.duplicate_records_skipped == 0


def test_aggregate_by_building():
    out = analyze_consumption(multi_records(), FULL_CFG)
    summary = aggregate_impact(out, by_building=True)
    keys = {(t.building, t.resource_type.value): t.savings_value for t in summary.totals}
    assert keys == {
        ("Hostel", "electricity"): pytest.approx(75.0),
        ("Hostel", "waste"): pytest.approx(75.0),
        ("Library", "electricity"): pytest.approx(75.0),
        ("Library", "water"): pytest.approx(750.0),
    }
    # Same overall CO2e as the per-resource view: nothing counted twice.
    assert summary.total_co2e_reduction_kg == pytest.approx(
        aggregate_impact(out).total_co2e_reduction_kg
    )


def test_aggregate_skips_duplicate_records():
    out = analyze_consumption(series(NOISY + [250]), FULL_CFG)
    once = aggregate_impact(out)
    dup = aggregate_impact(out + [out[-1]])  # same result submitted twice
    assert dup.duplicate_records_skipped == 1
    assert dup.totals[0].savings_value == pytest.approx(once.totals[0].savings_value)
    assert dup.total_co2e_reduction_kg == pytest.approx(once.total_co2e_reduction_kg)
    everything_twice = aggregate_impact(out + out)
    assert everything_twice.duplicate_records_skipped == len(out)
    assert everything_twice.total_co2e_reduction_kg == pytest.approx(once.total_co2e_reduction_kg)


def test_aggregate_is_order_independent():
    out = analyze_consumption(multi_records(), FULL_CFG)
    a = aggregate_impact(out)
    b = aggregate_impact(list(reversed(out)))
    assert a.total_co2e_reduction_kg == pytest.approx(b.total_co2e_reduction_kg)
    assert [t.model_dump() for t in a.totals] == [t.model_dump() for t in b.totals]


def test_aggregate_partial_co2e_is_flagged():
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=EF)  # no waste factor
    recs = (
        series(NOISY + [250], building="A")
        + series(NOISY + [250], building="B", resource="waste")
    )
    summary = aggregate_impact(analyze_consumption(recs, cfg))
    assert summary.records_counted == 2
    assert summary.co2e_complete is False
    assert summary.total_co2e_reduction_kg == pytest.approx(75.0 * EF)  # electricity only
    assert any("partial sum" in n for n in summary.notes)


def test_aggregate_with_no_savings_is_empty():
    summary = aggregate_impact(analyze_consumption(series(NOISY + [101]), FULL_CFG))
    assert summary.totals == []
    assert summary.total_co2e_reduction_kg is None
    assert summary.records_counted == 0
    assert aggregate_impact([]).records_counted == 0


def test_aggregate_never_negative():
    out = analyze_consumption(series(NOISY + [20, 0, 250]), FULL_CFG)
    summary = aggregate_impact(out)
    assert all(t.savings_value >= 0 for t in summary.totals)
    assert summary.total_co2e_reduction_kg is None or summary.total_co2e_reduction_kg >= 0


# ---- integration / compatibility -------------------------------------------- #
def test_detection_fields_unchanged_by_impact_step():
    recs = multi_records()
    cfg = AnalysisConfig()
    detected = detect_anomalies(recs, cfg)
    final = analyze_consumption(recs, cfg)
    for d, f in zip(detected, final):
        for name in (
            "timestamp", "building", "resource_type", "unit", "observed_value",
            "baseline_value", "deviation_percentage", "anomaly_score", "severity",
            "explanation", "data_quality_status",
        ):
            assert getattr(d, name) == getattr(f, name)
        # Original assumptions stay first; impact notes are appended.
        assert f.calculation_assumptions[: len(d.calculation_assumptions)] == d.calculation_assumptions


def test_analyze_consumption_signature_and_defaults_still_work():
    recs = series(NOISY + [250])
    assert analyze_consumption(recs) == analyze_consumption(recs, None)
    assert analyze_consumption(recs) == analyze_consumption(recs, AnalysisConfig())
    assert analyze_consumption([dict(
        timestamp="2026-03-01T10:00:00", building="Library",
        resource_type="electricity", consumption_value=5, unit="kWh",
    )])[0].calculation_status is CalculationStatus.INSUFFICIENT_BASELINE


def test_apply_impact_does_not_mutate_input():
    detected = detect_anomalies(series(NOISY + [250]), AnalysisConfig())[-1]
    before = detected.model_dump()
    out = apply_impact(detected, FULL_CFG)
    assert detected.model_dump() == before
    assert out is not detected
    assert out.estimated_savings is not None


def test_output_schema_fields_present_and_serializable():
    r = last(NOISY + [250], FULL_CFG)
    d = r.model_dump(mode="json")
    for name in (
        "observed_value", "baseline_value", "deviation_percentage", "anomaly_score",
        "severity", "data_quality_status", "estimated_savings", "savings_unit",
        "estimated_co2e_reduction_kg", "calculation_assumptions", "calculation_status",
    ):
        assert name in d
    assert d["calculation_status"] == "calculated"
    assert AnalysisResult.model_validate_json(r.model_dump_json()) == r
    s = aggregate_impact([r])
    assert ImpactSummary.model_validate_json(s.model_dump_json()) == s


def test_assumptions_document_formula_fraction_and_factor():
    r = last(NOISY + [250], FULL_CFG)
    text = " ".join(r.calculation_assumptions)
    assert "max(0, observed_value - baseline_value)" in text
    assert "achievable_reduction_fraction (0.5)" in text
    assert "0.4 kg CO2e/kWh" in text and "test value" in text


def test_results_are_deterministic_and_finite():
    recs = multi_records()
    a = analyze_consumption(recs, FULL_CFG)
    assert a == analyze_consumption(recs, FULL_CFG)
    for r in a:
        if r.estimated_savings is not None:
            assert math.isfinite(r.estimated_savings.value)
        if r.estimated_co2e_reduction_kg is not None:
            assert math.isfinite(r.estimated_co2e_reduction_kg)
