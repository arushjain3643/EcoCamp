"""Tests for Phase 2 anomaly detection (MAD baseline, Isolation Forest, API)."""
import random
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from ai_engine import (
    AnalysisConfig,
    AnomalyMethod,
    ConsumptionRecord,
    DataQualityStatus,
    RecordValidationError,
    Severity,
    analyze_consumption,
)
from ai_engine.anomaly_detector import detect_anomalies

BASE = datetime(2026, 1, 1)
# Median 100, MAD 2 -> well-behaved noisy history.
NOISY = [100, 102, 98, 101, 99, 103, 97, 100, 102, 98]
ORDER = [Severity.NONE, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


def rec(value, day=0, building="Library", resource="electricity", unit=None):
    unit = unit or {"electricity": "kWh", "water": "liters", "waste": "kg"}[resource]
    return ConsumptionRecord(
        timestamp=BASE + timedelta(days=day),
        building=building,
        resource_type=resource,
        consumption_value=value,
        unit=unit,
    )


def series(values, **kw):
    return [rec(v, day=i, **kw) for i, v in enumerate(values)]


def last(values, config=None, **kw):
    return analyze_consumption(series(values, **kw), config)[-1]


# ---- normal / high / zero -------------------------------------------------- #
def test_normal_consumption_not_flagged():
    r = last(NOISY + [101])
    assert r.severity is Severity.NONE
    assert r.data_quality_status is DataQualityStatus.OK
    assert r.baseline_value == 100.0
    assert r.observed_value == 101
    assert r.deviation_percentage == pytest.approx(1.0)
    assert r.anomaly_score < 0.5
    assert "Within normal range" in r.explanation


def test_unusually_high_consumption_flagged():
    r = last(NOISY + [250])
    assert r.baseline_value == 100.0
    assert r.deviation_percentage == pytest.approx(150.0)
    assert r.severity is Severity.HIGH
    assert r.anomaly_score == pytest.approx(0.75)
    assert "150.0% above" in r.explanation
    assert "Robust z-score" in r.explanation
    assert "250.00 kWh" in r.explanation


def test_severity_increases_with_magnitude():
    sev = [last(NOISY + [v]).severity for v in (101, 160, 250, 1000)]
    assert [ORDER.index(s) for s in sev] == sorted(ORDER.index(s) for s in sev)
    assert sev[-1] is Severity.CRITICAL


def test_unusually_low_consumption_flagged():
    r = last(NOISY + [20])
    assert r.severity is not Severity.NONE
    assert r.deviation_percentage == pytest.approx(-80.0)
    assert "below" in r.explanation


def test_large_percentage_but_not_statistical_outlier_not_flagged():
    # Very noisy history: a 60% rise is within normal variation (small z).
    noisy = [20, 180, 40, 160, 60, 140, 80, 120, 100, 100]
    r = last(noisy + [160])
    assert r.deviation_percentage >= 50
    assert r.severity is Severity.NONE


def test_small_percentage_but_huge_z_not_flagged():
    # Tiny MAD makes z huge, but a 3% change is below the % threshold.
    stable = [100, 100.1, 99.9, 100, 100.1, 99.9, 100, 100.1, 99.9, 100]
    r = last(stable + [103])
    assert r.severity is Severity.NONE


def test_zero_consumption_after_normal_history_is_suspect():
    r = last(NOISY + [0])
    assert r.observed_value == 0
    assert r.deviation_percentage == pytest.approx(-100.0)
    assert r.data_quality_status is DataQualityStatus.SUSPECT
    assert "meter outage" in r.explanation
    assert r.severity is not Severity.NONE


def test_zero_consumption_with_zero_baseline_is_normal():
    r = last([0] * 10 + [0])
    assert r.baseline_value == 0.0
    assert r.severity is Severity.NONE
    assert r.data_quality_status is DataQualityStatus.OK


def test_positive_value_on_zero_baseline_does_not_fabricate_deviation():
    r = last([0] * 10 + [5])
    assert r.baseline_value == 0.0
    assert r.deviation_percentage is None
    assert r.data_quality_status is DataQualityStatus.INSUFFICIENT_DATA
    assert r.severity is Severity.NONE


# ---- insufficient history -------------------------------------------------- #
def test_insufficient_history_returns_status_without_baseline():
    results = analyze_consumption(series([100, 101, 102]))
    for r in results:
        assert r.data_quality_status is DataQualityStatus.INSUFFICIENT_DATA
        assert r.baseline_value is None
        assert r.deviation_percentage is None
        assert r.severity is Severity.NONE
        assert r.anomaly_score == 0.0
        assert "at least 7" in r.explanation


def test_minimum_history_boundary():
    results = analyze_consumption(series(NOISY[:7] + [100, 100]))
    assert results[6].data_quality_status is DataQualityStatus.INSUFFICIENT_DATA  # 6 prior
    assert results[7].data_quality_status is DataQualityStatus.OK  # 7 prior
    assert results[7].baseline_value is not None


def test_custom_minimum_history():
    cfg = AnalysisConfig(minimum_history=3)
    r = analyze_consumption(series([100, 101, 99, 250]), cfg)[-1]
    assert r.baseline_value == 100.0
    assert r.severity is not Severity.NONE


# ---- multiple buildings / resources ---------------------------------------- #
def test_buildings_are_analyzed_independently():
    a = series(NOISY + [250], building="A")
    b = series([5000, 5100, 4900, 5050, 4950, 5000, 5100, 4900, 5000, 5050, 5000], building="B")
    results = analyze_consumption(a + b)
    ra, rb = results[len(a) - 1], results[-1]
    assert ra.severity is not Severity.NONE and ra.baseline_value == 100.0
    assert rb.severity is Severity.NONE and rb.baseline_value == 5000.0
    assert ra.building == "A" and rb.building == "B"


def test_resource_types_are_analyzed_independently():
    elec = series(NOISY + [250], resource="electricity")
    water = series([5000, 5100, 4900, 5050, 4950, 5000, 5100, 4900, 5000, 5050, 5000], resource="water")
    results = analyze_consumption(elec + water)
    assert results[len(elec) - 1].severity is not Severity.NONE
    assert results[-1].severity is Severity.NONE
    assert results[-1].unit == "liters" and results[0].unit == "kWh"


def test_interleaved_groups_keep_input_order():
    recs = []
    for i in range(11):
        recs.append(rec(NOISY[i] if i < 10 else 250, day=i, building="A"))
        recs.append(rec(1000, day=i, building="B", resource="waste"))
    results = analyze_consumption(recs)
    assert [(r.building, r.resource_type.value) for r in results] == [
        (x.building, x.resource_type.value) for x in recs
    ]
    assert results[20].severity is not Severity.NONE  # A day 10 spike
    assert results[21].severity is Severity.NONE


# ---- ordering --------------------------------------------------------------- #
def test_unsorted_input_gives_same_results_as_sorted():
    ordered = series(NOISY + [250, 101])
    shuffled = ordered[:]
    random.Random(1).shuffle(shuffled)
    by_ts = {r.timestamp: r for r in analyze_consumption(ordered)}
    out = analyze_consumption(shuffled)
    assert len(out) == len(shuffled)
    for src, res in zip(shuffled, out):  # aligned with input order
        assert res.timestamp == src.timestamp
        assert res == by_ts[src.timestamp]


# ---- constant history ------------------------------------------------------- #
def test_constant_history_same_value_is_normal():
    r = last([100] * 10 + [100])
    assert r.baseline_value == 100.0
    assert r.deviation_percentage == 0.0
    assert r.severity is Severity.NONE


def test_constant_history_spike_flagged_by_percentage_only():
    r = last([100] * 10 + [160])
    assert r.severity is not Severity.NONE
    assert "MAD is 0" in r.explanation
    assert r.deviation_percentage == pytest.approx(60.0)


def test_constant_history_small_change_not_flagged():
    r = last([100] * 10 + [110])
    assert r.severity is Severity.NONE


# ---- invalid records -------------------------------------------------------- #
def good_dict(**kw):
    d = {"timestamp": "2026-01-01T00:00:00", "building": "A",
         "resource_type": "water", "consumption_value": 10, "unit": "liters"}
    d.update(kw)
    return d


def test_dict_records_are_validated_and_accepted():
    out = analyze_consumption([good_dict()])
    assert len(out) == 1 and out[0].resource_type.value == "water"


@pytest.mark.parametrize(
    "bad",
    [
        good_dict(consumption_value=-5),
        good_dict(resource_type="gas"),
        good_dict(unit="kWh"),
        good_dict(timestamp="garbage"),
    ],
)
def test_invalid_dict_raises_with_index(bad):
    with pytest.raises(RecordValidationError) as exc:
        analyze_consumption([good_dict(), bad])
    assert exc.value.errors[0]["index"] == 1
    assert "record 1" in str(exc.value)


def test_all_invalid_records_reported_and_no_partial_results():
    with pytest.raises(RecordValidationError) as exc:
        analyze_consumption([good_dict(consumption_value=-1), good_dict(), good_dict(unit="x"), 42])
    assert [e["index"] for e in exc.value.errors] == [0, 2, 3]


def test_validation_error_is_a_value_error():
    with pytest.raises(ValueError):
        analyze_consumption([good_dict(consumption_value=-1)])


@pytest.mark.parametrize("bad", [None, "records", 5, good_dict()])
def test_non_sequence_records_rejected(bad):
    with pytest.raises(TypeError):
        analyze_consumption(bad)


def test_bad_config_type_rejected():
    with pytest.raises(TypeError):
        analyze_consumption([], config={"baseline_window": 5})


def test_empty_input_returns_empty_list():
    assert analyze_consumption([]) == []


# ---- no future-data leakage ------------------------------------------------- #
def test_future_records_do_not_change_earlier_results():
    base = series(NOISY + [101, 99, 250, 100])
    with_future = base + [rec(1_000_000, day=50), rec(0, day=51)]
    before = analyze_consumption(base)
    after = analyze_consumption(with_future)
    assert after[: len(base)] == before


def test_changing_a_future_value_does_not_change_past_results():
    a = series(NOISY + [101, 102, 103])
    b = series(NOISY + [101, 999, 103])
    ra, rb = analyze_consumption(a), analyze_consumption(b)
    assert ra[:11] == rb[:11]
    assert ra[11] != rb[11]


def test_baseline_excludes_the_record_itself():
    r = last([100] * 10 + [1000])
    assert r.baseline_value == 100.0  # not pulled up by the 1000 itself


def test_same_timestamp_records_are_not_each_others_history():
    same = [rec(100, day=0) for _ in range(9)]
    out = analyze_consumption(same)
    assert all(r.data_quality_status is DataQualityStatus.INSUFFICIENT_DATA for r in out)


def test_rolling_window_uses_only_recent_observations():
    cfg = AnalysisConfig(baseline_window=5, minimum_history=3)
    r = last([1000] * 10 + [100] * 5 + [100], cfg)
    assert r.baseline_value == 100.0
    assert r.severity is Severity.NONE


def test_mixed_naive_and_aware_timestamps_do_not_crash():
    recs = series(NOISY)
    aware = ConsumptionRecord(
        timestamp=datetime(2026, 1, 20, tzinfo=timezone.utc), building="Library",
        resource_type="electricity", consumption_value=250, unit="kWh",
    )
    out = analyze_consumption(recs + [aware])
    assert out[-1].severity is not Severity.NONE


# ---- phase boundaries / output contract ------------------------------------ #
def test_recommendation_and_savings_only_for_flagged_readings():
    # Phase 3/4: savings exist only for a qualifying above-baseline anomaly;
    # recommendations exist for flagged readings (including a suspect zero,
    # where the recommendation is to verify the meter), never for normal ones.
    out = analyze_consumption(series(NOISY + [250, 0]))
    normal, spike, zero = out[-3], out[-2], out[-1]
    assert normal.recommended_action is None and normal.recommendation is None
    assert normal.estimated_savings is None
    assert spike.recommended_action is not None and spike.recommendation is not None
    assert spike.estimated_savings is not None  # 250 vs baseline 100
    assert zero.recommendation is not None
    assert zero.estimated_savings is None  # 0 is below baseline


def test_result_fields_populated_and_serializable():
    r = last(NOISY + [250])
    assert 0.0 <= r.anomaly_score <= 1.0
    assert r.explanation and r.calculation_assumptions
    assert r.building == "Library" and r.resource_type.value == "electricity"
    r2 = type(r).model_validate_json(r.model_dump_json())
    assert r2 == r


def test_deterministic():
    recs = series(NOISY + [250])
    assert analyze_consumption(recs) == analyze_consumption(recs)


def test_detect_anomalies_direct_call_matches_analyze():
    # analyze_consumption = detect_anomalies + Phase 3 impact fields, so the two
    # outputs are no longer identical. Compare only the detection fields.
    recs = series(NOISY + [250])
    raw = detect_anomalies(recs)
    full = analyze_consumption(recs)
    assert len(raw) == len(full)
    for d, f in zip(raw, full):
        for name in (
            "timestamp", "building", "resource_type", "unit", "observed_value",
            "baseline_value", "deviation_percentage", "anomaly_score", "severity",
            "explanation", "data_quality_status",
        ):
            assert getattr(d, name) == getattr(f, name), name
    # The raw detector adds no impact fields; the analyzer does.
    assert raw[-1].estimated_savings is None
    assert full[-1].estimated_savings is not None


# ---- config ----------------------------------------------------------------- #
def test_config_defaults():
    c = AnalysisConfig()
    assert (c.baseline_window, c.minimum_history) == (30, 7)
    assert c.deviation_threshold_percentage == 50.0
    assert c.anomaly_method is AnomalyMethod.MAD


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_history": 2},
        {"baseline_window": 5, "minimum_history": 10},
        {"deviation_threshold_percentage": 0},
        {"deviation_threshold_percentage": float("nan")},
        {"anomaly_method": "kmeans"},
        {"robust_z_threshold": -1},
        {"anomaly_method": "isolation_forest"},  # default window 30 < IF min 50
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValidationError):
        AnalysisConfig(**kwargs)


def test_threshold_is_respected():
    strict = AnalysisConfig(deviation_threshold_percentage=10)
    lenient = AnalysisConfig(deviation_threshold_percentage=500)
    values = NOISY + [130]
    assert last(values, strict).severity is not Severity.NONE
    assert last(values, lenient).severity is Severity.NONE


# ---- Isolation Forest ------------------------------------------------------- #
IF_NOISE = [100 + ((i * 37) % 11) - 5 for i in range(70)]
IF_CFG = AnalysisConfig(anomaly_method="isolation_forest", baseline_window=60)


def test_isolation_forest_flags_spike_with_enough_history():
    r = last(IF_NOISE + [300], IF_CFG)
    assert r.severity is Severity.HIGH
    assert "Isolation Forest score" in r.explanation
    assert "outlier" in r.explanation


def test_isolation_forest_normal_value_not_flagged():
    r = last(IF_NOISE + [101], IF_CFG)
    assert r.severity is Severity.NONE


def test_isolation_forest_not_forced_on_small_history():
    r = last(IF_NOISE[:20] + [300], IF_CFG)
    assert r.severity is not Severity.NONE  # still detected via MAD fallback
    assert "Isolation Forest score" not in r.explanation
    assert any("robust MAD scoring used instead" in a for a in r.calculation_assumptions)


def test_isolation_forest_uses_mad_early_and_forest_later():
    out = analyze_consumption(series(IF_NOISE + [300]), IF_CFG)
    assert any("MAD scoring used" in a for a in out[30].calculation_assumptions)
    assert not any("MAD scoring used" in a for a in out[-1].calculation_assumptions)


def test_isolation_forest_constant_history_falls_back_to_percentage():
    r = last([100] * 60 + [200], IF_CFG)
    assert r.severity is not Severity.NONE
    assert any("history is constant" in a for a in r.calculation_assumptions)


def test_isolation_forest_is_deterministic_and_leak_free():
    recs = series(IF_NOISE + [300])
    assert analyze_consumption(recs, IF_CFG) == analyze_consumption(recs, IF_CFG)
    extended = recs + [rec(5000, day=200)]
    assert analyze_consumption(extended, IF_CFG)[: len(recs)] == analyze_consumption(recs, IF_CFG)
