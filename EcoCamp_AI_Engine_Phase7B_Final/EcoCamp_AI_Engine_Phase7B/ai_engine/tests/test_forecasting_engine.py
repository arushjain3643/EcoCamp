"""Phase 6: forecasting engine tests (data handling, models, selection, leakage, backtesting).

All data is synthetic and deterministic. Nothing here needs network access or an API key.
Tests that need scikit-learn skip themselves when it is not installed.
"""
import math
import random
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import pytest

import ai_engine.forecasting_engine as fe
from ai_engine import (
    DataQualityStatus,
    ForecastConfig,
    ForecastMethod,
    RecordValidationError,
    forecast_consumption,
    rolling_origin_backtest,
)

START = datetime(2026, 3, 2, tzinfo=timezone.utc)  # a Monday
WEEK = [100.0, 110.0, 105.0, 108.0, 102.0, 60.0, 55.0]  # Mon..Sun
UNIT = {"electricity": "kWh", "water": "liters", "waste": "kg"}


def rec(ts, value, building="Library", resource="electricity", unit=None):
    return {
        "timestamp": ts,
        "building": building,
        "resource_type": resource,
        "consumption_value": value,
        "unit": unit or UNIT[resource],
    }


def daily(n, fn, building="Library", resource="electricity", start=START, skip=()):
    return [
        rec(start + timedelta(days=i), fn(i), building, resource)
        for i in range(n)
        if i not in set(skip)
    ]


def hourly(days, fn, start=START, drop=(), building="Library", resource="electricity"):
    out = []
    for d in range(days):
        for h in range(24):
            if (d, h) in set(drop):
                continue
            out.append(rec(start + timedelta(days=d, hours=h), fn(d, h), building, resource))
    return out


def weekly(i):
    return WEEK[i % 7]


def noisy_weekly(seed, sigma):
    rng = random.Random(seed)
    return lambda i: max(0.0, WEEK[i % 7] + rng.gauss(0, sigma))


def one(records, horizon=7, config=None):
    results = forecast_consumption(records, horizon, config)
    assert len(results) == 1
    return results[0]


def has_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=None)
def _noisy_result(n, seed, sigma):
    return one(daily(n, noisy_weekly(seed, sigma)))


# --------------------------------------------------------------------------- #
# Normal behaviour
# --------------------------------------------------------------------------- #
def test_normal_daily_consumption_seven_day_forecast():
    f = _noisy_result(30, 1, 3.0)
    assert f.forecast_available and f.data_quality_status is DataQualityStatus.OK
    assert f.forecast_horizon_days == 7 and len(f.predicted_values) == 7
    assert f.forecast_start_date == (START + timedelta(days=30)).date()
    assert f.forecast_dates == [f.forecast_start_date + timedelta(days=i) for i in range(7)]
    assert f.predicted_total == pytest.approx(sum(f.predicted_values), abs=1e-3)
    assert all(v >= 0 for v in f.predicted_values)
    for lo, v, hi in zip(f.lower_bound, f.predicted_values, f.upper_bound):
        assert 0 <= lo <= v <= hi
    assert f.unit == "kWh" and f.building == "Library" and f.resource_type.value == "electricity"
    assert f.predicted_total_lower_bound <= f.predicted_total <= f.predicted_total_upper_bound
    assert f.interval_level == 0.8 and "residual" in f.uncertainty_method


def test_next_day_forecast_horizon_one():
    f = one(daily(30, weekly), horizon=1)
    assert len(f.predicted_values) == 1 and f.forecast_horizon_days == 1
    assert len(f.lower_bound) == len(f.upper_bound) == 1
    assert f.predicted_values[0] == pytest.approx(WEEK[30 % 7])  # day 30 is a Wednesday-pattern day


def test_longer_horizon_is_supported_and_flagged():
    f = one(daily(20, weekly), horizon=21)  # backtests only reach 13 days ahead here
    assert len(f.predicted_values) == 21
    assert f.predicted_values[:7] == f.predicted_values[7:14] == f.predicted_values[14:]
    assert any("exceeds the longest backtested lead" in n for n in f.confidence_notes)


def test_weekly_seasonal_pattern_is_reproduced_by_seasonal_naive():
    f = one(daily(28, weekly))  # 4 full weeks, Monday start -> forecast starts on a Monday
    assert f.baseline_method is ForecastMethod.SEASONAL_NAIVE
    assert f.model_used is ForecastMethod.SEASONAL_NAIVE
    assert f.predicted_values == pytest.approx(WEEK)
    assert f.predicted_values[5] < f.predicted_values[0]  # weekend dip preserved
    # perfect backtest -> zero-width bounds, and the engine says so
    assert f.lower_bound == pytest.approx(WEEK) and f.upper_bound == pytest.approx(WEEK)
    assert any("errors were zero" in n for n in f.confidence_notes)


def test_weekday_alignment_when_history_does_not_start_on_monday():
    start = START + timedelta(days=3)  # Thursday
    f = one(daily(35, weekly, start=start))
    first = f.forecast_start_date
    expected = [WEEK[(35 + i) % 7] for i in range(7)]  # same position in the weekly cycle as 7 days earlier
    assert first == (start + timedelta(days=35)).date()
    assert first.weekday() == 3
    assert f.predicted_values == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Insufficient data and missing dates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 3, 5, 13])
def test_insufficient_history_returns_status_not_a_forecast(n):
    f = one(daily(n, weekly))
    assert f.data_quality_status is DataQualityStatus.INSUFFICIENT_DATA
    assert f.forecast_available is False
    assert f.predicted_values == [] and f.lower_bound == [] and f.upper_bound == []
    assert f.predicted_total is None and f.forecast_start_date is None and f.model_used is None
    assert f.history.observed_days == n
    assert any("No forecast produced" in note for note in f.confidence_notes)
    assert f.forecast_horizon_days == 7


def test_lower_minimum_history_uses_trailing_mean_baseline():
    cfg = ForecastConfig(minimum_history_days=7)
    f = one(daily(10, lambda i: 50.0), config=cfg)
    assert f.forecast_available
    assert f.baseline_method is ForecastMethod.TRAILING_MEAN and f.model_used is ForecastMethod.TRAILING_MEAN
    assert f.predicted_values == pytest.approx([50.0] * 7)
    assert any("weekly seasonality cannot be verified" in n for n in f.confidence_notes)


def test_too_few_backtest_residuals_means_no_interval_and_no_forecast():
    cfg = ForecastConfig(minimum_history_days=7)
    f = one(daily(7, lambda i: 50.0), config=cfg)  # only 6 residuals available
    assert f.data_quality_status is DataQualityStatus.INSUFFICIENT_DATA
    assert any("prediction interval" in n for n in f.confidence_notes)


def test_missing_days_are_left_missing_not_imputed():
    f = one(daily(42, weekly, skip=(10, 25)))
    assert f.forecast_available
    h = f.history
    assert h.observed_days == 40 and h.span_days == 42 and h.missing_days == 2
    assert h.coverage == pytest.approx(40 / 42, abs=1e-3)
    assert any("not imputed" in n for n in f.confidence_notes)
    assert f.predicted_values == pytest.approx([WEEK[(42 + i) % 7] for i in range(7)])
    assert f.data_quality_status is DataQualityStatus.OK  # under 5% missing: noted, not suspect


def test_many_missing_days_mark_data_suspect():
    f = one(daily(60, weekly, skip=range(5, 55, 6)))  # 9 of 60 missing (> 5%)
    assert f.forecast_available and f.data_quality_status is DataQualityStatus.SUSPECT
    assert f.history.missing_days == 9


def test_seasonal_naive_falls_back_to_older_weeks_when_same_weekday_missing():
    # day 28 (the target of step 1, a Monday) has its last-week value (day 21) missing
    f = one(daily(28, weekly, skip=(21,)))
    assert f.predicted_values[0] == pytest.approx(WEEK[0])  # taken from day 14 (two weeks back)
    assert f.forecast_available


def test_coverage_below_minimum_is_insufficient():
    # 12 days observed, 10 missing, 12 observed, 10 missing, 12 observed -> 64% coverage, gaps <= 14 days
    f = one(daily(56, weekly, skip=list(range(12, 22)) + list(range(34, 44))))
    assert f.data_quality_status is DataQualityStatus.INSUFFICIENT_DATA
    assert any("calendar days are observed" in n for n in f.confidence_notes)


def test_every_other_day_readings_are_treated_as_coarser_than_daily():
    f = one(daily(60, weekly, skip=[i for i in range(60) if i % 2 == 1]))
    assert f.data_quality_status is DataQualityStatus.INSUFFICIENT_DATA
    assert any("coarser than daily" in n for n in f.confidence_notes)


def test_long_gap_restarts_history_after_the_gap():
    records = daily(30, lambda i: 500.0) + daily(30, weekly, start=START + timedelta(days=60))
    f = one(records)
    assert f.forecast_available
    assert f.history.first_date == (START + timedelta(days=60)).date()
    assert f.history.observed_days == 30
    assert any("max_gap_days" in n for n in f.confidence_notes)
    assert max(f.predicted_values) < 200  # old 500-level data was not used
    assert f.data_quality_status is DataQualityStatus.SUSPECT


def test_history_window_limits_old_data():
    cfg = ForecastConfig(history_window_days=60, advanced_min_days=60, minimum_history_days=14)
    f = one(daily(100, weekly), config=cfg)
    assert f.history.observed_days == 60
    assert any("most recent 60 days" in n for n in f.confidence_notes)


# --------------------------------------------------------------------------- #
# Multiple buildings / resources
# --------------------------------------------------------------------------- #
def test_multiple_buildings_are_forecast_separately_and_sorted():
    records = (
        daily(30, lambda i: 1000.0 + WEEK[i % 7], building="Hostel")
        + daily(30, weekly, building="Admin")
    )
    results = forecast_consumption(records)
    assert [r.building for r in results] == ["Admin", "Hostel"]
    admin, hostel = results
    assert max(admin.predicted_values) < 200 and min(hostel.predicted_values) > 1000
    assert admin.history.records_received == hostel.history.records_received == 30


def test_multiple_resources_for_one_building_keep_their_own_units():
    records = (
        daily(30, weekly, resource="electricity")
        + daily(30, lambda i: 9000.0 + 10 * (i % 7), resource="water")
        + daily(30, lambda i: 40.0 + (i % 7), resource="waste")
    )
    results = forecast_consumption(records)
    by_res = {r.resource_type.value: r for r in results}
    assert set(by_res) == {"electricity", "water", "waste"}
    assert {k: v.unit for k, v in by_res.items()} == UNIT
    assert 9000 <= by_res["water"].predicted_values[0] <= 9100
    assert by_res["waste"].predicted_values[0] < 50
    assert all(r.forecast_available for r in results)


def test_one_short_series_does_not_block_other_series():
    records = daily(30, weekly, building="A") + daily(4, weekly, building="B")
    a, b = forecast_consumption(records)
    assert a.forecast_available and not b.forecast_available


def test_input_order_does_not_matter():
    records = daily(30, noisy_weekly(3, 4.0)) + daily(30, noisy_weekly(4, 4.0), building="B")
    shuffled = list(records)
    random.Random(0).shuffle(shuffled)
    a = [r.model_dump(mode="json") for r in forecast_consumption(records)]
    b = [r.model_dump(mode="json") for r in forecast_consumption(shuffled)]
    assert a == b


def test_timestamps_are_sorted_chronologically_before_use():
    records = daily(28, weekly)
    f_sorted = one(records)
    f_reversed = one(list(reversed(records)))
    assert f_sorted.predicted_values == f_reversed.predicted_values
    assert f_sorted.forecast_start_date == f_reversed.forecast_start_date


# --------------------------------------------------------------------------- #
# Timestamps: frequency, aggregation, irregularity
# --------------------------------------------------------------------------- #
def test_hourly_readings_are_summed_into_complete_days_only():
    # 28 days x 24 hourly readings of 5 kWh = 120 per day; day 10 loses two hours, day 20 one.
    records = hourly(28, lambda d, h: 5.0, drop={(10, 3), (10, 4), (20, 23)})
    f = one(records)
    assert f.forecast_available
    assert f.history.granularity == "24_readings_per_day"
    assert f.history.observed_days == 26 and f.history.excluded_days == 2
    assert f.predicted_values == pytest.approx([120.0] * 7)  # a partial day was not used as a low total
    assert f.data_quality_status is DataQualityStatus.SUSPECT
    assert any("excluded rather than estimated" in n for n in f.confidence_notes)


def test_incompatible_measurement_periods_are_not_mixed():
    daily_part = daily(20, lambda i: 100.0)  # 100 per day, one reading per day
    hourly_part = hourly(20, lambda d, h: 5.0, start=START + timedelta(days=20))  # 120 per day
    f = one(daily_part + hourly_part)
    assert f.forecast_available
    assert f.predicted_values == pytest.approx([120.0] * 7)  # the recent regime only
    assert f.history.excluded_days == 20
    assert any("not uniform" in n for n in f.confidence_notes)
    assert f.data_quality_status is DataQualityStatus.SUSPECT


def test_readings_coarser_than_daily_are_not_disaggregated():
    weekly_readings = daily(40, lambda i: 700.0)
    weekly_readings = [r for i, r in enumerate(weekly_readings) if i % 7 == 0]
    f = one(weekly_readings)
    assert f.data_quality_status is DataQualityStatus.INSUFFICIENT_DATA
    assert any("coarser than daily" in n for n in f.confidence_notes)
    assert f.predicted_values == []


def test_irregular_time_of_day_on_daily_series_is_flagged():
    rng = random.Random(5)
    records = []
    for i in range(30):
        ts = START + timedelta(days=i, hours=rng.choice([0, 1, 6, 9, 14]), minutes=rng.randrange(60))
        records.append(rec(ts, WEEK[i % 7]))
    f = one(records)
    assert f.forecast_available and f.data_quality_status is DataQualityStatus.SUSPECT
    assert any("Time of day" in n for n in f.confidence_notes)
    assert f.history.observed_days == 30


def test_irregular_subdaily_timestamps_exclude_the_unreliable_days():
    rng = random.Random(8)
    records = []
    for d in range(30):
        if d % 2 == 0:
            for h in (0, 12):  # regular twice-a-day
                records.append(rec(START + timedelta(days=d, hours=h), 50.0))
        else:
            for h in (1, 7):  # not on the 12-hour grid
                records.append(rec(START + timedelta(days=d, hours=h, minutes=rng.randrange(60)), 50.0))
    f = one(records)
    assert f.history.excluded_days == 15
    assert f.history.observed_days == 15
    assert f.data_quality_status in (DataQualityStatus.SUSPECT, DataQualityStatus.INSUFFICIENT_DATA)


def test_duplicate_identical_readings_are_counted_once():
    records = daily(28, weekly)
    records += [dict(records[3]), dict(records[9])]
    f = one(records)
    assert f.predicted_values == pytest.approx(WEEK)
    assert any("duplicate" in n for n in f.confidence_notes)


def test_conflicting_duplicate_readings_exclude_that_day():
    records = daily(30, weekly)
    records.append(rec(records[12]["timestamp"], records[12]["consumption_value"] + 40.0))
    f = one(records)
    assert f.history.observed_days == 29 and f.history.excluded_days == 1
    assert f.data_quality_status is DataQualityStatus.SUSPECT


def test_calendar_day_follows_the_timestamp_offset_not_utc():
    tz = timezone(timedelta(hours=5, minutes=30))
    start = datetime(2026, 3, 2, 0, 0, tzinfo=tz)  # 18:30 UTC the previous day
    f = one(daily(28, weekly, start=start))
    assert f.history.first_date == start.date()
    assert f.forecast_start_date == (start + timedelta(days=28)).date()
    assert f.predicted_values == pytest.approx(WEEK)


def test_mixed_utc_offsets_are_normalised_and_flagged():
    tz = timezone(timedelta(hours=5, minutes=30))
    records = daily(14, weekly) + daily(14, weekly, start=START.astimezone(tz) + timedelta(days=14))
    f = one(records)
    assert any("different UTC offsets" in n for n in f.confidence_notes)
    assert f.data_quality_status is DataQualityStatus.SUSPECT


def test_naive_timestamps_are_accepted_as_utc():
    naive = START.replace(tzinfo=None)
    f = one(daily(28, weekly, start=naive))
    assert f.forecast_available and f.predicted_values == pytest.approx(WEEK)


def test_iso_string_timestamps_are_accepted():
    records = [rec((START + timedelta(days=i)).isoformat().replace("+00:00", "Z"), weekly(i)) for i in range(28)]
    assert one(records).predicted_values == pytest.approx(WEEK)


# --------------------------------------------------------------------------- #
# Zero consumption
# --------------------------------------------------------------------------- #
def test_all_zero_history_forecasts_zero_and_is_suspect():
    f = one(daily(30, lambda i: 0.0))
    assert f.forecast_available
    assert f.predicted_values == [0.0] * 7 and f.predicted_total == 0.0
    assert f.lower_bound == [0.0] * 7 and f.upper_bound == [0.0] * 7
    assert f.data_quality_status is DataQualityStatus.SUSPECT
    assert any("constant zero history" in n for n in f.confidence_notes)
    assert all(m.wape is None for m in f.evaluation.metrics)  # WAPE undefined when actuals sum to 0
    assert all(not math.isnan(m.mae) for m in f.evaluation.metrics)


def test_zero_days_inside_normal_history_are_kept_as_real_readings():
    f = one(daily(35, lambda i: 0.0 if i % 7 in (5, 6) else 80.0))
    assert f.predicted_values == pytest.approx([80, 80, 80, 80, 80, 0, 0])
    assert f.data_quality_status is DataQualityStatus.OK
    assert any("zero consumption" in n for n in f.confidence_notes)
    assert f.lower_bound[5] == 0.0 and f.upper_bound[5] >= 0.0


def test_lower_bound_never_negative_for_noisy_low_values():
    f = one(daily(40, noisy_weekly(9, 30.0)))
    assert all(lo >= 0 for lo in f.lower_bound) and f.predicted_total_lower_bound >= 0


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
def test_small_dataset_never_uses_an_advanced_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("advanced model must not be touched for small data")

    monkeypatch.setattr(fe, "_predict_hgb", boom)
    f = _noisy_result(60, 2, 5.0)
    f2 = one(daily(60, noisy_weekly(2, 5.0)))
    assert f.model_used is ForecastMethod.SEASONAL_NAIVE and f2.model_used is ForecastMethod.SEASONAL_NAIVE
    assert f2.evaluation.advanced_model_considered is False
    assert "below advanced_min_days" in f2.evaluation.selection_reason


class _FakeHGB:
    """Stand-in so selection logic is tested without depending on scikit-learn numerics."""


def _install_fake_advanced(monkeypatch, predictor):
    monkeypatch.setattr(fe, "_import_hgb", lambda: _FakeHGB)
    monkeypatch.setattr(fe, "_predict_hgb", predictor)


def _pattern_oracle(hist, horizon, first_dow, seed):
    origin = hist.size - 1
    return np.array([WEEK[(first_dow + origin + h) % 7] for h in range(1, horizon + 1)], dtype=float)


def test_advanced_model_selected_only_when_backtest_justifies_it(monkeypatch):
    _install_fake_advanced(monkeypatch, _pattern_oracle)
    f = one(daily(120, noisy_weekly(11, 6.0)))
    assert f.model_used is ForecastMethod.HIST_GRADIENT_BOOSTING
    assert f.baseline_method is ForecastMethod.SEASONAL_NAIVE
    metrics = {m.method: m for m in f.evaluation.metrics}
    base, adv = metrics[ForecastMethod.SEASONAL_NAIVE], metrics[ForecastMethod.HIST_GRADIENT_BOOSTING]
    assert adv.mae <= 0.9 * base.mae  # the required 10% improvement was met
    assert adv.n_origins == base.n_origins and adv.n_points == base.n_points  # same origins
    assert adv.is_selected and not base.is_selected and base.is_baseline
    assert "selected" in f.evaluation.selection_reason


def test_advanced_model_rejected_when_not_better_than_baseline(monkeypatch):
    def worse(hist, horizon, first_dow, seed):
        return np.full(horizon, float(np.nanmean(hist)))  # ignores the weekly pattern

    _install_fake_advanced(monkeypatch, worse)
    f = one(daily(120, noisy_weekly(11, 2.0)))
    assert f.model_used is ForecastMethod.SEASONAL_NAIVE
    assert f.evaluation.advanced_model_considered is True
    assert "baseline kept" in f.evaluation.selection_reason


def test_advanced_model_rejected_when_improvement_below_threshold(monkeypatch):
    _install_fake_advanced(monkeypatch, _pattern_oracle)
    cfg = ForecastConfig(advanced_min_relative_improvement=0.95)
    f = one(daily(120, noisy_weekly(11, 6.0)), config=cfg)
    assert f.model_used is ForecastMethod.SEASONAL_NAIVE
    assert "does not meet the required 95%" in f.evaluation.selection_reason


def test_perfect_baseline_is_never_replaced(monkeypatch):
    _install_fake_advanced(monkeypatch, _pattern_oracle)
    f = one(daily(120, weekly))  # seasonal naive has zero error
    assert f.model_used is ForecastMethod.SEASONAL_NAIVE


def test_advanced_model_can_be_disabled(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("advanced model is disabled")

    monkeypatch.setattr(fe, "_predict_hgb", boom)
    f = one(daily(150, noisy_weekly(1, 4.0)), config=ForecastConfig(advanced_model_enabled=False))
    assert f.model_used is ForecastMethod.SEASONAL_NAIVE
    assert "disabled" in f.evaluation.selection_reason


def test_missing_scikit_learn_falls_back_to_baseline(monkeypatch):
    def no_sklearn():
        raise ImportError("no sklearn")

    monkeypatch.setattr(fe, "_import_hgb", no_sklearn)
    f = one(daily(150, noisy_weekly(1, 4.0)))
    assert f.model_used is ForecastMethod.SEASONAL_NAIVE and f.forecast_available
    assert any("scikit-learn is not installed" in n for n in f.confidence_notes)


def test_advanced_model_failure_never_breaks_the_forecast(monkeypatch):
    def broken(hist, horizon, first_dow, seed):
        raise RuntimeError("model blew up")

    _install_fake_advanced(monkeypatch, broken)
    f = one(daily(150, noisy_weekly(1, 4.0)))
    assert f.model_used is ForecastMethod.SEASONAL_NAIVE and f.forecast_available
    assert any("failed" in n for n in f.confidence_notes)


def test_real_hist_gradient_boosting_is_selected_on_large_noisy_weekly_data():
    pytest.importorskip("sklearn")
    f = _noisy_result(200, 4, 4.0)  # seed chosen for a wide margin (about 30% lower MAE)
    assert f.forecast_available
    assert f.evaluation.advanced_model_considered is True
    assert f.model_used is ForecastMethod.HIST_GRADIENT_BOOSTING
    metrics = {m.method: m for m in f.evaluation.metrics}
    assert metrics[ForecastMethod.HIST_GRADIENT_BOOSTING].mae < 0.9 * metrics[ForecastMethod.SEASONAL_NAIVE].mae
    # the forecast still follows the weekly shape
    weekday_idx = [(f.forecast_start_date + timedelta(days=i)).weekday() for i in range(7)]
    for v, i in zip(f.predicted_values, range(7)):
        assert abs(v - WEEK[(200 + i) % 7]) < 15
    assert weekday_idx[0] == (START + timedelta(days=200)).weekday()
    assert any("extrapolate" in n for n in f.confidence_notes)


@pytest.mark.parametrize("seed", [0, 2, 3])
def test_real_model_selection_is_consistent_with_reported_metrics(seed):
    pytest.importorskip("sklearn")
    f = _noisy_result(150, seed, 4.0)
    metrics = {m.method: m for m in f.evaluation.metrics}
    base = metrics[ForecastMethod.SEASONAL_NAIVE]
    adv = metrics[ForecastMethod.HIST_GRADIENT_BOOSTING]
    justified = adv.mae <= 0.9 * base.mae
    assert (f.model_used is ForecastMethod.HIST_GRADIENT_BOOSTING) == justified
    assert adv.is_selected == justified and base.is_selected == (not justified)


def test_real_hist_gradient_boosting_is_deterministic():
    pytest.importorskip("sklearn")
    records = daily(150, noisy_weekly(21, 4.0))
    fe.clear_forecast_cache()  # force two genuine fits instead of a memoised replay
    a = one(records).model_dump(mode="json")
    fe.clear_forecast_cache()
    b = one(records).model_dump(mode="json")
    assert a == b


# --------------------------------------------------------------------------- #
# No future-data leakage
# --------------------------------------------------------------------------- #
def _points_by_origin(result, limit_origin):
    return {
        (p["origin"], p["step"]): p["predicted"] for p in result["points"] if p["origin"] <= limit_origin
    }


@pytest.mark.parametrize("method", ["seasonal_naive", "trailing_mean"])
def test_backtest_predictions_ignore_data_after_the_origin(method):
    base = [WEEK[i % 7] + (i % 5) for i in range(80)]
    altered = list(base)
    for i in range(60, 80):  # rewrite the "future" with extreme values
        altered[i] = 10_000.0 + i
    a = rolling_origin_backtest(base, method=method, step_days=3, max_origins=40)
    b = rolling_origin_backtest(altered, method=method, step_days=3, max_origins=40)
    assert a["origins"] == b["origins"]
    assert _points_by_origin(a, 58) == _points_by_origin(b, 58)  # origins <= 58 only see days <= 58
    assert _points_by_origin(a, 58)  # and there are such points
    assert a["mae"] != b["mae"]  # sanity: the altered tail does change scores


def test_hist_gradient_boosting_predictions_ignore_data_after_the_origin():
    pytest.importorskip("sklearn")
    base = [WEEK[i % 7] + (i * 7 % 11) for i in range(110)]
    altered = list(base)
    for i in range(95, 110):
        altered[i] = 5_000.0
    kw = dict(method="hist_gradient_boosting", horizon_days=7, step_days=7, max_origins=6)
    a = rolling_origin_backtest(base, **kw)
    b = rolling_origin_backtest(altered, **kw)
    assert a["origins"] == b["origins"]
    safe = max(o for o in a["origins"] if o <= 94)
    assert _points_by_origin(a, safe) == _points_by_origin(b, safe) and _points_by_origin(a, safe)


def test_predictors_only_receive_history_up_to_the_origin(monkeypatch):
    seen = []
    real = fe._predict_seasonal_naive

    def spy(hist, horizon, lookback):
        seen.append(hist.copy())
        return real(hist, horizon, lookback)

    monkeypatch.setattr(fe, "_predict_seasonal_naive", spy)
    values = np.arange(40, dtype=float)
    rolling_origin_backtest(values, method="seasonal_naive", max_origins=10)
    assert seen
    for hist in seen:
        # the history handed over ends at the origin: it never contains later values
        assert hist[-1] == hist.size - 1 and hist.size < values.size


def test_final_forecast_ignores_values_that_come_after_the_series_end_and_is_pure():
    records = daily(30, noisy_weekly(6, 3.0))
    before = [dict(r) for r in records]
    a = one(records)
    b = one(records)
    assert a.model_dump(mode="json") == b.model_dump(mode="json")
    assert records == before  # inputs are not mutated


# --------------------------------------------------------------------------- #
# Rolling-origin backtesting and metrics
# --------------------------------------------------------------------------- #
def _brute_force_seasonal_naive(values, horizon, min_train, step=1, max_origins=28):
    n = len(values)
    origins = sorted(list(range(n - 2, min_train - 2, -step))[:max_origins])
    abs_err, actual = [], []
    for o in origins:
        for h in range(1, horizon + 1):
            if o + h >= n:
                break
            lag = o + h - 7 * (-(-h // 7))
            abs_err.append(abs(values[o + h] - values[lag]))
            actual.append(values[o + h])
    return origins, abs_err, actual


def test_backtest_matches_independent_seasonal_naive_calculation():
    values = [WEEK[i % 7] + (3 * i % 7) for i in range(35)]
    res = rolling_origin_backtest(values, method="seasonal_naive", horizon_days=7, max_origins=28)
    origins, abs_err, actual = _brute_force_seasonal_naive(values, 7, 7)
    assert res["origins"] == origins
    assert res["n_points"] == len(abs_err)
    assert res["mae"] == pytest.approx(sum(abs_err) / len(abs_err))
    assert res["wape"] == pytest.approx(sum(abs_err) / sum(actual))


def test_backtest_is_expanding_window_with_min_train_and_latest_origins():
    res = rolling_origin_backtest(list(range(100)), method="seasonal_naive", step_days=1, max_origins=10)
    assert res["origins"] == list(range(89, 99))  # the 10 most recent origins, ascending
    short = rolling_origin_backtest(list(range(14)), method="seasonal_naive")
    assert short["origins"][0] == 6  # at least 7 training days
    assert short["origins"][-1] == 12  # needs one later day to score


def test_backtest_not_possible_on_tiny_series():
    res = rolling_origin_backtest([1.0, 2.0, 3.0], method="seasonal_naive")
    assert res["n_points"] == 0 and res["mae"] is None and res["wape"] is None and res["origins"] == []


def test_backtest_skips_missing_actuals():
    values = [WEEK[i % 7] for i in range(30)]
    values[20] = float("nan")
    res = rolling_origin_backtest(values, method="seasonal_naive")
    assert all(not math.isnan(p["actual"]) and not math.isnan(p["predicted"]) for p in res["points"])
    assert all(p["origin"] + p["step"] != 20 for p in res["points"])


def test_wape_is_none_when_actuals_are_all_zero_and_mae_is_zero():
    res = rolling_origin_backtest([0.0] * 30, method="seasonal_naive")
    assert res["mae"] == 0.0 and res["wape"] is None


def test_mae_and_wape_values_on_a_hand_checked_example():
    # constant 10 with one spike of 20 on the last day; trailing mean (last 7 observed) predicts 10
    values = [10.0] * 11 + [20.0]
    res = rolling_origin_backtest(values, method="trailing_mean", horizon_days=1, max_origins=50)
    # origins 3..10 predict day o+1: errors 0 except the spike day (origin 10 -> actual 20, pred 10)
    assert res["n_points"] == 8
    assert res["mae"] == pytest.approx(10.0 / 8)
    assert res["wape"] == pytest.approx(10.0 / (7 * 10.0 + 20.0))


def test_result_reports_backtest_metrics_for_baseline():
    f = _noisy_result(30, 1, 3.0)
    ev = f.evaluation
    assert ev.backtest_performed and ev.interval_residuals_used > 0
    assert len(ev.metrics) == 1
    m = ev.metrics[0]
    assert m.method is ForecastMethod.SEASONAL_NAIVE and m.is_baseline and m.is_selected
    assert m.mae > 0 and m.wape is not None and 0 < m.wape < 1 and m.n_points == ev.interval_residuals_used


def test_interval_reflects_backtest_error_level():
    quiet = one(daily(60, noisy_weekly(1, 1.0)))
    noisy = one(daily(60, noisy_weekly(1, 12.0)))
    width = lambda r: sum(u - l for l, u in zip(r.lower_bound, r.upper_bound))  # noqa: E731
    assert width(noisy) > 3 * width(quiet)


def test_interval_has_reasonable_empirical_coverage_on_held_out_days():
    # Fit on 70 days, score the next 7 many times with different noise draws.
    inside = total = 0
    for seed in range(40):
        rng = random.Random(seed)
        series = [max(0.0, WEEK[i % 7] + rng.gauss(0, 6)) for i in range(77)]
        records = daily(70, lambda i: series[i])
        f = one(records)
        for k in range(7):
            total += 1
            inside += f.lower_bound[k] <= series[70 + k] <= f.upper_bound[k]
    assert total == 280
    assert 0.65 <= inside / total <= 0.95  # nominal 80%, empirical not guaranteed


def test_conformal_quantile_helper():
    q, limited = fe._conformal_quantile(np.arange(1.0, 11.0), 0.8)  # n=10 -> k=ceil(8.8)=9
    assert q == 9.0 and limited is False
    q, limited = fe._conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.9)  # k=ceil(3.6)=4 > n
    assert q == 3.0 and limited is True


# --------------------------------------------------------------------------- #
# Validation, configuration, errors
# --------------------------------------------------------------------------- #
def test_invalid_records_raise_record_validation_error_like_analyze_consumption():
    bad = daily(20, weekly) + [rec(START, 10.0, unit="Wh")]
    with pytest.raises(RecordValidationError) as exc:
        forecast_consumption(bad)
    assert exc.value.errors[0]["index"] == 20


def test_units_are_never_converted_or_mixed():
    with pytest.raises(RecordValidationError):
        forecast_consumption([rec(START, 1.0, resource="water", unit="gallons")])
    f = one(daily(20, weekly, resource="water"))
    assert f.unit == "liters"


@pytest.mark.parametrize("horizon", [0, -1, 31])
def test_horizon_out_of_range_is_rejected(horizon):
    with pytest.raises(ValueError):
        forecast_consumption(daily(20, weekly), horizon)


@pytest.mark.parametrize("horizon", [True, 7.0, "7", None])
def test_horizon_must_be_an_integer(horizon):
    with pytest.raises(TypeError):
        forecast_consumption(daily(20, weekly), horizon)


def test_programmer_errors_raise_type_error():
    with pytest.raises(TypeError):
        forecast_consumption(daily(20, weekly), 7, config={"minimum_history_days": 7})
    with pytest.raises(TypeError):
        forecast_consumption("not records")
    with pytest.raises(TypeError):
        forecast_consumption(None)


def test_empty_records_give_empty_result():
    assert forecast_consumption([]) == []


def test_forecast_config_validation():
    assert ForecastConfig().minimum_history_days == 14 and ForecastConfig().interval_level == 0.8
    for bad in (
        {"minimum_history_days": 3},
        {"interval_level": 1.0},
        {"interval_level": 0},
        {"minimum_history_days": True},
        {"advanced_min_days": 20},
        {"history_window_days": 30, "advanced_min_days": 90},
        {"unknown_option": 1},
        {"min_residuals_for_interval": 2},
        {"max_horizon_days": 0},
    ):
        with pytest.raises(Exception):
            ForecastConfig(**bad)


def test_forecast_config_is_frozen():
    with pytest.raises(Exception):
        ForecastConfig().minimum_history_days = 3


def test_custom_horizon_limit():
    cfg = ForecastConfig(max_horizon_days=10)
    with pytest.raises(ValueError):
        forecast_consumption(daily(30, weekly), 11, cfg)
    assert len(one(daily(30, weekly), 10, cfg).predicted_values) == 10


def test_exact_bool_numbers_not_accepted_in_config():
    with pytest.raises(Exception):
        ForecastConfig(random_seed=True)
