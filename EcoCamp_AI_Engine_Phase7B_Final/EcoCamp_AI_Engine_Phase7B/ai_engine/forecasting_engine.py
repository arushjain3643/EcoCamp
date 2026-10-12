"""Phase 6: predictive resource forecasting (next-day and multi-day).

Public API (additive; ``analyze_consumption`` is untouched)::

    forecast_consumption(records, horizon_days=7, config=None) -> List[ForecastResult]
    forecast_payload(payload, config=None) -> AnalysisOutcome      # JSON in / JSON out
    serialize_forecast(s)                                           # JSON-safe dicts

One forecast is produced per (building, resource_type) series, never across
buildings or resources, so kWh, liters and kg are never mixed.

How a series is forecast
------------------------
1. **Data handling** (``_build_daily_series``): timestamps are sorted; readings
   are aggregated to calendar-day totals *only* when a day is fully covered at
   the series' own reading frequency. Days that are partial, ambiguous (several
   readings on a daily series) or contradictory (same timestamp, different
   values) are excluded, never imputed. Missing days stay missing. Nothing is
   fabricated.
2. **Gates**: too few days, too many gaps, or readings coarser than daily ->
   ``data_quality_status = "insufficient_data"`` and no forecast.
3. **Candidate models** (all trained only on data before the forecast origin):
   * ``seasonal_naive``: value of the same weekday last week (baseline).
   * ``trailing_mean``: mean of the last 7 observed days (baseline when fewer
     than two full weeks of history exist).
   * ``hist_gradient_boosting``: scikit-learn ``HistGradientBoostingRegressor``
     (direct multi-horizon, lag features). Considered only with at least
     ``advanced_min_days`` observed days and an installed scikit-learn.
4. **Selection**: rolling-origin backtesting (expanding window). The advanced
   model is selected only if its MAE beats the baseline's by at least
   ``advanced_min_relative_improvement`` on the same origins.
5. **Uncertainty**: empirical prediction intervals from the selected model's own
   backtest residuals (split-conformal style, absolute errors). See
   ``_interval_half_widths``. No arbitrary percentage bands are used.

Leakage guard: every predictor receives a *copy of the history truncated at the
origin* (``cal[: origin + 1]``), so it cannot see later data; training rows for
the advanced model only use targets inside that truncated history.
"""
from __future__ import annotations

import math
import threading
import warnings
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ._version import CONTRACT_VERSION, FORECAST_CONTRACT_VERSION, __version__
from .analyzer import RecordValidationError
from .config import STANDARD_UNITS, ResourceType
from .schemas import ConsumptionRecord, DataQualityStatus

__all__ = [
    "FORECAST_CONTRACT_VERSION",
    "ForecastConfig",
    "ForecastEvaluation",
    "ForecastMethod",
    "ForecastResult",
    "HistorySummary",
    "MethodMetrics",
    "forecast_consumption",
    "forecast_payload",
    "clear_forecast_cache",
    "rolling_origin_backtest",
    "serialize_forecast",
    "serialize_forecasts",
]

SEASONAL_PERIOD_DAYS = 7
_SECONDS_PER_DAY = 86400
_TRAILING_MEAN_WINDOW = 7
# Smallest training spans (calendar days) a method needs before it is backtested.
_MIN_TRAIN_DAYS_SEASONAL = 7
_MIN_TRAIN_DAYS_TRAILING = 4
_MIN_TRAIN_DAYS_HGB = 42
_HGB_FIRST_TRAINING_ORIGIN = 6
_HGB_MIN_TRAIN_ROWS = 100
_HGB_PARAMS: Dict[str, Any] = {
    "loss": "absolute_error",  # matches the MAE selection metric, robust to spikes
    "max_iter": 100,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "early_stopping": False,  # no random internal validation split; deterministic
}
_TOD_VARIATION_NOTE_SECONDS = 3 * 3600
_MISSING_SUSPECT_FRACTION = 0.05
_ZERO_SHARE_SUSPECT = 0.5
_MIN_ADVANCED_BACKTEST_ORIGINS = 3

UNCERTAINTY_METHOD = (
    "empirical_backtest_residual_quantile (split-conformal style): half-width = the "
    "ceil((n+1)*level)-th smallest absolute rolling-origin backtest error of the selected "
    "model, per forecast step when the step has enough residuals, otherwise pooled over "
    "steps; lower bound clipped at 0"
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class ForecastMethod(str, Enum):
    SEASONAL_NAIVE = "seasonal_naive"
    TRAILING_MEAN = "trailing_mean"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"


class ForecastConfig(BaseModel):
    """Settings for ``forecast_consumption``. Defaults are engineering starting points.

    Attributes:
        minimum_history_days: Observed complete days needed to forecast at all
            (default 14: two full weeks, the least that can show a weekly
            pattern). Values 7-13 are allowed; the baseline is then the
            trailing mean because weekly seasonality cannot be checked.
        history_window_days: Only the most recent N calendar days are used.
        max_gap_days: A run of missing days longer than this restarts the usable
            history after the gap (older data is dropped and a note is added).
        min_coverage: Min share of calendar days (inside the usable window) that
            must be observed. Missing days are never imputed.
        timestamp_tolerance_seconds: Allowed jitter when checking that
            sub-daily readings sit on a regular grid.
        max_horizon_days: Upper limit for ``horizon_days``.
        seasonal_max_lookback_weeks: How many weeks back the seasonal naive
            baseline may look for the same weekday when a value is missing.
        interval_level: Nominal coverage of the prediction interval (0.8 = 80%).
        min_residuals_for_interval: Min backtest residuals needed to build an
            interval; fewer -> ``insufficient_data`` (no defensible bounds).
        backtest_max_origins: Max rolling origins (daily step) used to backtest
            the baseline.
        advanced_model_enabled: Allow ``HistGradientBoostingRegressor``.
        advanced_min_days: Min observed days before the advanced model is even
            considered ("not for small datasets").
        advanced_backtest_origins / advanced_backtest_step_days: Origins (and
            spacing) used to compare the advanced model with the baseline.
        advanced_min_relative_improvement: The advanced model must reduce MAE by
            at least this fraction versus the baseline to be selected.
        random_seed: Seed for the advanced model (reproducible output).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    minimum_history_days: int = Field(default=14, ge=7)
    history_window_days: int = Field(default=365, ge=28)
    max_gap_days: int = Field(default=14, ge=1)
    min_coverage: float = Field(default=0.7, gt=0, le=1, allow_inf_nan=False)
    timestamp_tolerance_seconds: float = Field(default=60.0, ge=0, allow_inf_nan=False)
    max_horizon_days: int = Field(default=30, ge=1, le=366)
    seasonal_max_lookback_weeks: int = Field(default=4, ge=1, le=12)
    interval_level: float = Field(default=0.8, gt=0, lt=1, allow_inf_nan=False)
    min_residuals_for_interval: int = Field(default=10, ge=4)
    backtest_max_origins: int = Field(default=28, ge=1)
    advanced_model_enabled: bool = True
    advanced_min_days: int = Field(default=90, ge=45)
    advanced_backtest_origins: int = Field(default=12, ge=_MIN_ADVANCED_BACKTEST_ORIGINS)
    advanced_backtest_step_days: int = Field(default=5, ge=1)
    advanced_min_relative_improvement: float = Field(default=0.10, ge=0, lt=1, allow_inf_nan=False)
    random_seed: int = 42

    @field_validator(
        "minimum_history_days", "history_window_days", "max_gap_days", "min_coverage",
        "timestamp_tolerance_seconds", "max_horizon_days", "seasonal_max_lookback_weeks",
        "interval_level", "min_residuals_for_interval", "backtest_max_origins",
        "advanced_min_days", "advanced_backtest_origins", "advanced_backtest_step_days",
        "advanced_min_relative_improvement", "random_seed",
        mode="before",
    )
    @classmethod
    def _reject_bool_numbers(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("must be a number, not a boolean")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> "ForecastConfig":
        if self.history_window_days < self.minimum_history_days:
            raise ValueError("history_window_days must be >= minimum_history_days")
        if self.history_window_days < self.advanced_min_days:
            raise ValueError("history_window_days must be >= advanced_min_days")
        if self.advanced_min_days < self.minimum_history_days:
            raise ValueError("advanced_min_days must be >= minimum_history_days")
        return self


# --------------------------------------------------------------------------- #
# Output models
# --------------------------------------------------------------------------- #
class HistorySummary(BaseModel):
    """What the engine actually used from the input history."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    records_received: int = Field(ge=0)
    first_date: Optional[date] = None
    last_date: Optional[date] = None
    observed_days: int = Field(ge=0)
    span_days: int = Field(ge=0)
    missing_days: int = Field(ge=0)
    excluded_days: int = Field(ge=0)
    coverage: Optional[float] = Field(default=None, ge=0, le=1)
    granularity: Optional[str] = None


class MethodMetrics(BaseModel):
    """Rolling-origin backtest accuracy of one method (same origins for all rows)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    method: ForecastMethod
    is_baseline: bool
    is_selected: bool
    mae: float = Field(ge=0)
    wape: Optional[float] = Field(default=None, ge=0)
    n_origins: int = Field(ge=0)
    n_points: int = Field(ge=0)


class ForecastEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    backtest_performed: bool
    scheme: str
    advanced_model_considered: bool
    metrics: List[MethodMetrics] = Field(default_factory=list)
    selection_reason: str
    interval_residuals_used: int = Field(ge=0)


class ForecastResult(BaseModel):
    """Forecast for one (building, resource_type) series. Every key is always present."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    building: str
    resource_type: ResourceType
    unit: str
    forecast_available: bool
    forecast_start_date: Optional[date] = None
    forecast_horizon_days: int = Field(ge=1)
    forecast_dates: List[date] = Field(default_factory=list)
    predicted_values: List[float] = Field(default_factory=list)
    predicted_total: Optional[float] = Field(default=None, ge=0)
    baseline_method: Optional[ForecastMethod] = None
    model_used: Optional[ForecastMethod] = None
    lower_bound: List[float] = Field(default_factory=list)
    upper_bound: List[float] = Field(default_factory=list)
    predicted_total_lower_bound: Optional[float] = Field(default=None, ge=0)
    predicted_total_upper_bound: Optional[float] = Field(default=None, ge=0)
    interval_level: Optional[float] = None
    uncertainty_method: Optional[str] = None
    data_quality_status: DataQualityStatus
    confidence_notes: List[str] = Field(default_factory=list)
    history: HistorySummary
    evaluation: Optional[ForecastEvaluation] = None


# --------------------------------------------------------------------------- #
# Record validation (same rules and error type as analyze_consumption)
# --------------------------------------------------------------------------- #
def _coerce_records(records: Any) -> List[ConsumptionRecord]:
    if records is None or isinstance(records, (str, bytes, Mapping)):
        raise TypeError("records must be a sequence of ConsumptionRecord or dict")
    try:
        items = list(records)
    except TypeError:
        raise TypeError("records must be a sequence of ConsumptionRecord or dict") from None
    validated: List[ConsumptionRecord] = []
    errors: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        if isinstance(item, ConsumptionRecord):
            validated.append(item)
        elif isinstance(item, Mapping):
            try:
                validated.append(ConsumptionRecord.model_validate(dict(item)))
            except ValidationError as exc:
                errors.append({
                    "index": i,
                    "errors": [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()],
                })
        else:
            errors.append({
                "index": i,
                "errors": [{"loc": (), "msg": f"expected ConsumptionRecord or dict, got {type(item).__name__}"}],
            })
    if errors:
        raise RecordValidationError(errors)
    return validated


# --------------------------------------------------------------------------- #
# Step 1: records -> clean daily series (no imputation)
# --------------------------------------------------------------------------- #
@dataclass
class _DailySeries:
    dates: List[date] = field(default_factory=list)  # observed complete days, sorted
    values: List[float] = field(default_factory=list)
    granularity: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    suspect: bool = False
    excluded: Dict[str, int] = field(default_factory=dict)
    insufficient_reason: Optional[str] = None
    last_record_date: Optional[date] = None


def _wall_clock_readings(
    records: Sequence[ConsumptionRecord], series: _DailySeries
) -> List[Tuple[datetime, float]]:
    """Sorted (wall-clock datetime, value) pairs with one consistent UTC offset.

    Naive timestamps are treated as UTC (as documented for the analyze contract).
    If offsets differ, everything is converted to the most common offset so the
    calendar day of a reading is well defined.
    """
    aware: List[Tuple[datetime, float]] = []
    n_naive = 0
    for r in records:
        ts = r.timestamp
        if ts.tzinfo is None or ts.utcoffset() is None:
            n_naive += 1
            ts = ts.replace(tzinfo=timezone.utc)
        aware.append((ts, r.consumption_value))
    if 0 < n_naive < len(aware):
        series.notes.append(
            f"{n_naive} timestamp(s) had no UTC offset and were treated as UTC while others "
            "carry offsets."
        )
        series.suspect = True
    offsets = Counter(ts.utcoffset() for ts, _ in aware)
    if len(offsets) > 1:
        modal = sorted(offsets.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        tz = timezone(modal)
        aware = [(ts.astimezone(tz), v) for ts, v in aware]
        series.notes.append(
            f"Timestamps carried {len(offsets)} different UTC offsets; all were converted to the "
            "most common offset before assigning calendar days."
        )
        series.suspect = True
    wall = [(ts.replace(tzinfo=None), v) for ts, v in aware]
    wall.sort(key=lambda p: p[0])  # chronological; stable for equal timestamps
    return wall


def _build_daily_series(records: Sequence[ConsumptionRecord], cfg: ForecastConfig) -> _DailySeries:
    series = _DailySeries()
    wall = _wall_clock_readings(records, series)
    series.last_record_date = wall[-1][0].date() if wall else None

    # Same timestamp: identical values collapse (exact duplicate); different values conflict.
    by_ts: Dict[datetime, List[float]] = defaultdict(list)
    for ts, v in wall:
        by_ts[ts].append(v)
    exact_dupes = sum(len(v) - 1 for v in by_ts.values() if len(set(v)) == 1)
    conflict_dates = {ts.date() for ts, v in by_ts.items() if len(set(v)) > 1}
    if exact_dupes:
        series.notes.append(f"{exact_dupes} exact duplicate reading(s) (same timestamp and value) were counted once.")

    per_day: Dict[date, List[Tuple[datetime, float]]] = defaultdict(list)
    for ts in sorted(by_ts):
        if ts.date() in conflict_dates:
            continue
        per_day[ts.date()].append((ts, by_ts[ts][0]))
    excluded: Dict[str, int] = {}
    if conflict_dates:
        excluded["conflicting_duplicate_readings"] = len(conflict_dates)

    if not per_day:
        series.excluded = excluded
        series.insufficient_reason = "no usable readings"
        return series

    counts = Counter(len(v) for v in per_day.values())
    top = max(counts.values())
    tied = [c for c, n in counts.items() if n == top]
    if len(tied) == 1:
        per_day_count = tied[0]
    else:  # tie between measurement regimes: follow the most recent day's regime
        latest = max(per_day)
        per_day_count = len(per_day[latest]) if len(per_day[latest]) in tied else max(tied)
        series.notes.append("Readings per day differ between periods; the most recent regime was used.")
    if len(counts) > 1:
        series.suspect = True
        series.notes.append(
            "Measurement frequency is not uniform (readings per day vary); only days that match "
            f"the dominant pattern ({per_day_count} per day) were used."
        )

    dates: List[date] = []
    values: List[float] = []
    if per_day_count == 1:
        series.granularity = "daily"
        multi = 0
        tods: List[int] = []
        for d in sorted(per_day):
            readings = per_day[d]
            if len(readings) != 1:
                multi += 1
                continue
            ts, v = readings[0]
            dates.append(d)
            values.append(v)
            tods.append(ts.hour * 3600 + ts.minute * 60 + ts.second)
        if multi:
            excluded["multiple_readings_on_daily_series"] = multi
        if len(dates) >= 2:
            gaps = np.diff([d.toordinal() for d in dates])
            if float(np.median(gaps)) >= 2:
                series.insufficient_reason = (
                    f"readings are coarser than daily (median spacing {float(np.median(gaps)):g} days); "
                    "a daily forecast would need disaggregation, which would fabricate data"
                )
        if tods and (max(tods) - min(tods)) > _TOD_VARIATION_NOTE_SECONDS:
            series.suspect = True
            series.notes.append(
                f"Time of day of the daily readings varies by {(max(tods) - min(tods)) / 3600:.1f} h; "
                "each reading is assumed to cover the calendar date it is stamped with."
            )
    else:
        if _SECONDS_PER_DAY % per_day_count != 0:
            series.excluded = excluded
            series.insufficient_reason = (
                f"{per_day_count} readings per day does not correspond to a regular cadence"
            )
            return series
        cadence = _SECONDS_PER_DAY // per_day_count
        tol = min(cfg.timestamp_tolerance_seconds, 0.25 * cadence)
        series.granularity = f"{per_day_count}_readings_per_day"
        incomplete = 0
        for d in sorted(per_day):
            readings = per_day[d]
            ok = len(readings) == per_day_count
            if ok:
                gaps_s = [
                    (readings[i + 1][0] - readings[i][0]).total_seconds()
                    for i in range(len(readings) - 1)
                ]
                ok = all(abs(g - cadence) <= tol for g in gaps_s)
            if not ok:
                incomplete += 1
                continue
            dates.append(d)
            values.append(float(sum(v for _, v in readings)))
        if incomplete:
            excluded["incomplete_or_irregular_day"] = incomplete

    series.dates, series.values, series.excluded = dates, values, excluded
    n_excl = sum(excluded.values())
    if n_excl:
        series.suspect = True
        detail = ", ".join(f"{k}: {n}" for k, n in sorted(excluded.items()))
        series.notes.append(
            f"{n_excl} day(s) excluded rather than estimated ({detail}). Daily totals are summed "
            "only for fully covered days; each reading is assumed to be the consumption of its own "
            "interval (not a cumulative meter value)."
        )
    return series


def _apply_window_and_gaps(series: _DailySeries, cfg: ForecastConfig) -> None:
    """Keep the most recent usable stretch: trailing window, then after the last long gap."""
    if not series.dates:
        return
    cutoff = series.dates[-1] - timedelta(days=cfg.history_window_days - 1)
    if series.dates[0] < cutoff:
        keep = [i for i, d in enumerate(series.dates) if d >= cutoff]
        dropped = len(series.dates) - len(keep)
        series.dates = [series.dates[i] for i in keep]
        series.values = [series.values[i] for i in keep]
        series.notes.append(
            f"Only the most recent {cfg.history_window_days} days were used ({dropped} older day(s) ignored)."
        )
    start = 0
    for i in range(1, len(series.dates)):
        if (series.dates[i] - series.dates[i - 1]).days - 1 > cfg.max_gap_days:
            start = i
    if start:
        gap = (series.dates[start] - series.dates[start - 1]).days - 1
        series.notes.append(
            f"A gap of {gap} missing days (> max_gap_days={cfg.max_gap_days}) was found; only data "
            f"after it ({series.dates[start].isoformat()} onward) was used."
        )
        series.suspect = True
        series.dates = series.dates[start:]
        series.values = series.values[start:]


# --------------------------------------------------------------------------- #
# Step 2: predictors. Each sees ONLY the history it is handed (array ends at the origin).
# --------------------------------------------------------------------------- #
def _trailing_mean(hist: np.ndarray, k: int = _TRAILING_MEAN_WINDOW) -> float:
    obs = hist[~np.isnan(hist)]
    return float(obs[-k:].mean()) if obs.size else float("nan")


def _predict_trailing_mean(hist: np.ndarray, horizon: int) -> np.ndarray:
    return np.full(horizon, _trailing_mean(hist), dtype=float)


def _predict_seasonal_naive(
    hist: np.ndarray, horizon: int, lookback_weeks: int
) -> Tuple[np.ndarray, int]:
    """Same weekday last week; older weeks if missing; trailing mean as last resort."""
    origin = hist.size - 1
    fallback = _trailing_mean(hist)
    out = np.empty(horizon, dtype=float)
    n_fallback = 0
    for h in range(1, horizon + 1):
        target = origin + h
        first_lag_weeks = -(-h // SEASONAL_PERIOD_DAYS)  # ceil(h / 7): newest same weekday <= origin
        value = float("nan")
        for w in range(lookback_weeks):
            idx = target - SEASONAL_PERIOD_DAYS * (first_lag_weeks + w)
            if idx < 0:
                break
            if not np.isnan(hist[idx]):
                value = float(hist[idx])
                break
        if np.isnan(value):
            value = fallback
            n_fallback += 1
        out[h - 1] = value
    return out, n_fallback


def _nanmean_or_nan(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


def _hgb_features(hist: np.ndarray, origin: int, h: int, first_dow: int) -> List[float]:
    """Features for target day ``origin + h``; reads hist[<= origin] only."""
    target = origin + h
    weeks = -(-h // SEASONAL_PERIOD_DAYS)
    lag1 = target - SEASONAL_PERIOD_DAYS * weeks
    lag2 = lag1 - SEASONAL_PERIOD_DAYS

    def at(i: int) -> float:
        return float(hist[i]) if 0 <= i <= origin else float("nan")

    return [
        float(h),
        float((first_dow + target) % 7),  # day of week of the target (calendar fact)
        at(lag1),
        at(lag2),
        at(origin),
        _nanmean_or_nan(hist[max(0, origin - 6): origin + 1]),
        _nanmean_or_nan(hist[max(0, origin - 27): origin + 1]),
    ]


def _import_hgb() -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor  # lazy: optional dependency

    return HistGradientBoostingRegressor


def _hgb_training_matrix(
    hist: np.ndarray, horizon: int, first_dow: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Training rows for every (origin p, step h) whose target ``hist[p + h]`` lies inside ``hist``.

    Identical to calling ``_hgb_features`` row by row, but the two rolling means are computed
    once per origin instead of once per (origin, step) row. Row ``(p, h)`` reads ``hist[<= p]``
    only, so no later value can leak into a feature.
    """
    origin = hist.size - 1
    rows: List[List[float]] = []
    ys: List[float] = []
    for p in range(_HGB_FIRST_TRAINING_ORIGIN, origin):
        mean7 = _nanmean_or_nan(hist[max(0, p - 6): p + 1])
        mean28 = _nanmean_or_nan(hist[max(0, p - 27): p + 1])
        last = float(hist[p])
        for h in range(1, min(horizon, origin - p) + 1):
            y = hist[p + h]
            if np.isnan(y):
                continue
            target = p + h
            lag1 = target - SEASONAL_PERIOD_DAYS * (-(-h // SEASONAL_PERIOD_DAYS))
            lag2 = lag1 - SEASONAL_PERIOD_DAYS
            rows.append([
                float(h),
                float((first_dow + target) % 7),
                float(hist[lag1]) if 0 <= lag1 <= p else float("nan"),
                float(hist[lag2]) if 0 <= lag2 <= p else float("nan"),
                last,
                mean7,
                mean28,
            ])
            ys.append(float(y))
    return np.asarray(rows, dtype=float).reshape(-1, 7), np.asarray(ys, dtype=float)


# Fits are deterministic functions of (history, horizon, weekday offset, seed). Re-fitting the
# same series (a retried API call, the same series in two requests) is pure waste, so recent
# results are memoised. The key is the full history content, so a different history can never
# hit another series' entry.
_HGB_CACHE_SIZE = 64
_hgb_cache: "OrderedDict[Tuple[bytes, int, int, int], Optional[np.ndarray]]" = OrderedDict()
_hgb_cache_lock = threading.Lock()


def clear_forecast_cache() -> None:
    """Drop memoised advanced-model results (mainly for tests and benchmarks)."""
    with _hgb_cache_lock:
        _hgb_cache.clear()


def _predict_hgb(hist: np.ndarray, horizon: int, first_dow: int, seed: int) -> Optional[np.ndarray]:
    """Fit on (origin, step) rows whose targets lie inside ``hist``; predict the next ``horizon`` days."""
    key = (np.ascontiguousarray(hist, dtype=float).tobytes(), int(horizon), int(first_dow), int(seed))
    with _hgb_cache_lock:
        if key in _hgb_cache:
            _hgb_cache.move_to_end(key)
            cached = _hgb_cache[key]
            return None if cached is None else cached.copy()

    origin = hist.size - 1
    x_train, y_train = _hgb_training_matrix(hist, horizon, first_dow)
    result: Optional[np.ndarray]
    if y_train.size < _HGB_MIN_TRAIN_ROWS:
        result = None
    else:
        model = _import_hgb()(categorical_features=[1], random_state=seed, **_HGB_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)
            x_pred = np.asarray(
                [_hgb_features(hist, origin, h, first_dow) for h in range(1, horizon + 1)], dtype=float
            )
            pred = model.predict(x_pred)
        result = np.clip(pred, 0.0, None)
    with _hgb_cache_lock:
        _hgb_cache[key] = None if result is None else result.copy()
        while len(_hgb_cache) > _HGB_CACHE_SIZE:
            _hgb_cache.popitem(last=False)
    return result


def _predict(
    method: ForecastMethod, hist: np.ndarray, horizon: int, first_dow: int, cfg: ForecastConfig
) -> Tuple[Optional[np.ndarray], int]:
    """Dispatch. Returns (predictions or None, seasonal-fallback count)."""
    if method is ForecastMethod.SEASONAL_NAIVE:
        return _predict_seasonal_naive(hist, horizon, cfg.seasonal_max_lookback_weeks)
    if method is ForecastMethod.TRAILING_MEAN:
        return _predict_trailing_mean(hist, horizon), 0
    return _predict_hgb(hist, horizon, first_dow, cfg.random_seed), 0


# --------------------------------------------------------------------------- #
# Step 3: rolling-origin backtesting
# --------------------------------------------------------------------------- #
def _choose_origins(n_cal: int, min_train_days: int, step: int, max_origins: int) -> List[int]:
    """Indices of the last training day, most recent first stepping back, returned ascending."""
    last = n_cal - 2  # an origin needs at least one later day to score
    first_ok = min_train_days - 1
    if last < first_ok:
        return []
    return sorted(list(range(last, first_ok - 1, -step))[:max_origins])


@dataclass
class _BacktestRun:
    method: ForecastMethod
    origins: List[int]
    # parallel arrays, one entry per scored (origin, step)
    steps: List[int] = field(default_factory=list)
    actual: List[float] = field(default_factory=list)
    predicted: List[float] = field(default_factory=list)
    origin_index: List[int] = field(default_factory=list)

    def abs_errors(self) -> np.ndarray:
        return np.abs(np.asarray(self.actual) - np.asarray(self.predicted))

    @property
    def n_points(self) -> int:
        return len(self.actual)

    @property
    def n_origins_scored(self) -> int:
        return len(set(self.origin_index))

    def mae(self) -> float:
        return float(self.abs_errors().mean()) if self.actual else float("nan")

    def wape(self) -> Optional[float]:
        total = float(np.sum(self.actual)) if self.actual else 0.0
        if total <= 0:
            return None
        return float(self.abs_errors().sum() / total)


def _run_backtest(
    cal: np.ndarray, first_dow: int, method: ForecastMethod, origins: Sequence[int],
    horizon: int, cfg: ForecastConfig,
) -> _BacktestRun:
    run = _BacktestRun(method=method, origins=list(origins))
    for o in origins:
        hist = cal[: o + 1].copy()  # the predictor physically cannot see later days
        pred, _ = _predict(method, hist, horizon, first_dow, cfg)
        if pred is None:
            continue
        for h in range(1, horizon + 1):
            if o + h >= cal.size:
                break
            actual = cal[o + h]
            if np.isnan(actual) or np.isnan(pred[h - 1]):
                continue
            run.steps.append(h)
            run.actual.append(float(actual))
            run.predicted.append(float(pred[h - 1]))
            run.origin_index.append(o)
    return run


def rolling_origin_backtest(
    values: Sequence[float],
    *,
    method: Union[ForecastMethod, str] = ForecastMethod.SEASONAL_NAIVE,
    horizon_days: int = 7,
    first_weekday: int = 0,
    config: Optional[ForecastConfig] = None,
    step_days: int = 1,
    max_origins: Optional[int] = None,
) -> Dict[str, Any]:
    """Public helper: rolling-origin (expanding window) backtest of one method on a daily series.

    ``values`` is a calendar-day sequence (``nan`` for missing days), ``first_weekday``
    the weekday (Monday=0) of ``values[0]``. Returns MAE, WAPE (``None`` if the actuals
    sum to zero) and the scored points. At each origin the method only receives
    ``values[: origin + 1]``.
    """
    cfg = config or ForecastConfig()
    m = ForecastMethod(method)
    cal = np.asarray(values, dtype=float)
    min_train = {
        ForecastMethod.SEASONAL_NAIVE: _MIN_TRAIN_DAYS_SEASONAL,
        ForecastMethod.TRAILING_MEAN: _MIN_TRAIN_DAYS_TRAILING,
        ForecastMethod.HIST_GRADIENT_BOOSTING: _MIN_TRAIN_DAYS_HGB,
    }[m]
    origins = _choose_origins(cal.size, min_train, step_days, max_origins or cfg.backtest_max_origins)
    run = _run_backtest(cal, first_weekday, m, origins, horizon_days, cfg)
    return {
        "method": m.value,
        "origins": run.origins,
        "n_points": run.n_points,
        "mae": None if not run.n_points else run.mae(),
        "wape": run.wape(),
        "points": [
            {"origin": o, "step": s, "actual": a, "predicted": p}
            for o, s, a, p in zip(run.origin_index, run.steps, run.actual, run.predicted)
        ],
    }


# --------------------------------------------------------------------------- #
# Step 4: uncertainty
# --------------------------------------------------------------------------- #
def _conformal_quantile(abs_res: np.ndarray, level: float) -> Tuple[float, bool]:
    """ceil((n+1)*level)-th smallest absolute residual; ``True`` flag if n was too small."""
    n = int(abs_res.size)
    k = math.ceil((n + 1) * level - 1e-9)
    srt = np.sort(abs_res)
    if k <= n:
        return float(srt[k - 1]), False
    return float(srt[-1]), True


def _interval_half_widths(
    run: _BacktestRun, horizon: int, cfg: ForecastConfig
) -> Optional[Tuple[np.ndarray, int, int, bool]]:
    """Per-step half-widths from backtest residuals.

    Returns (half_widths, n_pooled_steps, n_residuals, limited_sample) or ``None`` when
    there are fewer than ``min_residuals_for_interval`` residuals overall.
    """
    abs_err = run.abs_errors()
    if abs_err.size < cfg.min_residuals_for_interval:
        return None
    steps = np.asarray(run.steps)
    pooled_q, pooled_limited = _conformal_quantile(abs_err, cfg.interval_level)
    widths = np.empty(horizon, dtype=float)
    n_pooled = 0
    limited = pooled_limited
    for h in range(1, horizon + 1):
        sel = abs_err[steps == h]
        if sel.size >= cfg.min_residuals_for_interval:
            q, lim = _conformal_quantile(sel, cfg.interval_level)
            limited = limited or lim
        else:
            q = pooled_q
            n_pooled += 1
        widths[h - 1] = q
    return widths, n_pooled, int(abs_err.size), limited


# --------------------------------------------------------------------------- #
# Step 5: one series end to end
# --------------------------------------------------------------------------- #
def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def _metrics(run: _BacktestRun, baseline: ForecastMethod, selected: ForecastMethod) -> MethodMetrics:
    return MethodMetrics(
        method=run.method,
        is_baseline=run.method is baseline,
        is_selected=run.method is selected,
        mae=_r(run.mae(), 6),
        wape=None if run.wape() is None else _r(run.wape(), 6),
        n_origins=run.n_origins_scored,
        n_points=run.n_points,
    )


def _insufficient(
    building: str, resource: ResourceType, horizon: int, history: HistorySummary,
    reasons: List[str], notes: List[str],
) -> ForecastResult:
    return ForecastResult(
        building=building,
        resource_type=resource,
        unit=STANDARD_UNITS[resource],
        forecast_available=False,
        forecast_horizon_days=horizon,
        data_quality_status=DataQualityStatus.INSUFFICIENT_DATA,
        confidence_notes=[f"No forecast produced: {r}." for r in reasons] + notes,
        history=history,
        evaluation=None,
    )


def _forecast_group(
    building: str, resource: ResourceType, group: Sequence[ConsumptionRecord],
    horizon: int, cfg: ForecastConfig,
) -> ForecastResult:
    series = _build_daily_series(group, cfg)
    if series.insufficient_reason is None:
        _apply_window_and_gaps(series, cfg)

    n_obs = len(series.dates)
    n_excluded = sum(series.excluded.values())
    if n_obs:
        span = (series.dates[-1] - series.dates[0]).days + 1
        coverage = n_obs / span
    else:
        span, coverage = 0, None
    history = HistorySummary(
        records_received=len(group),
        first_date=series.dates[0] if n_obs else None,
        last_date=series.dates[-1] if n_obs else None,
        observed_days=n_obs,
        span_days=span,
        missing_days=span - n_obs,
        excluded_days=n_excluded,
        coverage=None if coverage is None else _r(coverage, 4),
        granularity=series.granularity,
    )

    if series.insufficient_reason:
        return _insufficient(building, resource, horizon, history, [series.insufficient_reason], series.notes)
    if n_obs < cfg.minimum_history_days:
        return _insufficient(
            building, resource, horizon, history,
            [f"only {n_obs} usable day(s) of history; at least {cfg.minimum_history_days} are required"],
            series.notes,
        )
    if coverage is not None and coverage < cfg.min_coverage:
        return _insufficient(
            building, resource, horizon, history,
            [f"only {coverage:.0%} of calendar days are observed (min_coverage={cfg.min_coverage:.0%}); "
             "missing days are never imputed"],
            series.notes,
        )

    # Calendar array: NaN marks days with no usable reading (kept missing, not filled).
    first = series.dates[0]
    cal = np.full(span, np.nan, dtype=float)
    for d, v in zip(series.dates, series.values):
        cal[(d - first).days] = v
    first_dow = first.weekday()
    notes = list(series.notes)
    suspect = series.suspect

    missing = span - n_obs
    if missing:
        notes.append(
            f"{missing} missing day(s) inside the history were left missing (not imputed); "
            "forecasts use only observed days."
        )
        if missing / span > _MISSING_SUSPECT_FRACTION:
            suspect = True
    zero_share = float(np.mean(np.asarray(series.values) == 0.0))
    if zero_share >= _ZERO_SHARE_SUSPECT:
        suspect = True
        notes.append(
            f"{zero_share:.0%} of observed days are exactly zero"
            + (" (constant zero history: a possible meter outage or unused building)." if zero_share == 1.0 else ".")
        )
    elif zero_share > 0:
        notes.append(f"{int((zero_share * n_obs) + 0.5)} observed day(s) have zero consumption; zeros were kept as real readings.")

    baseline = (
        ForecastMethod.SEASONAL_NAIVE if n_obs >= 2 * SEASONAL_PERIOD_DAYS else ForecastMethod.TRAILING_MEAN
    )
    base_min_train = (
        _MIN_TRAIN_DAYS_SEASONAL if baseline is ForecastMethod.SEASONAL_NAIVE else _MIN_TRAIN_DAYS_TRAILING
    )
    if baseline is ForecastMethod.TRAILING_MEAN:
        notes.append(
            "Fewer than two full weeks of history: weekly seasonality cannot be verified, so the "
            "trailing mean is used as the baseline."
        )

    # --- baseline backtest (cheap; used for intervals if the baseline is selected) -------- #
    base_origins = _choose_origins(cal.size, base_min_train, 1, cfg.backtest_max_origins)
    base_run = _run_backtest(cal, first_dow, baseline, base_origins, horizon, cfg)

    selected = baseline
    selected_run = base_run
    metrics: List[MethodMetrics] = []
    advanced_considered = False
    selection_reason = f"{baseline.value} baseline used (advanced model not considered)."
    backtest_performed = base_run.n_points > 0
    eval_origins, eval_points = base_run.n_origins_scored, base_run.n_points

    sklearn_ok = True
    if cfg.advanced_model_enabled and n_obs >= cfg.advanced_min_days:
        try:
            _import_hgb()
        except ImportError:
            sklearn_ok = False
            selection_reason = (
                f"{baseline.value} baseline used (scikit-learn is not installed; advanced model skipped)."
            )
            notes.append("scikit-learn is not installed, so the advanced model was not evaluated.")
        if sklearn_ok:
            advanced_considered = True
            adv_origins = _choose_origins(
                cal.size, _MIN_TRAIN_DAYS_HGB, cfg.advanced_backtest_step_days, cfg.advanced_backtest_origins
            )
            adv_base = _run_backtest(cal, first_dow, baseline, adv_origins, horizon, cfg)
            try:
                adv_run = _run_backtest(cal, first_dow, ForecastMethod.HIST_GRADIENT_BOOSTING, adv_origins, horizon, cfg)
                adv_failed = False
            except Exception as exc:  # noqa: BLE001 - optional model must never break the baseline
                adv_run, adv_failed = None, True
                notes.append(f"Advanced model failed during backtesting ({type(exc).__name__}); baseline used.")
            enough = (
                adv_run is not None
                and adv_run.n_origins_scored >= _MIN_ADVANCED_BACKTEST_ORIGINS
                and adv_run.n_points >= cfg.min_residuals_for_interval
                and adv_base.n_points > 0
            )
            if adv_failed:
                selection_reason = f"{baseline.value} baseline used (advanced model failed during backtesting)."
            elif not enough:
                selection_reason = (
                    f"{baseline.value} baseline used (too few backtest origins to validate the advanced model)."
                )
            else:
                assert adv_run is not None
                mae_b, mae_a = adv_base.mae(), adv_run.mae()
                needed = (1.0 - cfg.advanced_min_relative_improvement) * mae_b
                if mae_b > 0 and mae_a <= needed:
                    selected, selected_run = ForecastMethod.HIST_GRADIENT_BOOSTING, adv_run
                    selection_reason = (
                        f"hist_gradient_boosting selected: backtest MAE {mae_a:.4g} vs {mae_b:.4g} for the "
                        f"{baseline.value} baseline ({1 - mae_a / mae_b:.0%} lower; required "
                        f"{cfg.advanced_min_relative_improvement:.0%}) on {adv_run.n_origins_scored} origins."
                    )
                else:
                    selection_reason = (
                        f"{baseline.value} baseline kept: advanced backtest MAE {mae_a:.4g} vs baseline "
                        f"{mae_b:.4g} does not meet the required {cfg.advanced_min_relative_improvement:.0%} "
                        "improvement."
                    )
                metrics = [_metrics(adv_base, baseline, selected), _metrics(adv_run, baseline, selected)]
                eval_origins, eval_points = adv_run.n_origins_scored, adv_run.n_points
    elif not cfg.advanced_model_enabled:
        selection_reason = f"{baseline.value} baseline used (advanced model disabled in configuration)."
    else:
        selection_reason = (
            f"{baseline.value} baseline used: {n_obs} observed days is below advanced_min_days="
            f"{cfg.advanced_min_days}, too little data for a gradient-boosting model."
        )
    if not metrics and base_run.n_points:
        metrics = [_metrics(base_run, baseline, selected)]

    # --- intervals must exist, otherwise the forecast is not defensible --------------------- #
    widths_info = _interval_half_widths(selected_run, horizon, cfg)
    if widths_info is None:
        return _insufficient(
            building, resource, horizon, history,
            [f"only {selected_run.n_points} backtest residual(s) are available; at least "
             f"{cfg.min_residuals_for_interval} are needed for a documented prediction interval"],
            notes,
        )
    widths, n_pooled_steps, n_resid, limited = widths_info

    # --- final forecast from the full history ---------------------------------------------- #
    n_fallback = 0
    pred: Optional[np.ndarray] = None
    if selected is ForecastMethod.HIST_GRADIENT_BOOSTING:
        try:
            pred, _ = _predict(selected, cal.copy(), horizon, first_dow, cfg)
        except Exception as exc:  # noqa: BLE001
            pred = None
            notes.append(f"Advanced model failed at forecast time ({type(exc).__name__}); baseline used.")
        if pred is None:
            selected, selected_run = baseline, base_run
            selection_reason += " Fell back to the baseline because the final model fit failed."
            widths_info = _interval_half_widths(base_run, horizon, cfg)
            if widths_info is None:
                return _insufficient(building, resource, horizon, history,
                                     ["no defensible prediction interval after falling back to the baseline"], notes)
            widths, n_pooled_steps, n_resid, limited = widths_info
            for m in metrics:
                m.is_selected = m.method is baseline
    if pred is None:
        pred, n_fallback = _predict(selected, cal.copy(), horizon, first_dow, cfg)
    assert pred is not None
    if n_fallback:
        notes.append(
            f"{n_fallback} forecast day(s) had no same-weekday observation within "
            f"{cfg.seasonal_max_lookback_weeks} weeks; the trailing mean of the last observed days was used for them."
        )
    pred = np.clip(pred, 0.0, None)

    start = series.dates[-1] + timedelta(days=1)
    dates = [start + timedelta(days=i) for i in range(horizon)]
    values = [_r(v) for v in pred]
    lower = [_r(max(0.0, v - w)) for v, w in zip(pred, widths)]
    upper = [_r(v + w) for v, w in zip(pred, widths)]

    if series.last_record_date and series.last_record_date > series.dates[-1]:
        notes.append(
            f"The latest day with readings ({series.last_record_date.isoformat()}) is not fully covered, so "
            f"the forecast starts after the last complete day ({series.dates[-1].isoformat()})."
        )

    level_pct = f"{cfg.interval_level:.0%}"
    notes.append(
        f"{level_pct} prediction interval from {n_resid} backtest residual(s) of {selected.value} "
        f"(rolling-origin, expanding window); coverage is empirical and not guaranteed."
    )
    if n_pooled_steps:
        notes.append(
            f"{n_pooled_steps} of {horizon} forecast step(s) had too few residuals of their own, so "
            "residuals pooled over all steps were used for them."
        )
    if limited:
        notes.append(
            f"Too few residuals for a full {level_pct} guarantee; the largest observed error was used as the bound."
        )
    if n_resid and float(np.max(widths)) == 0.0:
        notes.append("All backtest errors were zero, so the bounds equal the point forecast.")
    max_step = int(max(selected_run.steps)) if selected_run.steps else 0
    if horizon > max_step:
        notes.append(f"Horizon ({horizon} days) exceeds the longest backtested lead ({max_step} days); later steps are less reliable.")
    notes.append(f"Model selection: {selection_reason}")

    if selected is ForecastMethod.HIST_GRADIENT_BOOSTING:
        notes.append("Tree models cannot extrapolate beyond the range of training data (e.g. strong trends).")

    quality = DataQualityStatus.SUSPECT if suspect else DataQualityStatus.OK
    evaluation = ForecastEvaluation(
        backtest_performed=backtest_performed,
        scheme="rolling-origin, expanding window; each origin sees only earlier days",
        advanced_model_considered=advanced_considered,
        metrics=metrics,
        selection_reason=selection_reason,
        interval_residuals_used=n_resid,
    )
    return ForecastResult(
        building=building,
        resource_type=resource,
        unit=STANDARD_UNITS[resource],
        forecast_available=True,
        forecast_start_date=start,
        forecast_horizon_days=horizon,
        forecast_dates=dates,
        predicted_values=values,
        predicted_total=_r(sum(values)),
        baseline_method=baseline,
        model_used=selected,
        lower_bound=lower,
        upper_bound=upper,
        predicted_total_lower_bound=_r(sum(lower)),
        predicted_total_upper_bound=_r(sum(upper)),
        interval_level=cfg.interval_level,
        uncertainty_method=UNCERTAINTY_METHOD,
        data_quality_status=quality,
        confidence_notes=notes,
        history=history,
        evaluation=evaluation,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def forecast_consumption(
    records: Sequence[Union[ConsumptionRecord, Mapping[str, Any]]],
    horizon_days: int = 7,
    config: Optional[ForecastConfig] = None,
) -> List[ForecastResult]:
    """Forecast the next ``horizon_days`` days for every (building, resource) series.

    Args:
        records: ``ConsumptionRecord`` objects or dicts, any order, any mix of buildings and
            resources. Validated exactly like ``analyze_consumption``.
        horizon_days: 1 (next day) to ``config.max_horizon_days`` (7 by default).
        config: Optional ``ForecastConfig``.

    Returns:
        One ``ForecastResult`` per (building, resource_type), sorted by building then
        resource. A series without enough defensible history gets
        ``data_quality_status="insufficient_data"``, ``forecast_available=False`` and
        empty forecast lists; nothing is estimated for it.

    Raises:
        RecordValidationError: if any record is invalid (no partial output).
        TypeError / ValueError: wrong ``records``/``config``/``horizon_days`` types or range.
    """
    if config is not None and not isinstance(config, ForecastConfig):
        raise TypeError("config must be a ForecastConfig or None")
    cfg = config or ForecastConfig()
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int):
        raise TypeError("horizon_days must be an integer")
    if not 1 <= horizon_days <= cfg.max_horizon_days:
        raise ValueError(f"horizon_days must be between 1 and {cfg.max_horizon_days}")
    validated = _coerce_records(records)

    groups: Dict[Tuple[str, ResourceType], List[ConsumptionRecord]] = defaultdict(list)
    for r in validated:
        groups[(r.building, r.resource_type)].append(r)
    results: List[ForecastResult] = []
    for (building, resource) in sorted(groups, key=lambda k: (k[0], k[1].value)):
        results.append(_forecast_group(building, resource, groups[(building, resource)], horizon_days, cfg))
    return results


def serialize_forecast(result: ForecastResult) -> Dict[str, Any]:
    """JSON-safe dict (dates -> ``YYYY-MM-DD``, enums -> strings, ``None`` -> ``null``)."""
    return result.model_dump(mode="json")


def serialize_forecasts(results: Sequence[ForecastResult]) -> List[Dict[str, Any]]:
    return [serialize_forecast(r) for r in results]


# --------------------------------------------------------------------------- #
# JSON facade for the backend (same envelope style as analyze_payload)
# --------------------------------------------------------------------------- #
_FORECAST_REQUEST_KEYS = ("records", "options")
_FORECAST_OPTION_KEYS = ("horizon_days",)


def forecast_payload(
    payload: Any,
    config: Optional[ForecastConfig] = None,
    *,
    max_records: Optional[int] = None,
) -> Any:
    """Decoded JSON request body -> ``AnalysisOutcome`` (never raises for client mistakes).

    Request: ``{"records": [...], "options": {"horizon_days": 7}}``. Errors use the
    standard envelope (HTTP 422/413/500). See ``docs/JSON_CONTRACT.md`` section 6.
    """
    from .contract import DEFAULT_MAX_RECORDS, AnalysisOutcome, _detail, error_outcome, validation_error_body

    if config is not None and not isinstance(config, ForecastConfig):
        raise TypeError("config must be a ForecastConfig or None")
    cfg = config or ForecastConfig()
    limit = DEFAULT_MAX_RECORDS if max_records is None else max_records

    if not isinstance(payload, Mapping):
        return error_outcome(422, "invalid_request", "Request body must be a JSON object.",
                             [_detail([], "must be an object")])
    unknown = sorted(str(k) for k in payload if k not in _FORECAST_REQUEST_KEYS)
    if unknown:
        return error_outcome(
            422, "invalid_request",
            f"Unknown field(s): {', '.join(unknown)}. Allowed: {', '.join(_FORECAST_REQUEST_KEYS)}.",
            [_detail([k], "unknown field") for k in unknown],
        )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return error_outcome(422, "invalid_request", "records must be a non-empty JSON array.",
                             [_detail(["records"], "must be a non-empty array")])
    if len(records) > limit:
        return error_outcome(413, "too_many_records",
                             f"At most {limit} records per request; received {len(records)}.",
                             [_detail(["records"], f"maximum {limit} items")])
    options = payload.get("options")
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        return error_outcome(422, "invalid_options", "options must be a JSON object.",
                             [_detail(["options"], "must be an object")])
    bad = sorted(str(k) for k in options if k not in _FORECAST_OPTION_KEYS)
    if bad:
        return error_outcome(
            422, "invalid_options",
            f"Unknown option(s): {', '.join(bad)}. Allowed: {', '.join(_FORECAST_OPTION_KEYS)}.",
            [_detail(["options", k], "unknown option") for k in bad],
        )
    horizon = options.get("horizon_days", 7)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= cfg.max_horizon_days:
        return error_outcome(
            422, "invalid_options",
            f"options.horizon_days must be an integer between 1 and {cfg.max_horizon_days}.",
            [_detail(["options", "horizon_days"], f"must be an integer in 1..{cfg.max_horizon_days}")],
        )
    try:
        results = forecast_consumption(records, horizon, cfg)
    except RecordValidationError as exc:
        return error_outcome(
            422, "invalid_records",
            f"{len(exc.errors)} record(s) failed validation; no forecast was produced.",
            validation_error_body(exc),
        )
    except Exception:  # noqa: BLE001 - never leak internals to clients
        return error_outcome(500, "forecast_failed", "The forecast could not be completed.")

    status_counts = Counter(r.data_quality_status.value for r in results)
    model_counts = Counter(r.model_used.value for r in results if r.model_used is not None)
    body: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "forecast_contract_version": FORECAST_CONTRACT_VERSION,
        "status": "ok",
        "engine_version": __version__,
        "summary": {
            "series_count": len(results),
            "forecasts_produced": sum(1 for r in results if r.forecast_available),
            "data_quality_counts": {s.value: status_counts.get(s.value, 0) for s in DataQualityStatus},
            "model_counts": {m.value: model_counts.get(m.value, 0) for m in ForecastMethod},
        },
        "forecasts": serialize_forecasts(results),
    }
    return AnalysisOutcome(ok=True, http_status=200, body=body)
