"""What-if sustainability simulator (Phase 7A).

``simulate_scenario(records, scenario, config=None)`` answers one question:

    "If this building (or the whole campus) cut this resource by X %, how much
    would that save, relative to its recent consumption?"

It is deterministic, offline and additive: it does not change detection,
impact, recommendation or forecasting behaviour.

Formulas (the only arithmetic the simulator performs)
-----------------------------------------------------
::

    estimated_savings       = baseline x reduction_percentage / 100
    projected_consumption   = baseline - estimated_savings
    estimated_co2e_savings  = estimated_savings x emission factor   (only if configured)

Savings are never negative and never exceed the baseline (the percentage is
limited to 0-100).

Hypothetical, not measured
--------------------------
Every result is a *hypothetical projection*. It assumes the stated reduction
happens; it does not confirm that any reduction occurred. ``projection_type`` is
always ``"hypothetical"`` and ``measured_savings`` is always ``null``. No
monetary figure is ever produced.

Baseline
--------
Either supplied explicitly (``baseline_consumption`` with value, unit and
period) or measured from the records:

* the records for the resource (and building, unless campus-wide) are summed
  over the ``baseline_period_days`` calendar days (UTC) that end on the most
  recent matching reading (not on today's date, so results are reproducible);
* gaps are never filled or extrapolated: if only part of the window has
  readings, the baseline is the sum of what exists, the result is marked
  ``suspect`` and says how much of the window was covered; below
  ``MIN_COVERAGE_FRACTION`` no baseline is reported at all;
* identical duplicate readings count once; conflicting readings for the same
  building and timestamp are excluded and reported;
* units are the engine's standard units (kWh, liters, kg). Records go through
  ``ConsumptionRecord`` so incompatible units are rejected, never converted;
  an explicit baseline in another unit or period is rejected, never rescaled.

CO2e uses the same emission-factor configuration as ``analyze_consumption``
(``AnalysisConfig``). No factor is built in: without one the CO2e field is
``null`` and a warning says so.

Errors
------
A malformed scenario (reduction outside 0-100, unknown resource, bad baseline,
unknown keys...) raises ``ScenarioValidationError``; invalid records raise the
existing ``RecordValidationError``. Problems with the *data* (empty dataset,
unknown building, no records for the resource, too little coverage) are not
exceptions: the result has ``calculation_status = "not_calculated"``, a
``not_calculated_reason``, no numbers, and explanatory warnings.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pydantic import ValidationError

from ._version import CONTRACT_VERSION, SIMULATION_CONTRACT_VERSION, __version__
from .analyzer import RecordValidationError
from .anomaly_detector import _utc
from .config import STANDARD_UNITS, UNIT_ALIASES, AnalysisConfig, ResourceType
from .contract import AnalysisOutcome, error_outcome, validation_error_body
from .forecasting_engine import ForecastConfig, forecast_consumption
from .impact_calculator import _co2e_factor
from .schemas import ConsumptionRecord, DataQualityStatus

__all__ = [
    "DEFAULT_BASELINE_PERIOD_DAYS",
    "MAX_BASELINE_PERIOD_DAYS",
    "MIN_COVERAGE_FRACTION",
    "SCENARIO_KEYS",
    "ScenarioValidationError",
    "compare_payload",
    "compare_scenarios",
    "simulate_payload",
    "simulate_scenario",
]

_ROUND = 6

# Engineering defaults, not values derived from campus data.
DEFAULT_BASELINE_PERIOD_DAYS = 30
MAX_BASELINE_PERIOD_DAYS = 366
# Share of the baseline window's days that must contain at least one reading
# before a measured baseline is reported (placeholder; tune against real data).
MIN_COVERAGE_FRACTION = 0.8

SCENARIO_KEYS = (
    "resource_type",
    "building_code",
    "reduction_percentage",
    "baseline_period_days",
    "baseline_consumption",
    "scenario_id",
    "baseline_source",
    "forecast_horizon_days",
    "forecast_config",
)
_BASELINE_KEYS = ("value", "unit", "period_days")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
_MAX_NAMES_LISTED = 5


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ScenarioValidationError(ValueError):
    """Raised when the scenario definition is invalid.

    ``errors`` is a list of ``{"loc": [field, ...], "msg": str}``, one per
    problem (all problems are reported together). Nothing is simulated.
    """

    def __init__(self, errors: List[Dict[str, Any]]) -> None:
        self.errors = errors
        summary = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or '<scenario>'}: {e['msg']}" for e in errors
        )
        super().__init__(f"{len(errors)} invalid scenario field(s): {summary}")


# --------------------------------------------------------------------------- #
# Scenario parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Explicit:
    value: float
    unit: str
    period_days: int


@dataclass(frozen=True)
class _Spec:
    resource: ResourceType
    building: Optional[str]  # None -> campus-wide
    reduction: float
    period_days: int
    explicit: Optional[_Explicit]
    scenario_id: str
    baseline_source: str = "historical"
    forecast_horizon_days: Optional[int] = None
    forecast_config: Optional[ForecastConfig] = None


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _as_days(v: Any) -> Optional[int]:
    """A whole number of days in 1..MAX_BASELINE_PERIOD_DAYS, else ``None``."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        n = v
    elif isinstance(v, float) and math.isfinite(v) and v.is_integer():
        n = int(v)
    else:
        return None
    return n if 1 <= n <= MAX_BASELINE_PERIOD_DAYS else None


_DAYS_MSG = f"must be a whole number of days between 1 and {MAX_BASELINE_PERIOD_DAYS}"


def _make_scenario_id(
    resource: ResourceType,
    building: Optional[str],
    reduction: float,
    period_days: int,
    explicit: Optional[_Explicit],
    baseline_source: str = "historical",
) -> str:
    """Deterministic ID: the same scenario definition always gets the same ID."""
    canonical: Dict[str, Any] = {
        "resource_type": resource.value,
        "building_code": building.casefold() if building else None,
        "reduction_percentage": float(reduction),
        "baseline_period_days": period_days,
        "baseline_consumption": (
            None
            if explicit is None
            else {"value": float(explicit.value), "unit": explicit.unit, "period_days": explicit.period_days}
        ),
    }
    if baseline_source == "forecast":
        canonical["baseline_source"] = baseline_source
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"scn-{digest[:12]}"


def _parse_scenario(scenario: Mapping[str, Any]) -> _Spec:
    errors: List[Dict[str, Any]] = []

    def err(loc: Sequence[Any], msg: str) -> None:
        errors.append({"loc": list(loc), "msg": msg})

    for key in sorted(str(k) for k in scenario if k not in SCENARIO_KEYS):
        err([key], f"unknown field; allowed: {', '.join(SCENARIO_KEYS)}")

    # resource_type ----------------------------------------------------------
    resource: Optional[ResourceType] = None
    raw_resource = scenario.get("resource_type")
    if raw_resource is None:
        err(["resource_type"], "required")
    else:
        candidate = raw_resource.strip().lower() if isinstance(raw_resource, str) else raw_resource
        try:
            resource = ResourceType(candidate)
        except ValueError:
            allowed = ", ".join(r.value for r in ResourceType)
            err(["resource_type"], f"unsupported resource_type {raw_resource!r}; supported: {allowed}")

    # building_code ----------------------------------------------------------
    building: Optional[str] = None
    raw_building = scenario.get("building_code")
    if raw_building is not None:
        if not isinstance(raw_building, str) or not raw_building.strip():
            err(["building_code"], "must be a non-empty string, or omitted/null for a campus-wide scenario")
        else:
            building = raw_building.strip()

    # reduction_percentage ---------------------------------------------------
    reduction = 0.0
    raw_reduction = scenario.get("reduction_percentage")
    if raw_reduction is None:
        err(["reduction_percentage"], "required")
    elif not _is_number(raw_reduction):
        err(["reduction_percentage"], "must be a finite number (not a string or boolean)")
    elif not 0 <= raw_reduction <= 100:
        err(["reduction_percentage"], f"must be between 0 and 100 inclusive; got {raw_reduction}")
    else:
        reduction = float(raw_reduction)

    # baseline_period_days ---------------------------------------------------
    given_days: Optional[int] = None
    raw_days = scenario.get("baseline_period_days")
    if raw_days is not None:
        given_days = _as_days(raw_days)
        if given_days is None:
            err(["baseline_period_days"], _DAYS_MSG)

    # baseline_consumption (explicit baseline) -------------------------------
    explicit: Optional[_Explicit] = None
    raw_explicit = scenario.get("baseline_consumption")
    if raw_explicit is not None:
        if not isinstance(raw_explicit, Mapping):
            err(["baseline_consumption"], "must be an object with value, unit and period_days")
        else:
            for key in sorted(str(k) for k in raw_explicit if k not in _BASELINE_KEYS):
                err(["baseline_consumption", key], f"unknown field; allowed: {', '.join(_BASELINE_KEYS)}")
            missing = [k for k in _BASELINE_KEYS if raw_explicit.get(k) is None]
            for key in missing:
                err(["baseline_consumption", key], "required (value, unit and period_days are all needed so units and periods are never guessed)")
            if not missing:
                value, unit, period = (raw_explicit[k] for k in _BASELINE_KEYS)
                ok = True
                if not _is_number(value) or value < 0:
                    err(["baseline_consumption", "value"], "must be a finite number >= 0")
                    ok = False
                canonical_unit: Optional[str] = None
                if resource is not None:
                    if isinstance(unit, str):
                        canonical_unit = UNIT_ALIASES[resource].get(unit.strip().lower())
                    if canonical_unit is None:
                        err(
                            ["baseline_consumption", "unit"],
                            f"incompatible unit {unit!r} for {resource.value}; "
                            f"expected {STANDARD_UNITS[resource]!r} (units are never converted)",
                        )
                        ok = False
                days = _as_days(period)
                if days is None:
                    err(["baseline_consumption", "period_days"], _DAYS_MSG)
                    ok = False
                if ok and canonical_unit is not None and days is not None:
                    explicit = _Explicit(float(value), canonical_unit, days)
                    if given_days is not None and given_days != days:
                        err(
                            ["baseline_period_days"],
                            f"conflicts with baseline_consumption.period_days ({days}); periods are never rescaled",
                        )

    # baseline_source --------------------------------------------------------
    baseline_source = "historical"
    raw_source = scenario.get("baseline_source")
    if raw_source is not None:
        if not isinstance(raw_source, str) or not raw_source.strip():
            err(["baseline_source"], "must be a string: 'historical', 'forecast', or 'explicit'")
        else:
            cand_source = raw_source.strip().lower()
            if cand_source in ("historical", "measured_records", "measured"):
                baseline_source = "historical"
            elif cand_source in ("forecast", "forecasting", "predictive"):
                baseline_source = "forecast"
            elif cand_source in ("explicit",):
                baseline_source = "explicit"
            else:
                err(["baseline_source"], f"unsupported baseline_source {raw_source!r}; supported: 'historical', 'forecast', 'explicit'")
    elif explicit is not None:
        baseline_source = "explicit"

    if baseline_source == "forecast" and (explicit is not None or raw_explicit is not None):
        err(["baseline_source"], "baseline_consumption cannot be provided when baseline_source is 'forecast'")
    if baseline_source == "explicit" and explicit is None and raw_explicit is None:
        err(["baseline_source"], "baseline_consumption must be provided when baseline_source is 'explicit'")

    # forecast_horizon_days --------------------------------------------------
    forecast_horizon_days: Optional[int] = None
    raw_f_days = scenario.get("forecast_horizon_days")
    if raw_f_days is not None:
        forecast_horizon_days = _as_days(raw_f_days)
        if forecast_horizon_days is None:
            err(["forecast_horizon_days"], _DAYS_MSG)

    # forecast_config --------------------------------------------------------
    forecast_cfg: Optional[ForecastConfig] = None
    raw_f_cfg = scenario.get("forecast_config")
    if raw_f_cfg is not None:
        if not isinstance(raw_f_cfg, Mapping):
            err(["forecast_config"], "must be an object containing ForecastConfig settings")
        else:
            try:
                forecast_cfg = ForecastConfig(**dict(raw_f_cfg))
            except (ValidationError, ValueError, TypeError) as exc:
                err(["forecast_config"], f"invalid ForecastConfig: {exc}")

    # scenario_id ------------------------------------------------------------
    custom_id: Optional[str] = None
    raw_id = scenario.get("scenario_id")
    if raw_id is not None:
        if isinstance(raw_id, str) and _ID_PATTERN.match(raw_id.strip()):
            custom_id = raw_id.strip()
        else:
            err(["scenario_id"], "must be 1-64 characters: letters, digits, '_', '.', '-' (starting with a letter or digit)")

    if errors:
        raise ScenarioValidationError(errors)
    assert resource is not None

    if explicit is not None:
        period_days = explicit.period_days
    elif baseline_source == "forecast":
        if forecast_horizon_days is not None:
            period_days = forecast_horizon_days
        elif given_days is not None:
            period_days = given_days
        else:
            period_days = 7
    else:
        period_days = given_days or DEFAULT_BASELINE_PERIOD_DAYS

    scenario_id = custom_id or _make_scenario_id(
        resource, building, reduction, period_days, explicit, baseline_source
    )
    return _Spec(
        resource,
        building,
        reduction,
        period_days,
        explicit,
        scenario_id,
        baseline_source,
        forecast_horizon_days,
        forecast_cfg,
    )


# --------------------------------------------------------------------------- #
# Record validation (same rules and error type as analyze_consumption)
# --------------------------------------------------------------------------- #
def _validate_records(records: Any) -> List[ConsumptionRecord]:
    if records is None or isinstance(records, (str, bytes, Mapping)):
        raise TypeError("records must be a sequence of ConsumptionRecord or dict (use [] for none)")
    try:
        items = list(records)
    except TypeError:
        raise TypeError("records must be a sequence of ConsumptionRecord or dict (use [] for none)") from None
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
# Measured baseline
# --------------------------------------------------------------------------- #
@dataclass
class _Measured:
    value: Optional[float] = None
    reason: Optional[str] = None  # set when no baseline can be reported
    suspect: bool = False
    window_start: Optional[date] = None
    window_end: Optional[date] = None
    days_with_data: Optional[int] = None
    coverage: Optional[float] = None
    readings_used: Optional[int] = None
    building_name: Optional[str] = None  # canonical spelling for a building scope
    rows: List[Dict[str, Any]] = field(default_factory=list)  # per-building rows
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)


def _names(names: Sequence[str]) -> str:
    shown = ", ".join(repr(n) for n in names[:_MAX_NAMES_LISTED])
    extra = len(names) - _MAX_NAMES_LISTED
    return shown + (f" and {extra} more" if extra > 0 else "")


def _measured_baseline(items: Sequence[ConsumptionRecord], spec: _Spec) -> _Measured:
    m = _Measured()
    unit = STANDARD_UNITS[spec.resource]
    resource = spec.resource.value
    if not items:
        m.reason = "no_records"
        m.warnings.append("No consumption records were supplied, so no baseline can be measured.")
        return m

    # Buildings are matched case-insensitively; the data's own spelling is reported.
    all_spellings: Dict[str, Counter] = defaultdict(Counter)
    for r in items:
        all_spellings[r.building.casefold()][r.building] += 1

    def display(spellings: Counter) -> str:
        return sorted(spellings.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    if spec.building is not None:
        key = spec.building.casefold()
        if key not in all_spellings:
            known = sorted(display(c) for c in all_spellings.values())
            m.reason = "unknown_building"
            m.warnings.append(
                f"Building {spec.building!r} does not appear in the supplied records; "
                f"known buildings: {_names(known)}."
            )
            return m
        selected = [r for r in items if r.resource_type is spec.resource and r.building.casefold() == key]
        none_msg = f"Building {display(all_spellings[key])!r} has no {resource} records."
    else:
        selected = [r for r in items if r.resource_type is spec.resource]
        none_msg = f"No {resource} records were supplied for the campus."
    if not selected:
        m.reason = "no_resource_records"
        m.warnings.append(none_msg)
        return m

    sel_spellings: Dict[str, Counter] = defaultdict(Counter)
    for r in selected:
        sel_spellings[r.building.casefold()][r.building] += 1
    names = {k: display(c) for k, c in sel_spellings.items()}
    variants = sorted(names[k] for k, c in sel_spellings.items() if len(c) > 1)
    if variants:
        m.suspect = True
        m.warnings.append(
            "Building names differing only in letter case were treated as one building: "
            f"{_names(variants)}."
        )
    if spec.building is not None:
        m.building_name = names[spec.building.casefold()]

    # Timestamps: naive -> UTC (engine convention); calendar days are UTC days.
    naive = sum(1 for r in selected if r.timestamp.tzinfo is None)
    if 0 < naive < len(selected):
        m.suspect = True
        m.warnings.append(
            f"{naive} timestamp(s) had no UTC offset and were treated as UTC while others carry offsets."
        )
    offsets = {r.timestamp.utcoffset() for r in selected if r.timestamp.tzinfo is not None}
    if len(offsets) > 1:
        m.warnings.append(
            f"Timestamps carried {len(offsets)} different UTC offsets; all were converted to UTC "
            "before assigning calendar days."
        )

    # Duplicates: identical value -> counted once; conflicting values -> excluded.
    groups: Dict[Tuple[str, datetime], List[float]] = defaultdict(list)
    for r in selected:
        groups[(r.building.casefold(), _utc(r.timestamp))].append(r.consumption_value)
    exact_dupes = 0
    conflicts = 0
    readings: List[Tuple[str, datetime, float]] = []
    for (bkey, ts), values in sorted(groups.items()):
        if len(set(values)) == 1:
            exact_dupes += len(values) - 1
            readings.append((bkey, ts, values[0]))
        else:
            conflicts += 1
    if exact_dupes:
        m.warnings.append(f"{exact_dupes} exact duplicate reading(s) (same building, timestamp and value) were counted once.")
    if conflicts:
        m.suspect = True
        m.warnings.append(
            f"{conflicts} timestamp(s) had conflicting values for the same building; those readings were "
            "excluded (excluding them can only understate the baseline)."
        )
    if not readings:
        m.reason = "no_usable_readings"
        m.warnings.append("Every matching reading was excluded as a conflicting duplicate; no baseline can be measured.")
        return m

    # Window: period_days calendar days ending on the latest reading.
    n = spec.period_days
    end = max(ts.date() for _, ts, _ in readings)
    start = end - timedelta(days=n - 1)
    in_window = [(b, ts, v) for b, ts, v in readings if start <= ts.date() <= end]
    before = len(readings) - len(in_window)

    per_building_values: Dict[str, List[float]] = defaultdict(list)
    per_building_days: Dict[str, set] = defaultdict(set)
    per_day_counts: Counter = Counter()
    days_any: set = set()
    for b, ts, v in in_window:
        d = ts.date()
        per_building_values[b].append(v)
        per_building_days[b].add(d)
        per_day_counts[(b, d)] += 1
        days_any.add(d)

    coverage = len(days_any) / n
    m.window_start, m.window_end = start, end
    m.days_with_data = len(days_any)
    m.coverage = round(coverage, _ROUND)
    m.readings_used = len(in_window)

    m.assumptions.append(
        f"Baseline = sum of the {resource} readings ({unit}) in the {n} calendar days (UTC) ending on the "
        f"most recent reading ({end.isoformat()}), not on today's date."
    )
    m.assumptions.append(
        "Each record's consumption_value is consumption for its own interval (not a cumulative meter "
        "value); gaps are not filled or extrapolated."
    )
    if naive:
        m.assumptions.append("Timestamps without a UTC offset are treated as UTC.")
    if before:
        m.assumptions.append(f"{before} reading(s) older than the baseline window were not used.")

    if coverage < MIN_COVERAGE_FRACTION:
        m.reason = "insufficient_coverage"
        m.warnings.append(
            f"Only {len(days_any)} of the {n} days in the baseline window have readings "
            f"({coverage:.0%}); at least {MIN_COVERAGE_FRACTION:.0%} is required, so no baseline is reported."
        )
        return m

    if coverage < 1.0:
        m.suspect = True
        m.warnings.append(
            f"Readings exist for {len(days_any)} of the {n} days in the baseline window ({coverage:.0%}). "
            "The baseline is the sum of the available readings (no gap filling), so it understates a "
            "complete period and the projected savings are understated by the same share."
        )
    incomplete = sorted(names[b] for b, days in per_building_days.items() if len(days) < n)
    if spec.building is None and incomplete:
        m.suspect = True
        m.warnings.append(
            f"Building(s) with readings on fewer than {n} days of the window: {_names(incomplete)}; "
            "their baselines are partial."
        )
    stale = sorted(names[b] for b in names if b not in per_building_days)
    if spec.building is None and stale:
        m.suspect = True
        m.warnings.append(f"Building(s) with no readings inside the baseline window: {_names(stale)}.")
    counts = set(per_day_counts.values())
    if len(counts) > 1:
        m.suspect = True
        m.warnings.append(
            f"Reading frequency is irregular ({min(counts)} to {max(counts)} readings per building per day); "
            "the baseline is a plain sum of the readings present."
        )

    m.value = math.fsum(v for _, _, v in in_window)
    m.rows = [
        {
            "building": names[b],
            "baseline_value": round(math.fsum(per_building_values[b]), _ROUND),
            "days_with_data": len(per_building_days[b]),
            "readings_used": len(per_building_values[b]),
        }
        for b in sorted(per_building_values, key=lambda k: names[k])
    ]
    return m


# --------------------------------------------------------------------------- #
# Forecast-based baseline (Phase 7B)
# --------------------------------------------------------------------------- #
@dataclass
class _ForecastBaseline:
    value: Optional[float] = None
    reason: Optional[str] = None
    quality: DataQualityStatus = DataQualityStatus.INSUFFICIENT_DATA
    window_start: Optional[date] = None
    window_end: Optional[date] = None
    days_with_data: Optional[int] = None
    coverage: Optional[float] = None
    readings_used: Optional[int] = None
    building_name: Optional[str] = None
    included: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    interval_level: Optional[float] = None
    uncertainty_method: Optional[str] = None
    model_used: Optional[str] = None


def _forecast_baseline(items: Sequence[ConsumptionRecord], spec: _Spec) -> _ForecastBaseline:
    fb = _ForecastBaseline()
    resource = spec.resource.value
    unit = STANDARD_UNITS[spec.resource]
    n = spec.period_days

    if not items:
        fb.reason = "no_records"
        fb.warnings.append("No consumption records were supplied, so no baseline could be forecast.")
        return fb

    all_spellings: Dict[str, Counter] = defaultdict(Counter)
    for r in items:
        all_spellings[r.building.casefold()][r.building] += 1

    def display(spellings: Counter) -> str:
        return sorted(spellings.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    if spec.building is not None:
        key = spec.building.casefold()
        if key not in all_spellings:
            known = sorted(display(c) for c in all_spellings.values())
            fb.reason = "unknown_building"
            fb.warnings.append(
                f"Building {spec.building!r} does not appear in the supplied records; "
                f"known buildings: {_names(known)}."
            )
            return fb
        selected = [r for r in items if r.resource_type is spec.resource and r.building.casefold() == key]
        none_msg = f"Building {display(all_spellings[key])!r} has no {resource} records."
        fb.building_name = display(all_spellings[key])
    else:
        selected = [r for r in items if r.resource_type is spec.resource]
        none_msg = f"No {resource} records were supplied for the campus."

    if not selected:
        fb.reason = "no_resource_records"
        fb.warnings.append(none_msg)
        return fb

    forecast_cfg = spec.forecast_config or ForecastConfig(max_horizon_days=max(30, n))
    try:
        forecasts = forecast_consumption(items, horizon_days=n, config=forecast_cfg)
    except Exception as exc:  # defensive
        fb.reason = "forecast_failed"
        fb.warnings.append(f"Forecasting engine failed: {exc}")
        return fb

    res_forecasts = [f for f in forecasts if f.resource_type is spec.resource]

    if spec.building is not None:
        target_f = [f for f in res_forecasts if f.building.casefold() == spec.building.casefold()]
        f_res = target_f[0]
        if not f_res.forecast_available or f_res.predicted_total is None:
            fb.reason = "insufficient_forecast_data"
            fb.warnings.append(
                f"Insufficient historical data to forecast {resource} for {f_res.building}."
            )
            for note in f_res.confidence_notes:
                if note not in fb.warnings:
                    fb.warnings.append(note)
            fb.quality = DataQualityStatus.INSUFFICIENT_DATA
            return fb

        fb.value = f_res.predicted_total
        fb.quality = f_res.data_quality_status
        fb.building_name = f_res.building
        fb.included = [f_res.building]
        if f_res.forecast_dates:
            fb.window_start = f_res.forecast_dates[0]
            fb.window_end = f_res.forecast_dates[-1]
        fb.days_with_data = f_res.history.observed_days
        fb.coverage = f_res.history.coverage
        fb.readings_used = f_res.history.records_received
        fb.model_used = f_res.model_used.value if f_res.model_used else None

        fb.lower_bound = f_res.predicted_total_lower_bound
        fb.upper_bound = f_res.predicted_total_upper_bound
        fb.interval_level = f_res.interval_level
        fb.uncertainty_method = f_res.uncertainty_method

        fb.assumptions.append(
            f"Baseline = {f_res.predicted_total:g} {unit} forecast over {n} days "
            f"({fb.window_start} to {fb.window_end}) by the Phase 6 forecasting engine "
            f"(model: {fb.model_used or 'baseline'})."
        )
        if fb.lower_bound is not None and fb.upper_bound is not None:
            fb.assumptions.append(
                f"Baseline uncertainty: [{fb.lower_bound:g}, {fb.upper_bound:g}] {unit} at "
                f"{fb.interval_level:.0%} nominal level using {fb.uncertainty_method}."
            )
        else:
            fb.assumptions.append(
                "Forecast uncertainty bounds were not available from backtest residuals; "
                "bounds are not fabricated."
            )
        return fb
    else:
        # Campus-wide forecast baseline
        if not res_forecasts:
            fb.reason = "no_resource_records"
            fb.warnings.append(none_msg)
            return fb

        available = [f for f in res_forecasts if f.forecast_available and f.predicted_total is not None]
        unavailable = [f for f in res_forecasts if not f.forecast_available or f.predicted_total is None]

        for f in res_forecasts:
            for note in f.confidence_notes:
                if note not in fb.warnings:
                    fb.warnings.append(note)

        if not available:
            fb.reason = "insufficient_forecast_data"
            fb.warnings.append(f"No buildings had sufficient historical data to forecast campus {resource}.")
            fb.quality = DataQualityStatus.INSUFFICIENT_DATA
            return fb

        fb.value = math.fsum(f.predicted_total for f in available)
        fb.included = sorted(f.building for f in available)
        dates = [d for f in available for d in f.forecast_dates]
        if dates:
            fb.window_start = min(dates)
            fb.window_end = max(dates)
        fb.days_with_data = sum(f.history.observed_days for f in available)
        fb.readings_used = sum(f.history.records_received for f in available)

        fb.rows = [
            {
                "building": f.building,
                "baseline_value": round(f.predicted_total, _ROUND),
                "days_with_data": f.history.observed_days,
                "readings_used": f.history.records_received,
                "forecast_available": True,
                "model_used": f.model_used.value if f.model_used else None,
            }
            for f in sorted(available, key=lambda x: x.building)
        ]

        if unavailable:
            fb.quality = DataQualityStatus.SUSPECT
            unav_names = sorted(f.building for f in unavailable)
            fb.warnings.append(
                f"Building(s) with insufficient forecast data were excluded from campus baseline: {_names(unav_names)}."
            )
            for f in sorted(unavailable, key=lambda x: x.building):
                fb.rows.append({
                    "building": f.building,
                    "baseline_value": None,
                    "days_with_data": f.history.observed_days,
                    "readings_used": f.history.records_received,
                    "forecast_available": False,
                    "model_used": None,
                })
        else:
            fb.quality = (
                DataQualityStatus.SUSPECT
                if any(f.data_quality_status == DataQualityStatus.SUSPECT for f in available)
                else DataQualityStatus.OK
            )

        has_all_bounds = all(
            f.predicted_total_lower_bound is not None and f.predicted_total_upper_bound is not None
            for f in available
        )
        levels = {f.interval_level for f in available if f.interval_level is not None}
        if has_all_bounds and len(levels) == 1 and not unavailable:
            fb.lower_bound = math.fsum(f.predicted_total_lower_bound for f in available)
            fb.upper_bound = math.fsum(f.predicted_total_upper_bound for f in available)
            fb.interval_level = next(iter(levels))
            fb.uncertainty_method = available[0].uncertainty_method
            fb.assumptions.append(
                f"Campus forecast interval [{fb.lower_bound:g}, {fb.upper_bound:g}] {unit} is the conservative sum "
                f"of individual building intervals ({fb.interval_level:.0%}); building correlations are not modeled."
            )
        else:
            fb.lower_bound = None
            fb.upper_bound = None
            if not has_all_bounds:
                fb.assumptions.append("One or more buildings lack prediction intervals; campus uncertainty bounds are omitted.")
            elif unavailable:
                fb.assumptions.append("Partial campus forecast (some buildings excluded); campus uncertainty bounds are omitted.")

        fb.assumptions.append(
            f"Campus baseline = sum of forecasts for {len(available)} building(s) ({fb.value:g} {unit}) "
            f"over {n} days ({fb.window_start} to {fb.window_end})."
        )
        return fb


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def simulate_scenario(
    records: Sequence[Union[ConsumptionRecord, Mapping[str, Any]]],
    scenario: Mapping[str, Any],
    config: Optional[AnalysisConfig] = None,
) -> Dict[str, Any]:
    """Project the savings of a proposed percentage reduction (hypothetical).

    Args:
        records: ``ConsumptionRecord`` objects or dicts (validated exactly as in
            ``analyze_consumption``). May be empty, e.g. when the scenario
            carries an explicit ``baseline_consumption``.
        scenario: JSON-style mapping::

            {"resource_type": "electricity",          # electricity | water | waste
             "building_code": "BLOCK_A",              # omit or null for campus-wide
             "reduction_percentage": 15,              # 0-100 inclusive
             "baseline_period_days": 30,              # optional, default 30
             "baseline_consumption": {"value": 4200,  # optional explicit baseline;
                                      "unit": "kWh",  # value, unit and period are
                                      "period_days": 30},  # all required together
             "scenario_id": "my-id"}                  # optional; else derived

        config: ``AnalysisConfig``; only its emission factors are used. No
            factor is built in; without one the CO2e field is ``null``.

    Returns:
        A JSON-safe ``dict`` (see ``docs/JSON_CONTRACT.md`` section 7). When the
        data cannot support a baseline, ``calculation_status`` is
        ``"not_calculated"`` and the numbers are ``null``; this is a normal
        result, not an exception.

    Raises:
        ScenarioValidationError: the scenario is malformed (reduction outside
            0-100, unknown resource type or keys, bad explicit baseline...).
        RecordValidationError: a record is invalid (all failures listed).
        TypeError: ``records`` is not a sequence, ``scenario`` is not a mapping
            or ``config`` is not an ``AnalysisConfig``.
    """
    if config is not None and not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig or None")
    if not isinstance(scenario, Mapping):
        raise TypeError("scenario must be a mapping (JSON object)")
    spec = _parse_scenario(scenario)
    items = _validate_records(records)
    cfg = config or AnalysisConfig()

    unit = STANDARD_UNITS[spec.resource]
    n = spec.period_days
    warnings: List[str] = []
    assumptions: List[str] = [
        "Hypothetical projection: it assumes the stated reduction is achieved over the whole baseline "
        "period. It is not measured savings and does not show that any reduction happened.",
        "estimated_savings = baseline x reduction_percentage / 100; "
        "projected_consumption = baseline - estimated_savings.",
        f"All consumption values are in {unit}. The baseline period is {n} days and is never rescaled.",
        "No monetary savings are estimated.",
    ]

    baseline: Optional[float]
    quality: DataQualityStatus
    reason: Optional[str] = None
    source: str
    window_start = window_end = None
    days_with_data: Optional[int] = None
    coverage: Optional[float] = None
    readings_used: Optional[int] = None
    included: List[str] = []
    building_code: Optional[str] = spec.building
    breakdown: Optional[List[Dict[str, Any]]] = None

    uncertainty_dict: Optional[Dict[str, Any]] = None
    forecast_details: Optional[Dict[str, Any]] = None

    if spec.explicit is not None:
        source = "explicit"
        baseline = spec.explicit.value
        quality = DataQualityStatus.OK
        assumptions.append(
            f"Baseline is the caller-supplied {spec.explicit.value:g} {unit} over {n} days; "
            "it was not verified against any records."
        )
        if items:
            warnings.append("Consumption records were supplied but not used, because an explicit baseline_consumption was given.")
        included = [spec.building] if spec.building else []
    elif spec.baseline_source == "forecast":
        source = "forecast"
        fb = _forecast_baseline(items, spec)
        warnings.extend(fb.warnings)
        assumptions.extend(fb.assumptions)
        reason = fb.reason
        baseline = fb.value
        window_start, window_end = fb.window_start, fb.window_end
        days_with_data, coverage, readings_used = fb.days_with_data, fb.coverage, fb.readings_used
        if fb.building_name:
            building_code = fb.building_name
        if baseline is not None:
            included = fb.included
            if spec.building is None:
                breakdown = fb.rows
            quality = fb.quality
        else:
            quality = DataQualityStatus.INSUFFICIENT_DATA
        if fb.model_used or fb.window_start:
            forecast_details = {
                "horizon_days": n,
                "model_used": fb.model_used,
            }
    else:
        source = "measured_records"
        m = _measured_baseline(items, spec)
        warnings.extend(m.warnings)
        assumptions.extend(m.assumptions)
        reason = m.reason
        baseline = m.value
        window_start, window_end = m.window_start, m.window_end
        days_with_data, coverage, readings_used = m.days_with_data, m.coverage, m.readings_used
        if m.building_name:
            building_code = m.building_name
        if baseline is not None:
            included = sorted(row["building"] for row in m.rows)
            if spec.building is None:
                breakdown = m.rows
            quality = DataQualityStatus.SUSPECT if m.suspect else DataQualityStatus.OK
        else:
            quality = DataQualityStatus.INSUFFICIENT_DATA

    savings: Optional[float] = None
    projected: Optional[float] = None
    baseline_out: Optional[float] = None
    co2e: Optional[float] = None
    factor_out: Optional[Dict[str, Any]] = None

    if baseline is not None:
        baseline_out = round(baseline, _ROUND)
        savings = round(min(baseline_out, max(0.0, baseline_out * spec.reduction / 100.0)), _ROUND)
        projected = round(max(0.0, baseline_out - savings), _ROUND)
        if baseline_out == 0:
            if quality is DataQualityStatus.OK:
                quality = DataQualityStatus.SUSPECT
            warnings.append(
                f"The baseline is 0 {unit}, so there is nothing to reduce: savings are 0. "
                "If the building was in use, check the meter or the data."
            )
        factor, factor_unit, lines = _co2e_factor(spec.resource, cfg)
        if factor is None:
            warnings.extend(lines)
        else:
            assumptions.extend(lines)
            co2e = round(savings * factor, _ROUND)
            factor_out = {"value": factor, "unit": factor_unit}

        if spec.baseline_source == "forecast" and fb.lower_bound is not None and fb.upper_bound is not None:
            u_b_lower = round(max(0.0, fb.lower_bound), _ROUND)
            u_b_upper = round(max(0.0, fb.upper_bound), _ROUND)
            u_s_lower = round(max(0.0, u_b_lower * spec.reduction / 100.0), _ROUND)
            u_s_upper = round(max(0.0, u_b_upper * spec.reduction / 100.0), _ROUND)
            u_p_lower = round(max(0.0, u_b_lower * (1.0 - spec.reduction / 100.0)), _ROUND)
            u_p_upper = round(max(0.0, u_b_upper * (1.0 - spec.reduction / 100.0)), _ROUND)
            u_c_lower = round(u_s_lower * factor, _ROUND) if factor is not None else None
            u_c_upper = round(u_s_upper * factor, _ROUND) if factor is not None else None
            uncertainty_dict = {
                "interval_level": fb.interval_level,
                "method": fb.uncertainty_method,
                "baseline": {"lower_bound": u_b_lower, "upper_bound": u_b_upper},
                "estimated_savings": {"lower_bound": u_s_lower, "upper_bound": u_s_upper},
                "projected_consumption": {"lower_bound": u_p_lower, "upper_bound": u_p_upper},
                "estimated_co2e_savings_kg": (
                    {"lower_bound": u_c_lower, "upper_bound": u_c_upper}
                    if factor is not None
                    else None
                ),
            }

        calculation_status = "calculated"
    else:
        calculation_status = "not_calculated"

    res = {
        "simulation_contract_version": SIMULATION_CONTRACT_VERSION,
        "engine_version": __version__,
        "scenario_id": spec.scenario_id,
        "projection_type": "hypothetical",
        "calculation_status": calculation_status,
        "not_calculated_reason": reason,
        "scope": {
            "level": "campus" if spec.building is None else "building",
            "building_code": building_code,
            "buildings_included": included,
        },
        "resource_type": spec.resource.value,
        "unit": unit,
        "baseline": {
            "value": baseline_out,
            "unit": unit,
            "period_days": n,
            "source": source,
            "window_start": window_start.isoformat() if window_start else None,
            "window_end": window_end.isoformat() if window_end else None,
            "days_with_data": days_with_data,
            "coverage_fraction": coverage,
            "readings_used": readings_used,
        },
        "reduction_percentage": spec.reduction,
        "projected_consumption": {"value": projected, "unit": unit},
        "estimated_savings": {"value": savings, "unit": unit},
        "estimated_co2e_savings_kg": co2e,
        "emission_factor": factor_out,
        "measured_savings": None,
        "units": {"consumption": unit, "co2e": "kg CO2e", "period": "days"},
        "building_breakdown": breakdown,
        "assumptions": assumptions,
        "warnings": warnings,
        "data_quality_status": quality.value,
    }
    if spec.baseline_source == "forecast":
        res["simulation_type"] = "forecast"
        if uncertainty_dict is not None:
            res["uncertainty"] = uncertainty_dict
            res["baseline"]["lower_bound"] = uncertainty_dict["baseline"]["lower_bound"]
            res["baseline"]["upper_bound"] = uncertainty_dict["baseline"]["upper_bound"]
            res["baseline"]["interval_level"] = uncertainty_dict["interval_level"]
            res["baseline"]["uncertainty_method"] = uncertainty_dict["method"]
            res["estimated_savings"]["lower_bound"] = uncertainty_dict["estimated_savings"]["lower_bound"]
            res["estimated_savings"]["upper_bound"] = uncertainty_dict["estimated_savings"]["upper_bound"]
            res["projected_consumption"]["lower_bound"] = uncertainty_dict["projected_consumption"]["lower_bound"]
            res["projected_consumption"]["upper_bound"] = uncertainty_dict["projected_consumption"]["upper_bound"]
        if forecast_details is not None:
            res["forecast_details"] = forecast_details
    return res


_SIM_REQUEST_KEYS = ("records", "scenario")


def _detail(loc: Sequence[Any], msg: str) -> Dict[str, Any]:
    return {"index": None, "errors": [{"loc": list(loc), "msg": msg}]}


def simulate_payload(
    payload: Any,
    config: Optional[AnalysisConfig] = None,
    *,
    max_records: Optional[int] = None,
) -> AnalysisOutcome:
    """Decoded JSON request body -> ``AnalysisOutcome`` (never raises for client mistakes).

    Request: ``{"records": [...], "scenario": {...}}``. ``records`` is optional
    (omit it or send ``[]`` when the scenario has an explicit baseline).
    Errors use the standard envelope (HTTP 422/413/500); a scenario that cannot
    be calculated from the data is a normal HTTP 200 with
    ``calculation_status = "not_calculated"``. See ``docs/JSON_CONTRACT.md``
    section 7.
    """
    from .contract import DEFAULT_MAX_RECORDS

    if config is not None and not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig or None")
    limit = DEFAULT_MAX_RECORDS if max_records is None else max_records

    if not isinstance(payload, Mapping):
        return error_outcome(422, "invalid_request", "Request body must be a JSON object.",
                             [_detail([], "must be an object")])
    unknown = sorted(str(k) for k in payload if k not in _SIM_REQUEST_KEYS)
    if unknown:
        return error_outcome(
            422, "invalid_request",
            f"Unknown field(s): {', '.join(unknown)}. Allowed: {', '.join(_SIM_REQUEST_KEYS)}.",
            [_detail([k], "unknown field") for k in unknown],
        )
    scenario = payload.get("scenario")
    if not isinstance(scenario, Mapping):
        return error_outcome(422, "invalid_request", "scenario must be a JSON object.",
                             [_detail(["scenario"], "must be an object")])
    records = payload.get("records")
    if records is None:
        records = []
    if not isinstance(records, list):
        return error_outcome(422, "invalid_request", "records must be a JSON array when present.",
                             [_detail(["records"], "must be an array")])
    if len(records) > limit:
        return error_outcome(413, "too_many_records",
                             f"At most {limit} records per request; received {len(records)}.",
                             [_detail(["records"], f"maximum {limit} items")])
    try:
        result = simulate_scenario(records, scenario, config)
    except ScenarioValidationError as exc:
        return error_outcome(
            422, "invalid_scenario",
            f"The scenario failed validation ({len(exc.errors)} problem(s)); nothing was simulated.",
            [_detail(["scenario", *e["loc"]], e["msg"]) for e in exc.errors],
        )
    except RecordValidationError as exc:
        return error_outcome(
            422, "invalid_records",
            f"{len(exc.errors)} record(s) failed validation; no simulation was performed.",
            validation_error_body(exc),
        )
    except Exception:  # noqa: BLE001 - never leak internals to clients
        return error_outcome(500, "simulation_failed", "The simulation could not be completed.")

    body: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "simulation_contract_version": SIMULATION_CONTRACT_VERSION,
        "status": "ok",
        "engine_version": __version__,
        "result": result,
    }
    return AnalysisOutcome(ok=True, http_status=200, body=body)


# --------------------------------------------------------------------------- #
# Multi-Scenario Comparison (Phase 7B)
# --------------------------------------------------------------------------- #
def compare_scenarios(
    records: Sequence[Union[ConsumptionRecord, Mapping[str, Any]]],
    scenarios: Sequence[Mapping[str, Any]],
    config: Optional[AnalysisConfig] = None,
) -> Dict[str, Any]:
    """Compare multiple percentage-reduction sustainability scenarios (Phase 7B).

    Supports side-by-side comparison across multiple buildings, resources
    (electricity, water, waste), percentage reductions, and campus-wide scopes.
    Evaluates each scenario projection, computes non-overlapping portfolio
    savings totals preventing double counting, ranks scenarios, and reports
    uncertainty, data quality, and assumptions.

    Double-Counting Prevention Policy (Deterministic Portfolio Selection):
    ---------------------------------------------------------------------
    1. Scenarios targeting different resource types (electricity, water, waste)
       are independent and aggregated by resource in their native units.
    2. Within the same resource:
       - Competing scenarios for the SAME building: the scenario with the highest
         estimated savings is selected (tie-break: higher reduction percentage,
         then deterministic alphabetical scenario_id).
       - Campus-wide vs building-specific scopes: if a campus-wide scenario
         achieves greater or equal total savings than the sum of non-overlapping
         building scenarios, the campus scenario is selected; otherwise, the
         combined building scenarios are selected.
       - Mutually exclusive scenarios are excluded from portfolio totals to
         prevent double-counting. Excluded scenarios and conflict reasons are
         recorded in ``portfolio.overlap_groups``.
       - Both deduplicated ``portfolio_savings`` (also in ``total_estimated_savings``)
         and ``unadjusted_gross_savings`` are reported for full transparency.

    Args:
        records: Sequence of ``ConsumptionRecord`` objects or dicts (or empty list).
        scenarios: Sequence of scenario JSON-style mappings.
        config: Optional ``AnalysisConfig`` (emission factors).

    Returns:
        JSON-compatible dict matching the Phase 7B contract.
    """
    if config is not None and not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig or None")
    if scenarios is None or isinstance(scenarios, (str, bytes, Mapping)):
        raise TypeError("scenarios must be a sequence of scenario mappings (use [] for none)")
    try:
        scenario_list = list(scenarios)
    except TypeError:
        raise TypeError("scenarios must be a sequence of scenario mappings (use [] for none)") from None

    for i, s in enumerate(scenario_list):
        if not isinstance(s, Mapping):
            raise TypeError(f"scenario at index {i} must be a mapping (JSON object), got {type(s).__name__}")

    all_scenario_errors: List[Dict[str, Any]] = []
    for i, scn in enumerate(scenario_list):
        try:
            _parse_scenario(scn)
        except ScenarioValidationError as exc:
            for e in exc.errors:
                all_scenario_errors.append({"loc": [i, *e["loc"]], "msg": e["msg"]})
    if all_scenario_errors:
        raise ScenarioValidationError(all_scenario_errors)

    items = _validate_records(records)
    cfg = config or AnalysisConfig()

    if not scenario_list:
        return {
            "simulation_contract_version": SIMULATION_CONTRACT_VERSION,
            "engine_version": __version__,
            "projection_type": "hypothetical",
            "scenarios_count": 0,
            "scenarios": [],
            "resource_totals": {
                r.value: {
                    "resource_type": r.value,
                    "unit": STANDARD_UNITS[r],
                    "scenarios_count": 0,
                    "calculated_scenarios_count": 0,
                    "included_in_portfolio_count": 0,
                    "excluded_from_portfolio_count": 0,
                    "total_baseline": 0.0,
                    "total_estimated_savings": 0.0,
                    "portfolio_savings": 0.0,
                    "unadjusted_gross_savings": 0.0,
                    "total_projected_consumption": 0.0,
                    "total_estimated_co2e_savings_kg": None,
                    "overlapping_detected": False,
                }
                for r in ResourceType
            },
            "total_co2e_savings_kg": None,
            "co2e_supported": False,
            "rankings": {
                "overall": [],
                "by_resource": {r.value: [] for r in ResourceType},
                "by_co2e": [],
                "by_percentage": [],
            },
            "portfolio": {
                "policy": "deterministic_max_savings_deduplication",
                "description": (
                    "Prevent double-counting across overlapping scopes (same building or campus vs building) "
                    "by deterministically selecting maximum-saving non-overlapping initiatives."
                ),
                "overlapping_scenarios_detected": False,
                "included_scenario_ids": [],
                "excluded_scenario_ids": [],
                "overlap_groups": [],
            },
            "data_quality_status": DataQualityStatus.OK.value,
            "assumptions": [
                "Hypothetical projections: all results assume stated percentage reductions are fully achieved.",
                "Not measured savings: simulations do not confirm that any physical reductions took place.",
                "Deterministic overlap prevention: mutually exclusive scenarios for the same resource and scope are deduplicated in portfolio totals.",
                "No monetary figures or financial projections are calculated.",
            ],
            "warnings": ["No scenarios were provided for comparison."],
        }

    sim_results: List[Dict[str, Any]] = [
        simulate_scenario(items, s, cfg) for s in scenario_list
    ]

    portfolio_included_ids: List[str] = []
    portfolio_excluded_ids: List[str] = []
    overlap_groups: List[Dict[str, Any]] = []
    resource_totals: Dict[str, Dict[str, Any]] = {}
    has_any_co2e_factor = False

    for r_type in ResourceType:
        r_val = r_type.value
        r_unit = STANDARD_UNITS[r_type]
        r_scenarios = [s for s in sim_results if s["resource_type"] == r_val]
        r_calc = [s for s in r_scenarios if s["calculation_status"] == "calculated"]

        if not r_calc:
            resource_totals[r_val] = {
                "resource_type": r_val,
                "unit": r_unit,
                "scenarios_count": len(r_scenarios),
                "calculated_scenarios_count": 0,
                "included_in_portfolio_count": 0,
                "excluded_from_portfolio_count": len(r_scenarios),
                "total_baseline": 0.0,
                "total_estimated_savings": 0.0,
                "portfolio_savings": 0.0,
                "unadjusted_gross_savings": 0.0,
                "total_projected_consumption": 0.0,
                "total_estimated_co2e_savings_kg": None,
                "overlapping_detected": False,
            }
            continue

        campus_scns = [s for s in r_calc if s["scope"]["level"] == "campus"]
        bldg_scns = [s for s in r_calc if s["scope"]["level"] == "building"]

        bldg_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in bldg_scns:
            b_code = (s["scope"]["building_code"] or "").strip().casefold()
            bldg_groups[b_code].append(s)

        selected_bldg_scns: List[Dict[str, Any]] = []
        for b_code, group in sorted(bldg_groups.items()):
            if len(group) == 1:
                selected_bldg_scns.append(group[0])
            else:
                sorted_group = sorted(
                    group,
                    key=lambda x: (
                        -(x["estimated_savings"]["value"] or 0.0),
                        -x["reduction_percentage"],
                        x["scenario_id"],
                    ),
                )
                winner = sorted_group[0]
                losers = sorted_group[1:]
                selected_bldg_scns.append(winner)
                for loser in losers:
                    portfolio_excluded_ids.append(loser["scenario_id"])
                overlap_groups.append({
                    "resource_type": r_val,
                    "conflict_type": "competing_building_scenarios",
                    "scope": winner["scope"]["building_code"],
                    "conflicting_scenario_ids": [s["scenario_id"] for s in sorted_group],
                    "selected_scenario_id": winner["scenario_id"],
                    "excluded_scenario_ids": [s["scenario_id"] for s in losers],
                    "reason": (
                        f"Selected scenario with higher estimated savings "
                        f"({winner['estimated_savings']['value']:g} {r_unit} vs "
                        f"{', '.join(f'{l['estimated_savings']['value']:g} {r_unit}' for l in losers)})."
                    ),
                })

        selected_campus_scn: Optional[Dict[str, Any]] = None
        if campus_scns:
            if len(campus_scns) == 1:
                selected_campus_scn = campus_scns[0]
            else:
                sorted_campus = sorted(
                    campus_scns,
                    key=lambda x: (
                        -(x["estimated_savings"]["value"] or 0.0),
                        -x["reduction_percentage"],
                        x["scenario_id"],
                    ),
                )
                selected_campus_scn = sorted_campus[0]
                campus_losers = sorted_campus[1:]
                for cl in campus_losers:
                    portfolio_excluded_ids.append(cl["scenario_id"])
                overlap_groups.append({
                    "resource_type": r_val,
                    "conflict_type": "competing_campus_scenarios",
                    "scope": "campus",
                    "conflicting_scenario_ids": [s["scenario_id"] for s in sorted_campus],
                    "selected_scenario_id": selected_campus_scn["scenario_id"],
                    "excluded_scenario_ids": [s["scenario_id"] for s in campus_losers],
                    "reason": (
                        f"Selected campus scenario with higher estimated savings "
                        f"({selected_campus_scn['estimated_savings']['value']:g} {r_unit})."
                    ),
                })

        r_included: List[Dict[str, Any]] = []
        if selected_campus_scn is not None and selected_bldg_scns:
            campus_savings = selected_campus_scn["estimated_savings"]["value"] or 0.0
            sum_bldg_savings = sum((s["estimated_savings"]["value"] or 0.0) for s in selected_bldg_scns)

            if campus_savings >= sum_bldg_savings:
                r_included = [selected_campus_scn]
                for bs in selected_bldg_scns:
                    portfolio_excluded_ids.append(bs["scenario_id"])
                overlap_groups.append({
                    "resource_type": r_val,
                    "conflict_type": "campus_vs_building_scope",
                    "scope": "campus_hierarchy",
                    "conflicting_scenario_ids": [selected_campus_scn["scenario_id"]] + [s["scenario_id"] for s in selected_bldg_scns],
                    "selected_scenario_id": selected_campus_scn["scenario_id"],
                    "excluded_scenario_ids": [s["scenario_id"] for s in selected_bldg_scns],
                    "reason": (
                        f"Campus-wide scenario savings ({campus_savings:g} {r_unit}) >= sum of "
                        f"individual building scenarios ({sum_bldg_savings:g} {r_unit}); "
                        "campus scenario chosen to avoid double-counting."
                    ),
                })
            else:
                r_included = selected_bldg_scns
                portfolio_excluded_ids.append(selected_campus_scn["scenario_id"])
                overlap_groups.append({
                    "resource_type": r_val,
                    "conflict_type": "campus_vs_building_scope",
                    "scope": "campus_hierarchy",
                    "conflicting_scenario_ids": [selected_campus_scn["scenario_id"]] + [s["scenario_id"] for s in selected_bldg_scns],
                    "selected_scenario_ids": [s["scenario_id"] for s in selected_bldg_scns],
                    "excluded_scenario_ids": [selected_campus_scn["scenario_id"]],
                    "reason": (
                        f"Combined building scenarios savings ({sum_bldg_savings:g} {r_unit}) > "
                        f"campus-wide scenario ({campus_savings:g} {r_unit}); "
                        "building scenarios chosen to maximize savings without double-counting."
                    ),
                })
        elif selected_campus_scn is not None:
            r_included = [selected_campus_scn]
        else:
            r_included = selected_bldg_scns

        for inc in r_included:
            portfolio_included_ids.append(inc["scenario_id"])

        p_savings = round(sum((s["estimated_savings"]["value"] or 0.0) for s in r_included), _ROUND)
        gross_savings = round(sum((s["estimated_savings"]["value"] or 0.0) for s in r_calc), _ROUND)
        tot_baseline = round(sum((s["baseline"]["value"] or 0.0) for s in r_included), _ROUND)
        tot_projected = round(sum((s["projected_consumption"]["value"] or 0.0) for s in r_included), _ROUND)

        co2e_vals = [s["estimated_co2e_savings_kg"] for s in r_included if s["estimated_co2e_savings_kg"] is not None]
        if co2e_vals:
            r_co2e = round(sum(co2e_vals), _ROUND)
            has_any_co2e_factor = True
        else:
            r_co2e = None

        overlap_detected = len(r_calc) > len(r_included)
        resource_totals[r_val] = {
            "resource_type": r_val,
            "unit": r_unit,
            "scenarios_count": len(r_scenarios),
            "calculated_scenarios_count": len(r_calc),
            "included_in_portfolio_count": len(r_included),
            "excluded_from_portfolio_count": len(r_scenarios) - len(r_included),
            "total_baseline": tot_baseline,
            "total_estimated_savings": p_savings,
            "portfolio_savings": p_savings,
            "unadjusted_gross_savings": gross_savings,
            "total_projected_consumption": tot_projected,
            "total_estimated_co2e_savings_kg": r_co2e,
            "overlapping_detected": overlap_detected,
        }

    if has_any_co2e_factor:
        total_co2e_vals = [
            resource_totals[r.value]["total_estimated_co2e_savings_kg"]
            for r in ResourceType
            if resource_totals[r.value]["total_estimated_co2e_savings_kg"] is not None
        ]
        total_portfolio_co2e = round(sum(total_co2e_vals), _ROUND)
    else:
        total_portfolio_co2e = None

    rankings_by_resource: Dict[str, List[Dict[str, Any]]] = {}
    for r_type in ResourceType:
        r_val = r_type.value
        r_scns = [s for s in sim_results if s["resource_type"] == r_val]
        sorted_r = sorted(
            r_scns,
            key=lambda x: (
                0 if x["calculation_status"] == "calculated" else 1,
                -(x["estimated_savings"]["value"] or 0.0),
                -x["reduction_percentage"],
                x["scenario_id"],
            ),
        )
        r_ranked = []
        for rank_idx, s in enumerate(sorted_r, start=1):
            r_ranked.append({
                "rank": rank_idx,
                "scenario_id": s["scenario_id"],
                "building_code": s["scope"]["building_code"],
                "reduction_percentage": s["reduction_percentage"],
                "estimated_savings": s["estimated_savings"]["value"],
                "unit": s["unit"],
                "estimated_co2e_savings_kg": s["estimated_co2e_savings_kg"],
                "calculation_status": s["calculation_status"],
                "included_in_portfolio": s["scenario_id"] in portfolio_included_ids,
            })
        rankings_by_resource[r_val] = r_ranked

    co2e_ranked: List[Dict[str, Any]] = []
    scns_with_co2e = [s for s in sim_results if s["estimated_co2e_savings_kg"] is not None]
    if scns_with_co2e:
        sorted_co2e = sorted(
            scns_with_co2e,
            key=lambda x: (-x["estimated_co2e_savings_kg"], -x["reduction_percentage"], x["scenario_id"]),
        )
        for rank_idx, s in enumerate(sorted_co2e, start=1):
            co2e_ranked.append({
                "rank": rank_idx,
                "scenario_id": s["scenario_id"],
                "resource_type": s["resource_type"],
                "building_code": s["scope"]["building_code"],
                "reduction_percentage": s["reduction_percentage"],
                "estimated_co2e_savings_kg": s["estimated_co2e_savings_kg"],
                "estimated_savings": s["estimated_savings"]["value"],
                "unit": s["unit"],
                "included_in_portfolio": s["scenario_id"] in portfolio_included_ids,
            })

    pct_ranked: List[Dict[str, Any]] = []
    sorted_pct = sorted(
        sim_results,
        key=lambda x: (
            0 if x["calculation_status"] == "calculated" else 1,
            -x["reduction_percentage"],
            x["scenario_id"],
        ),
    )
    for rank_idx, s in enumerate(sorted_pct, start=1):
        pct_ranked.append({
            "rank": rank_idx,
            "scenario_id": s["scenario_id"],
            "resource_type": s["resource_type"],
            "building_code": s["scope"]["building_code"],
            "reduction_percentage": s["reduction_percentage"],
            "estimated_savings": s["estimated_savings"]["value"],
            "unit": s["unit"],
            "calculation_status": s["calculation_status"],
            "included_in_portfolio": s["scenario_id"] in portfolio_included_ids,
        })

    rankings = {
        "by_resource": rankings_by_resource,
        "by_co2e": co2e_ranked,
        "by_percentage": pct_ranked,
        "overall": co2e_ranked if co2e_ranked else pct_ranked,
    }

    assumptions = [
        "Hypothetical projections: all results assume stated percentage reductions are fully achieved over baseline periods.",
        "Not measured savings: simulations do not confirm that any physical reductions took place.",
        "Deterministic overlap prevention policy: mutually exclusive scenarios for the same resource and scope are deduplicated in portfolio totals by selecting maximum-saving combinations.",
        "No monetary figures or financial projections are calculated.",
    ]
    for s in sim_results:
        for a in s.get("assumptions", []):
            if a not in assumptions:
                assumptions.append(a)

    warnings: List[str] = []
    if overlap_groups:
        warnings.append(
            f"Overlapping scenarios detected ({len(overlap_groups)} conflict group(s)). "
            "To prevent double-counting, deduplicated non-overlapping portfolio totals are reported in "
            "total_estimated_savings. Unadjusted gross savings are provided for reference."
        )
    not_calc_count = sum(1 for s in sim_results if s["calculation_status"] == "not_calculated")
    if not_calc_count > 0:
        warnings.append(
            f"{not_calc_count} scenario(s) could not be calculated from the supplied data; "
            "see individual scenario results for reasons."
        )
    if not has_any_co2e_factor:
        warnings.append("No emission factors configured; CO2e savings are not calculated.")
    for s in sim_results:
        for w in s.get("warnings", []):
            if w not in warnings:
                warnings.append(w)

    qualities = [s["data_quality_status"] for s in sim_results]
    if all(q == DataQualityStatus.OK.value for q in qualities):
        agg_quality = DataQualityStatus.OK.value
    elif all(q == DataQualityStatus.INSUFFICIENT_DATA.value for q in qualities):
        agg_quality = DataQualityStatus.INSUFFICIENT_DATA.value
    else:
        agg_quality = DataQualityStatus.SUSPECT.value

    portfolio = {
        "policy": "deterministic_max_savings_deduplication",
        "description": (
            "Prevent double-counting across overlapping scopes (same building or campus vs building) "
            "by deterministically selecting maximum-saving non-overlapping initiatives."
        ),
        "overlapping_scenarios_detected": len(overlap_groups) > 0,
        "included_scenario_ids": sorted(set(portfolio_included_ids)),
        "excluded_scenario_ids": sorted(set(portfolio_excluded_ids)),
        "overlap_groups": overlap_groups,
    }

    return {
        "simulation_contract_version": SIMULATION_CONTRACT_VERSION,
        "engine_version": __version__,
        "projection_type": "hypothetical",
        "scenarios_count": len(sim_results),
        "scenarios": sim_results,
        "resource_totals": resource_totals,
        "total_co2e_savings_kg": total_portfolio_co2e,
        "co2e_supported": has_any_co2e_factor,
        "rankings": rankings,
        "portfolio": portfolio,
        "data_quality_status": agg_quality,
        "assumptions": assumptions,
        "warnings": warnings,
    }


_COMPARE_REQUEST_KEYS = ("records", "scenarios")


def compare_payload(
    payload: Any,
    config: Optional[AnalysisConfig] = None,
    *,
    max_records: Optional[int] = None,
) -> AnalysisOutcome:
    """Decoded JSON request body -> ``AnalysisOutcome`` for scenario comparison (Phase 7B).

    Request: ``{"records": [...], "scenarios": [...]}``.
    """
    from .contract import DEFAULT_MAX_RECORDS

    if config is not None and not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig or None")
    limit = DEFAULT_MAX_RECORDS if max_records is None else max_records

    if not isinstance(payload, Mapping):
        return error_outcome(422, "invalid_request", "Request body must be a JSON object.",
                             [_detail([], "must be an object")])
    unknown = sorted(str(k) for k in payload if k not in _COMPARE_REQUEST_KEYS)
    if unknown:
        return error_outcome(
            422, "invalid_request",
            f"Unknown field(s): {', '.join(unknown)}. Allowed: {', '.join(_COMPARE_REQUEST_KEYS)}.",
            [_detail([k], "unknown field") for k in unknown],
        )
    scenarios = payload.get("scenarios")
    if scenarios is None or not isinstance(scenarios, list):
        return error_outcome(422, "invalid_request", "scenarios must be a JSON array.",
                             [_detail(["scenarios"], "must be an array")])
    for i, s in enumerate(scenarios):
        if not isinstance(s, Mapping):
            return error_outcome(422, "invalid_request", f"Scenario at index {i} must be a JSON object.",
                                 [_detail(["scenarios", i], "must be an object")])
    records = payload.get("records")
    if records is None:
        records = []
    if not isinstance(records, list):
        return error_outcome(422, "invalid_request", "records must be a JSON array when present.",
                             [_detail(["records"], "must be an array")])
    if len(records) > limit:
        return error_outcome(413, "too_many_records",
                             f"At most {limit} records per request; received {len(records)}.",
                             [_detail(["records"], f"maximum {limit} items")])
    try:
        result = compare_scenarios(records, scenarios, config)
    except ScenarioValidationError as exc:
        return error_outcome(
            422, "invalid_scenario",
            f"Scenario validation failed ({len(exc.errors)} problem(s)); nothing was simulated.",
            [_detail(["scenarios", *e["loc"]], e["msg"]) for e in exc.errors],
        )
    except RecordValidationError as exc:
        return error_outcome(
            422, "invalid_records",
            f"{len(exc.errors)} record(s) failed validation; no simulation was performed.",
            validation_error_body(exc),
        )
    except Exception:  # noqa: BLE001
        return error_outcome(500, "simulation_failed", "The scenario comparison could not be completed.")

    body: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "simulation_contract_version": SIMULATION_CONTRACT_VERSION,
        "status": "ok",
        "engine_version": __version__,
        "result": result,
    }
    return AnalysisOutcome(ok=True, http_status=200, body=body)
