"""Environmental impact estimation (Phase 3).

Formula
-------
::

    estimated_savings = max(0, observed_value - baseline_value)
                        x achievable_reduction_fraction

Savings are expressed in the resource's own standard unit (kWh, liters, kg).
They are calculated **only** when

* a valid baseline exists (not ``None`` and greater than 0), and
* the record is a qualifying above-baseline anomaly (``severity != none``,
  ``observed_value > baseline_value`` and ``data_quality_status == ok``).

Anything else yields ``estimated_savings = None`` with an explanatory
``calculation_status``; savings are never negative.

CO2e
----
``estimated_co2e_reduction_kg = estimated_savings x factor`` is reported only
when the caller configured a valid factor for that resource:

* electricity: ``electricity_emission_factor_kg_co2e_per_kwh``
* water: ``water_emission_factor_kg_co2e_per_liter`` (a source is mandatory)
* waste: ``waste_emission_factor`` (explicit unit and source)

No factor is built into the engine. With a factor of ``None`` the CO2e field
is ``None`` and no carbon-equivalence claim is made.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .anomaly_detector import _utc
from .config import STANDARD_UNITS, AnalysisConfig, ResourceType
from .schemas import (
    AnalysisResult,
    CalculationStatus,
    ConsumptionRecord,
    DataQualityStatus,
    EstimatedSavings,
    ImpactSummary,
    ImpactTotal,
    Severity,
)

_ROUND = 6


@dataclass(frozen=True)
class ImpactEstimate:
    """Outcome of an impact calculation for a single record."""

    status: CalculationStatus
    savings_value: Optional[float] = None
    savings_unit: Optional[str] = None
    co2e_reduction_kg: Optional[float] = None
    assumptions: List[str] = field(default_factory=list)


def _co2e_factor(
    resource: ResourceType, cfg: AnalysisConfig
) -> Tuple[Optional[float], str, List[str]]:
    """Return ``(factor, unit_label, assumption_lines)`` for a resource.

    ``factor`` is ``None`` when unavailable; the lines say why or what was used.
    """
    if resource is ResourceType.ELECTRICITY:
        f = cfg.electricity_emission_factor_kg_co2e_per_kwh
        unit = "kg CO2e/kWh"
        if f is None:
            return None, unit, [
                "No electricity emission factor configured; CO2e reduction not calculated."
            ]
        src = cfg.electricity_emission_factor_source
        return f, unit, [
            "CO2e reduction = estimated savings (kWh) x "
            f"{f:g} {unit}"
            + (f" (source: {src})." if src else " (no source recorded).")
        ]
    if resource is ResourceType.WATER:
        f = cfg.water_emission_factor_kg_co2e_per_liter
        unit = "kg CO2e/liter"
        if f is None:
            return None, unit, [
                "No documented water CO2e conversion factor configured; "
                "CO2e reduction not calculated."
            ]
        return f, unit, [
            "CO2e reduction = estimated savings (liters) x "
            f"{f:g} {unit} (source: {cfg.water_emission_factor_source})."
        ]
    w = cfg.waste_emission_factor
    unit = "kg CO2e/kg"
    if w is None:
        return None, unit, [
            "No waste emission factor configured; no carbon-equivalence is "
            "claimed for avoided waste."
        ]
    return w.value, unit, [
        "CO2e reduction = estimated savings (kg waste) x "
        f"{w.value:g} {unit} (source: {w.source})."
    ]


def estimate_impact(
    record: ConsumptionRecord,
    baseline_value: Optional[float],
    config: Optional[AnalysisConfig] = None,
) -> ImpactEstimate:
    """Apply the savings formula to one record and a baseline.

    This is the pure calculation. It does **not** decide whether the record is
    an anomaly; ``apply_impact`` adds that gate. Observed values at or below the
    baseline give ``NOT_APPLICABLE`` (never a negative saving).
    """
    cfg = config or AnalysisConfig()
    resource = record.resource_type
    unit = STANDARD_UNITS[resource]
    observed = record.consumption_value

    if baseline_value is None or not math.isfinite(baseline_value) or baseline_value < 0:
        return ImpactEstimate(
            CalculationStatus.INSUFFICIENT_BASELINE,
            assumptions=["Savings not estimated: no valid baseline is available."],
        )
    if baseline_value == 0:
        status = (
            CalculationStatus.NOT_APPLICABLE
            if observed == 0
            else CalculationStatus.INSUFFICIENT_BASELINE
        )
        return ImpactEstimate(
            status,
            assumptions=["Savings not estimated: the baseline is 0, so there is no reference level."],
        )
    if observed <= baseline_value:
        return ImpactEstimate(
            CalculationStatus.NOT_APPLICABLE,
            assumptions=["Savings not estimated: observed value is not above the baseline."],
        )

    fraction = cfg.achievable_reduction_fraction
    savings = round(max(0.0, observed - baseline_value) * fraction, _ROUND)
    assumptions = [
        "estimated_savings = max(0, observed_value - baseline_value) x "
        f"achievable_reduction_fraction ({fraction:g}); expressed in {unit}.",
        "achievable_reduction_fraction is a configured assumption, not a measured value.",
    ]
    factor, _, lines = _co2e_factor(resource, cfg)
    assumptions.extend(lines)
    if factor is None:
        return ImpactEstimate(
            CalculationStatus.CALCULATED_WITHOUT_CO2E, savings, unit, None, assumptions
        )
    co2e = round(savings * factor, _ROUND)
    return ImpactEstimate(CalculationStatus.CALCULATED, savings, unit, co2e, assumptions)


def apply_impact(result: AnalysisResult, config: Optional[AnalysisConfig] = None) -> AnalysisResult:
    """Return a copy of ``result`` with the Phase 3 fields filled in.

    Only a qualifying above-baseline anomaly gets savings. All other fields of
    ``result`` are left untouched (the input object is not mutated).
    """
    cfg = config or AnalysisConfig()
    qualifying = (
        result.severity is not Severity.NONE
        and result.data_quality_status is DataQualityStatus.OK
        and result.baseline_value is not None
        and result.baseline_value > 0
        and result.observed_value > result.baseline_value
    )

    if qualifying:
        record = ConsumptionRecord(
            timestamp=result.timestamp,
            building=result.building,
            resource_type=result.resource_type,
            consumption_value=result.observed_value,
            unit=result.unit,
        )
        est = estimate_impact(record, result.baseline_value, cfg)
    elif result.baseline_value is None or (
        result.baseline_value == 0 and result.observed_value > 0
    ):
        est = ImpactEstimate(
            CalculationStatus.INSUFFICIENT_BASELINE,
            assumptions=["Savings not estimated: no valid baseline is available."],
        )
    else:
        est = ImpactEstimate(
            CalculationStatus.NOT_APPLICABLE,
            assumptions=[
                "Savings not estimated: reading is not a qualifying above-baseline anomaly."
            ],
        )

    savings = (
        None
        if est.savings_value is None
        else EstimatedSavings(value=est.savings_value, unit=est.savings_unit or "")
    )
    return result.model_copy(
        update={
            "estimated_savings": savings,
            "savings_unit": est.savings_unit,
            "estimated_co2e_reduction_kg": est.co2e_reduction_kg,
            "calculation_status": est.status,
            "calculation_assumptions": list(result.calculation_assumptions) + est.assumptions,
        }
    )


def aggregate_impact(
    results: Sequence[AnalysisResult], by_building: bool = False
) -> ImpactSummary:
    """Aggregate per-record savings without double counting.

    * Only records with a calculated saving are counted.
    * A record is counted once: results sharing (building, resource type,
      timestamp) are treated as duplicates and only the first (input order)
      counts. Duplicates are reported in ``duplicate_records_skipped``.
    * Savings are summed only within one resource (same unit); kWh, liters and
      kg are never added together.
    * CO2e is summed from each record's own CO2e figure. If any counted record
      lacks one, ``co2e_complete`` is ``False`` and the total is a partial sum
      (``None`` if no record has CO2e).
    """
    seen: Set[Tuple[str, ResourceType, object]] = set()
    duplicates = 0
    groups: Dict[Tuple[str, ResourceType], List[AnalysisResult]] = {}
    counted: List[AnalysisResult] = []

    for r in results:
        key = (r.building, r.resource_type, _utc(r.timestamp))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if r.estimated_savings is None or r.calculation_status not in (
            CalculationStatus.CALCULATED,
            CalculationStatus.CALCULATED_WITHOUT_CO2E,
        ):
            continue
        counted.append(r)
        gkey = (r.building if by_building else "", r.resource_type)
        groups.setdefault(gkey, []).append(r)

    totals: List[ImpactTotal] = []
    for (building, resource), items in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1].value)
    ):
        co2e_vals = [i.estimated_co2e_reduction_kg for i in items if i.estimated_co2e_reduction_kg is not None]
        totals.append(
            ImpactTotal(
                building=building if by_building else None,
                resource_type=resource,
                savings_value=round(math.fsum(i.estimated_savings.value for i in items), _ROUND),
                savings_unit=STANDARD_UNITS[resource],
                co2e_reduction_kg=round(math.fsum(co2e_vals), _ROUND) if co2e_vals else None,
                co2e_complete=len(co2e_vals) == len(items),
                records_counted=len(items),
            )
        )

    all_co2e = [c.estimated_co2e_reduction_kg for c in counted if c.estimated_co2e_reduction_kg is not None]
    complete = len(all_co2e) == len(counted)
    notes = [
        "Savings are totalled per resource in its own unit; kWh, liters and kg are never summed together.",
        "Each (building, resource, timestamp) is counted once.",
    ]
    if duplicates:
        notes.append(f"{duplicates} duplicate record(s) skipped to avoid double counting.")
    if counted and not complete:
        notes.append(
            "total_co2e_reduction_kg covers only records with a configured emission "
            "factor; it is a partial sum."
        )
    return ImpactSummary(
        totals=totals,
        total_co2e_reduction_kg=round(math.fsum(all_co2e), _ROUND) if all_co2e else None,
        co2e_complete=complete,
        records_counted=len(counted),
        duplicate_records_skipped=duplicates,
        notes=notes,
    )
