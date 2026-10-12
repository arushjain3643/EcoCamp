"""Pydantic input and output schemas for the EcoCamp AI engine."""
from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import STANDARD_UNITS, UNIT_ALIASES, ResourceType


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
class ConsumptionRecord(BaseModel):
    """A single resource-consumption measurement for one building."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: datetime
    building: str
    resource_type: ResourceType
    consumption_value: float
    unit: str

    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamp(cls, v: Any) -> Any:
        # Accept datetime objects or ISO-8601 strings only. Pydantic would
        # otherwise coerce ints/floats as Unix epochs, which hides malformed data.
        if isinstance(v, datetime):
            return v
        if isinstance(v, str) and v.strip():
            return v.strip()
        raise ValueError("timestamp must be a datetime or a non-empty ISO-8601 string")

    @field_validator("building")
    @classmethod
    def _validate_building(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("building must be a non-empty string")
        return v

    @field_validator("resource_type", mode="before")
    @classmethod
    def _normalize_resource_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().lower()
        try:
            return ResourceType(v)
        except ValueError:
            allowed = ", ".join(r.value for r in ResourceType)
            raise ValueError(
                f"unsupported resource_type {v!r}; supported: {allowed}"
            ) from None

    @field_validator("consumption_value", mode="before")
    @classmethod
    def _reject_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("consumption_value must be a number, not a boolean")
        return v

    @field_validator("consumption_value")
    @classmethod
    def _validate_value(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("consumption_value must be a finite number")
        if v < 0:
            raise ValueError("consumption_value must not be negative")
        return v

    @model_validator(mode="after")
    def _validate_unit(self) -> "ConsumptionRecord":
        aliases = UNIT_ALIASES[self.resource_type]
        canonical = aliases.get(self.unit.strip().lower())
        if canonical is None:
            raise ValueError(
                f"incompatible unit {self.unit!r} for {self.resource_type.value}; "
                f"expected {STANDARD_UNITS[self.resource_type]!r}"
            )
        # Frozen model: bypass __setattr__ to store the canonical spelling.
        object.__setattr__(self, "unit", canonical)
        return self


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataQualityStatus(str, Enum):
    OK = "ok"
    INSUFFICIENT_DATA = "insufficient_data"
    SUSPECT = "suspect"


class CalculationStatus(str, Enum):
    """Outcome of the Phase 3 environmental-impact calculation for one record."""

    # Savings and CO2e reduction both calculated.
    CALCULATED = "calculated"
    # Savings calculated; CO2e not reported because no valid factor is configured.
    CALCULATED_WITHOUT_CO2E = "calculated_without_co2e"
    # No qualifying above-baseline anomaly (normal, below baseline, or flagged
    # data-quality case), so nothing to estimate.
    NOT_APPLICABLE = "not_applicable"
    # No valid baseline (insufficient history, or zero baseline with positive use).
    INSUFFICIENT_BASELINE = "insufficient_baseline"


class EstimatedSavings(BaseModel):
    """Estimated savings, expressed in the resource's own standard unit."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0)
    unit: str


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class AnomalyType(str, Enum):
    """What kind of finding a recommendation responds to."""

    ABOVE_BASELINE = "above_baseline"
    BELOW_BASELINE = "below_baseline"
    SUSPECT_READING = "suspect_reading"


class RecommendationSource(str, Enum):
    RULES = "rules"  # deterministic rules only
    RULES_LLM = "rules_llm"  # rules plus validated LLM explanation


class LLMStatus(str, Enum):
    """Why the LLM text is, or is not, present on a recommendation."""

    DISABLED = "disabled"  # default: LLM not enabled, nothing attempted
    USED = "used"  # validated LLM output attached
    MISSING_API_KEY = "missing_api_key"
    SDK_UNAVAILABLE = "sdk_unavailable"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    INVALID_RESPONSE = "invalid_response"
    CALL_LIMIT_REACHED = "call_limit_reached"


class Recommendation(BaseModel):
    """Evidence-based recommendation for one flagged reading.

    ``supporting_evidence``, ``recommended_actions``, ``estimated_savings``,
    ``explanation`` and every number are produced by deterministic Python code.
    The ``llm_*`` fields are optional additions that were validated; they never
    replace or change the deterministic fields.
    """

    model_config = ConfigDict(extra="forbid")

    resource_type: ResourceType
    building: str
    severity: Severity
    priority: Priority
    anomaly_type: AnomalyType
    supporting_evidence: List[str]
    recommended_actions: List[str] = Field(min_length=1)
    estimated_savings: Optional[EstimatedSavings] = None
    estimated_co2e_reduction_kg: Optional[float] = Field(default=None, ge=0)
    explanation: str
    caveats: List[str] = Field(default_factory=list)
    source: RecommendationSource = RecommendationSource.RULES
    llm_status: LLMStatus = LLMStatus.DISABLED
    llm_explanation: Optional[str] = None
    llm_suggestions: List[str] = Field(default_factory=list)
    llm_impact_summary: Optional[str] = None


class AnalysisResult(BaseModel):
    """Structured analysis of one consumption record."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    building: str
    resource_type: ResourceType
    unit: str

    observed_value: float = Field(ge=0)
    baseline_value: Optional[float] = Field(default=None, ge=0)
    deviation_percentage: Optional[float] = None
    anomaly_score: float = Field(ge=0.0, le=1.0)
    severity: Severity
    explanation: str
    recommended_action: Optional[str] = None
    estimated_savings: Optional[EstimatedSavings] = None
    calculation_assumptions: List[str] = Field(default_factory=list)
    data_quality_status: DataQualityStatus
    # --- Phase 3 (appended so existing field order is unchanged) ----------- #
    savings_unit: Optional[str] = None
    estimated_co2e_reduction_kg: Optional[float] = Field(default=None, ge=0)
    calculation_status: CalculationStatus = CalculationStatus.NOT_APPLICABLE
    # --- Phase 4 (appended; ``None`` for normal / insufficient-data records) -- #
    recommendation: Optional[Recommendation] = None


class ImpactTotal(BaseModel):
    """Aggregated avoidable consumption for one resource (optionally one building).

    Savings are summed only within a single resource and unit, never across
    resources (kWh + liters + kg is meaningless).
    """

    model_config = ConfigDict(extra="forbid")

    building: Optional[str] = None
    resource_type: ResourceType
    savings_value: float = Field(ge=0)
    savings_unit: str
    co2e_reduction_kg: Optional[float] = Field(default=None, ge=0)
    co2e_complete: bool
    records_counted: int = Field(ge=0)


class ImpactSummary(BaseModel):
    """Result of ``aggregate_impact``: de-duplicated totals, no double counting."""

    model_config = ConfigDict(extra="forbid")

    totals: List[ImpactTotal]
    total_co2e_reduction_kg: Optional[float] = Field(default=None, ge=0)
    co2e_complete: bool
    records_counted: int = Field(ge=0)
    duplicate_records_skipped: int = Field(ge=0)
    notes: List[str] = Field(default_factory=list)
