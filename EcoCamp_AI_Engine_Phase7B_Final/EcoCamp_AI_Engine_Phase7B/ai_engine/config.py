"""Configuration and constants for the EcoCamp AI engine.

Definitional facts (which resources are supported, which unit is the standard
for each), detection settings and impact-calculation settings live here.

No environmental conversion factor (e.g. kg CO2e per kWh) is built in. Every
factor defaults to ``None`` (unavailable) and must be supplied by the caller
from a source they can cite. When a factor is ``None`` the engine reports the
physical saving (kWh, liters, kg) but no CO2e figure.
"""
from __future__ import annotations

import os
import re
from enum import Enum
from typing import Any, Dict, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ResourceType(str, Enum):
    """Supported campus resource categories."""

    ELECTRICITY = "electricity"
    WATER = "water"
    WASTE = "waste"


# Standard (canonical) unit for each resource type.
STANDARD_UNITS: Dict[ResourceType, str] = {
    ResourceType.ELECTRICITY: "kWh",
    ResourceType.WATER: "liters",
    ResourceType.WASTE: "kg",
}

# Accepted spellings of the *same* unit, mapped to the canonical unit.
# These are pure spelling/case aliases, NOT conversions: "Wh", "MWh",
# "gallons", "lbs", etc. are deliberately absent and are rejected.
UNIT_ALIASES: Dict[ResourceType, Dict[str, str]] = {
    ResourceType.ELECTRICITY: {"kwh": "kWh"},
    ResourceType.WATER: {"liters": "liters", "litres": "liters", "l": "liters"},
    ResourceType.WASTE: {"kg": "kg"},
}


class AnomalyMethod(str, Enum):
    """Supported anomaly scoring methods."""

    MAD = "mad"
    ISOLATION_FOREST = "isolation_forest"


class LLMProvider(str, Enum):
    """Supported LLM providers (official SDKs: ``openai`` and ``google-genai``)."""

    GEMINI = "gemini"
    OPENAI = "openai"


# Default environment variable that holds each provider's API key. The key itself
# is never stored in configuration or code; it is read from the environment at
# call time.
DEFAULT_API_KEY_ENV_VARS: Dict[LLMProvider, str] = {
    LLMProvider.GEMINI: "GEMINI_API_KEY",
    LLMProvider.OPENAI: "OPENAI_API_KEY",
}

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}


class LLMConfig(BaseModel):
    """Optional LLM settings. **Disabled by default.**

    When disabled (the default) the engine never reads API keys, never imports
    an LLM SDK and never touches the network.

    Attributes:
        enabled: Turn LLM explanations on.
        provider: ``gemini`` or ``openai``.
        model: Provider model name. Required when enabled; no default is
            shipped because model names change over time.
        api_key_env_var: Name of the environment variable holding the key
            (defaults to ``GEMINI_API_KEY`` / ``OPENAI_API_KEY``). Must be a
            variable *name*, so pasting a real key here is rejected.
        timeout_seconds: Per-request timeout passed to the SDK.
        max_llm_recommendations: Max LLM calls per ``analyze_consumption`` call
            (highest priority first); the rest use deterministic text only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    provider: LLMProvider = LLMProvider.GEMINI
    model: Optional[str] = None
    api_key_env_var: Optional[str] = None
    timeout_seconds: float = Field(default=20.0, gt=0, le=300, allow_inf_nan=False)
    max_llm_recommendations: int = Field(default=10, ge=1, le=500)

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def _reject_bool_timeout(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("timeout_seconds must be a number, not a boolean")
        return v

    @field_validator("model")
    @classmethod
    def _clean_model(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("model must be a non-empty string or None")
        return v

    @field_validator("api_key_env_var")
    @classmethod
    def _check_env_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not _ENV_NAME.match(v):
            raise ValueError(
                "api_key_env_var must be an environment variable NAME "
                "(letters, digits, underscore); never put the key itself here"
            )
        return v

    @model_validator(mode="after")
    def _check_enabled(self) -> "LLMConfig":
        if self.enabled and self.model is None:
            raise ValueError("llm.model is required when the LLM is enabled")
        return self

    @property
    def resolved_api_key_env_var(self) -> str:
        return self.api_key_env_var or DEFAULT_API_KEY_ENV_VARS[self.provider]

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "LLMConfig":
        """Build a config from ``ECOCAMP_LLM_*`` environment variables.

        ``ECOCAMP_LLM_ENABLED`` (true/false), ``ECOCAMP_LLM_PROVIDER``,
        ``ECOCAMP_LLM_MODEL``, ``ECOCAMP_LLM_API_KEY_ENV``,
        ``ECOCAMP_LLM_TIMEOUT_SECONDS``, ``ECOCAMP_LLM_MAX_RECOMMENDATIONS``.
        Unset variables keep their defaults. The API key is *not* read here.
        """
        env = os.environ if environ is None else environ
        raw_enabled = env.get("ECOCAMP_LLM_ENABLED", "").strip().lower()
        if raw_enabled in _TRUE:
            enabled = True
        elif raw_enabled in _FALSE:
            enabled = False
        else:
            raise ValueError("ECOCAMP_LLM_ENABLED must be true or false")
        kwargs: Dict[str, Any] = {"enabled": enabled}
        if env.get("ECOCAMP_LLM_PROVIDER", "").strip():
            kwargs["provider"] = env["ECOCAMP_LLM_PROVIDER"].strip().lower()
        if env.get("ECOCAMP_LLM_MODEL", "").strip():
            kwargs["model"] = env["ECOCAMP_LLM_MODEL"]
        if env.get("ECOCAMP_LLM_API_KEY_ENV", "").strip():
            kwargs["api_key_env_var"] = env["ECOCAMP_LLM_API_KEY_ENV"]
        if env.get("ECOCAMP_LLM_TIMEOUT_SECONDS", "").strip():
            kwargs["timeout_seconds"] = float(env["ECOCAMP_LLM_TIMEOUT_SECONDS"])
        if env.get("ECOCAMP_LLM_MAX_RECOMMENDATIONS", "").strip():
            kwargs["max_llm_recommendations"] = int(env["ECOCAMP_LLM_MAX_RECOMMENDATIONS"])
        return cls(**kwargs)


class WasteEmissionFactor(BaseModel):
    """A caller-supplied waste CO2e factor with an explicit unit and source.

    Only one unit is accepted (``kg_co2e_per_kg``: kg CO2e per kg of waste
    avoided) so that a factor can never be applied with the wrong unit. A
    ``source`` is mandatory because waste carbon equivalence depends heavily on
    the waste stream and disposal route; the engine will not guess it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(ge=0, allow_inf_nan=False)
    unit: Literal["kg_co2e_per_kg"]
    source: str

    @field_validator("value", mode="before")
    @classmethod
    def _reject_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("value must be a number, not a boolean")
        return v

    @field_validator("source")
    @classmethod
    def _source_required(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("source must be a non-empty citation or description")
        return v


class AnalysisConfig(BaseModel):
    """Runtime configuration for ``analyze_consumption``.

    All defaults are engineering starting points, not values derived from
    campus data. Tune them against real data before relying on the output.
    Unknown fields are rejected so typos do not pass silently.

    Attributes:
        baseline_window: Max number of earlier observations (per building and
            resource type) used to build the rolling baseline.
        minimum_history: Min earlier observations needed to compute a baseline.
            Fewer -> ``insufficient_data``. Must be >= 3 (MAD is degenerate
            below that).
        deviation_threshold_percentage: |deviation from baseline| (in %) at or
            above which a record can be flagged.
        anomaly_method: ``mad`` (default) or ``isolation_forest``.
        robust_z_threshold: Robust z-score cutoff for the MAD method. 3.5 is
            the conventional Iglewicz-Hoaglin value.
        isolation_forest_min_history: Min earlier observations before the
            Isolation Forest is used; below it, MAD scoring is used instead.
        random_seed: Seed for Isolation Forest, so results are reproducible.
        achievable_reduction_fraction: Fraction (0 to 1 inclusive) of the
            above-baseline excess assumed avoidable. 0.5 is a placeholder, not
            a measured value; set it from campus experience or a cited study.
        electricity_emission_factor_kg_co2e_per_kwh: kg CO2e per kWh, or
            ``None`` if unavailable (then no electricity CO2e is reported).
        electricity_emission_factor_source: Optional citation for that factor
            (recorded in ``calculation_assumptions`` when given).
        water_emission_factor_kg_co2e_per_liter: kg CO2e per liter of water
            supplied/treated, or ``None``. Requires a documented source.
        water_emission_factor_source: Citation for the water factor. Required
            whenever the water factor is set.
        waste_emission_factor: Optional ``WasteEmissionFactor`` (value, explicit
            unit, source). ``None`` means no waste CO2e is claimed.
        llm: ``LLMConfig`` for optional LLM explanations (disabled by default).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_window: int = Field(default=30, ge=3)
    minimum_history: int = Field(default=7, ge=3)
    deviation_threshold_percentage: float = Field(default=50.0, gt=0, allow_inf_nan=False)
    anomaly_method: AnomalyMethod = AnomalyMethod.MAD
    robust_z_threshold: float = Field(default=3.5, gt=0, allow_inf_nan=False)
    isolation_forest_min_history: int = Field(default=50, ge=10)
    random_seed: int = 42

    # --- Phase 3: environmental impact ------------------------------------ #
    achievable_reduction_fraction: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    electricity_emission_factor_kg_co2e_per_kwh: Optional[float] = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    electricity_emission_factor_source: Optional[str] = None
    water_emission_factor_kg_co2e_per_liter: Optional[float] = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    water_emission_factor_source: Optional[str] = None
    waste_emission_factor: Optional[WasteEmissionFactor] = None

    # --- Phase 4: optional LLM (disabled by default) ------------------------ #
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @field_validator(
        "achievable_reduction_fraction",
        "electricity_emission_factor_kg_co2e_per_kwh",
        "water_emission_factor_kg_co2e_per_liter",
        mode="before",
    )
    @classmethod
    def _reject_bool_numbers(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("must be a number, not a boolean")
        return v

    @field_validator(
        "electricity_emission_factor_source", "water_emission_factor_source"
    )
    @classmethod
    def _clean_source(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("source must be a non-empty string or None")
        return v

    @model_validator(mode="after")
    def _check_impact_consistency(self) -> "AnalysisConfig":
        if (
            self.water_emission_factor_kg_co2e_per_liter is not None
            and self.water_emission_factor_source is None
        ):
            raise ValueError(
                "water_emission_factor_source is required when "
                "water_emission_factor_kg_co2e_per_liter is set "
                "(the conversion must be documented)"
            )
        if (
            self.electricity_emission_factor_source is not None
            and self.electricity_emission_factor_kg_co2e_per_kwh is None
        ):
            raise ValueError(
                "electricity_emission_factor_source given without "
                "electricity_emission_factor_kg_co2e_per_kwh"
            )
        if (
            self.water_emission_factor_source is not None
            and self.water_emission_factor_kg_co2e_per_liter is None
        ):
            raise ValueError(
                "water_emission_factor_source given without "
                "water_emission_factor_kg_co2e_per_liter"
            )
        return self

    @model_validator(mode="after")
    def _check_consistency(self) -> "AnalysisConfig":
        if self.baseline_window < self.minimum_history:
            raise ValueError("baseline_window must be >= minimum_history")
        if (
            self.anomaly_method is AnomalyMethod.ISOLATION_FOREST
            and self.baseline_window < self.isolation_forest_min_history
        ):
            raise ValueError(
                "anomaly_method='isolation_forest' needs baseline_window >= "
                f"isolation_forest_min_history ({self.isolation_forest_min_history})"
            )
        return self
