"""JSON contract helpers for the backend integration (Phase 5).

This module is an *additive* layer on top of ``analyze_consumption``. It does
not change detection, impact or recommendation behaviour. It gives the FastAPI
backend one well-defined, JSON-safe entry point:

    outcome = analyze_payload(request_json, config)
    return JSONResponse(outcome.body, status_code=outcome.http_status)

* ``analyze_payload`` takes the decoded JSON request body and returns an
  ``AnalysisOutcome`` (never raises for client mistakes). The body is plain
  Python ``dict``/``list``/``str``/``float``/``bool``/``None`` only, so it
  serializes with ``json.dumps`` or any web framework.
* ``serialize_results`` / ``serialize_result`` convert ``AnalysisResult``
  objects to JSON-safe dicts (``model_dump(mode="json")``): timestamps become
  ISO-8601 strings, enums become their string values, optional fields stay
  present as ``null`` and nested objects become nested dicts.
* ``engine_info`` describes the engine (versions, units, LLM state) for a
  health or diagnostics endpoint. It never exposes an API key.

The full contract is documented in ``docs/JSON_CONTRACT.md``.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import ValidationError

from ._version import CONTRACT_VERSION, __version__
from .analyzer import RecordValidationError, analyze_consumption
from .config import STANDARD_UNITS, AnalysisConfig
from .impact_calculator import aggregate_impact
from .schemas import (
    AnalysisResult,
    ConsumptionRecord,
    DataQualityStatus,
    ImpactSummary,
    LLMStatus,
    Severity,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_MAX_RECORDS",
    "TUNABLE_OPTIONS",
    "AnalysisOutcome",
    "analyze_payload",
    "engine_info",
    "error_outcome",
    "json_schemas",
    "serialize_result",
    "serialize_results",
    "summarize_results",
    "to_json_string",
    "validation_error_body",
]

# Upper bound on records per request. Detection is O(n) per group with a small
# constant, but the LLM layer and very large bodies are not free; the backend
# can pass its own ``max_records``.
DEFAULT_MAX_RECORDS = 20000

# Per-request overrides a client may send in ``options.config``. Deliberately
# excluded: emission factors (need a cited source, set server-side), the LLM
# settings and API keys (server-side only), and ``anomaly_method`` (Isolation
# Forest refits a model per record and is expensive).
TUNABLE_OPTIONS = (
    "baseline_window",
    "minimum_history",
    "deviation_threshold_percentage",
    "robust_z_threshold",
    "achievable_reduction_fraction",
)

_REQUEST_KEYS = ("records", "options")
_OPTION_KEYS = ("include_impact_summary", "impact_by_building", "config")


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def serialize_result(result: AnalysisResult) -> Dict[str, Any]:
    """One ``AnalysisResult`` as a JSON-safe dict.

    Uses ``model_dump(mode="json")``: datetimes -> ISO-8601 strings, enums ->
    their string values, nested models -> dicts, ``None`` kept as ``null``.
    (Plain ``model_dump()`` leaves ``datetime`` objects, which ``json.dumps``
    cannot encode.)
    """
    return result.model_dump(mode="json")


def serialize_results(results: Sequence[AnalysisResult]) -> List[Dict[str, Any]]:
    """Serialize results, preserving order (``results[i]`` <-> ``records[i]``)."""
    return [serialize_result(r) for r in results]


def to_json_string(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` for contract bodies. Rejects NaN/Infinity (invalid JSON)."""
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("allow_nan", False)
    return json.dumps(obj, **kwargs)


def validation_error_body(exc: RecordValidationError) -> List[Dict[str, Any]]:
    """``RecordValidationError.errors`` as JSON-safe ``details`` (loc tuples -> lists)."""
    details: List[Dict[str, Any]] = []
    for item in exc.errors:
        details.append({
            "index": int(item["index"]),
            "errors": [
                {
                    "loc": [p if isinstance(p, (str, int)) else str(p) for p in e["loc"]],
                    "msg": str(e["msg"]),
                }
                for e in item["errors"]
            ],
        })
    return details


def summarize_results(results: Sequence[AnalysisResult]) -> Dict[str, Any]:
    """Counts for dashboards. Every enum value is present (zero-filled)."""
    severities = Counter(r.severity.value for r in results)
    qualities = Counter(r.data_quality_status.value for r in results)
    llm_states = Counter(
        r.recommendation.llm_status.value for r in results if r.recommendation is not None
    )
    return {
        "records_analyzed": len(results),
        "anomalies_flagged": sum(1 for r in results if r.severity is not Severity.NONE),
        "recommendations_generated": sum(1 for r in results if r.recommendation is not None),
        "severity_counts": {s.value: severities.get(s.value, 0) for s in Severity},
        "data_quality_counts": {s.value: qualities.get(s.value, 0) for s in DataQualityStatus},
        "llm_status_counts": {s.value: llm_states.get(s.value, 0) for s in LLMStatus},
    }


# --------------------------------------------------------------------------- #
# Request/response facade
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnalysisOutcome:
    """Result of ``analyze_payload``.

    Attributes:
        ok: ``True`` for a successful analysis.
        http_status: Suggested HTTP status (200, 413, 422 or 500).
        body: JSON-safe response body (success or error envelope).
    """

    ok: bool
    http_status: int
    body: Dict[str, Any]


def error_outcome(
    status: int, code: str, message: str, details: Optional[List[Dict[str, Any]]] = None
) -> AnalysisOutcome:
    """Build the standard error envelope (for errors the backend detects itself)."""
    error: Dict[str, Any] = {"code": code, "message": message, "details": details or []}
    return AnalysisOutcome(
        ok=False,
        http_status=status,
        body={"contract_version": CONTRACT_VERSION, "status": "error", "error": error},
    )


def _detail(loc: Sequence[Any], msg: str) -> Dict[str, Any]:
    return {"index": None, "errors": [{"loc": list(loc), "msg": msg}]}


def _parse_options(
    options: Any, base: AnalysisConfig
) -> "tuple[Optional[AnalysisConfig], bool, bool, Optional[AnalysisOutcome]]":
    """Validate ``options``. Returns (config, include_impact, by_building, error)."""
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        return None, False, False, error_outcome(
            422, "invalid_options", "options must be a JSON object.",
            [_detail(["options"], "must be an object")],
        )
    unknown = sorted(str(k) for k in options if k not in _OPTION_KEYS)
    if unknown:
        return None, False, False, error_outcome(
            422, "invalid_options",
            f"Unknown option(s): {', '.join(unknown)}. Allowed: {', '.join(_OPTION_KEYS)}.",
            [_detail(["options", k], "unknown option") for k in unknown],
        )
    flags: Dict[str, bool] = {}
    for key in ("include_impact_summary", "impact_by_building"):
        value = options.get(key, False)
        if not isinstance(value, bool):
            return None, False, False, error_outcome(
                422, "invalid_options", f"options.{key} must be true or false.",
                [_detail(["options", key], "must be a boolean")],
            )
        flags[key] = value

    cfg = base
    overrides = options.get("config")
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            return None, False, False, error_outcome(
                422, "invalid_options", "options.config must be a JSON object.",
                [_detail(["options", "config"], "must be an object")],
            )
        bad = sorted(str(k) for k in overrides if k not in TUNABLE_OPTIONS)
        if bad:
            return None, False, False, error_outcome(
                422, "invalid_options",
                f"options.config may only set: {', '.join(TUNABLE_OPTIONS)}.",
                [_detail(["options", "config", k], "not adjustable per request") for k in bad],
            )
        if overrides:
            try:
                cfg = AnalysisConfig.model_validate({**base.model_dump(), **dict(overrides)})
            except ValidationError as exc:
                return None, False, False, error_outcome(
                    422, "invalid_options", "options.config failed validation.",
                    [
                        _detail(["options", "config", *map(str, e["loc"])], str(e["msg"]))
                        for e in exc.errors()
                    ],
                )
    return cfg, flags["include_impact_summary"], flags["impact_by_building"], None


def analyze_payload(
    payload: Any,
    config: Optional[AnalysisConfig] = None,
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> AnalysisOutcome:
    """Run the full pipeline on a decoded JSON request body.

    Request body (see ``docs/JSON_CONTRACT.md``)::

        {"records": [{timestamp, building, resource_type, consumption_value, unit}, ...],
         "options": {"include_impact_summary": false, "impact_by_building": false,
                     "config": {"baseline_window": 30, ...}}}

    Client mistakes (bad shape, invalid records, bad options, too many records)
    come back as an error envelope with a suggested status; they never raise.
    Programmer mistakes (``config`` of the wrong type) still raise ``TypeError``.
    Detection never fails because of the LLM: see ``llm_status`` on each
    recommendation.
    """
    if config is not None and not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig or None")
    base = config or AnalysisConfig()

    if not isinstance(payload, Mapping):
        return error_outcome(422, "invalid_request", "Request body must be a JSON object.",
                      [_detail([], "must be an object")])
    unknown = sorted(str(k) for k in payload if k not in _REQUEST_KEYS)
    if unknown:
        return error_outcome(
            422, "invalid_request",
            f"Unknown field(s): {', '.join(unknown)}. Allowed: {', '.join(_REQUEST_KEYS)}.",
            [_detail([k], "unknown field") for k in unknown],
        )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return error_outcome(422, "invalid_request", "records must be a non-empty JSON array.",
                      [_detail(["records"], "must be a non-empty array")])
    if len(records) > max_records:
        return error_outcome(
            413, "too_many_records",
            f"At most {max_records} records per request; received {len(records)}.",
            [_detail(["records"], f"maximum {max_records} items")],
        )

    cfg, include_impact, by_building, option_error = _parse_options(payload.get("options"), base)
    if option_error is not None:
        return option_error
    assert cfg is not None

    try:
        results = analyze_consumption(records, cfg)
    except RecordValidationError as exc:
        return error_outcome(
            422, "invalid_records",
            f"{len(exc.errors)} record(s) failed validation; no analysis was performed.",
            validation_error_body(exc),
        )
    except Exception as exc:  # noqa: BLE001 - never leak internals to clients
        logger.error("analysis failed (%s)", type(exc).__name__)
        return error_outcome(500, "analysis_failed", "The analysis could not be completed.")

    impact: Optional[Dict[str, Any]] = None
    if include_impact:
        impact = aggregate_impact(results, by_building=by_building).model_dump(mode="json")

    body: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "status": "ok",
        "engine_version": __version__,
        "summary": summarize_results(results),
        "results": serialize_results(results),
        "impact_summary": impact,
    }
    return AnalysisOutcome(ok=True, http_status=200, body=body)


# --------------------------------------------------------------------------- #
# Diagnostics and schemas
# --------------------------------------------------------------------------- #
def engine_info(config: Optional[AnalysisConfig] = None) -> Dict[str, Any]:
    """Static facts about the engine for a health endpoint. Never returns a key.

    ``llm.api_key_configured`` is only evaluated when the LLM is enabled (the
    engine does not read API keys otherwise); it reports presence, not value.
    """
    cfg = config or AnalysisConfig()
    llm = cfg.llm
    key_present: Optional[bool] = None
    if llm.enabled:
        key_present = bool((os.environ.get(llm.resolved_api_key_env_var) or "").strip())
    return {
        "engine_version": __version__,
        "contract_version": CONTRACT_VERSION,
        "supported_resources": {r.value: unit for r, unit in STANDARD_UNITS.items()},
        "recommendation_mode": "rules_llm_with_fallback" if llm.enabled else "rules",
        "llm": {
            "enabled": llm.enabled,
            "provider": llm.provider.value if llm.enabled else None,
            "model": llm.model if llm.enabled else None,
            "timeout_seconds": llm.timeout_seconds,
            "max_llm_recommendations": llm.max_llm_recommendations,
            "api_key_configured": key_present,
        },
    }


def json_schemas() -> Dict[str, Dict[str, Any]]:
    """JSON Schemas generated from the Pydantic models (see ``scripts/export_json_schema.py``).

    ``record`` is the schema of one input record; ``result`` and
    ``impact_summary`` describe the serialized output (serialization mode).
    """
    return {
        "record": ConsumptionRecord.model_json_schema(),
        "result": AnalysisResult.model_json_schema(mode="serialization"),
        "impact_summary": ImpactSummary.model_json_schema(mode="serialization"),
    }
