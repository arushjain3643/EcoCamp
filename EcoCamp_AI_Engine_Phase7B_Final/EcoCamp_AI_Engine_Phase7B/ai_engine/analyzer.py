"""Public entry point of the EcoCamp AI engine."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from pydantic import ValidationError

from .anomaly_detector import detect_anomalies
from .config import AnalysisConfig
from .impact_calculator import apply_impact
from .recommendation_engine import generate_recommendations
from .schemas import AnalysisResult, ConsumptionRecord


class RecordValidationError(ValueError):
    """Raised when one or more input records are invalid.

    ``errors`` is a list of ``{"index": int, "errors": [{"loc", "msg"}, ...]}``
    so the backend can report exactly which records failed. No partial results
    are produced when any record is invalid.
    """

    def __init__(self, errors: List[Dict[str, Any]]) -> None:
        self.errors = errors
        summary = "; ".join(
            f"record {e['index']}: "
            + ", ".join(f"{'.'.join(map(str, d['loc'])) or '<record>'}: {d['msg']}" for d in e["errors"])
            for e in errors
        )
        super().__init__(f"{len(errors)} invalid record(s): {summary}")


def analyze_consumption(
    records: Sequence[Union[ConsumptionRecord, Mapping[str, Any]]],
    config: Optional[AnalysisConfig] = None,
) -> List[AnalysisResult]:
    """Analyze consumption records: anomalies, potential savings and recommendations.

    Args:
        records: ``ConsumptionRecord`` objects or dicts (validated into them),
            in any order, possibly spanning several buildings and resources.
        config: Optional ``AnalysisConfig``; defaults used if ``None``.

    Returns:
        One ``AnalysisResult`` per record, in the same order as ``records``.
        Qualifying above-baseline anomalies also carry estimated savings and
        (when an emission factor is configured) estimated CO2e reduction.
        Flagged records also carry a ``recommendation`` (deterministic rules;
        plus validated LLM text only if ``config.llm.enabled`` is true, which
        it is not by default). Pipeline: validation -> anomaly detection ->
        impact calculation -> recommendation generation.

    Raises:
        RecordValidationError: if any record is invalid (lists all failures).
        TypeError: if ``records`` is not a sequence or ``config`` is not an
            ``AnalysisConfig``.
    """
    if config is not None and not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig or None")
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

    cfg = config or AnalysisConfig()
    detected = detect_anomalies(validated, cfg)
    with_impact = [apply_impact(r, cfg) for r in detected]
    return generate_recommendations(with_impact, cfg)
