"""Anomaly detection for campus consumption records.

Method
------
Records are grouped by (building, resource_type) and ordered by time. For each
record the baseline is the **median of up to ``baseline_window`` observations
with a strictly earlier timestamp** (no future-data leakage; records sharing a
timestamp never serve as each other's history).

Two independent signals are each mapped to a score in [0, 1) where 0.5 means
"exactly at the threshold" (``x -> e / (1 + e)``, ``e`` = value / threshold):

* percentage deviation from the baseline vs ``deviation_threshold_percentage``
* statistical extremity: robust z-score (MAD), or an Isolation Forest outlier
  gate (its scores saturate, so it decides *whether*, not *how much*)

The final ``anomaly_score`` is the **minimum** of the signals, so a record is
flagged (score >= 0.5) only when every signal agrees. This avoids flagging a
tiny absolute change in very stable data (huge z, small %) and a large
percentage change in very noisy data (big %, small z).

Special cases
-------------
* Fewer than ``minimum_history`` earlier observations -> insufficient data.
* Zero MAD (constant-ish history) -> z-score undefined; percentage only.
* Zero baseline -> percentage undefined; no deviation is fabricated.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import AnalysisConfig, AnomalyMethod, ResourceType
from .schemas import AnalysisResult, ConsumptionRecord, DataQualityStatus, Severity

# Scale factor making MAD consistent with the standard deviation for normal data.
_MAD_SCALE = 1.4826

# Score cut-offs for severity. With score = e/(1+e) these correspond to
# exceeding the detection threshold by 1x, 2x, 3x and 5x.
_SEVERITY_BANDS: Tuple[Tuple[float, Severity], ...] = (
    (5 / 6, Severity.CRITICAL),
    (3 / 4, Severity.HIGH),
    (2 / 3, Severity.MEDIUM),
    (1 / 2, Severity.LOW),
)
_EPS = 1e-12


def _utc(ts: datetime) -> datetime:
    """Timestamp used for ordering. Naive timestamps are assumed to be UTC."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _signal(exceedance: float) -> float:
    """Map exceedance ratio (>= 0) to a score in [0, 1); 1.0 -> 0.5."""
    return exceedance / (1.0 + exceedance)


def _severity(score: float) -> Severity:
    for cutoff, severity in _SEVERITY_BANDS:
        if score >= cutoff - _EPS:
            return severity
    return Severity.NONE


def _isolation_forest_score(history: np.ndarray, observed: float, seed: int) -> float:
    """Isolation Forest anomaly score in (0, 1) (> 0.5 = more isolated than typical).

    Fitted only on ``history`` (earlier observations), then scores ``observed``.
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "scikit-learn is required for anomaly_method='isolation_forest'"
        ) from exc
    model = IsolationForest(n_estimators=100, random_state=seed)
    model.fit(history.reshape(-1, 1))
    return float(-model.score_samples(np.array([[observed]]))[0])


def _fmt(x: float) -> str:
    return f"{x:,.2f}"


def _assumptions(cfg: AnalysisConfig, notes: List[str]) -> List[str]:
    base = [
        f"Baseline = median of up to {cfg.baseline_window} earlier observations "
        "(strictly earlier timestamps) for the same building and resource type.",
        "Deviation % = (observed - baseline) / baseline x 100.",
        f"Records are flagged only when |deviation| >= "
        f"{cfg.deviation_threshold_percentage:g}% and the statistical signal "
        f"({'robust z-score >= ' + format(cfg.robust_z_threshold, 'g') if cfg.anomaly_method is AnomalyMethod.MAD else 'Isolation Forest outlier gate (score > 0.5)'}) "
        "also indicates an outlier.",
        "anomaly_score is in [0, 1]; 0.5 is the detection threshold. Severity "
        "bands start at score 0.5 (low), 0.67 (medium), 0.75 (high), 0.83 (critical).",
        "Timestamps without a timezone are treated as UTC for ordering.",
    ]
    return base + notes


def _result(
    rec: ConsumptionRecord,
    cfg: AnalysisConfig,
    *,
    baseline: Optional[float],
    deviation: Optional[float],
    score: float,
    severity: Severity,
    status: DataQualityStatus,
    explanation: str,
    notes: Optional[List[str]] = None,
) -> AnalysisResult:
    return AnalysisResult(
        timestamp=rec.timestamp,
        building=rec.building,
        resource_type=rec.resource_type,
        unit=rec.unit,
        observed_value=rec.consumption_value,
        baseline_value=None if baseline is None else round(baseline, 6),
        deviation_percentage=None if deviation is None else round(deviation, 6),
        anomaly_score=round(score, 6),
        severity=severity,
        explanation=explanation,
        recommended_action=None,  # Not produced until the recommendation phase.
        estimated_savings=None,  # Not produced until the impact phase.
        calculation_assumptions=_assumptions(cfg, notes or []),
        data_quality_status=status,
    )


def _evaluate(
    rec: ConsumptionRecord, history: np.ndarray, cfg: AnalysisConfig
) -> AnalysisResult:
    obs = rec.consumption_value
    n = len(history)
    where = f"{rec.building} / {rec.resource_type.value}"

    if n < cfg.minimum_history:
        return _result(
            rec, cfg, baseline=None, deviation=None, score=0.0,
            severity=Severity.NONE, status=DataQualityStatus.INSUFFICIENT_DATA,
            explanation=(
                f"Only {n} earlier observation(s) for {where}; at least "
                f"{cfg.minimum_history} are required to compute a baseline. "
                f"Observed {_fmt(obs)} {rec.unit}; no anomaly assessment made."
            ),
        )

    median = float(np.median(history))
    mad = float(np.median(np.abs(history - median)))
    scale = _MAD_SCALE * mad
    basis = f"median of the previous {n} observation(s)"

    if median == 0:
        if obs == 0:
            return _result(
                rec, cfg, baseline=0.0, deviation=0.0, score=0.0,
                severity=Severity.NONE, status=DataQualityStatus.OK,
                explanation=(
                    f"Observed 0 {rec.unit} matches a zero baseline ({basis})."
                ),
            )
        return _result(
            rec, cfg, baseline=0.0, deviation=None, score=0.0,
            severity=Severity.NONE, status=DataQualityStatus.INSUFFICIENT_DATA,
            explanation=(
                f"Observed {_fmt(obs)} {rec.unit} but the historical baseline is 0 "
                f"({basis}), so percentage deviation is undefined and no "
                "anomaly assessment was made."
            ),
        )

    deviation = (obs - median) / median * 100.0
    direction = "above" if deviation > 0 else "below"
    signals = [_signal(abs(deviation) / cfg.deviation_threshold_percentage)]
    notes: List[str] = []
    evidence: List[str] = []

    use_if = cfg.anomaly_method is AnomalyMethod.ISOLATION_FOREST
    if use_if and n < cfg.isolation_forest_min_history:
        notes.append(
            f"isolation_forest requested but only {n} earlier observations "
            f"(< {cfg.isolation_forest_min_history}); robust MAD scoring used instead."
        )
        use_if = False
    if use_if and float(np.ptp(history)) == 0.0:
        notes.append(
            "isolation_forest requested but history is constant; percentage "
            "deviation alone was used."
        )
        evidence.append("History is constant, so only percentage deviation is used.")
        use_if = False
        stat_done = True
    else:
        stat_done = False

    if use_if:
        s_if = _isolation_forest_score(history, obs, cfg.random_seed)
        # Isolation Forest scores saturate (a 3x spike in 1-D typically scores
        # ~0.6-0.7), so they cannot express magnitude. IF therefore acts as a
        # gate: outlier (> 0.5) -> no limit on the score; inlier -> caps it
        # below the detection threshold. Magnitude comes from the % deviation.
        signals.append(s_if if s_if <= 0.5 else 1.0)
        evidence.append(
            f"Isolation Forest score {s_if:.2f} "
            f"({'outlier' if s_if > 0.5 else 'inlier'}; outlier if > 0.50)."
        )
    elif not stat_done:
        if scale > 0:
            z = (obs - median) / scale
            signals.append(_signal(abs(z) / cfg.robust_z_threshold))
            evidence.append(
                f"Robust z-score {z:+.1f} (MAD {_fmt(mad)}; threshold "
                f"{cfg.robust_z_threshold:g})."
            )
        else:
            evidence.append(
                "Historical MAD is 0 (near-constant history), so the robust "
                "z-score is undefined; only percentage deviation is used."
            )

    score = min(signals)
    severity = _severity(score)
    status = DataQualityStatus.OK
    extra = ""
    if obs == 0:
        status = DataQualityStatus.SUSPECT
        extra = (
            " Zero consumption against a non-zero baseline may indicate a "
            "meter outage or a missing reading."
        )

    verdict = (
        f"Flagged as {severity.value} severity."
        if severity is not Severity.NONE
        else "Within normal range."
    )
    explanation = (
        f"Observed {_fmt(obs)} {rec.unit} is {abs(deviation):.1f}% {direction} the "
        f"historical baseline of {_fmt(median)} {rec.unit} ({basis}; threshold "
        f"{cfg.deviation_threshold_percentage:g}%). {' '.join(evidence)} "
        f"{verdict}{extra}"
    )
    return _result(
        rec, cfg, baseline=median, deviation=deviation, score=score,
        severity=severity, status=status, explanation=explanation, notes=notes,
    )


def detect_anomalies(
    records: Sequence[ConsumptionRecord],
    config: Optional[AnalysisConfig] = None,
) -> List[AnalysisResult]:
    """Detect anomalies in already-validated records.

    Args:
        records: Validated ``ConsumptionRecord`` objects, in any order.
        config: Detection settings; defaults used when ``None``.

    Returns:
        One ``AnalysisResult`` per input record, **in the same order as the
        input** (not sorted by time). Recommendation and savings fields are
        always ``None``.
    """
    cfg = config or AnalysisConfig()
    groups: Dict[Tuple[str, ResourceType], List[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        groups[(rec.building, rec.resource_type)].append(idx)

    results: List[Optional[AnalysisResult]] = [None] * len(records)
    for indices in groups.values():
        indices.sort(key=lambda i: (_utc(records[i].timestamp), i))
        times = [_utc(records[i].timestamp) for i in indices]
        values = np.array([records[i].consumption_value for i in indices], dtype=float)
        for pos, idx in enumerate(indices):
            # Strictly earlier timestamps only: excludes the record itself,
            # all later records, and same-timestamp records.
            end = bisect_left(times, times[pos])
            history = values[max(0, end - cfg.baseline_window):end]
            results[idx] = _evaluate(records[idx], history, cfg)
    return results  # type: ignore[return-value]
