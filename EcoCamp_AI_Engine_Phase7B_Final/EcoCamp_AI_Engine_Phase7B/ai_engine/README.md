# ai_engine - EcoCamp intelligence module (Phase 4 engine, Phase 5 integration layer, Phase 6 forecasting)

Independent Python module for campus electricity, water, and waste analysis.
It is called from the FastAPI backend as ordinary Python functions. It has no
server, database, or frontend.

**Status: Phase 4 (anomaly detection + environmental impact + recommendations).**
`analyze_consumption` detects anomalies, estimates avoidable consumption (kWh,
liters, kg) and, only with a configured emission factor, CO2e, and attaches an
evidence-based `recommendation` to every flagged reading. Recommendations come
from deterministic Python rules and work fully offline. An **optional** LLM layer
(Gemini or OpenAI, **disabled by default**) can add plain-language text on top;
it can never change numbers, savings or the deterministic recommendation. **No
emission factor is built in** and **no API key is stored anywhere in the code**.

**Phase 5 (integration).** The engine itself is unchanged. `ai_engine/contract.py` adds
`analyze_payload` (JSON in, JSON-safe envelope out), `serialize_results`, `engine_info` and
`json_schemas`. The contract is documented in `../docs/JSON_CONTRACT.md`, LLM setup in
`../docs/LLM_CONFIGURATION.md`, and a FastAPI example is in `../examples/fastapi_integration/`.

## Install

```bash
pip install -r ai_engine/requirements.txt
# Only if you enable the optional LLM layer (install the SDK you use):
pip install -r ai_engine/requirements-llm.txt      # openai and/or google-genai
```

Requires Python 3.9+. This installs pydantic, numpy, scikit-learn (only used by
the optional Isolation Forest) and pytest. Place the `ai_engine/` folder so its parent directory is on `PYTHONPATH`
(e.g. the backend project root).

## Usage (for Navditya)

```python
from pydantic import ValidationError
from ai_engine import ConsumptionRecord, AnalysisConfig, analyze_consumption

# 1. Validate incoming data
try:
    record = ConsumptionRecord(
        timestamp="2026-03-01T10:00:00Z",   # ISO-8601 string or datetime
        building="Library",
        resource_type="electricity",        # electricity | water | waste
        consumption_value=120.5,            # >= 0, finite
        unit="kWh",
    )
except ValidationError as e:
    errors = e.errors()   # map to a 422 response

# 2. Analyze (records may be objects or dicts, any order, mixed buildings/resources)
results = analyze_consumption([record], config=AnalysisConfig())   # defaults
results = analyze_consumption(records, AnalysisConfig(baseline_window=14, minimum_history=5))
```

`analyze_consumption(records, config=None) -> list[AnalysisResult]`

- Returns **one result per input record, in the same order as the input**
  (not sorted by time), so `results[i]` belongs to `records[i]`.
- If any record is invalid it raises `RecordValidationError` (a `ValueError`)
  whose `.errors` lists `{"index", "errors": [{"loc", "msg"}]}` for **every**
  bad record; no partial results are returned. Map it to a 422 response.
- Raises `TypeError` for a non-sequence `records` or a non-`AnalysisConfig` config.
- Detection is CPU-bound and synchronous; in FastAPI call it from a sync
  endpoint or `run_in_threadpool` for large batches.

Because the models are Pydantic v2, FastAPI can use `ConsumptionRecord` as a
request body type and `AnalysisResult` as a `response_model` directly.

## Standard units

| Resource    | Unit     | Accepted spellings (same unit, not conversions) |
|-------------|----------|--------------------------------------------------|
| electricity | `kWh`    | `kWh`, `kwh` (case-insensitive)                  |
| water       | `liters` | `liters`, `litres`, `L`                          |
| waste       | `kg`     | `kg`                                             |

Any other unit (Wh, MWh, gallons, lbs, ...) is rejected. **No unit conversion
is performed.** Units are normalized to the canonical spelling.

## Validation rules (`ConsumptionRecord`)

- `consumption_value`: finite, non-negative number (zero allowed; booleans rejected).
- `resource_type`: one of the three supported values (case-insensitive).
- `unit`: must match the resource type (see table).
- `timestamp`: datetime or non-empty ISO-8601 string. Numbers (Unix epochs),
  empty strings, and unparseable values are rejected.
- `building`: non-empty after trimming.
- Unknown fields are rejected; records are immutable.

## Output schema (`AnalysisResult`)

Identifies the record (`timestamp`, `building`, `resource_type`, `unit`) and carries:

| Field | Type | Meaning |
|---|---|---|
| `observed_value` | float | Measured consumption |
| `baseline_value` | float \| null | Rolling-median baseline; `null` without a valid baseline |
| `deviation_percentage` | float \| null | `(observed - baseline) / baseline x 100`; `null` if undefined |
| `anomaly_score` | float 0-1 | 0.5 is the detection threshold |
| `severity` | enum | `none|low|medium|high|critical` |
| `data_quality_status` | enum | `ok|insufficient_data|suspect` |
| `explanation` | str | Evidence for the verdict |
| `recommended_action` | str \| null | First suggested action of the recommendation; `null` when there is none |
| `recommendation` | `Recommendation` \| null | Evidence-based recommendation for flagged readings; `null` for normal readings and insufficient history (see Phase 4) |
| `estimated_savings` | `{value, unit}` \| null | Avoidable consumption in the resource's own unit; `null` when not calculated |
| `savings_unit` | str \| null | `kWh`, `liters` or `kg`; `null` when not calculated |
| `estimated_co2e_reduction_kg` | float \| null | `null` unless a valid factor is configured |
| `calculation_assumptions` | list[str] | Detection notes, then formula, fraction and factor/source used |
| `calculation_status` | enum | See below |

`calculation_status`:

| Value | Meaning |
|---|---|
| `calculated` | Savings and CO2e both calculated |
| `calculated_without_co2e` | Savings calculated; no valid emission factor configured, so CO2e is `null` |
| `not_applicable` | Not a qualifying above-baseline anomaly (normal, below baseline, zero reading, or zero baseline with zero use) |
| `insufficient_baseline` | No valid baseline (too little history, or zero baseline with positive use) |

## Anomaly detection (Phase 2)

**Baseline.** Records are grouped by `(building, resource_type)` (building names
match exactly after trimming) and ordered by timestamp. For each record the
baseline is the **median of up to `baseline_window` observations with a strictly
earlier timestamp**. The record itself, later records, and records sharing its
timestamp are never used (no future-data leakage). Naive timestamps are treated
as UTC for ordering.

**Signals.** Two signals are each mapped to a score where 0.5 means "exactly at
the threshold" (`e / (1 + e)`, e = value / threshold):

1. Percentage deviation from baseline vs `deviation_threshold_percentage`.
2. Statistical extremity: robust z-score `(x - median) / (1.4826 * MAD)` vs
   `robust_z_threshold` (method `mad`), or an Isolation Forest outlier gate
   (method `isolation_forest`).

`anomaly_score` is the **minimum** of the signals, so a record is flagged
(score >= 0.5) only when all signals agree. Deviations in **both directions**
are flagged (unusually low use can matter too).

**Severity** (from `anomaly_score`): `none` < 0.5, `low` >= 0.5, `medium` >= 0.67,
`high` >= 0.75, `critical` >= 0.83 (i.e. exceeding the threshold by 1x, 2x, 3x, 5x).

**Edge cases**

| Situation | Behavior |
|---|---|
| Fewer than `minimum_history` earlier observations | `insufficient_data`, baseline/deviation `None`, score 0, severity `none` |
| MAD = 0 (constant or near-constant history) | z-score undefined; decision uses percentage deviation only (stated in the explanation) |
| Zero consumption, non-zero baseline | Evaluated normally (deviation -100%); `data_quality_status = suspect` (possible meter outage / missing reading) |
| Zero baseline, zero observed | Normal, deviation 0 |
| Zero baseline, positive observed | Percentage undefined, so no deviation is fabricated: `insufficient_data`, severity `none` |

**Isolation Forest (optional).** `anomaly_method="isolation_forest"` requires
`baseline_window >= isolation_forest_min_history` (default 50). For a record with
fewer earlier observations than that, MAD scoring is used instead (noted in
`calculation_assumptions`), so it is never forced onto tiny datasets. The forest
is fitted only on earlier observations, on the consumption value alone (no
time-of-day or seasonality features), with a fixed `random_seed`. Its scores
saturate, so it acts as an outlier **gate**; severity magnitude comes from the
percentage deviation. A forest is refit per record (roughly 0.1 s each), so
`mad` is the recommended default for large batches.

## Configuration (`AnalysisConfig`)

| Field | Default | Meaning |
|---|---|---|
| `baseline_window` | 30 | Max earlier observations in the rolling baseline |
| `minimum_history` | 7 | Min earlier observations to compute a baseline (>= 3) |
| `deviation_threshold_percentage` | 50.0 | Min \|deviation\| (%) to flag |
| `anomaly_method` | `mad` | `mad` or `isolation_forest` |
| `robust_z_threshold` | 3.5 | Robust z cutoff (conventional Iglewicz-Hoaglin value) |
| `isolation_forest_min_history` | 50 | Min earlier observations to use the forest |
| `random_seed` | 42 | Isolation Forest seed |
| `achievable_reduction_fraction` | 0.5 | Share of the above-baseline excess assumed avoidable; must be 0 to 1 (placeholder value) |
| `electricity_emission_factor_kg_co2e_per_kwh` | `None` | kg CO2e per kWh, `None` = unavailable; finite and >= 0 |
| `electricity_emission_factor_source` | `None` | Optional citation; only allowed with the factor |
| `water_emission_factor_kg_co2e_per_liter` | `None` | kg CO2e per liter, `None` = unavailable; requires a source |
| `water_emission_factor_source` | `None` | Citation; required when the water factor is set |
| `llm` | `LLMConfig()` (disabled) | Optional LLM settings; see Phase 4 |
| `waste_emission_factor` | `None` | `WasteEmissionFactor(value, unit="kg_co2e_per_kg", source)`; source required |

**All defaults are engineering starting points, not values fitted to campus
data.** Tune them on real data. Invalid values or combinations raise `ValidationError`: a reduction fraction
outside 0 to 1, NaN or infinite numbers, negative or boolean factors, a water factor without a
source, a source without its factor, a waste factor with a wrong unit or no
source, unknown fields, or `baseline_window < minimum_history`.

## Environmental impact (Phase 3)

**Formula**

```
estimated_savings = max(0, observed_value - baseline_value) x achievable_reduction_fraction
estimated_co2e_reduction_kg = estimated_savings x emission_factor   (only if a valid factor is configured)
```

Savings are in the resource's standard unit: electricity in **kWh**, water in
**liters**, waste in **kg**.

**When savings are calculated.** Only when all of these hold: the baseline is
valid and greater than 0; `severity` is not `none`; `observed_value` is above
the baseline; and `data_quality_status` is `ok`. Otherwise `estimated_savings`,
`savings_unit` and `estimated_co2e_reduction_kg` are `null` and
`calculation_status` says why. Savings are never negative. Unusually *low*
readings are still flagged by detection but produce no savings.

**CO2e per resource**

| Resource | CO2e reported when | Factor unit |
|---|---|---|
| electricity | `electricity_emission_factor_kg_co2e_per_kwh` is set | kg CO2e per kWh |
| water | `water_emission_factor_kg_co2e_per_liter` **and** `water_emission_factor_source` are set | kg CO2e per liter |
| waste | `waste_emission_factor` (value, unit `kg_co2e_per_kg`, source) is set | kg CO2e per kg |

A factor of `None` means "unavailable": the CO2e field is `null`, and for waste
no carbon-equivalence claim is made at all. A factor of `0` is allowed and
yields `0.0` (for example a documented zero-carbon supply).

**Aggregating without double counting.** `aggregate_impact(results, by_building=False)`

- counts only records with a calculated saving;
- counts each `(building, resource_type, timestamp)` once; repeated results are
  skipped and reported in `duplicate_records_skipped`;
- sums savings per resource in its own unit and never adds kWh, liters and kg together;
- sums CO2e from each record's own figure. If any counted record lacks a factor,
  `co2e_complete` is `false` and `total_co2e_reduction_kg` is a partial sum
  (`null` if none has CO2e).

Each reading is treated as a separate period of consumption, so a sustained
anomaly over several days contributes each day's excess once. That is a
reasonable total of avoidable consumption over those periods, not a forecast.

### Environmental assumptions

- `achievable_reduction_fraction` (default **0.5**) is a placeholder, not a
  measured value. Set it from campus experience or a cited study.
- The baseline is the rolling median of earlier readings. "Avoidable" means
  "above that baseline", which is an estimate and does not prove waste.
  Seasonality, occupancy and events are not modelled.
- No emission factor ships with the engine. Grid factors vary by country, year
  and method; water and waste factors vary even more. You must provide factors
  from a source you can cite. They are recorded in `calculation_assumptions`.
- Values in the examples below (e.g. 0.4 kg CO2e/kWh) are **illustrative
  placeholders, not real emission factors**.

### Configuration examples

```python
from ai_engine import AnalysisConfig, WasteEmissionFactor

# Savings only (default): no CO2e is produced.
cfg = AnalysisConfig()

# Electricity CO2e with a cited factor, 30% of the excess assumed avoidable.
cfg = AnalysisConfig(
    achievable_reduction_fraction=0.3,
    electricity_emission_factor_kg_co2e_per_kwh=0.4,            # illustrative
    electricity_emission_factor_source="<your cited source, year>",
)

# Water and waste factors (both need a source).
cfg = AnalysisConfig(
    electricity_emission_factor_kg_co2e_per_kwh=0.4,            # illustrative
    water_emission_factor_kg_co2e_per_liter=0.0003,             # illustrative
    water_emission_factor_source="<your cited source, year>",
    waste_emission_factor=WasteEmissionFactor(
        value=0.2, unit="kg_co2e_per_kg", source="<your cited source, year>"
    ),
)
```

Invalid values raise `pydantic.ValidationError` (see the table below).

### Example

Input: ten normal daily electricity readings for the Library (median 100 kWh)
followed by a reading of 250 kWh, with the config from the second example above
but `achievable_reduction_fraction=0.5`.

```python
results = analyze_consumption(records, cfg)
print(results[-1].model_dump(mode="json"))
```

Abridged output (`explanation`, most `calculation_assumptions` and timestamps shortened):

```json
{
  "building": "Library",
  "resource_type": "electricity",
  "unit": "kWh",
  "observed_value": 250.0,
  "baseline_value": 100.0,
  "deviation_percentage": 150.0,
  "anomaly_score": 0.75,
  "severity": "high",
  "data_quality_status": "ok",
  "recommended_action": "Check whether lighting is left on in unoccupied rooms, ...",
  "recommendation": { "...": "see the Phase 4 section" },
  "estimated_savings": {"value": 75.0, "unit": "kWh"},
  "savings_unit": "kWh",
  "estimated_co2e_reduction_kg": 30.0,
  "calculation_status": "calculated",
  "calculation_assumptions": [
    "...detection notes...",
    "estimated_savings = max(0, observed_value - baseline_value) x achievable_reduction_fraction (0.5); expressed in kWh.",
    "CO2e reduction = estimated savings (kWh) x 0.4 kg CO2e/kWh (source: <your cited source, year>)."
  ]
}
```

A normal reading, or one below baseline, returns `estimated_savings: null`,
`savings_unit: null`, `estimated_co2e_reduction_kg: null` and
`calculation_status: "not_applicable"`.

Aggregation:

```python
from ai_engine import aggregate_impact

summary = aggregate_impact(results)                      # per resource
summary = aggregate_impact(results, by_building=True)    # per building and resource
summary.totals                    # [ImpactTotal(resource_type, savings_value, savings_unit, co2e_reduction_kg, ...)]
summary.total_co2e_reduction_kg   # float | None
summary.co2e_complete             # False if some counted records had no factor
```

## Recommendations (Phase 4)

Pipeline: input validation -> anomaly detection -> impact calculation ->
recommendation generation -> structured output. `analyze_consumption(records,
config=None)` is unchanged; recommendations are an extra field on each result.

### Deterministic rules (always on)

| Reading | Recommendation |
|---|---|
| Normal, or insufficient history / no valid baseline | none (`recommendation: null`) |
| Electricity above baseline | Check lighting (unoccupied rooms, timers, sensors), HVAC schedules and setpoints, idle equipment, temporary loads |
| Water above baseline | Inspect for possible leaks, taps and flush valves, water-using schedules, overnight use. States that a reading **cannot confirm a leak** |
| Waste above baseline | Check segregation, run a bin audit, rule out one-off causes, reduction measures, verify recorded weights |
| Below baseline | Verify the meter or weighing process and look for a known reason (holiday, closure); no savings are estimated |
| Zero reading against a non-zero baseline | Treated as a possible outage or missing data (`suspect_reading`); verify the meter first |

Priority follows severity: `low`->`low`, `medium`->`medium`, `high`->`high`,
`critical`->`urgent`. Verification-type recommendations (below baseline,
suspect) are capped at `medium`. Estimated savings and CO2e are **copied from
the Phase 3 result**, never recomputed; if none were calculated, the
recommendation has none and its text says nothing about savings.

### Recommendation JSON schema

```json
{
  "resource_type": "electricity",
  "building": "Library",
  "severity": "high",
  "priority": "high",
  "anomaly_type": "above_baseline",
  "supporting_evidence": [
    "Observed 250.00 kWh versus a baseline of 100.00 kWh: 150.0% above baseline.",
    "Anomaly score 0.75 (0.50 is the detection threshold); severity high.",
    "The baseline is the median of up to 30 earlier readings for this building and resource.",
    "Estimated avoidable electricity: 75.00 kWh (assumes 50% of the above-baseline excess is avoidable).",
    "Estimated CO2e reduction: 30.00 kg (using the configured emission factor)."
  ],
  "recommended_actions": [
    "Check whether lighting is left on in unoccupied rooms, corridors and outdoor areas, and whether timers or occupancy sensors are working.",
    "Review HVAC schedules, setpoints and run times against actual occupancy and the academic timetable.",
    "..."
  ],
  "estimated_savings": {"value": 75.0, "unit": "kWh"},
  "estimated_co2e_reduction_kg": 30.0,
  "explanation": "Library's electricity use of 250.00 kWh is 150.0% above its baseline of 100.00 kWh. This is a statistical deviation from earlier readings; its cause has not been identified, and the actions below are suggested checks rather than confirmed causes.",
  "caveats": [
    "The cause has not been identified or confirmed; the actions are suggested checks.",
    "Savings are estimates based on the configured reduction fraction, not guaranteed results."
  ],
  "source": "rules",
  "llm_status": "disabled",
  "llm_explanation": null,
  "llm_suggestions": [],
  "llm_impact_summary": null
}
```

(Abridged and illustrative; the 0.4 kg CO2e/kWh factor behind the CO2e line is a
placeholder, not a real emission factor.) `source` is `rules` or `rules_llm`.
`priority` is `low|medium|high|urgent`; `anomaly_type` is
`above_baseline|below_baseline|suspect_reading`.

### Optional LLM layer

Disabled unless you enable it. When disabled the engine never reads API keys,
never imports an LLM SDK and makes no network calls.

**1. Install the official SDK you use** (`pip install openai` or `pip install google-genai`,
or `requirements-llm.txt`).

**2. Set environment variables** (see `.env.example`; real keys never go in code or git):

| Variable | Meaning |
|---|---|
| `ECOCAMP_LLM_ENABLED` | `true` to enable (default `false`) |
| `ECOCAMP_LLM_PROVIDER` | `gemini` (default) or `openai` |
| `ECOCAMP_LLM_MODEL` | Model name from your provider's docs. **Required** when enabled; no default is shipped because model names change |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | The provider key, read at call time |
| `ECOCAMP_LLM_API_KEY_ENV` | Optional: name of a different variable that holds the key (a name only; pasting a key here is rejected) |
| `ECOCAMP_LLM_TIMEOUT_SECONDS` | Per-request timeout (default 20) |
| `ECOCAMP_LLM_MAX_RECOMMENDATIONS` | Max LLM calls per `analyze_consumption` call (default 10), highest priority first |

**3. Pass the config** (the engine does not read `ECOCAMP_LLM_*` on its own):

```python
from ai_engine import AnalysisConfig, LLMConfig, analyze_consumption

cfg = AnalysisConfig(llm=LLMConfig.from_env())        # from environment variables
# or explicitly:
cfg = AnalysisConfig(llm=LLMConfig(
    enabled=True, provider="openai", model="<model from provider docs>",
    timeout_seconds=15, max_llm_recommendations=5,
))
results = analyze_consumption(records, cfg)
rec = results[-1].recommendation
print(rec.llm_status)        # used | missing_api_key | timeout | ...
print(rec.llm_explanation)   # plain-language explanation, or None
```

Load a `.env` file with your own tooling (for example `set -a; source .env; set +a`,
your process manager, or `python-dotenv` in the backend). This package does not
depend on it.

**What the LLM does:** explains the flagged reading in simple language, suggests
extra practical steps, and summarises the *already calculated* impact. Its text goes
into `llm_explanation`, `llm_suggestions` and `llm_impact_summary` only. All
other fields stay deterministic.

**What is sent:** per flagged record only the building name, resource, unit,
observed value, baseline, deviation, severity, calculated savings/CO2e (or
null), the reduction percentage and the deterministic action list. No
timestamps, history or other buildings. The building name leaves your system,
so use provider terms that suit that.

**Validation of every reply** (any failure -> fallback):

- valid JSON with exactly `explanation`, `recommended_actions`, `impact_summary` (extra keys rejected), bounded lengths;
- every number in the text must already appear in the facts sent, so invented consumption, dates or savings are rejected (extra digits such as "150.4" for 150 are rejected too);
- no wording that a cause is confirmed or certain, or that a leak/fault was found (negations such as "has not been confirmed" are fine);
- no carbon/CO2 wording unless a CO2e estimate exists.

These checks are heuristics, not proof. They cannot verify every factual or
scientific statement the model makes, so treat `llm_*` text as advisory and keep
it visually separate from the deterministic fields in your UI.

### API failure behavior

The application never fails because of the LLM. In every case below the
deterministic recommendation is returned unchanged and `llm_status` says why:

| `llm_status` | Cause |
|---|---|
| `disabled` | LLM not enabled (default); nothing attempted |
| `used` | Validated LLM text attached |
| `missing_api_key` | Key variable unset or empty |
| `sdk_unavailable` | Official SDK not installed |
| `timeout` | Request exceeded `timeout_seconds` |
| `provider_error` | Any other provider/SDK/network failure |
| `invalid_response` | Reply failed JSON/schema/content validation |
| `call_limit_reached` | Beyond `max_llm_recommendations` for this call |

After a missing key, missing SDK, timeout or provider error, the remaining LLM
calls in that batch are skipped, so an outage costs one timeout, not one per
record. Only the exception class name is logged or stored; provider error
messages are dropped because they can echo credentials. The library makes no
retries.

## Forecasting (Phase 6)

```python
from ai_engine import ForecastConfig, forecast_consumption

forecasts = forecast_consumption(records, horizon_days=7, config=ForecastConfig())
f = forecasts[0]
f.forecast_available, f.predicted_values, f.lower_bound, f.upper_bound, f.model_used
f.evaluation.metrics          # MAE / WAPE of each method on the same backtest origins
```

Public names: `forecast_consumption`, `forecast_payload`, `rolling_origin_backtest`, `serialize_forecast(s)`,
`ForecastConfig`, `ForecastResult`, `ForecastMethod`, `ForecastEvaluation`, `HistorySummary`, `MethodMetrics`.
Full behaviour, configuration fields and the JSON contract: top-level `README.md` ("Forecasting") and
`docs/JSON_CONTRACT.md` section 6. `analyze_consumption` is not affected.

## Integrating with a separate FastAPI backend

The engine is a plain Python package with no server. Put the folder that
*contains* `ai_engine/` on `PYTHONPATH`, install `ai_engine/requirements.txt`,
and call it from your backend:

```python
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from ai_engine import (
    AnalysisConfig, AnalysisResult, ConsumptionRecord, ImpactSummary, LLMConfig,
    RecordValidationError, aggregate_impact, analyze_consumption,
)

app = FastAPI()

# Load factors from your own settings or environment, with their sources.
CONFIG = AnalysisConfig(
    achievable_reduction_fraction=0.3,
    electricity_emission_factor_kg_co2e_per_kwh=0.4,        # illustrative
    electricity_emission_factor_source="<your cited source, year>",
    llm=LLMConfig.from_env(),   # disabled unless ECOCAMP_LLM_ENABLED=true
)


@app.post("/analyze", response_model=List[AnalysisResult])
async def analyze(records: List[ConsumptionRecord]):
    try:
        return await run_in_threadpool(analyze_consumption, records, CONFIG)
    except RecordValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors)


@app.post("/impact-summary", response_model=ImpactSummary)
async def impact_summary(records: List[ConsumptionRecord], by_building: bool = False):
    results = await run_in_threadpool(analyze_consumption, records, CONFIG)
    return aggregate_impact(results, by_building=by_building)
```

With the LLM enabled, a request takes up to one provider round trip per flagged
record (capped by `max_llm_recommendations`), which is why the examples use
`run_in_threadpool`. Keep the LLM off, or lower the cap, for large batches.
Each `AnalysisResult` already contains its `recommendation`, so `response_model`
needs no change. FastAPI itself is **not** a dependency of this package; install it in the backend.
Send the whole history of each building and resource in one request: a record's
baseline is built only from earlier records in the same call.

## Known limitations

- The baseline ignores daily/weekly/seasonal patterns (e.g. weekends, term breaks),
  so recurring seasonal swings can be flagged. Seasonal baselines are future work.
- Explanations and recommendations state evidence and suggested checks, not
  causes; the only causal hints are hedged (possible leak, possible meter outage).
- Rule-based recommendations are generic starting points; they do not know your
  buildings' equipment, schedules or occupancy.
- LLM text is validated heuristically (see Phase 4) and is advisory only.
- Savings are estimates relative to a rolling-median baseline and a configured
  reduction fraction; they are not measured or guaranteed reductions.
- Statistical results are only as good as the history; sparse or irregular
  sampling affects baselines.

## Run tests

From the directory that **contains** `ai_engine/`:

```bash
python -m pytest ai_engine/tests -q
```

## Layout

| File                       | Purpose                                   |
|----------------------------|-------------------------------------------|
| `schemas.py`               | Input/output models, `CalculationStatus`, `ImpactSummary`, `Recommendation` |
| `config.py`                | Resource types, standard units, `AnalysisConfig`, `WasteEmissionFactor`, `LLMConfig` |
| `analyzer.py`              | Public `analyze_consumption`: validate, detect, estimate impact, recommend |
| `anomaly_detector.py`      | Baseline + MAD / Isolation Forest detection |
| `impact_calculator.py`     | Savings and CO2e estimation, `aggregate_impact` |
| `recommendation_engine.py` | Deterministic rules + optional validated LLM text, `generate_recommendations` |
| `llm_client.py`            | Gemini / OpenAI adapters (official SDKs, lazy import, env-var keys) |
| `contract.py`              | Phase 5: `analyze_payload`, JSON serialization, error envelopes, `engine_info`, `json_schemas` |
| `_version.py`              | Engine version and JSON contract version |
| `.env.example`             | Example environment variables (no secrets) |
| `requirements-llm.txt`     | Optional provider SDKs                    |
| `tests/`                   | Schema, anomaly, impact and recommendation/LLM tests (all external calls mocked) |

## Phase 7A: what-if simulator

`scenario_simulator.py` adds `simulate_scenario(records, scenario, config=None)` and `simulate_payload(payload, config)`:
projected savings of a percentage reduction (0-100 %) per building or campus-wide, hypothetical and never measured. It reuses
`ConsumptionRecord`, the standard units and the `AnalysisConfig` emission factors, and does not touch analysis or forecasting.
See the top-level `README.md` and `docs/JSON_CONTRACT.md` section 7.
