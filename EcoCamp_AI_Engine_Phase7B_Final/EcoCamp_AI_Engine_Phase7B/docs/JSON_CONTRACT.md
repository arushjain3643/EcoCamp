# EcoCamp AI engine: JSON input/output contract

**Contract version:** `1.0` (constant `ai_engine.CONTRACT_VERSION`, returned in every response).
**Forecast contract version:** `1.0` (`ai_engine.FORECAST_CONTRACT_VERSION`), described in section 6.
**Simulation contract version:** `1.0` (`ai_engine.SIMULATION_CONTRACT_VERSION`), described in section 7.
**Audience:** the backend developer calling the engine from FastAPI.

The contract is implemented by `ai_engine.analyze_payload(payload, config)`
(`ai_engine/contract.py`). It wraps the unchanged `analyze_consumption(records, config=None)`
and adds request checking, a response envelope, counts and error bodies.
The HTTP path below is a **suggestion**; the backend owns routing and authentication
(see `examples/fastapi_integration/app.py`).

Machine-readable JSON Schemas of the record and result models can be generated with
`python scripts/export_json_schema.py` (writes `docs/schemas/`), or at run time with
`ai_engine.json_schemas()`. This document is the authoritative description.

Versioning: additive changes (new optional response fields) raise the minor number;
breaking changes raise the major number. **Clients must ignore response fields they do not know.**

---

## 1. Request

`POST /api/v1/ai/analyze` (suggested), `Content-Type: application/json`

```json
{
  "records": [
    {
      "timestamp": "2026-03-01T00:00:00Z",
      "building": "Library",
      "resource_type": "electricity",
      "consumption_value": 100.0,
      "unit": "kWh"
    }
  ],
  "options": {
    "include_impact_summary": false,
    "impact_by_building": false,
    "config": { "baseline_window": 30 }
  }
}
```

### Top-level fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `records` | array of record | yes | 1 to 20,000 items (`DEFAULT_MAX_RECORDS`; the backend may set its own limit). More -> HTTP 413 |
| `options` | object | no | See below. Unknown fields anywhere in the body are rejected |

### Record

| Field | Type | Rules |
|---|---|---|
| `timestamp` | string | ISO-8601 date-time, e.g. `2026-03-01T00:00:00Z` or `2026-03-01T00:00:00+05:30`. Numbers (Unix epochs), empty strings and unparseable values are rejected. A timestamp with no offset is accepted and treated as UTC for ordering |
| `building` | string | Non-empty after trimming. Series are matched on the exact trimmed name (`"Library"` and `"library"` are different buildings) |
| `resource_type` | string | `electricity`, `water` or `waste` (case-insensitive; normalized to lower case) |
| `consumption_value` | number | Finite and `>= 0`. Zero is allowed (and treated as a possible meter outage when the baseline is non-zero). Booleans are rejected |
| `unit` | string | Must match the resource: electricity `kWh` (also `kwh`), water `liters` (also `litres`, `L`), waste `kg`. Other units (Wh, MWh, gallons, lbs...) are rejected; **no unit conversion is performed**. Output always uses the canonical spelling |

Unknown record fields are rejected.

### `options`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `include_impact_summary` | boolean | `false` | Also return `impact_summary` (de-duplicated totals) |
| `impact_by_building` | boolean | `false` | With the summary: one total per building and resource instead of per resource |
| `config` | object | none | Per-request tuning. Only these keys are allowed: `baseline_window` (int >= 3), `minimum_history` (int >= 3), `deviation_threshold_percentage` (number > 0), `robust_z_threshold` (number > 0), `achievable_reduction_fraction` (0 to 1) |

Everything else (emission factors and their sources, LLM settings, API keys, the
Isolation Forest method) is **server-side configuration only** and is rejected if a client sends it.

### Rules the backend must respect

* **Send each series' history in the same request.** A record's baseline is the median
  of up to `baseline_window` (default 30) *strictly earlier* records for the same
  building and resource **in that request**. The engine keeps no state between calls.
* With fewer than `minimum_history` (default 7) earlier readings a record gets
  `data_quality_status: "insufficient_data"` and no verdict.
* Records may be in any order and mix buildings and resources. **`results[i]` always
  belongs to `records[i]`** (output is not sorted).

---

## 2. Success response (HTTP 200)

```json
{
  "contract_version": "1.0",
  "status": "ok",
  "engine_version": "0.5.0",
  "summary": { "...": "see below" },
  "results": [ { "...": "one object per input record, same order" } ],
  "impact_summary": null
}
```

| Field | Type | Notes |
|---|---|---|
| `contract_version` | string | `"1.0"` |
| `status` | string | `"ok"` |
| `engine_version` | string | Engine package version |
| `summary` | object | Counts, below |
| `results` | array | One result per record, **same order and length as `records`** |
| `impact_summary` | object or null | `null` unless `options.include_impact_summary` was `true` |

### `summary`

| Field | Type | Notes |
|---|---|---|
| `records_analyzed` | integer | `== len(results)` |
| `anomalies_flagged` | integer | Results with `severity != "none"` |
| `recommendations_generated` | integer | Results with a non-null `recommendation` |
| `severity_counts` | object | Keys `none`, `low`, `medium`, `high`, `critical` (all always present) |
| `data_quality_counts` | object | Keys `ok`, `insufficient_data`, `suspect` |
| `llm_status_counts` | object | Keys `disabled`, `used`, `missing_api_key`, `sdk_unavailable`, `timeout`, `provider_error`, `invalid_response`, `call_limit_reached`; counts recommendations only |

### Result object

Every key below is **always present**; optional values are `null`, never omitted.

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `timestamp` | string (ISO-8601) | no | The record's timestamp, with its original offset |
| `building` | string | no | Trimmed building name |
| `resource_type` | enum | no | `electricity`, `water`, `waste` |
| `unit` | string | no | `kWh`, `liters` or `kg` |
| `observed_value` | number | no | The measured value |
| `baseline_value` | number | yes | Rolling median; `null` without a valid baseline |
| `deviation_percentage` | number | yes | `(observed - baseline) / baseline x 100`; `null` if undefined |
| `anomaly_score` | number 0 to 1 | no | `0.5` is the detection threshold |
| `severity` | enum | no | `none`, `low`, `medium`, `high`, `critical` |
| `explanation` | string | no | Evidence for the verdict |
| `recommended_action` | string | yes | First action of `recommendation`; `null` if there is none |
| `estimated_savings` | object `{value, unit}` | yes | Avoidable consumption in the resource's own unit |
| `calculation_assumptions` | array of string | no | Detection notes, formula, fraction and factor/source used |
| `data_quality_status` | enum | no | `ok`, `insufficient_data`, `suspect` |
| `savings_unit` | string | yes | `kWh`, `liters` or `kg`; `null` when no savings |
| `estimated_co2e_reduction_kg` | number | yes | `null` unless an emission factor is configured **server-side** |
| `calculation_status` | enum | no | `calculated`, `calculated_without_co2e`, `not_applicable`, `insufficient_baseline` |
| `recommendation` | object | yes | `null` for normal readings and insufficient history |

`calculation_status`: `calculated` = savings and CO2e computed; `calculated_without_co2e` =
savings computed, no emission factor configured; `not_applicable` = not a qualifying
above-baseline anomaly; `insufficient_baseline` = no valid baseline.

### `recommendation` object

| Field | Type | Nullable | Notes |
|---|---|---|---|
| `resource_type`, `building` | enum, string | no | Same as the result |
| `severity` | enum | no | Same values as above |
| `priority` | enum | no | `low`, `medium`, `high`, `urgent` (from severity; verification-type findings never exceed `medium`) |
| `anomaly_type` | enum | no | `above_baseline`, `below_baseline`, `suspect_reading` |
| `supporting_evidence` | array of string | no | Facts taken from the analysis |
| `recommended_actions` | array of string | no | At least one; suggested checks, not confirmed causes |
| `estimated_savings` | object `{value, unit}` | yes | Copied from the result, never recomputed |
| `estimated_co2e_reduction_kg` | number | yes | Copied from the result |
| `explanation` | string | no | Deterministic text |
| `caveats` | array of string | no | Always includes that the cause is unconfirmed |
| `source` | enum | no | `rules` (deterministic only) or `rules_llm` |
| `llm_status` | enum | no | `disabled` (default), `used`, `missing_api_key`, `sdk_unavailable`, `timeout`, `provider_error`, `invalid_response`, `call_limit_reached` |
| `llm_explanation` | string | yes | Advisory LLM text; `null` unless `llm_status` is `used` |
| `llm_suggestions` | array of string | no | Empty unless `llm_status` is `used` |
| `llm_impact_summary` | string | yes | Advisory LLM text; `null` unless used |

**UI guidance:** show the deterministic fields as the authoritative recommendation and
show `llm_*` text, if any, visually separate and labelled as AI-generated. A missing or
failed LLM is normal; it never affects the other fields.

### `impact_summary` object (when requested)

```json
{
  "totals": [
    { "building": null, "resource_type": "electricity", "savings_value": 75.0,
      "savings_unit": "kWh", "co2e_reduction_kg": null, "co2e_complete": false,
      "records_counted": 1 }
  ],
  "total_co2e_reduction_kg": null,
  "co2e_complete": false,
  "records_counted": 1,
  "duplicate_records_skipped": 0,
  "notes": ["Savings are totalled per resource in its own unit; kWh, liters and kg are never summed together.", "..."]
}
```

Savings are summed per resource only (never kWh + liters + kg). A `(building, resource,
timestamp)` is counted once. `building` is `null` unless `impact_by_building` is `true`.
`total_co2e_reduction_kg` is `null` without a factor and a partial sum (with
`co2e_complete: false`) when only some records have one.

### Serialization rules

* **Timestamps:** ISO-8601 strings. UTC is written with `Z`; other offsets keep their
  offset (`+05:30`); a timestamp sent without an offset is returned without one.
* **Enums:** their lower-case string values, as listed above.
* **Numbers:** JSON numbers. Never `NaN` or `Infinity` (the engine rejects them on input and
  never produces them). Values are rounded (6 decimals) by the engine; do not re-round for storage.
* **Optional fields:** present with `null`.
* **Nested objects:** plain JSON objects/arrays; no class names or Python-specific types.
* **Key order:** not significant.
* In Python, serialize results with `ai_engine.serialize_results(results)` (equals
  `result.model_dump(mode="json")`). Plain `model_dump()` leaves `datetime` objects that
  `json.dumps` cannot encode.

---

## 3. Error response

All client errors use one envelope and never contain partial results:

```json
{
  "contract_version": "1.0",
  "status": "error",
  "error": {
    "code": "invalid_records",
    "message": "2 record(s) failed validation; no analysis was performed.",
    "details": [
      { "index": 1, "errors": [ { "loc": ["consumption_value"], "msg": "Value error, consumption_value must not be negative" } ] },
      { "index": 3, "errors": [ { "loc": [], "msg": "Value error, incompatible unit 'MWh' for electricity; expected 'kWh'" } ] }
    ]
  }
}
```

(`msg` wording comes from the validators and may change; branch on `code`, not on `msg`.)

| HTTP | `code` | When | `details` |
|---|---|---|---|
| 422 | `invalid_json` | Body is not valid JSON (example app) | empty |
| 422 | `invalid_request` | Body is not an object, `records` missing/empty/not an array, unknown top-level field | `[{"index": null, "errors": [{"loc": [...], "msg": "..."}]}]` |
| 422 | `invalid_records` | One or more records fail validation | **every** bad record: `index` (0-based position in `records`) and its `errors` (`loc` is a list of field names) |
| 422 | `invalid_options` | Unknown option, wrong type, a non-adjustable `config` key, or a value the config validation rejects | one entry per problem, `loc` starts with `options` |
| 413 | `too_many_records` | More than the configured maximum | one entry |
| 500 | `analysis_failed` | Unexpected internal failure | empty (no internals are leaked; the exception class name is logged) |

The LLM can never cause an error response: LLM problems degrade to deterministic
recommendations and are reported per recommendation in `llm_status`.

---

## 4. Examples

Request (ten flat readings, then a spike; server config has an *illustrative* emission
factor of 0.4 kg CO2e/kWh and `achievable_reduction_fraction` 0.5):

```json
{ "records": [
  { "timestamp": "2026-03-01T00:00:00Z", "building": "Library", "resource_type": "electricity", "consumption_value": 100.0, "unit": "kWh" },
  "... nine more readings of 100.0 on 2026-03-02 .. 2026-03-10 ...",
  { "timestamp": "2026-03-11T00:00:00Z", "building": "Library", "resource_type": "electricity", "consumption_value": 250.0, "unit": "kWh" }
] }
```

Response (abridged: long text fields shortened with `...`; `results[1]` to `results[9]` omitted):

```json
{
  "contract_version": "1.0",
  "status": "ok",
  "engine_version": "0.5.0",
  "summary": {
    "records_analyzed": 11,
    "anomalies_flagged": 1,
    "recommendations_generated": 1,
    "severity_counts": { "none": 10, "low": 0, "medium": 0, "high": 1, "critical": 0 },
    "data_quality_counts": { "ok": 4, "insufficient_data": 7, "suspect": 0 },
    "llm_status_counts": { "disabled": 1, "used": 0, "missing_api_key": 0, "sdk_unavailable": 0,
                           "timeout": 0, "provider_error": 0, "invalid_response": 0, "call_limit_reached": 0 }
  },
  "results": [
    {
      "timestamp": "2026-03-01T00:00:00Z",
      "building": "Library",
      "resource_type": "electricity",
      "unit": "kWh",
      "observed_value": 100.0,
      "baseline_value": null,
      "deviation_percentage": null,
      "anomaly_score": 0.0,
      "severity": "none",
      "explanation": "Only 0 earlier observation(s) for Library / electricity; at least 7 are required to compute a baseline. ...",
      "recommended_action": null,
      "estimated_savings": null,
      "calculation_assumptions": ["...", "Savings not estimated: no valid baseline is available."],
      "data_quality_status": "insufficient_data",
      "savings_unit": null,
      "estimated_co2e_reduction_kg": null,
      "calculation_status": "insufficient_baseline",
      "recommendation": null
    },
    "...",
    {
      "timestamp": "2026-03-11T00:00:00Z",
      "building": "Library",
      "resource_type": "electricity",
      "unit": "kWh",
      "observed_value": 250.0,
      "baseline_value": 100.0,
      "deviation_percentage": 150.0,
      "anomaly_score": 0.75,
      "severity": "high",
      "explanation": "Observed 250.00 kWh is 150.0% above the historical baseline of 100.00 kWh (median of the previous 10 observation(s); threshold 50%). ... Flagged as high severity.",
      "recommended_action": "Check whether lighting is left on in unoccupied rooms, corridors and outdoor areas, ...",
      "estimated_savings": { "value": 75.0, "unit": "kWh" },
      "calculation_assumptions": ["...", "estimated_savings = max(0, observed_value - baseline_value) x achievable_reduction_fraction (0.5); expressed in kWh.", "CO2e reduction = estimated savings (kWh) x 0.4 kg CO2e/kWh (source: <your cited source, year>)."],
      "data_quality_status": "ok",
      "savings_unit": "kWh",
      "estimated_co2e_reduction_kg": 30.0,
      "calculation_status": "calculated",
      "recommendation": {
        "resource_type": "electricity",
        "building": "Library",
        "severity": "high",
        "priority": "high",
        "anomaly_type": "above_baseline",
        "supporting_evidence": [
          "Observed 250.00 kWh versus a baseline of 100.00 kWh: 150.0% above baseline.",
          "Anomaly score 0.75 (0.50 is the detection threshold); severity high.",
          "..."
        ],
        "recommended_actions": [
          "Check whether lighting is left on in unoccupied rooms, corridors and outdoor areas, and whether timers or occupancy sensors are working.",
          "Review HVAC schedules, setpoints and run times against actual occupancy and the academic timetable.",
          "..."
        ],
        "estimated_savings": { "value": 75.0, "unit": "kWh" },
        "estimated_co2e_reduction_kg": 30.0,
        "explanation": "Library's electricity use of 250.00 kWh is 150.0% above its baseline of 100.00 kWh. ...",
        "caveats": ["The cause has not been identified or confirmed; the actions are suggested checks.", "..."],
        "source": "rules",
        "llm_status": "disabled",
        "llm_explanation": null,
        "llm_suggestions": [],
        "llm_impact_summary": null
      }
    }
  ],
  "impact_summary": null
}
```

The 0.4 kg CO2e/kWh figure is an **illustrative placeholder, not a real emission factor**.
With the default configuration no factor is set, so `estimated_co2e_reduction_kg` is `null`
and `calculation_status` is `calculated_without_co2e`.

Ready-made request bodies for every scenario are in `samples/` (see `samples/README.md`).

---

## 5. Using it from Python

```python
from ai_engine import AnalysisConfig, LLMConfig, analyze_payload

CONFIG = AnalysisConfig(llm=LLMConfig.from_env())      # build once at start-up

outcome = analyze_payload(request_json, CONFIG)         # never raises for client mistakes
status_code, body = outcome.http_status, outcome.body   # body is plain JSON-safe data
```

`analyze_consumption(records, config=None)` remains available unchanged for code that
wants model objects instead of JSON; pass its output to `serialize_results` to get the
contract's result objects.

---

## 6. Forecasting (Phase 6)

Implemented by `ai_engine.forecast_payload(payload, config=None)` (`ai_engine/forecasting_engine.py`),
which wraps `forecast_consumption(records, horizon_days=7, config=None)`. It is **additive**: sections 1-5
and `analyze_consumption(records, config=None)` / `analyze_payload` are unchanged. The forecast contract
has its own version, `"forecast_contract_version": "1.0"`, so the analyze contract stays at `1.0`.

`POST /api/v1/ai/forecast` (suggested; see `examples/fastapi_integration/app.py`),
`Content-Type: application/json`.

### 6.1 Request

```json
{
  "records": [
    {"timestamp": "2026-06-01T00:00:00Z", "building": "Library",
     "resource_type": "electricity", "consumption_value": 420.0, "unit": "kWh"}
  ],
  "options": {"horizon_days": 7}
}
```

* `records`: the same record objects and validation rules as section 1 (any order, any mix of buildings and
  resources; spelling variants such as `"Electricity"` / `"kwh"` are normalized). One invalid record rejects
  the whole request with `invalid_records`; there is no partial output.
* `options.horizon_days`: integer from `1` (next day) up to `ForecastConfig.max_horizon_days` (default 30,
  hard limit 366). Default `7`. Booleans, strings and floats are rejected (`invalid_options`).
* No other top-level field or option is accepted (`invalid_request` / `invalid_options`).
* Send each series' **whole recent history** (at least 14 complete days; more is better). The engine keeps
  nothing between calls and uses at most the latest `history_window_days` (default 365) days.

### 6.2 Success response (HTTP 200)

```json
{
  "contract_version": "1.0",
  "forecast_contract_version": "1.0",
  "status": "ok",
  "engine_version": "0.7.0",
  "summary": {
    "series_count": 2,
    "forecasts_produced": 1,
    "data_quality_counts": {"ok": 1, "suspect": 0, "insufficient_data": 1},
    "model_counts": {"seasonal_naive": 1, "trailing_mean": 0, "hist_gradient_boosting": 0}
  },
  "forecasts": [ /* one object per (building, resource_type), sorted by building then resource */ ]
}
```

HTTP 200 is also returned when **no** series could be forecast; check `forecast_available` per series.
(`data_quality_counts` keys come from the `DataQualityStatus` enum.)

### 6.3 Forecast object

One object per `(building, resource_type)`. **Every key is always present** (`null` / `[]` when not applicable).
Series are never combined across buildings or resources, so kWh, liters and kg are never mixed.

| Field | Type | Meaning |
|---|---|---|
| `building`, `resource_type`, `unit` | string | Series identity; `unit` is the standard unit (`kWh`, `liters`, `kg`) |
| `forecast_available` | bool | `false` means **no forecast was produced** (see `confidence_notes`) |
| `forecast_start_date` | date or null | Day after the last *complete* observed day |
| `forecast_horizon_days` | int | Requested horizon |
| `forecast_dates` | date[] | `horizon_days` consecutive calendar days (`YYYY-MM-DD`) |
| `predicted_values` | number[] | Point forecast per day, never negative |
| `predicted_total` | number or null | Sum of `predicted_values` |
| `lower_bound`, `upper_bound` | number[] | Prediction interval per day (`lower_bound` is clipped at 0) |
| `predicted_total_lower_bound`, `predicted_total_upper_bound` | number or null | Sums of the daily bounds (a **conservative**, not statistical, range for the total) |
| `interval_level` | number or null | Nominal coverage, default `0.8` |
| `uncertainty_method` | string or null | How the interval was computed (section 6.5) |
| `baseline_method`, `model_used` | enum or null | `seasonal_naive`, `trailing_mean` or `hist_gradient_boosting` |
| `data_quality_status` | enum | `ok`, `suspect` or `insufficient_data` |
| `confidence_notes` | string[] | Plain-language caveats; always show them with the numbers |
| `history` | object | `records_received`, `first_date`, `last_date`, `observed_days`, `span_days`, `missing_days`, `excluded_days`, `coverage`, `granularity` |
| `evaluation` | object or null | Backtest evidence (section 6.4); `null` when no forecast was possible |

`forecast_available = false` forecasts have `data_quality_status = "insufficient_data"`, empty
`predicted_values` / bounds, `null` totals and model fields, and a note starting `No forecast produced:`.
**Clients must not substitute a number.**

### 6.4 Evidence: baselines, backtesting, model choice

`evaluation` = `{backtest_performed, scheme, advanced_model_considered, metrics[], selection_reason,
interval_residuals_used}`. Each `metrics` row: `{method, is_baseline, is_selected, mae, wape, n_origins, n_points}`
in the series' own unit. **MAE** is the mean absolute error; **WAPE** is `sum|error| / sum(actual)` (`null`
when the actuals sum to zero). Methods and rows are scored on the **same** rolling origins.

* Baselines: `seasonal_naive` (same weekday last week; used with at least 14 observed days) and
  `trailing_mean` (mean of the last 7 observed days; used with fewer than two weeks).
* Advanced: scikit-learn `HistGradientBoostingRegressor`, considered only with at least
  `advanced_min_days` (default 90) observed days and scikit-learn installed. It replaces the baseline **only if**
  its backtest MAE is at least `advanced_min_relative_improvement` (default 10%) lower on the same origins;
  otherwise the baseline is kept and `selection_reason` says why.
* **No future-data leakage:** every backtest origin gives the model only the days up to that origin.

### 6.5 Prediction intervals

Intervals are empirical, not a fixed percentage: half-width = the `ceil((n+1) * level)`-th smallest absolute
rolling-origin backtest error of the *selected* model (split-conformal style), per forecast step when enough
residuals exist for it, otherwise pooled over steps. If fewer than `min_residuals_for_interval` (default 10)
residuals exist, no forecast is returned. Coverage is empirical and **not guaranteed**; notes state how many
residuals were used and when steps were pooled or the sample was too small.

### 6.6 Data handling (nothing is invented)

* Readings are summed to calendar days **only** for fully covered days at the series' own frequency
  (e.g. 24 hourly readings). Partial, ambiguous or contradictory days are excluded, never estimated.
* Missing days stay missing (not imputed); a gap longer than `max_gap_days` (default 14) restarts the usable
  history after the gap; coverage below `min_coverage` (default 70%) gives `insufficient_data`.
* Readings coarser than daily are not disaggregated (`insufficient_data`).
* Each reading is assumed to be the consumption of its own interval, not a cumulative meter value.
  Timestamps without an offset are treated as UTC; mixed offsets are converted to the most common one.
* Irregularities that still allow a forecast set `data_quality_status = "suspect"` and are listed in the notes.

### 6.7 Errors

Same envelope as section 3. Codes: `invalid_request` (422, bad body or unknown field), `invalid_options` (422),
`invalid_records` (422, with per-record `details`), `too_many_records` (413), `forecast_failed` (500, no
internals leaked). The example app adds `invalid_json` (422) for a malformed body.

### 6.8 Example (illustrative excerpt, hand-written)

Four weeks of an exactly repeating weekly pattern starting on a Monday, `horizon_days = 7`. This excerpt shows
the **shape**; it is not captured engine output. Run `python scripts/generate_forecast_samples.py --outputs`
to store real responses for `samples/forecasting/`.

```json
{
  "building": "Library",
  "resource_type": "electricity",
  "unit": "kWh",
  "forecast_available": true,
  "forecast_start_date": "2026-03-30",
  "forecast_horizon_days": 7,
  "forecast_dates": ["2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04", "2026-04-05"],
  "predicted_values": [100.0, 110.0, 105.0, 108.0, 102.0, 60.0, 55.0],
  "predicted_total": 640.0,
  "baseline_method": "seasonal_naive",
  "model_used": "seasonal_naive",
  "lower_bound": [100.0, 110.0, 105.0, 108.0, 102.0, 60.0, 55.0],
  "upper_bound": [100.0, 110.0, 105.0, 108.0, 102.0, 60.0, 55.0],
  "interval_level": 0.8,
  "data_quality_status": "ok",
  "confidence_notes": ["... All backtest errors were zero, so the bounds equal the point forecast. ..."],
  "history": {"observed_days": 28, "missing_days": 0, "granularity": "daily"},
  "evaluation": {"backtest_performed": true, "advanced_model_considered": false}
}
```

(Excerpt: `uncertainty_method`, `predicted_total_*_bound` and the remaining `history` / `evaluation` keys are omitted.)

Ready-made requests: `samples/forecasting/` (see `samples/README.md`).

### 6.9 Using it from Python

```python
from ai_engine import ForecastConfig, forecast_consumption, forecast_payload, serialize_forecasts

CONFIG = ForecastConfig()                                  # build once at start-up
outcome = forecast_payload(request_json, CONFIG)           # never raises for client mistakes
status_code, body = outcome.http_status, outcome.body

results = forecast_consumption(records, horizon_days=1)    # model objects (next day)
json_ready = serialize_forecasts(results)                  # same objects as body["forecasts"]
```

Forecasting is CPU-bound (a long history can train and backtest a gradient-boosting model, roughly 0.1-0.3 s
per fit, 13 fits per long series on the default settings); call it from a worker thread like `analyze_payload`.

---

## 7. What-if simulation and scenario comparison (Phase 7B)

**Simulation contract version:** `1.0` (`ai_engine.SIMULATION_CONTRACT_VERSION`).
Implemented by:
- `ai_engine.simulate_scenario(records, scenario, config=None)` (single scenario simulation dict result)
- `ai_engine.simulate_payload(payload, config)` (single scenario envelope)
- `ai_engine.compare_scenarios(records, scenarios, config=None)` (multi-scenario comparison dict result)
- `ai_engine.compare_payload(payload, config)` (multi-scenario comparison envelope)

Suggested paths:
- `POST /api/simulate` (aliases: `POST /api/v1/ai/simulate`)
- `POST /api/simulate/compare` (aliases: `POST /api/v1/ai/simulate/compare`)

The analyze and forecast contracts are unchanged.

The simulator answers: *if this building (or the whole campus) cut this resource by X %, what would
that save relative to its recent consumption or future forecasted consumption?* It is deterministic and offline, and every number is
produced by Python arithmetic, never by an LLM.

> **A result is a hypothetical projection, never measured savings.** It assumes the stated
> reduction is achieved over the whole baseline period. `projection_type` is always `"hypothetical"`
> and `measured_savings` is always `null`. No monetary value is ever produced.

### 7.1 Request: Single scenario simulation (`POST /api/simulate`)

```json
{
  "records": [ { "timestamp": "...", "building": "BLOCK_A", "resource_type": "electricity",
                 "consumption_value": 420.0, "unit": "kWh" } ],
  "scenario": {
    "resource_type": "electricity",
    "building_code": "BLOCK_A",
    "reduction_percentage": 15,
    "baseline_period_days": 30
  }
}
```

Forecast-based scenario request:

```json
{
  "records": [ ... ],
  "scenario": {
    "resource_type": "electricity",
    "building_code": "BLOCK_A",
    "reduction_percentage": 15,
    "baseline_source": "forecast",
    "forecast_horizon_days": 7
  }
}
```

| Field | Required | Rules |
|---|---|---|
| `records` | no | Array of the same record objects as section 1 (same validation and canonical units). May be omitted or `[]`, e.g. with an explicit baseline. At most the server's record limit (413 `too_many_records`). |
| `scenario` | yes | Object, fields below. Unknown fields are rejected. |
| `scenario.resource_type` | yes | `electricity` (kWh), `water` (liters) or `waste` (kg); case-insensitive. |
| `scenario.building_code` | no | Matches the record field `building` (case-insensitive, trimmed). **Omit or `null` for a campus-wide scenario.** Empty strings are rejected. |
| `scenario.reduction_percentage` | yes | Number from **0 to 100 inclusive**. Anything else (negative, above 100, NaN, a string, a boolean) is rejected. |
| `scenario.baseline_period_days` | no | Whole number 1..366, default 30. Length of the baseline period (used for historical baselines). |
| `scenario.baseline_source` | no | Baseline origin: `"historical"` (default; measured records), `"forecast"` (Phase 6 forecasting engine projection), or `"explicit"`. |
| `scenario.forecast_horizon_days` | no | Whole number 1..366 (default 7). Horizon used when `baseline_source: "forecast"`. |
| `scenario.forecast_config` | no | Optional object with `ForecastConfig` tuning overrides (e.g. `advanced_model_enabled`). |
| `scenario.baseline_consumption` | no | Explicit baseline `{"value": number >= 0, "unit": "kWh", "period_days": 30}`. `value`, `unit` and `period_days` are all required together, so a unit or period is never guessed. The unit must be the resource's standard unit (spelling aliases such as `litres` are accepted; `Wh`, `MWh`, gallons... are rejected, never converted). When given, records are **not** used for the baseline. If `baseline_period_days` is also given it must equal `period_days` (periods are never rescaled). Cannot be combined with `baseline_source: "forecast"`. |
| `scenario.scenario_id` | no | Your own ID (1-64 characters: letters, digits, `_`, `.`, `-`). Otherwise a deterministic ID `scn-<12 hex>` is derived from the scenario definition, so the same scenario always gets the same ID and the data does not change it. |

### 7.2 Calculation

```
estimated_savings      = baseline x reduction_percentage / 100
projected_consumption  = baseline - estimated_savings
estimated_co2e_savings = estimated_savings x emission factor        (only if a factor is configured)
```

* Savings are never negative and never exceed the baseline. Values are rounded to 6 decimals.
* **Measured baseline** (no explicit baseline): the sum of the matching readings in the
  `baseline_period_days` calendar days (UTC) **ending on the most recent matching reading** (not on
  today's date, so results are reproducible). Naive timestamps are treated as UTC. Each reading is
  interval consumption (not a cumulative meter value). Records of other resources are never mixed in.
* **No gap filling.** If only part of the window has readings, the baseline is the sum of what exists, the
  result is `suspect`, and `baseline.days_with_data` / `coverage_fraction` say how much was covered
  (savings are understated by the same share). Below **80 %** of the window's days
  (`MIN_COVERAGE_FRACTION`, an engineering default) no baseline is reported (`insufficient_coverage`).
* **Duplicates:** identical readings (same building, timestamp and value) count once; conflicting readings
  for the same building and timestamp are excluded and reported (`suspect`).
* **CO2e** uses the engine's existing emission-factor configuration (`AnalysisConfig`:
  `electricity_emission_factor_kg_co2e_per_kwh`, `water_emission_factor_kg_co2e_per_liter` with a source,
  `waste_emission_factor` with unit and source). No factor ships with the engine; without one
  `estimated_co2e_savings_kg` is `null`, `emission_factor` is `null` and a warning says so. Emission factors
  are set server-side and cannot be sent in a request.

### 7.3 Success response (HTTP 200)

```json
{
  "contract_version": "1.0",
  "simulation_contract_version": "1.0",
  "status": "ok",
  "engine_version": "0.7.0",
  "result": { "...": "result object, see 7.4" }
}
```

A scenario the **data** cannot support (empty dataset, unknown building, no records for the resource,
too little coverage) is still **HTTP 200** with `result.calculation_status = "not_calculated"`; check that
field before reading numbers.

### 7.4 Result object

| Field | Type | Meaning |
|---|---|---|
| `simulation_contract_version`, `engine_version` | string | Versions. |
| `scenario_id` | string | Supplied or deterministic ID. |
| `projection_type` | `"hypothetical"` | Always. |
| `calculation_status` | `"calculated"` \| `"not_calculated"` | Whether numbers were produced. |
| `not_calculated_reason` | string \| null | `no_records`, `unknown_building`, `no_resource_records`, `no_usable_readings`, `insufficient_coverage`; `null` when calculated. |
| `scope` | object | `level` (`"building"` or `"campus"`), `building_code` (data's own spelling when found, else as requested, `null` for campus), `buildings_included` (sorted names that contributed). |
| `resource_type`, `unit` | string | Resource and its standard unit (`kWh`, `liters`, `kg`). |
| `baseline` | object | `value` (or `null`), `unit`, `period_days`, `source` (`"measured_records"` or `"explicit"`), `window_start` / `window_end` (ISO dates, `null` for explicit), `days_with_data`, `coverage_fraction`, `readings_used`. |
| `reduction_percentage` | number | The scenario's percentage. |
| `projected_consumption` | `{value, unit}` | `baseline - savings`; `value` is `null` when not calculated. |
| `estimated_savings` | `{value, unit}` | `baseline x pct / 100`; `value` is `null` when not calculated; never negative. |
| `estimated_co2e_savings_kg` | number \| null | `null` when no valid factor is configured or nothing was calculated. |
| `emission_factor` | object \| null | `{value, unit}` actually applied. |
| `measured_savings` | `null` | Always `null`: this is not a measurement. |
| `units` | object | `consumption`, `co2e` (`"kg CO2e"`), `period` (`"days"`). |
| `building_breakdown` | array \| null | Campus scenarios with a measured baseline: `{building, baseline_value, days_with_data, readings_used}` per building. Otherwise `null`. |
| `assumptions` | string[] | Formulas, units, period, window rule, "hypothetical", "no monetary savings", emission-factor source. |
| `warnings` | string[] | Data problems and omissions (gaps, duplicates, missing factor, unknown building...). Read these. |
| `data_quality_status` | `"ok"` \| `"suspect"` \| `"insufficient_data"` | Same values as the analyze contract. `suspect`: calculated but with gaps, conflicts, irregular frequency, mixed timestamp conventions, case-variant building names or a zero baseline. `insufficient_data`: not calculated. An explicit baseline is `ok` (it is caller-supplied and not verified; the assumptions say so). |

### 7.5 Errors

Same envelope as section 3. New code: **`invalid_scenario`** (HTTP 422); `details[].errors[].loc` is the
path, e.g. `["scenario", "reduction_percentage"]`. Others reused: `invalid_json`, `invalid_request`
(non-object body, unknown top-level field, missing/non-object `scenario`, non-array `records`),
`invalid_records`, `too_many_records` (413), `simulation_failed` (500).

### 7.6 Examples (real engine output; regenerate with `python scripts/generate_simulation_samples.py --outputs`)

Request: [`samples/simulation/sim_building_electricity_15pct.json`](../samples/simulation/sim_building_electricity_15pct.json)
(45 days of synthetic BLOCK_A readings; scenario as in 7.1). Response with the default config (no emission factor):

```json
{
  "contract_version": "1.0",
  "simulation_contract_version": "1.0",
  "status": "ok",
  "engine_version": "0.7.0",
  "result": {
    "simulation_contract_version": "1.0",
    "engine_version": "0.7.0",
    "scenario_id": "scn-eb07ac88ee72",
    "projection_type": "hypothetical",
    "calculation_status": "calculated",
    "not_calculated_reason": null,
    "scope": {
      "level": "building",
      "building_code": "BLOCK_A",
      "buildings_included": [
        "BLOCK_A"
      ]
    },
    "resource_type": "electricity",
    "unit": "kWh",
    "baseline": {
      "value": 11382.0,
      "unit": "kWh",
      "period_days": 30,
      "source": "measured_records",
      "window_start": "2026-06-16",
      "window_end": "2026-07-15",
      "days_with_data": 30,
      "coverage_fraction": 1.0,
      "readings_used": 30
    },
    "reduction_percentage": 15.0,
    "projected_consumption": {
      "value": 9674.7,
      "unit": "kWh"
    },
    "estimated_savings": {
      "value": 1707.3,
      "unit": "kWh"
    },
    "estimated_co2e_savings_kg": null,
    "emission_factor": null,
    "measured_savings": null,
    "units": {
      "consumption": "kWh",
      "co2e": "kg CO2e",
      "period": "days"
    },
    "building_breakdown": null,
    "assumptions": [
      "Hypothetical projection: it assumes the stated reduction is achieved over the whole baseline period. It is not measured savings and does not show that any reduction happened.",
      "estimated_savings = baseline x reduction_percentage / 100; projected_consumption = baseline - estimated_savings.",
      "All consumption values are in kWh. The baseline period is 30 days and is never rescaled.",
      "No monetary savings are estimated.",
      "Baseline = sum of the electricity readings (kWh) in the 30 calendar days (UTC) ending on the most recent reading (2026-07-15), not on today's date.",
      "Each record's consumption_value is consumption for its own interval (not a cumulative meter value); gaps are not filled or extrapolated.",
      "15 reading(s) older than the baseline window were not used."
    ],
    "warnings": [
      "No electricity emission factor configured; CO2e reduction not calculated."
    ],
    "data_quality_status": "ok"
  }
}
```

With an **illustrative** electricity factor of 0.5 kg CO2e/kWh (a placeholder for this documentation, not a real grid factor),
only these fields differ ([`...co2e_example.response.json`](../samples/simulation/sim_building_electricity_15pct.co2e_example.response.json)):

```json
{
  "estimated_co2e_savings_kg": 853.65,
  "emission_factor": {
    "value": 0.5,
    "unit": "kg CO2e/kWh"
  }
}
```

Not calculable (unknown building; HTTP 200), result excerpt ([`sim_unknown_building`](../samples/simulation/sim_unknown_building.response.json)):

```json
{
  "calculation_status": "not_calculated",
  "not_calculated_reason": "unknown_building",
  "baseline": {
    "value": null,
    "unit": "kWh",
    "period_days": 30,
    "source": "measured_records",
    "window_start": null,
    "window_end": null,
    "days_with_data": null,
    "coverage_fraction": null,
    "readings_used": null
  },
  "estimated_savings": {
    "value": null,
    "unit": "kWh"
  },
  "projected_consumption": {
    "value": null,
    "unit": "kWh"
  },
  "estimated_co2e_savings_kg": null,
  "data_quality_status": "insufficient_data",
  "warnings": [
    "Building 'BLOCK_Z' does not appear in the supplied records; known buildings: 'BLOCK_A'."
  ]
}
```

Invalid scenario (HTTP 422, [`sim_invalid_percentage`](../samples/simulation/sim_invalid_percentage.response.json)):

```json
{
  "contract_version": "1.0",
  "status": "error",
  "error": {
    "code": "invalid_scenario",
    "message": "The scenario failed validation (1 problem(s)); nothing was simulated.",
    "details": [
      {
        "index": null,
        "errors": [
          {
            "loc": [
              "scenario",
              "reduction_percentage"
            ],
            "msg": "must be between 0 and 100 inclusive; got 120"
          }
        ]
      }
    ]
  }
}
```

Campus-wide scenario: omit `building_code`; the response has `scope.level = "campus"` and a `building_breakdown`
([`sim_campus_water_10pct`](../samples/simulation/sim_campus_water_10pct.response.json)). Explicit baseline with no records:
[`sim_waste_explicit_baseline.json`](../samples/simulation/sim_waste_explicit_baseline.json). More: `sim_partial_coverage`, `sim_zero_baseline`.

### 7.7 Using it from Python

```python
from ai_engine import AnalysisConfig, ScenarioValidationError, simulate_payload, simulate_scenario

CONFIG = AnalysisConfig(...)                         # build once at start-up (emission factors with sources)
outcome = simulate_payload(request_json, CONFIG)     # never raises for client mistakes
status_code, body = outcome.http_status, outcome.body

try:                                                 # plain dict result; raises for a malformed scenario
    result = simulate_scenario(records, {"resource_type": "water", "reduction_percentage": 10}, CONFIG)
except ScenarioValidationError as exc:
    print(exc.errors)                                # [{"loc": ["reduction_percentage"], "msg": "..."}]
```

`simulate_scenario` raises `ScenarioValidationError` for a bad scenario, the existing `RecordValidationError`
for bad records, and `TypeError` for wrong argument types (`records=None`, a non-mapping scenario, a non-`AnalysisConfig`).
It is cheap and synchronous (a single pass over the records).

### 7.8 Distinguishing Projected Savings vs. Actual Measured Savings

The EcoCamp AI engine maintains a strict, mathematically sound boundary between **projected** savings and **actual measured** savings:

| Attribute | Projected Savings (`simulate_scenario`, `compare_scenarios`) | Actual Measured Savings (`analyze_consumption`) |
|---|---|---|
| **Nature** | **Hypothetical / Counterfactual target**. Answers: *"What would be saved IF a target percentage reduction is achieved?"* | **Observed historical reduction**. Identifies avoidable consumption from actual measured meter anomalies. |
| **Field indicator** | `projection_type: "hypothetical"`, `measured_savings: null`. | `severity`, `is_anomaly: true`, `estimated_avoidable_consumption`. |
| **Baseline origin** | Historical window sum or Phase 6 model forecast over a future horizon. | Rolling median of strictly earlier measured readings within the same series. |
| **Certainty** | Target assumption (or bounded by forecast empirical uncertainty when forecast-based). | Empirical measurement; never speculative. |
| **Financial values** | Strictly `null`. The engine never fabricates currency or monetary figures. | Strictly `null`. Monies are never calculated or guessed. |

Neither simulator results nor comparison outcomes ever represent real-world achieved reductions; they are planning targets.

---

### 7.9 Forecast-based scenarios & uncertainty propagation

When `scenario.baseline_source = "forecast"`:
1. **Engine integration**: The simulator invokes the Phase 6 forecasting engine (`forecast_consumption`) using the requested `forecast_horizon_days` (default 7).
2. **Forecast baseline**: The predicted total consumption over the horizon becomes the baseline:
   `baseline.value = forecast.predicted_total`.
3. **Handling insufficient forecast data**: If records have fewer than 14 usable days, gaps exceeding tolerance, or irregular resolution, the forecast engine marks `forecast_available = false`. The simulation returns:
   - `calculation_status: "not_calculated"`
   - `not_calculated_reason: "insufficient_forecast_data"`
   - `data_quality_status: "insufficient_data"`
   - Baseline, savings, and projected consumption values are strictly `null`.
4. **Uncertainty propagation**: If the Phase 6 model produced empirical prediction intervals from backtest residuals, the simulation propagates these bounds mathematically:
   - `baseline.lower_bound = forecast.total_lower_bound`
   - `baseline.upper_bound = forecast.total_upper_bound`
   - `estimated_savings.lower_bound = baseline.lower_bound * reduction_percentage / 100`
   - `estimated_savings.upper_bound = baseline.upper_bound * reduction_percentage / 100`
   - `projected_consumption.lower_bound = baseline.lower_bound * (1 - reduction_percentage / 100)`
   - `projected_consumption.upper_bound = baseline.upper_bound * (1 - reduction_percentage / 100)`
   - When an emission factor is configured, CO2e bounds are scaled by that factor.
   - **No fabrication**: If backtest residuals are insufficient or intervals are unavailable, `uncertainty` is omitted or bounds remain `null`. The engine never fabricates arbitrary confidence bands.

---

### 7.10 Advanced Scenario Comparison (`POST /api/simulate/compare`)

Implemented by `compare_scenarios(records, scenarios, config=None)` and `compare_payload(payload, config=None)`.
Suggested path: `POST /api/simulate/compare` (alias: `POST /api/v1/ai/simulate/compare`).

#### 7.10.1 Request

```json
{
  "records": [ ... ],
  "scenarios": [
    {
      "scenario_id": "opt-1",
      "resource_type": "electricity",
      "building_code": "BLOCK_A",
      "reduction_percentage": 10
    },
    {
      "scenario_id": "opt-2",
      "resource_type": "electricity",
      "building_code": "BLOCK_B",
      "reduction_percentage": 15
    },
    {
      "scenario_id": "opt-campus",
      "resource_type": "electricity",
      "reduction_percentage": 12
    }
  ]
}
```

| Field | Required | Rules |
|---|---|---|
| `records` | no | Array of consumption records (same schema as section 1). |
| `scenarios` | yes | Array of 1 to 50 scenario objects (each matching the single scenario schema in 7.1). |

#### 7.10.2 Deterministic Double-Counting Prevention Policy

When comparing multiple scenarios, naive summation of savings can double-count overlapping scopes:
- **Competing building scenarios**: Multiple proposed reductions targeting the same building and resource (e.g., 10% vs 20% on BLOCK_A electricity).
- **Scope overlap (Campus vs. Building)**: A campus-wide scenario overlaps with any building-level scenario for the same resource.

**The Policy (`deterministic_maximum_savings`)**:
1. Scenarios are evaluated independently first to establish projected savings and baselines.
2. For each resource, the engine detects overlaps:
   - Multiple scenarios on the same building are mutually exclusive.
   - A campus-wide scenario competes with building-specific scenarios for that resource.
3. Within each conflicting group:
   - The engine selects the subset yielding the **maximum achievable savings**. If a campus-wide scenario saves more than the sum of building-specific scenarios, the campus scenario is selected; otherwise, the non-overlapping building scenarios are selected.
   - Ties are broken deterministically by scenario ID.
4. **Reporting**:
   - `portfolio.included_scenario_ids`: Scenarios selected for the non-overlapping portfolio.
   - `portfolio.excluded_scenario_ids`: Overlapping scenarios excluded to prevent double-counting.
   - `portfolio.overlap_groups`: Explanations of each detected conflict, competing IDs, selected ID, and policy reason.
   - `resource_totals[resource].portfolio_savings`: The deduplicated portfolio savings.
   - `resource_totals[resource].unadjusted_gross_savings`: The naive sum of all calculated scenarios, provided for full transparency.

#### 7.10.3 Comparison Response Structure

```json
{
  "contract_version": "1.0",
  "simulation_contract_version": "1.0",
  "status": "ok",
  "engine_version": "0.7.0",
  "result": {
    "simulation_contract_version": "1.0",
    "engine_version": "0.7.0",
    "projection_type": "hypothetical",
    "measured_savings": null,
    "scenarios_count": 3,
    "portfolio": {
      "policy": "deterministic_maximum_savings",
      "overlapping_scenarios_detected": true,
      "included_scenario_ids": ["opt-campus"],
      "excluded_scenario_ids": ["opt-1", "opt-2"],
      "overlap_groups": [
        {
          "conflict_type": "campus_vs_building_overlap",
          "resource_type": "electricity",
          "scope_level": "mixed",
          "competing_scenario_ids": ["opt-campus", "opt-1", "opt-2"],
          "selected_scenario_id": "opt-campus",
          "excluded_scenario_ids": ["opt-1", "opt-2"],
          "policy_applied": "deterministic_maximum_savings",
          "reason": "Campus-wide scenario yielded higher total savings than competing building scenarios"
        }
      ]
    },
    "resource_totals": {
      "electricity": {
        "unit": "kWh",
        "scenarios_count": 3,
        "calculated_scenarios_count": 3,
        "included_in_portfolio_count": 1,
        "excluded_from_portfolio_count": 2,
        "portfolio_savings": 1440.0,
        "unadjusted_gross_savings": 1950.0,
        "total_baseline": 12000.0,
        "total_estimated_savings": 1440.0,
        "total_estimated_co2e_savings_kg": null,
        "overlapping_detected": true
      },
      "water": { ... },
      "waste": { ... }
    },
    "total_co2e_savings_kg": null,
    "co2e_supported": false,
    "rankings": {
      "by_resource": {
        "electricity": [
          { "rank": 1, "scenario_id": "opt-campus", "savings_value": 1440.0, "unit": "kWh", "reduction_percentage": 12.0 },
          { "rank": 2, "scenario_id": "opt-2", "savings_value": 300.0, "unit": "kWh", "reduction_percentage": 15.0 },
          { "rank": 3, "scenario_id": "opt-1", "savings_value": 210.0, "unit": "kWh", "reduction_percentage": 10.0 }
        ]
      },
      "by_percentage": [ ... ]
    },
    "scenarios": [ { "...": "individual simulation objects matching 7.4" } ],
    "assumptions": [ ... ],
    "warnings": [ ... ],
    "data_quality_status": "ok"
  }
}
```

---

### 7.11 Limitations

* **Projections, not measurements**: Neither simulation nor scenario comparison represents actual achieved savings; they are hypothetical targets.
* **Rankings**: Rankings are ordered strictly by absolute savings quantity (kWh, liters, kg) or percentage target. They do not evaluate operational feasibility, capital expenditure, or implementation difficulty.
* **Double-counting policy**: The `deterministic_maximum_savings` policy selects the mathematically highest-saving option when scenarios conflict. If a user wishes to implement an alternative combination, they should submit only the desired non-conflicting scenarios.
* **Forecast baselines**: Forecast-based scenarios inherit all Phase 6 forecasting limitations (weekly pattern extrapolation, no holiday calendars or weather covariates, sensitivity to regime changes).
* **Emissions & financials**: CO2e is computed only when emission factors are explicitly configured; no monetary values are ever calculated.
