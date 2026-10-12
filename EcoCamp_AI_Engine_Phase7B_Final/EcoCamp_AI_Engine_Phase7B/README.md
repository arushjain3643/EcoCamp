# EcoCamp AI Engine: scenario comparison + what-if simulator + forecasting + integration package (Phase 7B)

AI engine of **EcoCamp**, the campus sustainability intelligence platform (KAUSHAL 8.0).
It analyzes electricity, water and waste consumption: anomaly detection, estimated avoidable
consumption (and CO2e when you supply a cited emission factor), evidence-based
recommendations, next-day / multi-day consumption forecasts, a deterministic **what-if simulator**
("what would a 15 % cut in BLOCK_A's electricity save?"), and **advanced scenario comparison**
with deterministic overlap deduplication and forecast-based simulation. It is a plain Python package: **no server, database or frontend**.
The team's FastAPI backend calls it; this package only adds what that integration needs.

| Phase | Content |
|---|---|
| 1 | Architecture and validation |
| 2 | Anomaly detection |
| 3 | Environmental impact |
| 4 | Recommendations (deterministic rules + optional LLM) |
| 5 | Integration package: JSON contract, serialization, FastAPI example, samples, smoke tests, docs |
| 6 | Forecasting: next-day and multi-day predictions with baselines, backtesting, MAE/WAPE and evidence-based intervals |
| 7A | What-if simulator: projected resource and CO2e savings of a percentage reduction, per building or campus-wide (hypothetical, never measured) |
| **7B** | **Advanced scenario comparison & forecast-based scenarios: multi-building/resource comparison, deterministic overlap deduplication, forecast baseline integration with uncertainty propagation** |

Phases 5, 6, 7A and 7B changed no detection, impact, recommendation or forecasting logic.
`analyze_consumption(records, config=None)`, `forecast_consumption(records, horizon_days=7, config=None)`
and every earlier public name are unchanged; everything new is additive.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest                      # runs ai_engine/tests (pytest.ini sets the paths)
```

```python
from ai_engine import AnalysisConfig, LLMConfig, analyze_consumption, analyze_payload

# 1) Existing API, unchanged: returns AnalysisResult objects
results = analyze_consumption(records)                 # records: dicts or ConsumptionRecord

# 2) JSON-safe entry point for the backend
outcome = analyze_payload({"records": records}, AnalysisConfig(llm=LLMConfig.from_env()))
outcome.http_status, outcome.body                      # 200 + response, or 422/413/500 + error envelope
```

Forecasting (Phase 6) is a second, independent entry point:

```python
from ai_engine import ForecastConfig, forecast_consumption, forecast_payload

# Model objects: one ForecastResult per (building, resource), next day or several days
forecasts = forecast_consumption(records, horizon_days=7)          # horizon_days=1 -> next day

# JSON-safe entry point for the backend (POST /api/v1/ai/forecast in the example app)
outcome = forecast_payload({"records": records, "options": {"horizon_days": 7}}, ForecastConfig())
outcome.http_status, outcome.body                                  # 200 + forecasts, or 422/413/500 + error envelope
```

What-if simulation (Phase 7A & 7B) is a third, independent entry point:

```python
from ai_engine import simulate_payload, simulate_scenario

# Single scenario simulation: historical or forecast baseline
scenario = {
    "resource_type": "electricity",
    "building_code": "BLOCK_A",
    "reduction_percentage": 15,
    "baseline_period_days": 30,  # or baseline_source="forecast", forecast_horizon_days=7
}
result = simulate_scenario(records, scenario)         # plain JSON-safe dict; raises ScenarioValidationError if malformed
result["estimated_savings"], result["projected_consumption"]      # e.g. {"value": 1707.3, "unit": "kWh"}, ...

# JSON-safe entry point for the backend (POST /api/simulate in the example app)
outcome = simulate_payload({"records": records, "scenario": scenario}, AnalysisConfig())
outcome.http_status, outcome.body                      # 200 + result, or 422/413/500 + error envelope
```

Advanced scenario comparison (Phase 7B) is a fourth entry point:

```python
from ai_engine import compare_payload, compare_scenarios

scenarios = [
    {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 10},
    {"resource_type": "electricity", "building_code": "BLOCK_B", "reduction_percentage": 15},
    {"resource_type": "electricity", "reduction_percentage": 12},  # campus-wide
]
# Multi-scenario comparison with deduplicated portfolio and rankings
res = compare_scenarios(records, scenarios)
portfolio_savings = res["resource_totals"]["electricity"]["portfolio_savings"]
rankings = res["rankings"]["by_resource"]["electricity"]

# JSON-safe entry point for the backend (POST /api/simulate/compare in the example app)
outcome = compare_payload({"records": records, "scenarios": scenarios}, AnalysisConfig())
outcome.http_status, outcome.body                      # 200 + comparison result, or 422/413/500
```

No API key is needed. With the LLM off (default) recommendations are deterministic and the
engine makes no network calls.

## What to read

| For | File |
|---|---|
| The request/response contract (fields, enums, errors, examples) | [`docs/JSON_CONTRACT.md`](docs/JSON_CONTRACT.md) |
| Gemini/OpenAI setup, timeouts, fallback | [`docs/LLM_CONFIGURATION.md`](docs/LLM_CONFIGURATION.md) |
| How to call the engine from FastAPI | [`examples/fastapi_integration/`](examples/fastapi_integration/README.md) |
| Sample request bodies per scenario (analysis, forecasting, simulation) | [`samples/README.md`](samples/README.md) |
| Simulation & comparison contract (section 7) | [`docs/JSON_CONTRACT.md`](docs/JSON_CONTRACT.md#7-what-if-simulation-and-scenario-comparison-phase-7b) |
| Phase 7B verification & test report | [`docs/PHASE7B_VERIFICATION.md`](docs/PHASE7B_VERIFICATION.md) |
| Phase 7A verification archive | [`docs/PHASE7A_VERIFICATION.md`](docs/PHASE7A_VERIFICATION.md) |
| Forecasting contract (section 6) | [`docs/JSON_CONTRACT.md`](docs/JSON_CONTRACT.md#6-forecasting-phase-6) |
| Algorithms, units, validation rules, configuration reference (Phases 1-4) | [`ai_engine/README.md`](ai_engine/README.md) |
| Environment variables | [`.env.example`](.env.example) |

## Integrating into the backend (checklist)

1. Copy the `ai_engine/` folder into the backend project (or put this folder's root on `PYTHONPATH`)
   and install `requirements.txt` (pydantic, numpy, scikit-learn). FastAPI is the backend's own dependency.
2. Build **one** `AnalysisConfig` at start-up: emission factors with their sources from your settings,
   `llm=LLMConfig.from_env()`.
3. Call `analyze_payload(body, config)` from a worker thread (`run_in_threadpool`) and return
   `outcome.body` with `outcome.http_status` (see the example app).
4. Send each building/resource's **whole history** in one request: baselines are built only from earlier
   records in the same request, and the engine stores nothing between calls.
5. Store or forward results as JSON as they are; `results[i]` belongs to `records[i]`.
6. Show deterministic recommendation fields as primary content and `llm_*` text (if any) separately.

## Forecasting (Phase 6)

`forecast_consumption(records, horizon_days=7, config=None)` produces one forecast per
(building, resource_type) series for electricity (kWh), water (liters) and waste (kg), never mixing
buildings or units. Records are validated exactly as in `analyze_consumption`.

| Topic | What the engine does |
|---|---|
| Horizon | `1` (next day) up to `ForecastConfig.max_horizon_days` (default 30); default 7 |
| Data handling | Readings are summed to calendar days only when the day is fully covered; partial/ambiguous/contradictory days are excluded, missing days are **never imputed**, long gaps restart the usable history |
| Insufficient history | Fewer than 14 usable days, coverage below 70%, readings coarser than daily, or too few backtest residuals: `forecast_available=false`, `data_quality_status="insufficient_data"`, no numbers |
| Baselines | `seasonal_naive` (same weekday last week) and `trailing_mean` (last 7 days; used when under two weeks of history) |
| Advanced model | scikit-learn `HistGradientBoostingRegressor`, considered only with at least 90 observed days; it is **selected only if** its backtest MAE beats the baseline by at least 10% on the same origins |
| Evaluation | Rolling-origin (expanding window) backtest, MAE and WAPE per method, reported in every forecast |
| No leakage | Each backtest origin (and the final fit) sees only days up to that origin; tests check that altering later data cannot change earlier predictions |
| Intervals | Empirical, from the selected model's own backtest residuals (split-conformal style, default 80%); no arbitrary percentage bands; coverage is not guaranteed |
| Output | `ForecastResult` models; `serialize_forecasts` / `forecast_payload` give JSON-safe data (see the contract) |

### Performance notes

Backtesting the gradient-boosting model is the only expensive step: one fit costs roughly 0.1-0.3 s
(single core) and a long series needs `advanced_backtest_origins` (default 12) fits for the evidence plus
one for the final forecast. What the engine does to keep this bounded **without weakening the evidence**:

* The advanced model is never run for series under `advanced_min_days` (default 90), and only the most recent
  `history_window_days` (365) days are used.
* Backtest origins for the advanced model are spaced `advanced_backtest_step_days` (5) apart and capped at
  `advanced_backtest_origins` (12); the baseline is evaluated on exactly the same origins.
* Training features are computed with per-origin rolling means instead of per-row recomputation
  (results are identical to the previous implementation).
* Identical fits (same history, horizon, weekday offset and seed) are memoised in a small in-process LRU
  cache (64 entries), so a retried request or an unchanged series costs milliseconds. The key is the full
  history, so different data can never reuse another series' result. `ai_engine.forecasting_engine.clear_forecast_cache()`
  empties it.
* The advanced model is not disabled by default; set `ForecastConfig(advanced_model_enabled=False)` to skip it
  explicitly. Fitting is the dominant remaining cost; lowering `advanced_backtest_origins` trades evidence for speed.

Call forecasting from a worker thread (the example app does).

## What-if simulator & scenario comparison (Phase 7A & 7B)

`simulate_scenario(records, scenario, config=None)` projects the savings of a **proposed percentage
reduction** of electricity (kWh), water (liters) or waste (kg), for one building or the whole campus.
`compare_scenarios(records, scenarios, config=None)` evaluates and ranks multiple scenarios simultaneously,
resolving overlaps using a deterministic deduplication policy.
The full contract is documented in [`docs/JSON_CONTRACT.md`](docs/JSON_CONTRACT.md#7-what-if-simulation-and-scenario-comparison-phase-7b).

```
estimated_savings     = baseline x reduction_percentage / 100
projected_consumption = baseline - estimated_savings
estimated_co2e_savings = estimated_savings x emission factor      (only when you configured a factor)
```

### Distinguishing projected savings vs. actual measured savings

| Attribute | Projected Savings (`simulate_scenario`, `compare_scenarios`) | Actual Measured Savings (`analyze_consumption`) |
|---|---|---|
| **Nature** | **Hypothetical / Counterfactual target**. Answers: *"What would be saved IF a target percentage reduction is achieved?"* | **Observed historical reduction**. Identifies avoidable consumption from actual measured meter anomalies. |
| **Field indicator** | `projection_type: "hypothetical"`, `measured_savings: null`. | `severity`, `is_anomaly: true`, `estimated_avoidable_consumption`. |
| **Baseline origin** | Historical window sum or Phase 6 model forecast over a future horizon. | Rolling median of strictly earlier measured readings within the same series. |
| **Certainty** | Target assumption (or bounded by forecast empirical uncertainty when forecast-based). | Empirical measurement; never speculative. |
| **Financial values** | Strictly `null`. The engine never fabricates currency or monetary figures. | Strictly `null`. Monies are never calculated or guessed. |

**Sample single scenario input** (`samples/simulation/sim_building_electricity_15pct.json`):

```json
{"scenario": {"resource_type": "electricity", "building_code": "BLOCK_A",
              "reduction_percentage": 15, "baseline_period_days": 30}}
```

**Sample output** (excerpt of the real engine output, default config):

```json
{
  "scenario_id": "scn-eb07ac88ee72",
  "projection_type": "hypothetical",
  "calculation_status": "calculated",
  "scope": {
    "level": "building",
    "building_code": "BLOCK_A",
    "buildings_included": [
      "BLOCK_A"
    ]
  },
  "resource_type": "electricity",
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
  "units": {
    "consumption": "kWh",
    "co2e": "kg CO2e",
    "period": "days"
  },
  "warnings": [
    "No electricity emission factor configured; CO2e reduction not calculated."
  ],
  "data_quality_status": "ok",
  "...": "assumptions, emission_factor, ..."
}
```

### Forecast-based baselines (Phase 7B)

Scenarios can set `"baseline_source": "forecast"` to use the Phase 6 forecasting engine as the baseline.
- The forecasted consumption total over `forecast_horizon_days` (default 7) becomes the baseline.
- If data is insufficient for forecasting (< 14 days, severe gaps), the simulation returns `calculation_status: "not_calculated"` with `not_calculated_reason: "insufficient_forecast_data"`.
- When backtesting produces empirical intervals, uncertainty bounds (`total_lower_bound`, `total_upper_bound`) are propagated mathematically into baseline, savings, and projected consumption intervals.
- The engine **never fabricates** confidence intervals or emission factors.

### Advanced scenario comparison & double-counting prevention (Phase 7B)

`compare_scenarios(records, scenarios, config=None)` compares multiple scenarios across buildings, resources, percentage targets, and campus-wide scopes.

**Deterministic double-counting prevention policy (`deterministic_maximum_savings`)**:
- When multiple scenarios conflict on the same building/resource, or a campus-wide scenario overlaps with building-specific scenarios, naive addition would double-count savings.
- The engine groups conflicting scenarios and selects the non-overlapping combination that achieves the **maximum total savings**.
- The result reports both the deduplicated `portfolio_savings` (with `included_scenario_ids` and `excluded_scenario_ids`) and the naive `unadjusted_gross_savings` for complete transparency.
- Full details of every conflict and policy decision are reported in `portfolio.overlap_groups`.
- Resource-wise totals are maintained in their native units (`kWh`, `liters`, `kg`), with combined CO2e when configured.
- Scenarios are ranked both by resource savings and by reduction percentage.

## Behavior without an LLM or API key

* Default: `llm.enabled = false`. No key is read, no SDK is imported, no network call is made;
  `llm_status = "disabled"`.
* LLM enabled but key missing, SDK missing, timeout, provider error or invalid reply: the same
  deterministic recommendation is returned and `llm_status` says why. See `docs/LLM_CONFIGURATION.md`.
* No emission factor ships with the engine. Without one, savings are still reported (kWh, liters, kg)
  and `estimated_co2e_reduction_kg` is `null`.

## Tests

```bash
python -m pytest                                              # everything (from this folder)
python -m pytest ai_engine/tests/test_scenario_comparison.py  # Phase 7B comparison suite only
python -m pytest ai_engine/tests/test_scenario_simulator.py ai_engine/tests/test_scenario_integration.py   # Simulation suite
python -m pytest ai_engine/tests/test_fastapi_example.py      # Integration routes
```

| File | Covers |
|---|---|
| `test_schemas.py`, `test_anomaly_detector.py`, `test_impact_calculator.py`, `test_recommendation_engine.py` | Phases 1-4 (unchanged) |
| `test_integration_contract.py` | JSON serialization (timestamps, enums, nulls, nested objects, round trip), `analyze_payload`, error envelopes, options, `engine_info` |
| `test_integration_pipeline.py` | Full pipeline on every sample dataset; no-API-key and offline behavior; LLM timeout/provider-error fallback |
| `test_forecasting_engine.py` | Phase 6 algorithms: data handling, baselines, advanced-model selection, backtesting, leakage, intervals, serialization |
| `test_forecast_integration.py` | Phase 6 contract: `forecast_payload`, forecasting samples, error envelopes, unchanged analyze API |
| `test_scenario_simulator.py` | Phase 7A: formulas, 0-100 % validation, units and periods, building/campus scope, explicit baselines, empty/unknown/zero/missing data, duplicates and irregular records, CO2e factors, determinism, `simulate_payload` envelopes |
| `test_scenario_comparison.py` | Phase 7B: multi-scenario comparisons, multi-building, multi-resource, deterministic double-counting deduplication policy, forecast-based baselines, uncertainty propagation, input validation, non-interference |
| `test_scenario_integration.py` | Phase 7A & 7B: stored sample responses equal real engine output, docs match, earlier signatures and exports unchanged, scope guard |
| `test_fastapi_example.py` | The example app incl. `/api/simulate`, `/api/simulate/compare`, and `/api/v1/ai/...` routes (requires `fastapi` and `httpx`) |
| `test_project_hygiene.py` | No secrets or real `.env`, required files present, contract version matches |

All external calls in tests are mocked; no test needs network access or a key.

## Layout

```
ai_engine/            the engine package (+ original README) and tests/
  contract.py         Phase 5: analyze_payload, serialization, engine_info, schemas
  forecasting_engine.py  Phase 6: forecast_consumption, forecast_payload, backtesting
  scenario_simulator.py  Phase 7A & 7B: simulate_scenario, compare_scenarios, simulate_payload, compare_payload
  _version.py         engine and contract versions
docs/                 JSON_CONTRACT.md, LLM_CONFIGURATION.md, PHASE7B_VERIFICATION.md
examples/fastapi_integration/   example app with POST /api/simulate and POST /api/simulate/compare
samples/              synthetic datasets + manifest + README (forecasting/ Phase 6, simulation/ Phase 7A & 7B)
scripts/              generate_samples.py, generate_forecast_samples.py, generate_simulation_samples.py, export_json_schema.py
requirements.txt  requirements-llm.txt  pytest.ini  .env.example  .gitignore
```

## Limitations

Unchanged from Phase 4 (see `ai_engine/README.md`): the baseline ignores weekly/seasonal patterns;
recommendations are generic suggested checks, not diagnoses; savings are estimates against a rolling
median; LLM text is validated heuristically and is advisory. Sample data is synthetic.

Forecasting limitations (Phase 6): forecasts extrapolate recent weekly patterns; they do not know about
holidays, term breaks, weather, occupancy or planned changes, and a regime change (new equipment, a closed
building) is only reflected after it shows up in the data. Only a weekly cycle is modelled; there is no yearly
seasonality. Prediction intervals are empirical (limited by the backtest sample) and not guaranteed; the total
bounds are sums of daily bounds, i.e. conservative. Tree models cannot extrapolate beyond the training range.
Data is assumed to be per-interval consumption, not cumulative meter values. Sample data is synthetic.

Simulator limitations (Phase 7A): a projection is not a forecast or a measurement and does not judge whether a
reduction is realistic. The baseline is a plain sum over the window (no weekly, seasonal, weather or occupancy
adjustment), so a window containing a holiday or closure lowers it. Coverage is judged per UTC calendar day. The 80 %
coverage threshold is an engineering default, not a value derived from campus data.

Comparison limitations (Phase 7B): rankings are ordered strictly by absolute resource savings quantity or
percentage target without modeling equipment capital cost, operational disruption, or execution difficulty.
The deduplication policy chooses the mathematically highest-saving option among overlapping targets; if campus
managers prefer a specific alternate combination, they should submit only those target scenarios. Forecast-based
scenarios inherit all forecasting limitations. Financial savings are not calculated.
