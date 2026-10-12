# Sample datasets

Synthetic, deterministic request bodies for developing and testing the
integration. Magnitudes are plausible for a mid-sized campus but are **not real
measurements**. Regenerate byte-identical files with
`python scripts/generate_samples.py`.

Every file is a ready-to-POST body: `{"records": [ ... ]}` (the contract in
`docs/JSON_CONTRACT.md`). `manifest.json` lists the files and record counts.

| File | Scenario | What to expect with the **default** `AnalysisConfig()` |
|---|---|---|
| `normal_consumption.json` | 4 buildings x electricity/water/waste, 30 daily readings, mild noise | 0 anomalies. The first 7 readings of each series have `insufficient_data` (no baseline yet) |
| `electricity_unusual_usage.json` | Library 3 days at ~2.3-2.6x; Admin Block one ~1.9x day; Hostel A control | 4 flagged readings, each with savings (kWh) and an electricity recommendation |
| `water_wastage.json` | Hostel B ~1.9x for 6 days (`+05:30` timestamps); two controls | 6 flagged readings, savings in liters; caveat that a reading cannot confirm a leak |
| `waste_anomalies.json` | Canteen ~2.5x for 3 days; Hostel A one ~2.1x day; Admin Block control | 4 flagged readings, savings in kg |
| `insufficient_history.json` | Series of only 5, 6 and 3 readings, incl. a 3x spike | 0 flagged; all `insufficient_data`, `recommendation: null` (the spike is deliberately not judged) |
| `edge_cases.json` | Zero reading (meter outage), drop to 30% of normal, `"Electricity"`/`"kwh"` spellings | 2 flagged: a `suspect_reading` and a `below_baseline` recommendation (neither has savings); spellings normalized to `electricity` / `kWh` |
| `mixed_campus.json` | All six scenarios shuffled into one request (782 records) | 16 flagged; each result equals its separate-run result; output order equals input order |
| `invalid_records.json` | 6 records, 5 invalid | HTTP 422 `invalid_records`, details for indices 1-5 |

The expectations above are asserted by `ai_engine/tests/test_integration_pipeline.py`.
Emission factors are not configured by default, so `estimated_co2e_reduction_kg`
is `null` unless you pass a configuration with a cited factor.

Try one against the example app:

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/ai/analyze \
     -H "Content-Type: application/json" \
     --data @samples/electricity_unusual_usage.json
```

## Forecasting samples (Phase 6): `samples/forecasting/`

Synthetic request bodies for `forecast_payload` / `POST /api/v1/ai/forecast`:
`{"records": [...], "options": {"horizon_days": N}}`. Regenerate byte-identical files with
`python scripts/generate_forecast_samples.py`. They have their own `manifest.json` and do not change the analysis
samples above.

| File | Scenario | Expectation with the default `ForecastConfig()` |
|---|---|---|
| `forecast_normal_7day.json` | 4 series (4 buildings, 3 resources), 56 daily readings, shuffled | 4 forecasts of 7 days, baseline model (under 90 days) |
| `forecast_next_day.json` | Library electricity, 35 days, `horizon_days = 1` | 1 forecast with a single predicted day |
| `forecast_missing_irregular.json` | Canteen water, 3 missing days, `+05:30` timestamps, 1 exact duplicate | Forecast produced, `suspect`, missing days not imputed |
| `forecast_insufficient_history.json` | Series with 5 and 3 days | `forecast_available = false`, `insufficient_data`, no numbers |
| `forecast_long_history.json` | Library electricity, 150 days, 8% noise | Advanced model backtested against the baseline; the backtest decides which is used |

These expectations are asserted by `ai_engine/tests/test_forecast_integration.py`. Sample **responses** are not
shipped as static files, because they must come from the real engine: run
`python scripts/generate_forecast_samples.py --outputs` (needs `pip install -r requirements.txt`) to write
`*.response.json` next to each request.

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/ai/forecast \
     -H "Content-Type: application/json" \
     --data @samples/forecasting/forecast_normal_7day.json
```

## Simulation samples (Phase 7A & 7B): `samples/simulation/`

Synthetic, deterministic request bodies for `simulate_payload` (`POST /api/simulate` or `POST /api/v1/ai/simulate`)
and `compare_payload` (`POST /api/simulate/compare` or `POST /api/v1/ai/simulate/compare`), each with the exact engine
response stored next to it as `*.response.json`. Regenerate everything with `python scripts/generate_simulation_samples.py --outputs`;
`manifest.json` lists the files, expected HTTP status and calculation metadata.

| File | Type | Shows |
|---|---|---|
| `sim_building_electricity_15pct` | Single | The spec example: 15 % cut for BLOCK_A, 30-day baseline (+ a `.co2e_example` response with an illustrative factor) |
| `sim_campus_water_10pct` | Single | Campus-wide water, other resources ignored, per-building breakdown |
| `sim_waste_explicit_baseline` | Single | Explicit baseline (value, unit, period), no records |
| `sim_partial_coverage` | Single | 25 of 30 days present: `suspect`, no gap filling |
| `sim_unknown_building` | Single | HTTP 200, `not_calculated` |
| `sim_zero_baseline` | Single | Zero savings and a warning |
| `sim_invalid_percentage` | Single | 120 %: HTTP 422 `invalid_scenario` |
| `sim_forecast_baseline_electricity` | Single (Phase 7B) | Forecast-based scenario: 7-day projected baseline via Phase 6 forecasting engine with uncertainty propagation |
| `sim_compare_multiple_buildings` | Compare (Phase 7B) | Comparison across 3 buildings (BLOCK_A, BLOCK_B, BLOCK_C) with rankings and resource totals |
| `sim_compare_overlapping_scenarios` | Compare (Phase 7B) | Competing campus-wide and building scenarios with deterministic double-counting prevention |

Try simulation and comparison against the example app:

```bash
# Single scenario simulation
curl -s -X POST http://127.0.0.1:8001/api/simulate \
     -H "Content-Type: application/json" \
     --data @samples/simulation/sim_building_electricity_15pct.json

# Multi-scenario comparison
curl -s -X POST http://127.0.0.1:8001/api/simulate/compare \
     -H "Content-Type: application/json" \
     --data @samples/simulation/sim_compare_multiple_buildings.json
```
