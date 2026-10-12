# FastAPI integration example

**Example only. This is not the EcoCamp backend.** It shows how the team's FastAPI
service calls the AI engine; copy the pattern, not the file.

```bash
# from the project root
pip install -r examples/fastapi_integration/requirements.txt
cd examples/fastapi_integration
uvicorn app:app --reload --port 8001
```

| Method and path | Purpose |
|---|---|
| `POST /api/v1/ai/analyze` | Body and response exactly as in `docs/JSON_CONTRACT.md` |
| `POST /api/simulate` (or `/api/v1/ai/simulate`) | What-if single scenario simulation (Phase 7A/7B); body and response in `docs/JSON_CONTRACT.md` section 7 |
| `POST /api/simulate/compare` (or `/api/v1/ai/simulate/compare`) | What-if multi-scenario comparison (Phase 7B); body and response in `docs/JSON_CONTRACT.md` section 7 |
| `GET /health` | Liveness plus engine facts (versions, LLM on/off, `api_key_configured`; never a key) |
| `GET /api/v1/ai/schema` | JSON Schemas of the record, result and impact-summary models |

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/ai/analyze \
     -H "Content-Type: application/json" \
     --data @../../samples/water_wastage.json
```

Include totals with `{"records": [...], "options": {"include_impact_summary": true}}`.

## Forecast route (Phase 6)

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/ai/forecast \
     -H "Content-Type: application/json" \
     --data @../../samples/forecasting/forecast_next_day.json
```

```python
from ai_engine import ForecastConfig, forecast_payload

FORECAST_CONFIG = ForecastConfig()                            # once, at start-up

@app.post("/api/v1/ai/forecast")
async def forecast(request: Request):
    payload = await request.json()                            # handle ValueError -> 422
    outcome = await run_in_threadpool(forecast_payload, payload, FORECAST_CONFIG)
    return JSONResponse(outcome.body, status_code=outcome.http_status)
```

Body: `{"records": [...], "options": {"horizon_days": 7}}`; response and errors are in
`docs/JSON_CONTRACT.md` section 6. Forecasting can train a model on long histories, so keep it off the event loop.

## Simulation routes (Phase 7A & 7B)

### Single scenario: `POST /api/simulate`

```bash
curl -s -X POST http://127.0.0.1:8001/api/simulate \
     -H "Content-Type: application/json" \
     --data @../../samples/simulation/sim_building_electricity_15pct.json
```

```python
from ai_engine import simulate_payload

@app.post("/api/simulate")
async def simulate(request: Request):
    payload = await request.json()                            # handle ValueError -> 422
    outcome = await run_in_threadpool(simulate_payload, payload, ENGINE_CONFIG)
    return JSONResponse(outcome.body, status_code=outcome.http_status)
```

Body: `{"records": [...], "scenario": {...}}` (`records` optional with an explicit baseline). A scenario
the data cannot support (unknown building, no records, insufficient forecast history) is HTTP 200 with
`result.calculation_status = "not_calculated"`; a malformed scenario is HTTP 422 `invalid_scenario`.

### Multi-scenario comparison: `POST /api/simulate/compare`

```bash
curl -s -X POST http://127.0.0.1:8001/api/simulate/compare \
     -H "Content-Type: application/json" \
     --data @../../samples/simulation/sim_compare_multiple_buildings.json
```

```python
from ai_engine import compare_payload

@app.post("/api/simulate/compare")
async def simulate_compare(request: Request):
    payload = await request.json()                            # handle ValueError -> 422
    outcome = await run_in_threadpool(compare_payload, payload, ENGINE_CONFIG)
    return JSONResponse(outcome.body, status_code=outcome.http_status)
```

Body: `{"records": [...], "scenarios": [...]}`. Returns individual scenario projections, resource-wise
savings totals, rankings, and deduplicated portfolio totals preventing double-counting overlapping scenarios.

## The pattern to copy

```python
from fastapi.concurrency import run_in_threadpool
from ai_engine import AnalysisConfig, LLMConfig, analyze_payload

ENGINE_CONFIG = AnalysisConfig(llm=LLMConfig.from_env())     # once, at start-up

@app.post("/api/v1/ai/analyze")
async def analyze(request: Request):
    payload = await request.json()                            # handle ValueError -> 422
    outcome = await run_in_threadpool(analyze_payload, payload, ENGINE_CONFIG)
    return JSONResponse(outcome.body, status_code=outcome.http_status)
```

* The engine is synchronous; keep it off the event loop (`run_in_threadpool`).
* Reading the raw body keeps every error in the contract's envelope. If you prefer typed
  request models, validate the shape yourself and still pass the dict to `analyze_payload`.
* Emission factors are server-side settings (the example reads optional `ECOCAMP_*`
  variables); clients cannot set them.
* Without an API key everything works with deterministic recommendations.

Tests for this example are in `ai_engine/tests/test_fastapi_example.py` and are skipped
automatically when `fastapi` or `httpx` is not installed.
