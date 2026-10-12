"""EXAMPLE ONLY: how the team's FastAPI backend can call the EcoCamp AI engine.

This is not the EcoCamp backend. It shows the minimum glue a backend needs:

1. Build one ``AnalysisConfig`` at start-up (emission factors from your own
   settings, LLM settings from environment variables).
2. Call ``analyze_payload`` from a worker thread (the engine is synchronous and
   CPU-bound; with the LLM enabled it also performs blocking network calls).
3. Return ``outcome.body`` with ``outcome.http_status``.

Phase 6 adds a second, independent route, ``POST /api/v1/ai/forecast``, that calls
``forecast_payload`` the same way (same envelope, same error codes; see
``docs/JSON_CONTRACT.md`` section 6). Forecasting can train a gradient-boosting model on
long histories, so it is also run in a worker thread.

Phase 7A adds a third independent route, ``POST /api/v1/ai/simulate``, that calls
``simulate_payload`` (what-if percentage-reduction scenarios; ``docs/JSON_CONTRACT.md`` section 7).
It is cheap and deterministic, but is still run in a worker thread for consistency. It uses the
same ``AnalysisConfig`` (emission factors) as the analyze route.

Run it (from this folder)::

    pip install -r requirements.txt
    uvicorn app:app --reload --port 8001

Then ``POST http://127.0.0.1:8001/api/v1/ai/analyze`` with a body such as
``samples/mixed_campus.json`` (see README.md in this folder). The engine runs
fully offline with no API key: recommendations are deterministic rules unless
you enable the optional LLM layer (docs/LLM_CONFIGURATION.md).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

# Make ``ai_engine`` importable when running from this repository. In the real
# backend, install or copy the ``ai_engine`` package instead and delete this.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:  # optional convenience: load a local .env (never commit it)
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - python-dotenv is optional
    pass

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from ai_engine import (
    AnalysisConfig,
    ForecastConfig,
    LLMConfig,
    analyze_payload,
    compare_payload,
    engine_info,
    forecast_payload,
    json_schemas,
    simulate_payload,
)
from ai_engine.contract import DEFAULT_MAX_RECORDS, error_outcome

logger = logging.getLogger("ecocamp.ai")


def build_engine_config(environ: Optional[Mapping[str, str]] = None) -> AnalysisConfig:
    """Build the engine configuration once, at start-up.

    * LLM settings come from ``ECOCAMP_LLM_*`` variables (disabled by default).
      API keys are read later, by the engine, from ``GEMINI_API_KEY`` /
      ``OPENAI_API_KEY``; they are never put into this config.
    * Emission factors have NO built-in default. This example reads optional
      ``ECOCAMP_*`` variables so a factor is supplied together with its source;
      use your own settings system in the real backend. Without a factor the
      engine still reports savings in kWh / liters / kg, with CO2e ``null``.

    Raises ``ValueError`` (including pydantic's ``ValidationError``) for bad
    values, so a misconfiguration fails fast at start-up, not on a request.
    """
    env = os.environ if environ is None else environ
    kwargs: dict = {"llm": LLMConfig.from_env(env)}
    fraction = env.get("ECOCAMP_REDUCTION_FRACTION", "").strip()
    if fraction:
        kwargs["achievable_reduction_fraction"] = float(fraction)
    factor = env.get("ECOCAMP_ELECTRICITY_EF_KG_CO2E_PER_KWH", "").strip()
    if factor:
        kwargs["electricity_emission_factor_kg_co2e_per_kwh"] = float(factor)
        source = env.get("ECOCAMP_ELECTRICITY_EF_SOURCE", "").strip()
        if source:
            kwargs["electricity_emission_factor_source"] = source
    return AnalysisConfig(**kwargs)


def create_app(
    config: Optional[AnalysisConfig] = None,
    max_records: int = DEFAULT_MAX_RECORDS,
    forecast_config: Optional[ForecastConfig] = None,
) -> FastAPI:
    """Create the example app. Pass ``config`` / ``forecast_config`` explicitly in tests."""
    engine_config = config if config is not None else build_engine_config()
    forecast_cfg = forecast_config if forecast_config is not None else ForecastConfig()
    app = FastAPI(title="EcoCamp AI engine - integration example", version="1.0")

    @app.get("/health")
    async def health() -> dict:
        """Liveness plus engine facts. Never returns API keys."""
        return {"status": "ok", **engine_info(engine_config)}

    @app.post(
        "/api/v1/ai/analyze",
        summary="Analyze consumption records",
        description="Body and response are specified in docs/JSON_CONTRACT.md.",
    )
    async def analyze(request: Request) -> JSONResponse:
        # Read the raw body so that every error uses the contract's envelope
        # (FastAPI's own 422 shape would differ for malformed requests).
        try:
            payload: Any = await request.json()
        except ValueError:
            outcome = error_outcome(422, "invalid_json", "Request body is not valid JSON.")
        else:
            # The engine is synchronous: keep it off the event loop.
            outcome = await run_in_threadpool(
                analyze_payload, payload, engine_config, max_records=max_records
            )
        return JSONResponse(content=outcome.body, status_code=outcome.http_status)

    @app.post(
        "/api/v1/ai/forecast",
        summary="Forecast the next day(s) of consumption",
        description="Body and response are specified in docs/JSON_CONTRACT.md, section 6.",
    )
    async def forecast(request: Request) -> JSONResponse:
        try:
            payload: Any = await request.json()
        except ValueError:
            outcome = error_outcome(422, "invalid_json", "Request body is not valid JSON.")
        else:
            outcome = await run_in_threadpool(
                forecast_payload, payload, forecast_cfg, max_records=max_records
            )
        return JSONResponse(content=outcome.body, status_code=outcome.http_status)

    @app.post(
        "/api/simulate",
        summary="What-if simulation: projected savings of a percentage reduction",
        description="Body and response are specified in docs/JSON_CONTRACT.md, section 7. "
        "Results are hypothetical projections, not measured savings.",
    )
    @app.post(
        "/api/v1/ai/simulate",
        summary="What-if simulation (v1 alias)",
        include_in_schema=False,
    )
    async def simulate(request: Request) -> JSONResponse:
        try:
            payload: Any = await request.json()
        except ValueError:
            outcome = error_outcome(422, "invalid_json", "Request body is not valid JSON.")
        else:
            outcome = await run_in_threadpool(
                simulate_payload, payload, engine_config, max_records=max_records
            )
        return JSONResponse(content=outcome.body, status_code=outcome.http_status)

    @app.post(
        "/api/simulate/compare",
        summary="What-if scenario comparison: compare multiple reduction scenarios",
        description="Body and response are specified in docs/JSON_CONTRACT.md, section 7. "
        "Prevents double-counting overlapping scenarios using a deterministic policy.",
    )
    @app.post(
        "/api/v1/ai/simulate/compare",
        summary="What-if scenario comparison (v1 alias)",
        include_in_schema=False,
    )
    async def simulate_compare(request: Request) -> JSONResponse:
        try:
            payload: Any = await request.json()
        except ValueError:
            outcome = error_outcome(422, "invalid_json", "Request body is not valid JSON.")
        else:
            outcome = await run_in_threadpool(
                compare_payload, payload, engine_config, max_records=max_records
            )
        return JSONResponse(content=outcome.body, status_code=outcome.http_status)

    @app.get("/api/v1/ai/schema", summary="JSON Schemas of the record and result models")
    async def schema() -> dict:
        return json_schemas()

    return app


app = create_app()
