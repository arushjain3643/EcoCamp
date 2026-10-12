# Phase 7A verification notes

Scope: the what-if simulator only (`ai_engine/scenario_simulator.py`). Multi-scenario comparison (Phase 7B) is **not** implemented.

## What was run, and where

The Phase 7A build environment had Python 3.12, `numpy` and `scikit-learn`, but **no `pydantic`, no `pytest`, no
`fastapi`/`httpx`, and no network access** to install them. Consequently **the full test suite (Phases 1-6 and the new
files) was NOT run against the real dependencies in that environment.** What was done instead:

| Check | Result |
|---|---|
| `python -m py_compile` on every `.py` file in the project | passed |
| `ai_engine/tests/test_scenario_simulator.py` (166 test cases after parametrization) | **166 passed, 0 failed**, run with a throwaway stand-in, see below |
| `ai_engine/tests/test_scenario_integration.py` | 10 of 13 passed; the other 3 import `analyze_consumption` / `forecast_consumption` or `ai_engine.__all__`, which need the real package, and were **not run** |
| Mutation check (changed the 80 % coverage constant and the savings formula) | 34 of the 166 cases failed, so the tests do detect those faults; the file was restored afterwards |
| Existing Phase 1-6 test files, `test_fastapi_example.py` (incl. the 4 new simulate-route tests), `test_project_hygiene.py` | **not run** |

The stand-in ("harness") was sandbox-only and is **not part of the ZIP**. It loaded the real `scenario_simulator.py`
and the real `ResourceType`, `STANDARD_UNITS`, `UNIT_ALIASES`, `DataQualityStatus`, `RecordValidationError`, `_utc`,
`_co2e_factor`, `AnalysisOutcome`, `error_outcome` and `validation_error_body` source (extracted from the shipped files), but replaced
`ConsumptionRecord` and `AnalysisConfig` with small look-alikes and ran the tests with a minimal pytest substitute.
So the simulator's own logic was executed, but pydantic's real validation was not.

The files in `samples/simulation/*.response.json` were produced by `scripts/generate_simulation_samples.py --outputs`
under that same stand-in. They should be identical with the real engine; `test_stored_responses_match_what_the_engine_returns_today`
checks exactly that.

## Please run this on your machine

```bash
pip install -r requirements.txt
python -m pytest                       # everything: Phases 1-6 + 7A
python scripts/generate_simulation_samples.py --outputs   # optional: regenerate samples; should produce no diff
```

If `test_stored_responses_match_what_the_engine_returns_today` or `test_co2e_example_response_matches_the_engine_with_the_illustrative_factor`
fails, regenerate the samples with the command above and compare the diff: it would mean the stand-in differed from real
pydantic in some detail. Any other failure is a real defect and worth reporting.
