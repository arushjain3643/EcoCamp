# Phase 6 verification status

Written when Phase 6 was resumed and completed. Read this before trusting the package.

## NOT verified by a test run
The environment used to finish Phase 6 had no network, and `pytest`, `pydantic` and `fastapi` were not
installed, so **no pytest test (old or new) could be executed there**. Nothing in this package was
confirmed by a passing test run. Run these on a machine with `pip install -r requirements.txt` (and
`examples/fastapi_integration/requirements.txt` for the FastAPI tests):

```bash
python -m pytest ai_engine/tests/test_forecasting_engine.py ai_engine/tests/test_forecast_integration.py -q --durations=10
python -m pytest -q
```

## What WAS checked (outside pytest)
* Every Python file compiles.
* Performance change in `_predict_hgb`: the new training-matrix builder and the original per-row feature
  code were compared on 6 random series with NaNs (lengths 69-171, horizons 3-14): matrices and final
  predictions were **bit-identical**. Per-fit time dropped about 10% (the model fit itself dominates, about
  0.1-0.3 s per fit); a repeated identical fit is served from the cache in well under 1 ms.
* Profiling the 12-origin backtest showed the cost is the gradient-boosting fits themselves, not the
  feature construction; no threading effect (single core).
* Sample request files regenerate byte-identically and match `samples/forecasting/manifest.json`.

## Changes made in Phase 6 completion
* `forecasting_engine.py`: `_hgb_training_matrix` (per-origin rolling means), memoised `_predict_hgb`,
  `clear_forecast_cache()`. No change to models, selection rule, thresholds or output fields.
* `test_forecasting_engine.py`: the determinism test now clears the cache between runs so it still compares
  two genuine fits. No test was weakened or removed.
* New: `test_forecast_integration.py`, forecast route and tests in the FastAPI example, `samples/forecasting/`,
  `scripts/generate_forecast_samples.py`, JSON contract section 6, README sections.

## Known gaps
* Sample **responses** are not shipped (they must come from the real engine); generate them with
  `python scripts/generate_forecast_samples.py --outputs`. The example in contract section 6.8 is hand-written.
* Whether the real tests run within a given time limit is unmeasured. Tests that train real models
  (100-200 day series) cost roughly 13 fits each, about 2-4 s per series on one core.
