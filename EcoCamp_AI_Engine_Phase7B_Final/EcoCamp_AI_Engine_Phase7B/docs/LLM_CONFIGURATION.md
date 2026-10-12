# LLM configuration (Gemini / OpenAI), timeouts and fallback

The LLM layer is **optional and off by default**. Without it the engine produces complete,
deterministic recommendations (rule-based evidence, suggested checks, priority, savings). With it
enabled, the LLM only *adds* plain-language text in the `llm_*` fields of a recommendation; it can
never change a number, a severity, a saving, a priority or the deterministic actions.

## 1. Running with no API key (default)

Nothing to configure. With `LLMConfig()` (the default) the engine does not read any API key, does not
import any LLM SDK and makes no network calls. Every flagged result carries
`recommendation.source = "rules"` and `recommendation.llm_status = "disabled"`.
`ai_engine/tests/test_integration_pipeline.py` runs the whole pipeline on the sample datasets with
API keys removed from the environment and network connections blocked.

## 2. Enabling the LLM

1. **Install one SDK** (not needed otherwise): `pip install -r requirements-llm.txt`
   (`openai` and/or `google-genai`).
2. **Set environment variables** (see `.env.example`; keep real keys out of code and git):

| Variable | Meaning | Default |
|---|---|---|
| `ECOCAMP_LLM_ENABLED` | `true` to enable | `false` |
| `ECOCAMP_LLM_PROVIDER` | `gemini` or `openai` | `gemini` |
| `ECOCAMP_LLM_MODEL` | Model name from your provider's current documentation. **Required when enabled.** No default is shipped because model names change | none |
| `GEMINI_API_KEY` | Gemini key (provider `gemini`) | none |
| `OPENAI_API_KEY` | OpenAI key (provider `openai`) | none |
| `ECOCAMP_LLM_API_KEY_ENV` | Optional: *name* of another variable that holds the key. Putting a key here is rejected | none |
| `ECOCAMP_LLM_TIMEOUT_SECONDS` | Per-request timeout, `0 < value <= 300` | `20` |
| `ECOCAMP_LLM_MAX_RECOMMENDATIONS` | Max LLM calls per analysis call, `1..500` | `10` |

3. **Pass the config explicitly.** The engine does not read `ECOCAMP_LLM_*` by itself:

```python
from ai_engine import AnalysisConfig, LLMConfig

CONFIG = AnalysisConfig(llm=LLMConfig.from_env())     # reads ECOCAMP_LLM_* (not the key)
# or, explicitly:
CONFIG = AnalysisConfig(llm=LLMConfig(
    enabled=True, provider="gemini", model="<model name from provider docs>",
    timeout_seconds=15, max_llm_recommendations=5,
))
```

`LLMConfig.from_env()` raises `ValueError` for bad values (for example enabled without a model), so
build the config **at application start-up** and fail fast. The example app does this.
Load a `.env` file with your own tooling (the example app uses `python-dotenv` if installed).

4. **Check the setup without a real call:** `GET /health` in the example app (or
`ai_engine.engine_info(CONFIG)`) reports `llm.enabled`, provider, model, timeout and
`api_key_configured` (true/false only; the key itself is never returned).

### Provider notes

| | Gemini | OpenAI |
|---|---|---|
| SDK package | `google-genai` (`from google import genai`) | `openai` |
| Key variable | `GEMINI_API_KEY` | `OPENAI_API_KEY` |
| How the engine calls it | `models.generate_content` with `response_mime_type="application/json"` | Chat Completions with `response_format={"type": "json_object"}` |
| Timeout | passed as `HttpOptions(timeout=<milliseconds>)` | passed as `timeout=<seconds>`; `max_retries=0` |

The key is passed explicitly to the SDK client from the environment at call time. Billing, quotas and
which model names are available to your key are governed by your provider account.
**No live provider call was made while preparing this package**; the provider adapters are covered by
tests that use fake clients and fake SDK modules. Do one real smoke call with your own key before
a demo and check `llm_status`.

## 3. Timeouts

* `timeout_seconds` is applied per request by the provider SDK. The engine makes **no retries**.
* A request is made for each flagged recommendation, highest priority first, up to
  `max_llm_recommendations`; the rest get `llm_status = "call_limit_reached"`.
* After a missing key, missing SDK, **timeout** or provider error, the remaining LLM calls of that
  analysis are skipped. An outage therefore costs at most one timeout, not one per record.
* Worst case for a healthy but slow provider: about `max_llm_recommendations x timeout_seconds`
  added to the request. Budget this against your gateway/client timeout: for interactive endpoints
  use a low cap (for example 3 to 5) and a short timeout (for example 10 s), or keep the LLM off for
  large batches. Run the engine in a worker thread (`run_in_threadpool`) as in the example.

## 4. Fallback behavior

The application never fails because of the LLM. In every case the deterministic recommendation is
returned unchanged and `recommendation.llm_status` says why no LLM text is present:

| `llm_status` | Cause |
|---|---|
| `disabled` | LLM not enabled (default); nothing attempted |
| `used` | Validated LLM text attached (`source = "rules_llm"`) |
| `missing_api_key` | The key variable is unset or empty |
| `sdk_unavailable` | The official SDK is not installed |
| `timeout` | Request exceeded `timeout_seconds` |
| `provider_error` | Any other provider, SDK, auth, quota or network failure |
| `invalid_response` | Reply failed JSON/schema/content validation |
| `call_limit_reached` | Beyond `max_llm_recommendations` for this analysis |

`summary.llm_status_counts` in the response gives the totals for dashboards and alerting.
Only the exception **class name** of a provider failure is logged or kept; provider error messages are
dropped because they can echo credentials.

## 5. What is sent and how replies are checked

Per flagged record only: building name, resource, unit, observed value, baseline, deviation, severity,
calculated savings/CO2e (or null), the reduction percentage and the deterministic action list. No
timestamps, no history, no other buildings. The building name leaves your system, so choose provider
terms accordingly.

A reply is accepted only if it is JSON with exactly the expected keys, every number in it already
appears in the facts sent, it does not claim a cause was confirmed or a leak/fault found, and it does
not mention carbon unless a CO2e estimate exists. These are heuristics, not proof: treat `llm_*` text
as advisory, and show it separately from the deterministic fields.

## 6. Secrets

Never commit `.env`, keys or key-bearing URLs. `.env.example` has blank keys; `.gitignore` excludes
`.env`. A test (`test_project_hygiene.py`) scans the project for key-shaped strings.
