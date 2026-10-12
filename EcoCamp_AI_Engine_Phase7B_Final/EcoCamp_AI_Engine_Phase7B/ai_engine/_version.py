"""Single source of truth for version strings (kept import-free to avoid cycles)."""

__version__ = "0.7.0"

# Version of the JSON request/response contract (docs/JSON_CONTRACT.md).
# Bump the major number for breaking changes, the minor number for additions.
CONTRACT_VERSION = "1.0"

# Version of the forecasting JSON contract (docs/JSON_CONTRACT.md, section 6).
# Independent of CONTRACT_VERSION so the Phase 5 analyze contract stays at 1.0.
FORECAST_CONTRACT_VERSION = "1.0"

# Version of the what-if simulation JSON contract (docs/JSON_CONTRACT.md, section 7).
# Independent of the analyze and forecast contract versions, which stay at 1.0.
SIMULATION_CONTRACT_VERSION = "1.0"
