"""Shared helpers for the Phase 5 integration tests (no autouse fixtures)."""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "samples"


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def load_sample():
    """Return a function ``name -> list of record dicts`` for ``samples/<name>.json``."""

    def _load(name: str):
        path = SAMPLES_DIR / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))["records"]

    return _load
