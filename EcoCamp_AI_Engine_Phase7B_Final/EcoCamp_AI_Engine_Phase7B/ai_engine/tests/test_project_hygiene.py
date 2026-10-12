"""Phase 5: delivery hygiene. No secrets, no caches, documented files present."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),        # Google API key shape
    re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{20,}"),  # OpenAI project-key shape
    re.compile(r"\bsk-[A-Za-z0-9]{32,}"),           # OpenAI legacy-key shape
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".ini", ".example", ".cfg", ".toml", ".yml", ".yaml"}


def _project_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts:
            yield path


def test_no_secrets_in_any_project_file():
    for path in _project_files():
        if path.suffix in TEXT_SUFFIXES or path.name.startswith(".env"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                assert not pattern.search(text), f"possible secret in {path.relative_to(ROOT)}"


@pytest.mark.parametrize("env_path", [ROOT / ".env.example", ROOT / "ai_engine" / ".env.example"])
def test_env_example_has_blank_keys_and_safe_defaults(env_path):
    values = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    assert values["GEMINI_API_KEY"] == "" and values["OPENAI_API_KEY"] == ""
    assert values["ECOCAMP_LLM_ENABLED"] == "false"


def test_no_real_env_file_or_caches_shipped():
    # (__pycache__ / .pytest_cache are created by running the tests themselves,
    # so they are excluded from the ZIP at packaging time, not asserted here.)
    assert not (ROOT / ".env").exists()
    assert not [p for p in ROOT.rglob("*") if p.name in {"venv", ".venv", "env"} and p.is_dir()]


def test_required_deliverables_exist():
    for rel in (
        "README.md", "requirements.txt", ".env.example", "docs/JSON_CONTRACT.md",
        "docs/LLM_CONFIGURATION.md", "examples/fastapi_integration/app.py",
        "scripts/generate_samples.py", "samples/manifest.json", "samples/README.md",
        "ai_engine/analyzer.py", "ai_engine/contract.py",
    ):
        assert (ROOT / rel).is_file(), rel


def test_sample_manifest_lists_existing_valid_files():
    manifest = json.loads((ROOT / "samples" / "manifest.json").read_text(encoding="utf-8"))
    names = {entry["file"] for entry in manifest["files"]}
    assert {
        "normal_consumption.json", "electricity_unusual_usage.json", "water_wastage.json",
        "waste_anomalies.json", "insufficient_history.json", "edge_cases.json",
        "mixed_campus.json", "invalid_records.json",
    } == names
    for entry in manifest["files"]:
        data = json.loads((ROOT / "samples" / entry["file"]).read_text(encoding="utf-8"))
        assert len(data["records"]) == entry["records"]


def test_documented_contract_version_matches_code():
    from ai_engine import CONTRACT_VERSION

    doc = (ROOT / "docs" / "JSON_CONTRACT.md").read_text(encoding="utf-8")
    assert f'"contract_version": "{CONTRACT_VERSION}"' in doc
