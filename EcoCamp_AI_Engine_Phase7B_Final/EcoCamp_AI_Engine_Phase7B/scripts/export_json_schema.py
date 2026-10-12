#!/usr/bin/env python3
"""Write machine-readable JSON Schemas for the backend developer.

    python scripts/export_json_schema.py            # writes docs/schemas/*.json

The schemas are generated from the Pydantic models, so they always match the
code. Requires the engine dependencies (pydantic).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine import json_schemas  # noqa: E402


def main() -> None:
    out = ROOT / "docs" / "schemas"
    out.mkdir(parents=True, exist_ok=True)
    for name, schema in json_schemas().items():
        path = out / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
