"""Read the fixed online test stream shared by every experiment method."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_fixed_stream(path: Path, expected_count: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"expected {expected_count} records, found {len(records)}")
    return records
