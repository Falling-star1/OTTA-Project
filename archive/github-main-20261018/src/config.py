"""Experiment configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_CONFIG_FIELDS = {
    "model_name",
    "data_path",
    "artifacts_root",
    "seed",
    "batch_size",
    "max_input_length",
    "device",
}


def load_experiment_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    missing = sorted(REQUIRED_CONFIG_FIELDS - config.keys())
    if missing:
        raise ValueError(f"missing required config fields: {', '.join(missing)}")
    if int(config["batch_size"]) <= 0 or int(config["max_input_length"]) <= 0:
        raise ValueError("batch_size and max_input_length must be positive")
    return config
