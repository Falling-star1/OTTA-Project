"""Run the No Adaptation baseline on the fixed ARC smoke stream."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from src.config import load_experiment_config
from src.data.stream import load_fixed_stream
from src.evaluation.choices import make_causal_lm_option_scorer
from src.models.load import load_causal_lm
from src.runner import run_prediction_stream


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", required=True, choices=["no_adaptation"])
    return parser.parse_args()


def run_no_adaptation(config: dict[str, Any]) -> Path:
    records = load_fixed_stream(Path(config["data_path"]), expected_count=32)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config["artifacts_root"]) / "no_adaptation" / str(config["seed"]) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model, tokenizer, device = load_causal_lm(config["model_name"], config["device"])
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "model_name": config["model_name"],
        "model_revision": getattr(model.config, "_commit_hash", None),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "selections.jsonl").write_text("", encoding="utf-8")

    scorer = make_causal_lm_option_scorer(
        model, tokenizer, device, int(config["max_input_length"])
    )
    predictions_path = run_dir / "predictions.jsonl"
    start_time = time.perf_counter()
    stream_result = run_prediction_stream(
        records,
        scorer,
        int(config["batch_size"]),
        lambda predictions: write_jsonl(predictions_path, predictions),
    )
    elapsed_seconds = time.perf_counter() - start_time
    write_jsonl(run_dir / "metrics_by_batch.jsonl", stream_result["batch_metrics"])
    summary = {
        "method": "no_adaptation",
        "seed": config["seed"],
        **stream_result["summary"],
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_gib": (
            round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
            if device.type == "cuda"
            else None
        ),
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "run.log").write_text("No Adaptation completed successfully.\n", encoding="utf-8")
    return run_dir


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.method != "no_adaptation":
        raise SystemExit("Only no_adaptation is implemented in this stage")
    run_dir = run_no_adaptation(config)
    print(f"RUN_DIR: {run_dir}")


if __name__ == "__main__":
    main()
