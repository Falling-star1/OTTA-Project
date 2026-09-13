"""Online experiment runner with predict-before-adapt semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.evaluation.choices import OptionScorer, evaluate_predictions, predict_record


def run_prediction_stream(
    records: list[dict[str, Any]],
    score_option: OptionScorer,
    batch_size: int,
    persist_predictions: Callable[[list[dict[str, Any]]], None],
) -> dict[str, Any]:
    """Predict and persist each batch before allowing evaluation to access labels."""
    predictions: list[dict[str, Any]] = []
    batch_metrics: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(records), batch_size)):
        batch_records = records[start : start + batch_size]
        batch_predictions = [predict_record(record, score_option) for record in batch_records]
        predictions.extend(batch_predictions)
        persist_predictions(predictions)
        batch_metrics.append(
            {
                "batch_index": batch_index,
                "sample_ids": [record["sample_id"] for record in batch_records],
                **evaluate_predictions(batch_predictions, batch_records),
            }
        )
    return {
        "predictions": predictions,
        "batch_metrics": batch_metrics,
        "summary": evaluate_predictions(predictions, records),
    }
