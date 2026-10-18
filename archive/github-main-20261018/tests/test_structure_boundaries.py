import json

import pytest

from src.config import load_experiment_config
from src.data.stream import load_fixed_stream
from src.runner import run_prediction_stream


def test_load_experiment_config_rejects_missing_required_fields(tmp_path):
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text("model_name: example\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required config fields"):
        load_experiment_config(config_path)


def test_load_fixed_stream_rejects_wrong_record_count(tmp_path):
    data_path = tmp_path / "samples.jsonl"
    data_path.write_text(json.dumps({"sample_id": "one"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 32 records, found 1"):
        load_fixed_stream(data_path, expected_count=32)


def test_runner_persists_predictions_before_evaluating_each_batch():
    records = [
        {"sample_id": "one", "question": "q", "choices": [{"label": "A", "text": "a"}], "answer": "A"},
        {"sample_id": "two", "question": "q", "choices": [{"label": "A", "text": "a"}], "answer": "A"},
    ]
    saved_prediction_counts = []

    result = run_prediction_stream(
        records,
        score_option=lambda _question, _choice: 0.0,
        batch_size=1,
        persist_predictions=lambda predictions: saved_prediction_counts.append(len(predictions)),
    )

    assert saved_prediction_counts == [1, 2]
    assert result["summary"] == {"correct": 2, "total": 2, "accuracy": 1.0}
    assert len(result["batch_metrics"]) == 2
