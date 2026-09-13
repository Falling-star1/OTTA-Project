from src.evaluation.choices import evaluate_predictions, predict_record


RECORD = {
    "sample_id": "arc-1",
    "question": "Which animal can fly?",
    "choices": [
        {"label": "A", "text": "fish"},
        {"label": "B", "text": "bird"},
    ],
    "answer": "B",
}


def test_predict_record_selects_highest_scored_option_without_exposing_answer():
    scores = {"fish": -2.0, "bird": -0.1}

    prediction = predict_record(RECORD, lambda _question, choice_text: scores[choice_text])

    assert prediction == {
        "sample_id": "arc-1",
        "prediction": "B",
        "choice_scores": {"A": -2.0, "B": -0.1},
    }
    assert "answer" not in prediction


def test_evaluate_predictions_reads_answers_only_after_predictions_are_saved():
    predictions = [
        {"sample_id": "arc-1", "prediction": "B"},
        {"sample_id": "arc-2", "prediction": "A"},
    ]
    records = [RECORD, {**RECORD, "sample_id": "arc-2", "answer": "B"}]

    assert evaluate_predictions(predictions, records) == {
        "correct": 1,
        "total": 2,
        "accuracy": 0.5,
    }
