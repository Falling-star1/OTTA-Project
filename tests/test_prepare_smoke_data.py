from src.data.prepare_smoke_data import (
    build_smoke_records,
    model_input_view,
    validate_smoke_records,
)


RAW_ROWS = [
    {
        "id": "arc-1",
        "question": "Which option is correct?",
        "choices": {"label": ["A", "B"], "text": ["first", "second"]},
        "answerKey": "B",
    },
    {
        "id": "arc-2",
        "question": "Which animal can fly?",
        "choices": {"label": ["A", "B"], "text": ["fish", "bird"]},
        "answerKey": "B",
    },
    {
        "id": "arc-3",
        "question": "Which number is even?",
        "choices": {"label": ["A", "B"], "text": ["three", "four"]},
        "answerKey": "B",
    },
]


def test_build_smoke_records_is_seeded_and_uses_standard_schema():
    records = build_smoke_records(RAW_ROWS, count=2, seed=42)

    assert [record["sample_id"] for record in records] == ["arc-1", "arc-3"]
    assert records[0] == {
        "sample_id": "arc-1",
        "question": "Which option is correct?",
        "choices": [
            {"label": "A", "text": "first"},
            {"label": "B", "text": "second"},
        ],
        "answer": "B",
    }


def test_validate_smoke_records_rejects_duplicate_ids_and_invalid_answers():
    records = build_smoke_records(RAW_ROWS, count=2, seed=42)
    records[1]["sample_id"] = records[0]["sample_id"]
    records[1]["answer"] = "Z"

    errors = validate_smoke_records(records, expected_count=2)

    assert "duplicate sample_id: arc-1" in errors
    assert "record 2: answer 'Z' is not one of its choice labels" in errors


def test_model_input_view_excludes_answer():
    record = build_smoke_records(RAW_ROWS, count=1, seed=42)[0]

    assert model_input_view(record) == {
        "sample_id": "arc-3",
        "question": "Which number is even?",
        "choices": [
            {"label": "A", "text": "three"},
            {"label": "B", "text": "four"},
        ],
    }
