"""Create and validate the fixed ARC-Challenge smoke-test stream."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from src.data.stream import load_fixed_stream


DEFAULT_DATASET = "allenai/ai2_arc"
DEFAULT_CONFIG = "ARC-Challenge"
DEFAULT_SPLIT = "validation"
DEFAULT_COUNT = 32
DEFAULT_SEED = 42


def build_smoke_records(
    rows: Iterable[dict[str, Any]], count: int = DEFAULT_COUNT, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Deterministically select ARC rows and convert them to the project schema."""
    source_rows = list(rows)
    if count <= 0:
        raise ValueError("count must be positive")
    if len(source_rows) < count:
        raise ValueError(f"requested {count} records, but only {len(source_rows)} are available")

    selected_indices = sorted(random.Random(seed).sample(range(len(source_rows)), count))
    records: list[dict[str, Any]] = []
    for index in selected_indices:
        row = source_rows[index]
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        records.append(
            {
                "sample_id": row["id"],
                "question": row["question"],
                "choices": [
                    {"label": label, "text": text}
                    for label, text in zip(labels, texts, strict=True)
                ],
                "answer": row["answerKey"],
            }
        )
    return records


def validate_smoke_records(
    records: Iterable[dict[str, Any]], expected_count: int = DEFAULT_COUNT
) -> list[str]:
    """Return all schema and leakage-boundary errors; an empty list means valid."""
    records = list(records)
    errors: list[str] = []
    if len(records) != expected_count:
        errors.append(f"expected {expected_count} records, found {len(records)}")

    seen_ids: set[str] = set()
    for position, record in enumerate(records, start=1):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"record {position}: missing non-empty sample_id")
        elif sample_id in seen_ids:
            errors.append(f"duplicate sample_id: {sample_id}")
        else:
            seen_ids.add(sample_id)

        if not isinstance(record.get("question"), str) or not record["question"].strip():
            errors.append(f"record {position}: missing non-empty question")

        choices = record.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            errors.append(f"record {position}: choices must contain at least two options")
            continue
        labels = []
        for choice in choices:
            if not isinstance(choice, dict) or not isinstance(choice.get("label"), str) or not choice["label"]:
                errors.append(f"record {position}: every choice needs a non-empty label")
                continue
            if not isinstance(choice.get("text"), str) or not choice["text"].strip():
                errors.append(f"record {position}: every choice needs non-empty text")
            labels.append(choice["label"])

        answer = record.get("answer")
        if answer not in labels:
            errors.append(f"record {position}: answer '{answer}' is not one of its choice labels")
    return errors


def model_input_view(record: dict[str, Any]) -> dict[str, Any]:
    """Expose only fields permitted to prediction, selection, and adaptation code."""
    return {key: record[key] for key in ("sample_id", "question", "choices")}


def write_jsonl(records: Iterable[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/smoke_32.jsonl"))
    parser.add_argument("--ids-output", type=Path, default=Path("data/smoke_32_ids.txt"))
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--check", action="store_true", help="validate --output without downloading data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        records = load_fixed_stream(args.output)
    else:
        from datasets import load_dataset

        dataset = load_dataset(DEFAULT_DATASET, DEFAULT_CONFIG, split=DEFAULT_SPLIT)
        records = build_smoke_records(dataset, count=args.count, seed=args.seed)

    errors = validate_smoke_records(records, expected_count=args.count)
    if errors:
        raise SystemExit("Smoke data validation failed:\n- " + "\n- ".join(errors))

    if args.check:
        print(f"VALID: {args.output} ({len(records)} records)")
        return

    write_jsonl(records, args.output)
    args.ids_output.parent.mkdir(parents=True, exist_ok=True)
    args.ids_output.write_text("\n".join(record["sample_id"] for record in records) + "\n", encoding="utf-8")
    print(f"WROTE: {args.output} ({len(records)} records, seed={args.seed})")
    print(f"WROTE: {args.ids_output}")


if __name__ == "__main__":
    main()
