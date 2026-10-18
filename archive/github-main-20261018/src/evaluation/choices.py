"""Multiple-choice scoring for causal language models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


OptionScorer = Callable[[str, str], float]


def format_question(record: dict[str, Any]) -> str:
    options = "\n".join(
        f"{choice['label']}. {choice['text']}" for choice in record["choices"]
    )
    return f"Question: {record['question']}\nChoices:\n{options}\nAnswer:"


def predict_record(record: dict[str, Any], score_option: OptionScorer) -> dict[str, Any]:
    """Predict one choice without reading the record's answer field."""
    question = format_question(record)
    choice_scores = {
        choice["label"]: score_option(question, choice["text"])
        for choice in record["choices"]
    }
    prediction = max(choice_scores, key=choice_scores.get)
    return {
        "sample_id": record["sample_id"],
        "prediction": prediction,
        "choice_scores": choice_scores,
    }


def evaluate_predictions(
    predictions: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, float | int]:
    """Read labels only after a prediction record has been produced and persisted."""
    answers = {record["sample_id"]: record["answer"] for record in records}
    correct = sum(prediction["prediction"] == answers[prediction["sample_id"]] for prediction in predictions)
    total = len(predictions)
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def make_causal_lm_option_scorer(
    model: Any, tokenizer: Any, device: torch.device, max_input_length: int
) -> OptionScorer:
    """Return token-average log-likelihood scorer for each candidate option text."""
    model.eval()

    def score_option(question: str, choice_text: str) -> float:
        prompt_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(" " + choice_text, add_special_tokens=False)["input_ids"]
        if not target_ids:
            raise ValueError("choice text produced no tokens")
        prompt_budget = max_input_length - len(target_ids)
        if prompt_budget <= 0:
            raise ValueError("choice text exceeds max_input_length")
        prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = torch.tensor([prompt_ids + target_ids], device=device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits[:, :-1, :]
            targets = input_ids[:, 1:]
            token_log_probs = torch.log_softmax(logits, dim=-1).gather(
                dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)
        target_log_probs = token_log_probs[:, -len(target_ids) :]
        return float(target_log_probs.mean().item())

    return score_option
