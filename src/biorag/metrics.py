"""Metrics used by the local sanity-check demonstration."""

from __future__ import annotations

import re
from collections import Counter

from .text import normalize, tokenize


def reciprocal_rank(ranked_ids: list[str], gold_ids: set[str]) -> float:
    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / index
    return 0.0


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & gold_ids) / len(gold_ids)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction).strip(" .?!") == normalize(reference).strip(" .?!"))


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(tokenize(prediction, remove_stopwords=False))
    expected = Counter(tokenize(reference, remove_stopwords=False))
    overlap = sum((predicted & expected).values())
    if not overlap or not predicted or not expected:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    left = tokenize(prediction, remove_stopwords=False)
    right = tokenize(reference, remove_stopwords=False)
    matrix = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, left_token in enumerate(left, start=1):
        for j, right_token in enumerate(right, start=1):
            matrix[i][j] = matrix[i - 1][j - 1] + 1 if left_token == right_token else max(matrix[i - 1][j], matrix[i][j - 1])
    lcs = matrix[-1][-1]
    precision = lcs / max(1, len(left))
    recall = lcs / max(1, len(right))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def normalize_answer_for_type(answer: str, question_type: str) -> str:
    value = normalize(answer)
    value = re.sub(r"[^a-z0-9 ]", "", value)
    if question_type == "yesno":
        if value.startswith("yes"):
            return "yes"
        if value.startswith("no"):
            return "no"
    return value.strip()
