"""Small, dependency-free text processing utilities for the milestone baseline."""

from __future__ import annotations

import re
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "what", "which",
    "with", "does", "do", "how", "why", "can", "many", "often", "than", "into", "has", "have",
}


def normalize(text: str) -> str:
    """Normalize whitespace and a few common biomedical punctuation variants."""
    if not isinstance(text, str):
        return ""
    value = text.replace("–", "-").replace("—", "-").replace("µ", "u")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(normalize(text))
    if remove_stopwords:
        tokens = [token for token in tokens if token not in _STOPWORDS]
    return tokens


def sentences(text: str) -> list[str]:
    value = re.sub(r"\s+", " ", text or "").strip()
    if not value:
        return []
    parts = _SENTENCE_RE.split(value)
    return [part.strip() for part in parts if part.strip()]


def character_ngrams(tokens: Iterable[str], n: int = 3) -> list[str]:
    """Return boundary-aware character n-grams as a lightweight semantic proxy."""
    grams: list[str] = []
    for token in tokens:
        bounded = f"^{token}$"
        grams.extend(bounded[index:index + n] for index in range(max(0, len(bounded) - n + 1)))
    return grams
