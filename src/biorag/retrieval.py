"""Transparent BM25 + subword-similarity retrieval for the first milestone."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .data import Document
from .text import character_ngrams, tokenize


@dataclass
class RetrievalResult:
    doc_id: str
    title: str
    text: str
    score: float
    lexical_score: float
    semantic_score: float


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class HybridRetriever:
    """BM25 + character-gram cosine similarity.

    The second signal is intentionally small and inspectable: it is a
    orthographic subword representation for shared word fragments. It is NOT a
    learned semantic representation, and does not establish synonyms or entailment.
    Planned CBOW/Skip-gram vectors will replace it for the semantic experiment.
    """

    def __init__(self, documents: list[Document], lexical_weight: float = 0.6, semantic_weight: float = 0.4):
        if lexical_weight < 0 or semantic_weight < 0 or not math.isclose(lexical_weight + semantic_weight, 1.0):
            raise ValueError("Retrieval weights must be nonnegative and sum to 1")
        self.documents = documents
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self._tokens = [tokenize(doc.title + " " + doc.text) for doc in documents]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / max(1, len(self._lengths))
        self._df: Counter[str] = Counter()
        for tokens in self._tokens:
            self._df.update(set(tokens))
        self._vectors = [Counter(character_ngrams(tokens)) for tokens in self._tokens]

    def _bm25(self, query_tokens: list[str], index: int) -> float:
        k1, b = 1.5, 0.75
        counts = Counter(self._tokens[index])
        score = 0.0
        total_docs = len(self.documents)
        for term in query_tokens:
            if term not in counts:
                continue
            df = self._df.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denominator = counts[term] + k1 * (1 - b + b * self._lengths[index] / max(1.0, self._average_length))
            score += idf * (counts[term] * (k1 + 1)) / denominator
        return score

    @staticmethod
    def _minmax(values: list[float]) -> list[float]:
        if not values:
            return []
        low, high = min(values), max(values)
        if math.isclose(low, high):
            return [1.0 if high > 0 else 0.0 for _ in values]
        return [(value - low) / (high - low) for value in values]

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = Counter(character_ngrams(query_tokens))
        lexical = [self._bm25(query_tokens, index) for index in range(len(self.documents))]
        semantic = [_cosine(query_vector, vector) for vector in self._vectors]
        lexical_norm = self._minmax(lexical)
        semantic_norm = self._minmax(semantic)
        results = []
        for index, document in enumerate(self.documents):
            if not document.text.strip():
                continue
            score = self.lexical_weight * lexical_norm[index] + self.semantic_weight * semantic_norm[index]
            if score <= 0:
                continue
            results.append(RetrievalResult(document.doc_id, document.title, document.text, score, lexical[index], semantic[index]))
        results.sort(key=lambda item: (-item.score, -item.lexical_score, item.doc_id))
        return results[: max(0, top_k)]
