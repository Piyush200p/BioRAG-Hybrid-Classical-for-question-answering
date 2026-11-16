"""Validated NumPy cache helpers and CPU dense retrieval scoring.

Embedding generation deliberately lives in a separate GPU-oriented tool. The
runtime scorer only needs NumPy and accepts vectors produced by that tool.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data import Document
from .retrieval import HybridRetriever, RetrievalResult


@dataclass
class EmbeddingCache:
    ids: list[str]
    vectors: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingCache":
        path = Path(path)
        metadata_path = path.with_suffix(path.suffix + ".json")
        if not metadata_path.exists():
            raise ValueError(f"Missing embedding metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as data:
            ids = data["ids"].astype(str).tolist()
            vectors = np.asarray(data["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or len(ids) != vectors.shape[0]:
            raise ValueError("Embedding cache IDs and vectors have incompatible shapes")
        if len(set(ids)) != len(ids):
            raise ValueError("Embedding cache contains duplicate IDs")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding cache contains non-finite values")
        if metadata.get("count", len(ids)) != len(ids) or metadata.get("dimension", vectors.shape[1]) != vectors.shape[1]:
            raise ValueError("Embedding cache dimensions disagree with metadata")
        expected_hash = metadata.get("cache_sha256")
        if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Embedding cache checksum does not match metadata")
        return cls(ids, vectors, metadata)

    def aligned(self, ids: list[str]) -> np.ndarray:
        positions = {doc_id: index for index, doc_id in enumerate(self.ids)}
        missing = [doc_id for doc_id in ids if doc_id not in positions]
        if missing:
            raise ValueError(f"Embedding cache is missing {len(missing)} requested IDs; first: {missing[0]}")
        return self.vectors[[positions[doc_id] for doc_id in ids]]


def cosine_scores(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity, with zero vectors safely assigned a zero score."""
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    docs = np.asarray(document_vectors, dtype=np.float32)
    if docs.ndim != 2 or docs.shape[1] != query.shape[0]:
        raise ValueError("Query and document embedding dimensions do not match")
    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(docs, axis=1)
    if query_norm == 0:
        return np.zeros(len(docs), dtype=np.float32)
    denom = doc_norms * query_norm
    return np.divide(docs @ query, denom, out=np.zeros(len(docs), dtype=np.float32), where=denom > 0)


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    if np.isclose(low, high):
        return np.ones_like(values) if high > 0 else np.zeros_like(values)
    return (values - low) / (high - low)


@lru_cache(maxsize=4)
def _id_ranks(ids: tuple[str, ...]) -> np.ndarray:
    order = sorted(range(len(ids)), key=ids.__getitem__)
    ranks = np.empty(len(ids), dtype=np.int32)
    ranks[order] = np.arange(len(ids), dtype=np.int32)
    return ranks


def ranked_ids(scores: np.ndarray, ids: list[str], top_k: int) -> list[str]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    values = np.asarray(scores)
    if values.ndim != 1 or len(values) != len(ids):
        raise ValueError("Scores and document IDs must have matching one-dimensional shapes")
    k = min(top_k, len(ids))
    if k == 0:
        return []
    threshold_index = len(values) - k
    threshold = np.partition(values, threshold_index)[threshold_index]
    above = np.flatnonzero(values > threshold)
    ties = np.flatnonzero(values == threshold)
    needed = k - len(above)
    if len(ties) > needed:
        ranks = _id_ranks(tuple(ids))
        ties = ties[np.argpartition(ranks[ties], needed - 1)[:needed]]
    selected = np.concatenate((above, ties))
    selected = sorted(selected.tolist(), key=lambda i: (-float(values[i]), ids[i]))
    return [ids[index] for index in selected]


class CachedDenseRetriever:
    """BM25/dense score fusion using precomputed document and query vectors."""

    def __init__(self, documents: list[Document], document_cache: EmbeddingCache,
                 query_cache: EmbeddingCache, dense_weight: float, model_key: str):
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0 and 1")
        expected = {"biobert": ("biobert", "biobert"),
                    "medcpt": ("medcpt-article", "medcpt-query")}
        if model_key not in expected:
            raise ValueError("model_key must be biobert or medcpt")
        expected_docs, expected_queries = expected[model_key]
        if document_cache.metadata.get("model_key") != expected_docs or query_cache.metadata.get("model_key") != expected_queries:
            raise ValueError(f"{model_key} needs matching document/query encoder caches")
        if document_cache.metadata.get("kind") != "documents" or query_cache.metadata.get("kind") != "questions":
            raise ValueError("Expected document and question embedding caches")
        if document_cache.vectors.shape[1] != query_cache.vectors.shape[1]:
            raise ValueError("Query and document embedding dimensions differ")
        self.documents = documents
        self.document_ids = [document.doc_id for document in documents]
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("Corpus contains duplicate document IDs")
        if document_cache.ids != self.document_ids:
            raise ValueError("Document cache IDs and order must exactly match the corpus")
        self.document_vectors = document_cache.vectors
        self.query_cache = query_cache
        self.query_positions = {question_id: index for index, question_id in enumerate(query_cache.ids)}
        self.bm25 = HybridRetriever(documents, lexical_weight=1.0, semantic_weight=0.0, include_subword=False)
        self.dense_weight = dense_weight
        self.model_key = model_key

    def search(self, query: str, top_k: int = 5, question_id: str | None = None) -> list[RetrievalResult]:
        if question_id is None or question_id not in self.query_positions:
            raise ValueError(f"No cached query embedding for question ID {question_id!r}")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        lexical = np.asarray(self.bm25.bm25_scores(query), dtype=np.float32)
        query_vector = self.query_cache.vectors[self.query_positions[question_id]]
        dense = cosine_scores(query_vector, self.document_vectors)
        fused = (1.0 - self.dense_weight) * minmax(lexical) + self.dense_weight * minmax(dense)
        chosen = ranked_ids(fused, self.document_ids, top_k)
        position = {doc_id: index for index, doc_id in enumerate(self.document_ids)}
        output = []
        for doc_id in chosen:
            index = position[doc_id]
            document = self.documents[index]
            output.append(RetrievalResult(doc_id, document.title, document.text,
                                          float(fused[index]), float(lexical[index]), float(dense[index])))
        return output
