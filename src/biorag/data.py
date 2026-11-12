"""BioASQ-shaped data loading with validation and a documented fixture path."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class Question:
    question_id: str
    body: str
    question_type: str
    documents: list[Document] = field(default_factory=list)
    gold_snippets: list[str] = field(default_factory=list)
    exact_answer: list[str] = field(default_factory=list)
    ideal_answer: str = ""
    gold_document_ids: set[str] = field(default_factory=set)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def _canonical_document_id(value: str) -> str:
    """Use PMID as the stable key for both BioASQ URLs and corpus records."""
    value = value.strip()
    match = re.search(r"/([0-9]+)/?$", value)
    return match.group(1) if match else value


def _flatten_text(value: Any) -> list[str]:
    """Flatten BioASQ answer variants while preserving their textual content."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_text(item))
        return result
    if isinstance(value, dict):
        for key in ("text", "answer", "value"):
            if key in value:
                return _flatten_text(value[key])
    return []


def _answer_list(value: Any) -> list[str]:
    return _flatten_text(value)


def _documents(raw: Any) -> list[Document]:
    result: list[Document] = []
    if not isinstance(raw, list):
        return result
    for index, item in enumerate(raw):
        if isinstance(item, str):
            result.append(Document(_canonical_document_id(item), item, ""))
            continue
        if not isinstance(item, dict):
            continue
        doc_id = _canonical_document_id(_text(item.get("id") or item.get("document") or item.get("url") or f"doc-{index}"))
        title = _text(item.get("title") or doc_id)
        text = _text(item.get("text") or item.get("abstract") or item.get("contents"))
        result.append(Document(doc_id, title, text))
    return result


def load_questions(path: str | Path) -> list[Question]:
    """Load either the local fixture shape or common BioASQ JSON variants.

    Official BioASQ records may reference documents by URL and provide snippets
    separately; locally prepared records may include a ``documents`` list with
    article text. The loader keeps both forms and never downloads remote text.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        raw = raw.get("questions") or raw.get("data") or raw.get("items") or []
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list or a questions/data wrapper: {source}")

    questions: list[Question] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Record {index} is not an object")
        body = _text(item.get("body") or item.get("question") or item.get("text")).strip()
        if not body:
            raise ValueError(f"Record {index} has no question body")
        snippets = item.get("snippets") or []
        gold_snippets: list[str] = []
        gold_document_ids: set[str] = set()
        if isinstance(snippets, list):
            for snippet in snippets:
                if isinstance(snippet, str):
                    value = snippet
                elif isinstance(snippet, dict):
                    value = _text(snippet.get("text") or snippet.get("snippet"))
                    if isinstance(snippet.get("document"), str):
                        gold_document_ids.add(_canonical_document_id(snippet["document"]))
                else:
                    value = ""
                if value.strip():
                    gold_snippets.append(value.strip())
        raw_documents = item.get("documents") or item.get("articles") or []
        if all(isinstance(document, str) for document in raw_documents):
            gold_document_ids.update(_canonical_document_id(document) for document in raw_documents)
        questions.append(
            Question(
                question_id=_text(item.get("id") or item.get("question_id") or f"question-{index}"),
                body=body,
                question_type=_text(item.get("type") or item.get("question_type") or "summary").lower(),
                documents=_documents(item.get("documents") or item.get("articles") or []),
                gold_snippets=gold_snippets,
                exact_answer=_answer_list(item.get("exact_answer") or item.get("exactAnswer")),
                ideal_answer=" ".join(_flatten_text(item.get("ideal_answer") or item.get("idealAnswer") or "")).strip(),
                gold_document_ids=gold_document_ids,
            )
        )
    return questions
