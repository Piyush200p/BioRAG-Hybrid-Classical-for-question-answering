"""Inspect a BioASQ JSON release without copying biomedical text into reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def inspect(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("questions", raw) if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError("Expected a JSON list or a wrapper with a questions list")

    type_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    total_documents = 0
    total_snippets = 0
    nonempty_article_text = 0
    ids: list[str] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        type_counts[str(item.get("type", "unknown"))] += 1
        for field in ("documents", "snippets", "exact_answer", "ideal_answer"):
            if field in item and item[field]:
                field_counts[field] += 1
        documents = item.get("documents") or []
        snippets = item.get("snippets") or []
        total_documents += len(documents) if isinstance(documents, list) else 0
        total_snippets += len(snippets) if isinstance(snippets, list) else 0
        for document in documents if isinstance(documents, list) else []:
            if isinstance(document, dict) and any(document.get(key) for key in ("text", "abstract", "contents")):
                nonempty_article_text += 1
        ids.append(str(item.get("id", "")))

    duplicate_ids = len(ids) - len(set(ids))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "record_count": len(records),
        "question_type_counts": dict(sorted(type_counts.items())),
        "records_with_nonempty_field": dict(sorted(field_counts.items())),
        "document_reference_count": total_documents,
        "snippet_count": total_snippets,
        "embedded_article_text_count": nonempty_article_text,
        "duplicate_question_id_count": duplicate_ids,
        "source_note": "BioASQ11 training JSON is a question/evidence reference file; document references are URLs and do not embed article abstracts.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.path)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
