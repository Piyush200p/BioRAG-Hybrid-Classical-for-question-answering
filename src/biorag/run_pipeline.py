"""Execute BM25/subword ranking, candidate HMM selection and cited extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import Document, Question, load_questions
from .hmm import EvidenceHMM
from .metrics import recall_at_k, reciprocal_rank, rouge_l, token_f1
from .retrieval import HybridRetriever
from .text import sentences


def build_corpus(questions: list[Question]) -> list[Document]:
    """Fixture corpus: pool all candidates globally, independent of a query."""
    corpus: dict[str, Document] = {}
    for question in questions:
        for document in question.documents:
            if document.text.strip():
                if document.doc_id in corpus and corpus[document.doc_id].text != document.text:
                    raise ValueError(f"Conflicting corpus text for document {document.doc_id}")
                corpus[document.doc_id] = document
    if not corpus:
        raise ValueError("No article text found. Official BioASQ URLs are relevance references, not a searchable corpus. Supply a separate text corpus; do not silently index gold snippets.")
    return list(corpus.values())


def answer_question(question: Question, retriever: HybridRetriever, top_k: int = 3, selector: str = "hmm", max_sentences: int = 2) -> dict:
    ranked = retriever.search(question.body, top_k=top_k)
    evidence: list[dict] = []
    for document in ranked:
        pieces = sentences(document.text)
        if selector == "hmm":
            selected = EvidenceHMM().select(question.body, pieces)
        else:
            selected = []
            if pieces:
                from .hmm import EvidenceSentence
                index = max(range(len(pieces)), key=lambda position: EvidenceHMM.overlap(question.body, pieces[position]))
                score = EvidenceHMM.overlap(question.body, pieces[index])
                if score > 0:
                    selected = [EvidenceSentence(pieces[index], "TOP_OVERLAP", score, index)]
        for item in selected:
            evidence.append({"sentence": item.sentence, "document_id": document.doc_id, "state": item.state, "overlap": round(item.score, 6), "position": item.position})
    chosen: list[dict] = []
    seen: set[str] = set()
    for item in evidence:
        if item["sentence"] not in seen:
            chosen.append(item)
            seen.add(item["sentence"])
        if len(chosen) == max_sentences:
            break
    prediction = " ".join(item["sentence"] for item in chosen)
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.body,
        "ranked_documents": [{"id": item.doc_id, "title": item.title, "score": round(item.score, 6), "bm25_score": round(item.lexical_score, 6), "subword_cosine": round(item.semantic_score, 6)} for item in ranked],
        "selected_evidence": chosen,
        "predicted_summary": prediction,
        "short_answer": None,
        "reference_summary": question.ideal_answer,
        "status": "candidate_evidence" if prediction else "insufficient_evidence",
        "metrics": {
            "recall_at_3": round(recall_at_k([item.doc_id for item in ranked], question.gold_document_ids, 3), 6),
            "mrr_at_3": round(reciprocal_rank([item.doc_id for item in ranked], question.gold_document_ids), 6),
            "summary_token_f1": round(token_f1(prediction, question.ideal_answer), 6),
            "rouge_l_f1": round(rouge_l(prediction, question.ideal_answer), 6),
        },
    }


def run(data_path: str | Path, output_path: str | Path | None = None, corpus_path: str | Path | None = None, lexical_weight: float = 0.6, selector: str = "hmm", max_questions: int | None = None) -> dict:
    questions = load_questions(data_path)
    if max_questions is not None:
        if max_questions < 1:
            raise ValueError("max_questions must be positive")
        questions = questions[:max_questions]
    if corpus_path:
        raw = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        corpus = [Document(item["id"], item.get("title", ""), item["text"]) for item in raw]
    else:
        corpus = build_corpus(questions)
    retriever = HybridRetriever(corpus, lexical_weight, 1.0 - lexical_weight)
    records = [answer_question(question, retriever, selector=selector) for question in questions]
    names = ["recall_at_3", "mrr_at_3", "summary_token_f1", "rouge_l_f1"]
    result = {
        "status": "executed", "data_path": str(data_path), "record_count": len(records),
        "corpus_count": len(corpus), "lexical_weight": lexical_weight, "selector": selector,
        "input_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "evidence_note": "Candidate HMM states and verbatim extraction do not certify biomedical entailment. Short-answer generation and official BioASQ scoring remain pending.",
        "aggregate_metrics": {name: round(sum(record["metrics"][name] for record in records) / max(1, len(records)), 6) for name in names},
        "records": records,
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output")
    parser.add_argument("--corpus", help="Separate JSON array of id/title/text records")
    parser.add_argument("--lexical-weight", type=float, default=0.6)
    parser.add_argument("--selector", choices=["hmm", "overlap"], default="hmm")
    parser.add_argument("--max-questions", type=int, help="Optional official-data smoke limit")
    args = parser.parse_args()
    try:
        result = run(args.data, args.output, args.corpus, args.lexical_weight, args.selector, args.max_questions)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"Input error: {error}\n")
    print(json.dumps({"record_count": result["record_count"], "corpus_count": result["corpus_count"], "aggregate_metrics": result["aggregate_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
