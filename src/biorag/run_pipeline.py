"""Execute BM25/subword ranking, candidate HMM selection and cited extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import Document, Question, load_questions
from .dense_retrieval import CachedDenseRetriever, EmbeddingCache
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


def answer_question(question: Question, retriever, top_k: int = 3, selector: str = "hmm", max_sentences: int = 2) -> dict:
    if isinstance(retriever, CachedDenseRetriever):
        ranked = retriever.search(question.body, top_k=top_k, question_id=question.question_id)
    else:
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
        "ranked_documents": [{"id": item.doc_id, "title": item.title, "score": round(item.score, 6), "bm25_score": round(item.lexical_score, 6), ("dense_cosine" if isinstance(retriever, CachedDenseRetriever) else "subword_cosine"): round(item.semantic_score, 6)} for item in ranked],
        "selected_evidence": chosen,
        "predicted_summary": prediction,
        "short_answer": None,
        "reference_summary": question.ideal_answer,
        "status": "candidate_evidence" if prediction else "insufficient_evidence",
        "metrics": {
            f"recall_at_{top_k}": round(recall_at_k([item.doc_id for item in ranked], question.gold_document_ids, top_k), 6),
            f"mrr_at_{top_k}": round(reciprocal_rank([item.doc_id for item in ranked], question.gold_document_ids), 6),
            "summary_token_f1": round(token_f1(prediction, question.ideal_answer), 6),
            "rouge_l_f1": round(rouge_l(prediction, question.ideal_answer), 6),
        },
    }


def run(data_path: str | Path, output_path: str | Path | None = None, corpus_path: str | Path | None = None, lexical_weight: float = 0.6, selector: str = "hmm", max_questions: int | None = None,
        retriever_mode: str = "legacy", dense_model: str = "medcpt", document_embeddings: str | Path | None = None,
        query_embeddings: str | Path | None = None, evaluation_results: str | Path | None = None,
        fusion_weight: float | None = None, split_manifest: str | Path | None = None, split_name: str | None = None,
        top_k: int = 3) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    questions = load_questions(data_path)
    if split_manifest:
        if split_name not in {"train", "validation", "test"}:
            raise ValueError("split_name must be train, validation, or test when a split manifest is provided")
        manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
        question_ids = set(manifest.get(split_name, []))
        questions = [question for question in questions if question.question_id in question_ids]
        if not questions:
            raise ValueError(f"No questions matched the {split_name} split")
    if max_questions is not None:
        if max_questions < 1:
            raise ValueError("max_questions must be positive")
        questions = questions[:max_questions]
    if corpus_path:
        raw = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        corpus = [Document(item["id"], item.get("title", ""), item["text"]) for item in raw]
    else:
        corpus = build_corpus(questions)
    selected_weight = fusion_weight
    selected_model = None
    if retriever_mode == "legacy":
        retriever = HybridRetriever(corpus, lexical_weight, 1.0 - lexical_weight)
    elif retriever_mode == "bm25":
        retriever = HybridRetriever(corpus, 1.0, 0.0, include_subword=False)
    elif retriever_mode in {"biobert", "medcpt", "hybrid"}:
        if not corpus_path:
            raise ValueError("Dense retrieval requires --corpus so the embedding cache can be validated and aligned")
        model_key = dense_model if retriever_mode == "hybrid" else retriever_mode
        if model_key not in {"biobert", "medcpt"}:
            raise ValueError("dense_model must be biobert or medcpt")
        selected_model = model_key
        if not document_embeddings or not query_embeddings:
            raise ValueError("Dense retrieval requires document_embeddings and query_embeddings cache paths")
        document_cache = EmbeddingCache.load(document_embeddings)
        query_cache = EmbeddingCache.load(query_embeddings)
        if document_cache.metadata.get("input_sha256") != hashlib.sha256(Path(corpus_path).read_bytes()).hexdigest():
            raise ValueError("Document embedding cache was created from a different corpus file")
        if query_cache.metadata.get("input_sha256") != hashlib.sha256(Path(data_path).read_bytes()).hexdigest():
            raise ValueError("Query embedding cache was created from a different question file")
        if retriever_mode == "hybrid" and selected_weight is None:
            result_path = Path(evaluation_results or "artifacts/results/neural_retrieval_evaluation.json")
            tuned = json.loads(result_path.read_text(encoding="utf-8"))
            selected_weight = tuned["selected_hybrid_weights"][model_key]["dense_weight"]
        if selected_weight is None:
            selected_weight = 1.0 if retriever_mode in {"biobert", "medcpt"} else 0.0
        retriever = CachedDenseRetriever(corpus, document_cache, query_cache, selected_weight, model_key)
    else:
        raise ValueError(f"Unknown retriever mode: {retriever_mode}")
    records = [answer_question(question, retriever, top_k=top_k, selector=selector) for question in questions]
    names = [f"recall_at_{top_k}", f"mrr_at_{top_k}", "summary_token_f1", "rouge_l_f1"]
    result = {
        "status": "executed", "data_path": str(data_path), "record_count": len(records),
        "corpus_count": len(corpus), "lexical_weight": lexical_weight, "selector": selector,
        "retriever_mode": retriever_mode, "dense_model": selected_model,
        "dense_weight": selected_weight if retriever_mode in {"hybrid", "biobert", "medcpt"} else None,
        "split_name": split_name if split_manifest else None, "top_k": top_k,
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
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--retriever", choices=["legacy", "bm25", "biobert", "medcpt", "hybrid"], default="legacy")
    parser.add_argument("--dense-model", choices=["biobert", "medcpt"], default="medcpt", help="Dense model used by hybrid mode")
    parser.add_argument("--document-embeddings", help="Document NPZ cache for dense retrieval")
    parser.add_argument("--query-embeddings", help="Question NPZ cache for dense retrieval")
    parser.add_argument("--evaluation-results", help="Validation tuning result JSON; defaults to artifacts/results/neural_retrieval_evaluation.json")
    parser.add_argument("--fusion-weight", type=float, help="Optional explicit dense weight; otherwise hybrid mode reads validation-tuned weight")
    parser.add_argument("--split-manifest", help="Optional question split manifest")
    parser.add_argument("--split-name", choices=["train", "validation", "test"])
    args = parser.parse_args()
    try:
        result = run(args.data, args.output, args.corpus, args.lexical_weight, args.selector, args.max_questions,
                     args.retriever, args.dense_model, args.document_embeddings, args.query_embeddings,
                     args.evaluation_results, args.fusion_weight, args.split_manifest, args.split_name, args.top_k)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"Input error: {error}\n")
    print(json.dumps({"record_count": result["record_count"], "corpus_count": result["corpus_count"], "aggregate_metrics": result["aggregate_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
