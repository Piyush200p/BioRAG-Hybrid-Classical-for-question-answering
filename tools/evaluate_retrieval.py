"""Compare BM25, BioBERT, MedCPT and validation-tuned hybrid retrieval.

This script uses only cached embeddings and NumPy; it does not load models or
download weights. The test split is evaluated once after validation tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biorag.data import Document, load_questions  # noqa: E402
from biorag.dense_retrieval import EmbeddingCache, cosine_scores, minmax, ranked_ids  # noqa: E402
from biorag.metrics import recall_at_k, reciprocal_rank  # noqa: E402
from biorag.retrieval import HybridRetriever  # noqa: E402


def summarize(rows: list[dict], split: str, top_k: int) -> dict:
    def metrics(items: list[dict]) -> dict:
        return {
            f"recall@{top_k}": round(float(np.mean([row["recall"] for row in items])) if items else 0.0, 6),
            f"mrr@{top_k}": round(float(np.mean([row["mrr"] for row in items])) if items else 0.0, 6),
            "question_count": len(items),
            "unavailable_gold_document_references": sum(row["unavailable_gold"] for row in items),
        }
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["question_type"]].append(row)
    return {"overall": metrics(rows), "by_question_type": {key: metrics(value) for key, value in sorted(groups.items())}, "split": split}


def compute_rows(questions, split_ids, corpus, retriever, doc_ids, doc_vectors, query_cache, top_k):
    if not split_ids.issubset({question.question_id for question in questions}):
        raise ValueError("Split manifest contains question IDs absent from the loaded dataset")
    ordered_corpus_ids = [document.doc_id for document in corpus]
    corpus_id_set = set(ordered_corpus_ids)
    if ordered_corpus_ids != doc_ids:
        raise ValueError("Embedding cache document order/IDs must match corpus order exactly")
    query_positions = {item: index for index, item in enumerate(query_cache.ids)}
    missing_questions = split_ids - set(query_positions)
    if missing_questions:
        raise ValueError(f"Query cache missing {len(missing_questions)} IDs; first: {sorted(missing_questions)[0]}")
    rows = []
    scores = {}
    for question in questions:
        if question.question_id not in split_ids:
            continue
        dense = cosine_scores(query_cache.vectors[query_positions[question.question_id]], doc_vectors)
        lexical = np.asarray(retriever.bm25_scores(question.body), dtype=np.float32)
        gold = question.gold_document_ids & corpus_id_set
        model_scores = {"bm25": lexical, "dense": dense}
        for name, values in model_scores.items():
            ids = ranked_ids(values, ordered_corpus_ids, top_k)
            scores[(question.question_id, name)] = values
            rows.append({"question_id": question.question_id, "question_type": question.question_type, "method": name,
                         "recall": recall_at_k(ids, gold, top_k), "mrr": reciprocal_rank(ids, gold),
                         "unavailable_gold": len(question.gold_document_ids - gold)})
    return rows, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/official/training11b.json")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/official/pubmed_abstracts.json")
    parser.add_argument("--split", type=Path, default=ROOT / "data/official/question_split.json")
    parser.add_argument("--biobert-docs", type=Path, required=True)
    parser.add_argument("--biobert-queries", type=Path, required=True)
    parser.add_argument("--medcpt-docs", type=Path, required=True)
    parser.add_argument("--medcpt-queries", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/results/neural_retrieval_evaluation.json")
    args = parser.parse_args()
    if args.top_k < 1 or not 0 < args.weight_step <= 1:
        parser.error("--top-k must be positive and --weight-step must be in (0, 1]")

    questions = load_questions(args.questions)
    manifest = json.loads(args.split.read_text(encoding="utf-8"))
    split_sets = {key: set(value) for key, value in manifest.items()}
    required = {"train", "validation", "test"}
    if set(split_sets) != required:
        raise ValueError("Split manifest must contain train, validation, and test IDs")
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Question split sets overlap")
    question_ids = {question.question_id for question in questions}
    if set.union(*split_sets.values()) != question_ids:
        raise ValueError("Split manifest must partition all loaded question IDs")

    raw_corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    corpus = [Document(str(item["id"]), item.get("title", ""), item.get("text", "")) for item in raw_corpus]
    if len({item.doc_id for item in corpus}) != len(corpus):
        raise ValueError("Corpus has duplicate document IDs")
    document_ids = [document.doc_id for document in corpus]
    corpus_id_set = set(document_ids)
    retriever = HybridRetriever(corpus, lexical_weight=1.0, semantic_weight=0.0, include_subword=False)

    cache_specs = {
        "biobert": (args.biobert_docs, args.biobert_queries, "biobert"),
        "medcpt": (args.medcpt_docs, args.medcpt_queries, "medcpt"),
    }
    loaded = {}
    for name, (doc_path, query_path, expected) in cache_specs.items():
        docs, queries_cache = EmbeddingCache.load(doc_path), EmbeddingCache.load(query_path)
        if docs.metadata.get("kind") != "documents" or queries_cache.metadata.get("kind") != "questions":
            raise ValueError(f"{name} cache kinds must be documents and questions")
        model_key = str(docs.metadata.get("model_key", ""))
        query_model = str(queries_cache.metadata.get("model_key", ""))
        if expected == "medcpt" and (model_key != "medcpt-article" or query_model != "medcpt-query"):
            raise ValueError("MedCPT requires medcpt-article document vectors and medcpt-query question vectors")
        if expected == "biobert" and (model_key != "biobert" or query_model != "biobert"):
            raise ValueError("BioBERT caches must both use the biobert model")
        if docs.metadata.get("input_sha256") != hashlib.sha256(args.corpus.read_bytes()).hexdigest():
            raise ValueError(f"{name} document cache was generated from a different corpus file")
        if queries_cache.metadata.get("input_sha256") != hashlib.sha256(args.questions.read_bytes()).hexdigest():
            raise ValueError(f"{name} query cache was generated from a different question file")
        if docs.vectors.shape[1] != queries_cache.vectors.shape[1]:
            raise ValueError(f"{name} query/document dimensions differ")
        docs_matrix = docs.aligned(document_ids)
        if docs.ids != document_ids:
            raise ValueError(f"{name} document cache IDs/order do not exactly match corpus")
        loaded[name] = (docs_matrix, queries_cache)

    result = {"protocol": {"questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                            "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                            "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
                            "top_k": args.top_k, "validation_objective": f"mean MRR@{args.top_k}",
                            "weight_step": args.weight_step, "test_used_for_tuning": False,
                            "gold_relevance_scope": "gold document IDs present in the searchable PubMed corpus"},
              "cache_metadata": {name: {"documents": docs.metadata, "queries": queries.metadata}
                                 for name, (docs, queries) in ((name, (EmbeddingCache.load(dp), EmbeddingCache.load(qp))) for name, (dp, qp, _) in cache_specs.items())}}

    validation_per_model, test_per_model, selected = {}, {}, {}
    for name, (doc_vectors, query_cache) in loaded.items():
        val_rows, val_scores = compute_rows(questions, split_sets["validation"], corpus, retriever, document_ids, doc_vectors, query_cache, args.top_k)
        validation_per_model[name] = summarize([row for row in val_rows if row["method"] == "dense"], "validation", args.top_k)
        if name == "biobert":
            baseline_val_rows = val_rows
            baseline_val_scores = val_scores
        weights = np.arange(0.0, 1.0 + args.weight_step / 2, args.weight_step)
        trials = []
        for dense_weight in weights:
            hybrid_rows = []
            for question in questions:
                qid = question.question_id
                if qid not in split_sets["validation"]:
                    continue
                bm = minmax(baseline_val_scores[(qid, "bm25")])
                dn = minmax(val_scores[(qid, "dense")])
                ranked = ranked_ids((1.0 - dense_weight) * bm + dense_weight * dn, document_ids, args.top_k)
                hybrid_rows.append({"question_id": qid, "question_type": question.question_type,
                                    "recall": recall_at_k(ranked, question.gold_document_ids & corpus_id_set, args.top_k),
                                    "mrr": reciprocal_rank(ranked, question.gold_document_ids & corpus_id_set),
                                    "unavailable_gold": len(question.gold_document_ids - corpus_id_set)})
            summary = summarize(hybrid_rows, "validation", args.top_k)
            trials.append({"dense_weight": round(float(dense_weight), 6), "metrics": summary["overall"]})
        best = max(trials, key=lambda trial: (trial["metrics"][f"mrr@{args.top_k}"], trial["metrics"][f"recall@{args.top_k}"], -trial["dense_weight"]))
        selected[name] = {"dense_weight": best["dense_weight"], "validation_trials": trials,
                          "validation_metrics": best["metrics"]}

    # Only after both model weights are fixed, score the held-out test split.
    for name, (doc_vectors, query_cache) in loaded.items():
        test_rows, test_scores = compute_rows(questions, split_sets["test"], corpus, retriever, document_ids, doc_vectors, query_cache, args.top_k)
        test_per_model[name] = summarize([row for row in test_rows if row["method"] == "dense"], "test", args.top_k)
        if name == "biobert":
            baseline_test_rows, baseline_test_scores = test_rows, test_scores
        hybrid_test_rows = []
        dense_weight = selected[name]["dense_weight"]
        for question in questions:
            qid = question.question_id
            if qid not in split_sets["test"]:
                continue
            bm = minmax(baseline_test_scores[(qid, "bm25")])
            dn = minmax(test_scores[(qid, "dense")])
            ranked = ranked_ids((1.0 - dense_weight) * bm + dense_weight * dn, document_ids, args.top_k)
            hybrid_test_rows.append({"question_id": qid, "question_type": question.question_type,
                                     "recall": recall_at_k(ranked, question.gold_document_ids & corpus_id_set, args.top_k),
                                     "mrr": reciprocal_rank(ranked, question.gold_document_ids & corpus_id_set),
                                     "unavailable_gold": len(question.gold_document_ids - corpus_id_set)})
        test_per_model[name + "_hybrid"] = summarize(hybrid_test_rows, "test", args.top_k)

    # BM25 is shared between both dense-model evaluations; summarize once per split.
    result["baselines"] = {
        "bm25_validation": summarize([row for row in baseline_val_rows if row["method"] == "bm25"], "validation", args.top_k),
        "bm25_test": summarize([row for row in baseline_test_rows if row["method"] == "bm25"], "test", args.top_k),
        "biobert_validation": validation_per_model["biobert"],
        "biobert_test": test_per_model["biobert"],
        "medcpt_validation": validation_per_model["medcpt"],
        "medcpt_test": test_per_model["medcpt"],
    }
    result["selected_hybrid_weights"] = selected
    result["hybrid_test"] = {key: value for key, value in test_per_model.items() if key.endswith("_hybrid")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "baselines": result["baselines"],
                      "selected_weights": {key: value["dense_weight"] for key, value in selected.items()},
                      "hybrid_test": result["hybrid_test"]}, indent=2))


if __name__ == "__main__":
    main()
