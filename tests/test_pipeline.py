import json
import tempfile
import unittest
import itertools
import math
from pathlib import Path

import numpy as np

from biorag.data import load_questions
from biorag.dense_retrieval import EmbeddingCache, cosine_scores, minmax, ranked_ids
from biorag.hmm import EvidenceHMM
from biorag.metrics import recall_at_k, rouge_l
from biorag.retrieval import HybridRetriever
from biorag.run_pipeline import run
from biorag.text import normalize, tokenize


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "sample" / "bioasq_sample.json"


class BioRAGPipelineTests(unittest.TestCase):
    def test_loader_reads_all_fixture_questions(self):
        questions = load_questions(FIXTURE)
        self.assertEqual(len(questions), 6)
        self.assertEqual(questions[0].question_id, "sample-001")
        self.assertTrue(questions[0].documents)

    def test_loader_flattens_official_answer_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested.json"
            path.write_text(json.dumps([{
                "id": "nested-1",
                "body": "Which proteins are involved?",
                "type": "list",
                "documents": ["https://example.org/doc"],
                "snippets": [{"text": "Protein A and Protein B are involved."}],
                "exact_answer": [["Protein A"], ["Protein B"]],
                "ideal_answer": ["Protein A and Protein B are involved."],
            }]), encoding="utf-8")
            question = load_questions(path)[0]
            self.assertEqual(question.exact_answer, ["Protein A", "Protein B"])
            self.assertIn("Protein A", question.ideal_answer)

    def test_normalization_and_tokenization(self):
        self.assertEqual(normalize("  HMM–based  evidence "), "hmm-based evidence")
        self.assertIn("metformin", tokenize("Metformin in type 2 diabetes"))

    def test_retrieval_finds_expected_document(self):
        question = load_questions(FIXTURE)[0]
        result = HybridRetriever(question.documents).search(question.body, top_k=1)
        self.assertEqual(result[0].doc_id, "doc-metformin-1")

    def test_dense_cache_alignment_and_cosine_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npz"
            np.savez(path, ids=np.asarray(["b", "a"]), vectors=np.asarray([[0, 1], [1, 0]], dtype=np.float32))
            path.with_suffix(".npz.json").write_text(json.dumps({"kind": "documents"}), encoding="utf-8")
            cache = EmbeddingCache.load(path)
            aligned = cache.aligned(["a", "b"])
            np.testing.assert_array_equal(aligned, [[1, 0], [0, 1]])
            np.testing.assert_allclose(cosine_scores(np.asarray([1, 0]), aligned), [1, 0])
            with self.assertRaisesRegex(ValueError, "missing"):
                cache.aligned(["missing"])

    def test_dense_ranking_helpers_handle_constant_and_tied_scores(self):
        np.testing.assert_array_equal(minmax(np.asarray([0.0, 0.0])), [0.0, 0.0])
        self.assertEqual(ranked_ids(np.asarray([0.5, 0.5]), ["b", "a"], 2), ["a", "b"])

    def test_hmm_returns_supporting_evidence(self):
        selected = EvidenceHMM().select(
            "What is the role of metformin in type 2 diabetes?",
            ["Metformin reduces hepatic glucose production.", "The sky is blue."],
            retrieval_score=1.0,
        )
        self.assertTrue(selected)
        self.assertIn("metformin", selected[0].sentence.lower())

    def test_viterbi_matches_exhaustive_path_search(self):
        model = EvidenceHMM()
        observations = ["HIGH", "MID", "LOW", "HIGH"]
        def log_probability(path):
            score = math.log(model.start[path[0]]) + math.log(model.emission[path[0]][observations[0]])
            for index in range(1, len(path)):
                score += math.log(model.transition[path[index - 1]][path[index]]) + math.log(model.emission[path[index]][observations[index]])
            return score
        exhaustive = max(itertools.product(model.states, repeat=len(observations)), key=log_probability)
        self.assertEqual(model.decode(observations), list(exhaustive))
        for row in [model.start, *model.transition.values(), *model.emission.values()]:
            self.assertAlmostEqual(sum(row.values()), 1.0)

    def test_rouge_l_uses_precision_and_recall(self):
        self.assertAlmostEqual(rouge_l("a b c d", "a b"), 2 / 3)

    def test_empty_query_does_not_retrieve(self):
        question = load_questions(FIXTURE)[0]
        self.assertEqual(HybridRetriever(question.documents).search(""), [])

    def test_url_only_official_record_requires_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "urls.json"
            path.write_text(json.dumps({"questions": [{"body": "What is insulin?", "id": "q", "documents": ["https://pubmed.ncbi.nlm.nih.gov/1/"]}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No article text"):
                run(path)

    def test_metrics_have_expected_bounds(self):
        self.assertEqual(recall_at_k(["a", "b"], {"b"}, 2), 1.0)
        self.assertGreater(rouge_l("oxygen transport", "Hemoglobin transports oxygen"), 0.0)

    def test_end_to_end_output_is_reproducible_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = run(FIXTURE, output)
            self.assertEqual(result["status"], "executed")
            self.assertEqual(result["record_count"], 6)
            self.assertTrue(output.exists())
            parsed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(parsed["record_count"], 6)
            self.assertIn("aggregate_metrics", parsed)


if __name__ == "__main__":
    unittest.main()
