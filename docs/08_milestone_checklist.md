# F. Evaluation-1 implementation checklist

## Work-package estimate

| Module | Current status | Evidence | Approx. project work |
|---|---|---|---:|
| Project setup and reproducibility | **Completed** | README, `pyproject.toml`, config, standard-library run | 4% |
| Dataset schema loader/validation | **Completed on fixture** | `src/biorag/data.py`, tests | 6% |
| Data inspection and preprocessing | **Completed on fixture** | `src/biorag/text.py`, tests | 5% |
| Lexical baseline | **Completed** | BM25 implementation and demo | 7% |
| Semantic representation | **Partially completed** | deterministic n-gram proxy; real encoder pending | 4% |
| Hybrid fusion | **Completed in prototype** | `src/biorag/retrieval.py` | 3% |
| HMM/Viterbi evidence selector | **Completed in prototype** | `src/biorag/hmm.py`, test | 5% |
| Extractive answer/metrics | **Completed in prototype** | `metrics.py`, `run_pipeline.py` | 4% |
| Official BioASQ acquisition/inspection | **Completed for BioASQ11 JSON** | 4,719 records, type counts, checksum, zero embedded article texts | 8% |
| Leakage-safe official experiment | **Pending** | depends on official data | 8% |
| RNN/GRU summarizer | **Pending** | planned after baseline | 18% |
| Neural reranker/biomedical encoder | **Pending** | hardware/licence review | 10% |
| Support/negation/conflict analysis | **Partially planned** | design in docs | 8% |
| Final API/UI and report | **Pending** | not needed for E1 | 10% |

The work-package numbers are planning estimates that sum to 100%; they are not a claim that an accuracy percentage has been achieved. On the high-value core path needed to demonstrate the idea, the loader, preprocessing, lexical baseline, hybrid prototype, HMM selector, metrics, tests, and reproducibility artifacts are implemented. BioASQ11 JSON inspection is complete, but a separate searchable article corpus, official retrieval experiment, and neural training remain pending.

## Actual execution evidence

Command:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m biorag.run_pipeline \
  --data data/sample/bioasq_sample.json \
  --output artifacts/results/demo_results.json
```

Observed on 7 October 2026 in this workspace:

- 11 unit tests: **passed**.
- End-to-end records: **6**.
- Aggregate smoke metrics: Recall@3 **1.000000**, MRR **1.000000**, summary token F1 **0.443306**, ROUGE-L F1 **0.429417**.

These metrics are reproducible results on the included synthetic fixture and must not appear in the presentation as official BioASQ performance. The exact JSON is in `artifacts/results/demo_results.json`.
