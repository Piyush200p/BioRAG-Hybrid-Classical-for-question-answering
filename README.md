# BioRAG-Hybrid — Project Evaluation 1

Team 4 project workspace for **BioRAG-Hybrid: Classical–neural biomedical retrieval-augmented question answering**.

This repository contains the research review, dataset decision, reproducible diagrams, an editable presentation, and a small runnable implementation milestone. It deliberately separates **observed evidence**, **implemented baseline behaviour**, and **planned components**.

## Quick start

The initial implementation uses only the Python standard library, so it can run without GPU access or external model downloads.

```bash
cd /home/anmol/BioRAG-Hybrid-Evaluation1
PYTHONPATH=src python3 -m biorag.run_pipeline --data data/sample/bioasq_sample.json --output artifacts/results/demo_results.json
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The demonstration uses a six-question **synthetic smoke fixture** shaped like BioASQ JSON. It is not a claim about the official BioASQ test set. The output is evidence that the loader, normalization, BM25/subword ranking, fixed-parameter HMM/Viterbi candidate selection, extractive supported-answer stage, and evaluation checks execute end to end. The official BioASQ11 file is inspectable through `tools/inspect_bioasq.py`, but it cannot be passed to the retrieval demo without a separate article-text corpus.

## Project structure

```text
BioRAG-Hybrid-Evaluation1/
├── README.md
├── requirements.txt                 # standard-library baseline; optional stack documented
├── pyproject.toml
├── config/prototype.json
├── data/sample/bioasq_sample.json   # synthetic, non-sensitive fixture
├── src/biorag/
│   ├── data.py                       # BioASQ-shaped loader and validation
│   ├── text.py                       # biomedical-safe normalization and sentence splitting
│   ├── retrieval.py                  # BM25 + deterministic subword similarity proxy
│   ├── hmm.py                        # transparent HMM/Viterbi sentence evidence selector
│   ├── metrics.py                    # retrieval, exact-answer, F1 and ROUGE-L checks
│   └── run_pipeline.py               # executable end-to-end demo
├── tests/test_pipeline.py
├── tools/inspect_bioasq.py          # version/checksum/schema inspection
├── notebooks/                        # implementation status and follow-up plan
│   └── BioRAG_Hybrid_Implementation_Status.ipynb
├── docs/                             # research, slides, viva and milestone evidence
└── artifacts/
    ├── diagrams/                     # .drawio, .svg, .png and Mermaid sources
    ├── presentation/                 # editable .pptx and slide source
    └── results/                      # actual local execution output
```

## Scope locked for Evaluation 1

1. **Input:** English BioASQ Task B-style question with linked biomedical abstracts/snippets.
2. **Retrieval:** transparent BM25 plus a small corpus-derived n-gram distributional representation, combined by weighted rank-score fusion.
3. **Evidence:** sentence-level HMM/Viterbi selection with explicit `SUPPORT`, `CONTEXT`, and `NOISE` states.
4. **Answer:** safe extractive supported summary in the milestone; a trained RNN encoder–decoder is planned after the baseline and evaluation harness are stable.
5. **Evaluation:** document recall@k/MRR, exact-answer EM/F1 for short answers, ROUGE-L for summaries, and a manually inspected support/error sample.

The lightweight subword representation is an implementation-stage similarity proxy, not a claim that it is equivalent to BioBERT, MedCPT, CBOW/Skip-gram training, or a modern dense encoder. The upgrade path is specified in `docs/03_dataset_methodology.md`.

## Dataset access

The recommended public resource is **BioASQ-QA**, published as an open Scientific Data descriptor and distributed through Zenodo:

- Paper: https://doi.org/10.1038/s41597-023-02068-4
- Dataset record: https://doi.org/10.5281/zenodo.7655130
- BioASQ data portal: https://participants-area.bioasq.org/datasets/

The verified Zenodo record is BioASQ11, open access, and identifies the dataset licence as CC BY 2.5. Keep attribution and release metadata with any local copy.

The official paper reports 4,721 questions in the released corpus at publication time and notes that the challenge dataset is continuously extended. The downloaded BioASQ11 `training11b.json` was inspected locally: 4,719 records, SHA-256 `6df656862ca860efc355c7805d07ddca700d64ecc3785c519a49afccaaeeac98`. The file contains document URLs/snippets but no embedded article abstracts, so retrieval experiments require a separate versioned text corpus.

## Reproducibility notes

- No API keys, private data, or machine-specific paths are required.
- The sample fixture is included so the pipeline and tests work before the official dataset is obtained.
- The official BioASQ challenge test labels are not assumed to be public. Until access is confirmed, validation should use a leakage-safe split of the released training material and the sample output must be labelled accordingly.
- Inspect the downloaded release with `PYTHONPATH=src python3 tools/inspect_bioasq.py data/official/training11b.json --output artifacts/results/bioasq11_inspection.json`.
- For the planned neural upgrade, use pinned `transformers`, `torch`, `sentence-transformers`/MedCPT, and `faiss` versions only after hardware and model licensing are checked.
