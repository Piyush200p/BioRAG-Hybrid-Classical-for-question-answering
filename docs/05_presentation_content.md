# C. Presentation content — 16 slides

The editable presentation is `artifacts/presentation/BioRAG_Hybrid_Evaluation1.pptx`. This document is the text/source companion and can be used if slide edits are needed.

## Slide 1 — BioRAG-Hybrid

**Classical–neural biomedical retrieval-augmented question answering**  
Team 4 | Project Evaluation 1  
Student names and roll numbers: **[insert from team list]**

*Speaker cue:* Our system is a research prototype for literature QA and evidence-supported summaries, not a diagnostic tool.

## Slide 2 — Problem statement

- Biomedical questions require both **recall** and **support**.
- Similar text is not necessarily evidence that supports an answer.
- A useful pipeline must expose: question → documents → evidence sentences → answer.
- **Research question:** Does hybrid lexical–semantic retrieval plus HMM evidence selection improve supported answering over single-signal baselines?

Citation: Krithara et al. (2023); Xiong et al. (2024).

## Slide 3 — Motivation and applications

- Millions of biomedical records make manual search slow.
- Terminology, abbreviations, synonyms, and long abstracts challenge keyword-only search.
- Applications: literature review, biomedical information access, research question triage.
- Output must show evidence so users can inspect the answer path.

## Slide 4 — Objectives and scope

1. Build BioASQ-shaped loader and leakage-safe split.
2. Compare BM25, semantic proxy, and hybrid rankers.
3. Select coherent evidence with HMM/Viterbi.
4. Produce short answer + supporting snippets; add RNN summarization after baseline.
5. Evaluate retrieval, answer, summary, and support quality.

**Scope boundary:** English BioASQ Task B-style literature QA; no diagnosis, EHR, or private clinical data.

## Slide 5 — Existing approaches and shortcomings

| Approach | Strength | Limitation motivating us |
|---|---|---|
| BM25/TF-IDF | fast, interpretable | misses paraphrase/synonym links |
| Dense retrieval | semantic matching | can retrieve related but unsupported text |
| Vanilla RAG | grounds an LLM with context | context noise and corpus sensitivity |
| Learned filtering | can estimate evidence utility | added training/compute and less transparent |

Citation: Xiong et al. (2024); Sohn et al. (2025).

## Slide 6 — Paper 1: MedRAG/MIRAGE

- **Xiong et al., Findings ACL 2024.**
- MIRAGE: 7,663 questions from five medical QA datasets.
- MedRAG: PubMed/StatPearls/textbooks/Wikipedia; BM25, Contriever, SPECTER, MedCPT; RRF.
- Result: up to 18% improvement over CoT; GPT-3.5 MedCorp + RRF-4: 71.57 average accuracy.
- Limitation: vanilla RAG, high compute, multi-choice emphasis, limited retrieval labels.

## Slide 7 — Paper 2: RAG²

- **Sohn et al., NAACL 2025.**
- Rationale query → balanced four-corpus retrieval → MedCPT reranking → Flan-T5 helpful/not-helpful filter → answer.
- Rationale-guided filtering uses perplexity difference as a training signal.
- Llama-3-8B average: 66.3 RAG² vs 60.7 MedRAG on MedQA/MedMCQA/MMLU-Med.
- Limitation: rationale errors, per-snippet filtering, extra cost, no BioASQ paragraph focus.

## Slide 8 — Paper 3: BioASQ-QA

- **Krithara et al., Scientific Data 2023.**
- Expert English biomedical questions; four types: yes/no, factoid, list, summary.
- JSON records include documents, snippets, exact answers, and ideal paragraph answers.
- 4,721 questions reported at publication; continuously extended.
- Limitation: evidence agreement can be low and release counts evolve.

## Slide 9 — Comparison and research gap

- MedRAG tells us to measure corpus/retriever choices.
- RAG² tells us similarity alone is not enough; evidence utility matters.
- BioASQ gives the evidence and summary labels needed for this project.
- **Inferred gap:** a low-cost, inspectable BioASQ ablation of lexical/semantic fusion plus sequence-level evidence selection.
- **Hypothesis:** hybrid ranking + HMM improves evidence coverage/coherence; this remains to be tested.

## Slide 10 — Proposed methodology

1. Normalize and split by question ID.
2. BM25 lexical score.
3. Corpus-derived n-gram distributional score.
4. Weighted fusion → top-k documents.
5. Sentence segmentation → HMM emissions.
6. Viterbi → ordered `SUPPORT/CONTEXT/NOISE` sequence.
7. Extractive answer now; RNN/GRU summary planned.

## Slide 11 — Workflow diagram

Insert `artifacts/diagrams/simple_workflow.svg`.  
Caption: *The proposed data flow; solid blocks are Evaluation-1 implementation, dashed/outlined future block is the RNN summarizer.*

## Slide 12 — Dataset

**BioASQ-QA / Task B**

- Source: BioASQ portal + Zenodo 10.5281/zenodo.7655130.
- JSON, English, biomedical literature.
- Officially reported: 4,721 questions at 2023 publication; inspected BioASQ11 `training11b.json`: 4,719 records.
- Labels: document/snippet evidence, exact answer, ideal paragraph answer.
- Proposed local split: 80/10/10 by question ID from released training data; official test only when authorized.
- Version/licence: Zenodo record BioASQ11, open access, **CC BY 2.5**; preserve attribution, release metadata, and comply with current source/challenge terms.
- Downloaded JSON contains document URLs/snippets, not embedded article abstracts; retrieval needs a separate versioned text corpus.

## Slide 13 — Architecture and tools

- Data: JSON, Python standard library baseline.
- Retrieval: BM25 + deterministic n-gram proxy; planned MedCPT/BioBERT comparison.
- Evidence: explicit HMM/Viterbi.
- Output: extractive supported answer; planned RNN/GRU.
- Evaluation: Recall@k, MRR, EM/F1, ROUGE-L, supported-claim review.
- Editable architecture: `artifacts/diagrams/technical_architecture.drawio`.

## Slide 14 — Implementation completed

**Executed in this workspace:**

- BioASQ-shaped JSON loader and validation.
- Normalization/tokenization/sentence splitting.
- BM25 + n-gram hybrid retrieval.
- Three-state HMM/Viterbi evidence selector.
- Extractive answer and metric harness.
- Six-record synthetic smoke fixture; 11/11 unit tests passed.

**Observed smoke metrics:** Recall@3 1.000000; MRR 1.000000; summary token F1 0.443306; ROUGE-L F1 0.429417. These are fixture metrics, not official BioASQ results.

## Slide 15 — Remaining plan and risks

- Record and freeze the BioASQ11 release inspection.
- Obtain and version a searchable article-text corpus for the document URLs.
- Run leakage-safe split and baseline ablations.
- Add real semantic encoder after hardware/license review.
- Train/evaluate RNN/GRU on ideal answers.
- Add support/negation/conflict manual review.
- Risk: corpus access, GPU/storage cost, annotator disagreement, domain drift.

## Slide 16 — Evaluation, contribution, references

- Main comparison: BM25 vs semantic vs hybrid; top sentence vs HMM.
- Report per-question-type results and error cases.
- Contribution is a testable, inspectable pipeline—not an unsupported claim of state-of-the-art.
- Full references: `docs/references.bib` and `docs/02_literature_review.md`.
