# G. Viva preparation, team allocation, milestones, and risks

## Likely questions and defensible answers

### Why did you choose this topic?

Biomedical literature search needs both relevant retrieval and evidence-supported compression. The uploaded proposal already identifies BioASQ, hybrid retrieval, HMM/Viterbi, and RNN summarization; we refined it into measurable retrieval/evidence/summary experiments.

### Why is this NLP?

The system processes natural-language questions and abstracts, performs information retrieval and sentence selection, and produces answers/summaries. The core representations, ranking, sequence decoding, and evaluation are NLP tasks.

### Why these three papers?

MedRAG/MIRAGE gives a strong component benchmark; RAG² gives a recent reliability and filtering method; BioASQ-QA defines the exact evidence and answer data we will use. They are complementary rather than three papers repeating the same result.

### What is the main contribution of each paper?

MedRAG: systematic corpus/retriever/LLM evaluation and toolkit. RAG²: rationale queries, balanced retrieval, and a learned helpfulness filter. BioASQ-QA: expert-curated biomedical QA corpus with evidence and ideal answers.

### What are the main limitations of current methods?

Lexical retrieval misses paraphrases; semantic retrieval can return related but unsupported text; vanilla RAG can be distracted by context; learned filters add cost; BioASQ evidence annotations can disagree. These motivate an inspectable sequence-level evidence experiment.

### Why BioASQ?

It directly matches the proposed outputs: documents, snippets, exact answers, and ideal summaries. It is more aligned with literature QA than an exam-only multi-choice set.

### What preprocessing is required?

Schema validation, missing-value checks, whitespace/punctuation normalization, abbreviation-safe tokenization, sentence splitting, and duplicate/URL checks. We will preserve negation markers and IDs.

### Why choose this model/algorithm?

BM25 is a strong transparent baseline. The initial semantic proxy is deterministic and CPU-runnable; the planned controlled comparison is MedCPT or another licensed biomedical encoder. HMM/Viterbi is chosen for inspectable sentence-sequence selection, not because superiority is assumed.

### What baseline will you compare against?

At minimum: BM25-only; semantic-only; weighted hybrid; top-1/top-k sentence extraction versus HMM/Viterbi. If compute permits, compare MedCPT or a compact biomedical encoder under the same corpus and split.

### Which metrics are appropriate?

Recall@k and MRR for retrieval; snippet recall/precision for evidence; exact match and token F1 for factoid/list/yes-no outputs; ROUGE-L and manual information precision/recall/readability for summaries; supported-claim rate for grounding.

### How will you prevent data leakage?

Split by question ID before fitting document statistics, vocabularies, representations, thresholds, or any summarizer. Do not use official test ideal answers or gold snippets for tuning. If linked articles occur across splits, record that and consider a document-aware analysis.

### What is current implementation status?

The dependency-free baseline has executed on a six-record synthetic BioASQ-shaped fixture. Eleven unit tests pass. The public BioASQ11 JSON has been downloaded and inspected locally, but it contains document references/snippets rather than article text, so official retrieval performance is not claimed. The RNN summarizer is pending.

### What exactly is the 30% completion?

The completed high-value modules are setup, loader/validation, preprocessing, a BM25+semantic-proxy retrieval baseline, HMM/Viterbi evidence selection, answer formatting, metrics, tests, diagrams, and an executable demo. The remaining major modules are official dataset ingestion, full ablations, real neural semantic encoder, trained RNN summary model, and final error analysis. The percentage is an honest work-package estimate, not a measured accuracy claim.

### What is different from existing papers?

The project tests a low-cost, transparent combination of lexical/semantic ranking and sequence-level evidence selection specifically against BioASQ’s document/snippet/ideal-answer structure. We do not claim state-of-the-art or novelty until the ablation is executed and compared fairly.

### What are the main risks?

Dataset access/version drift, large-corpus compute, evidence disagreement, synonym/abbreviation mismatch, negation/conflict, and RNN overfitting. The fallback is a well-evaluated extractive system with a documented neural extension.

## Team allocation template

The images contain no names or roll numbers. Replace placeholders after the team list is available. Every member should read all three papers and defend their assigned paper in detail.

| Member | Primary responsibility | Concrete deliverable | Dependency/checkpoint |
|---|---|---|---|
| Student 1 — [Name/Roll] | Paper 1 + retrieval | MedRAG/MIRAGE 2-page review; BM25/semantic ablation explanation | Understands P1 method/results/limitations before slide freeze |
| Student 2 — [Name/Roll] | Paper 2 + evidence | RAG² review; HMM/Viterbi evidence module and error cases | Evidence module runs on fixture and explains limitations |
| Student 3 — [Name/Roll] | Paper 3 + dataset | BioASQ-QA review; versioned data inspection and dataset slide | Local counts recorded without leakage |
| Student 4 — [Name/Roll] | Evaluation/testing | Metric harness, tests, reproducibility log, support review | Test report and ablation table verified |
| Shared | Presentation/integration | Diagrams, slide integration, rehearsal, cross-question practice | Each student can explain all paper abstracts and own method |

If the team has fewer or more members, merge/split by deliverable rather than assigning all implementation to one person.

## Milestones

| Checkpoint | Deliverable | Acceptance evidence |
|---|---|---|
| E1 now | loader + sample fixture + baseline + tests + diagrams | `demo_results.json`, test output, editable diagrams |
| E1 + 1 week | article-text corpus, leakage-safe split, and schema report | version/date, aligned corpus, local counts |
| E1 + 2 weeks | lexical/semantic/hybrid ablation | per-type retrieval and evidence table |
| E1 + 3 weeks | neural semantic encoder | pinned environment, model licence, resource log |
| E1 + 4 weeks | RNN/GRU extractive/abstractive comparison | train/validation curves, ROUGE-L, error cases |
| Final | support/conflict review and presentation | reproducible report, final slides, rehearsal |
