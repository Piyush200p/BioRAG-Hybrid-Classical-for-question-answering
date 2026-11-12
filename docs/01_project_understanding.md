# A. Project understanding and validation

## What the uploaded images say

The three supplied images identify the project as:

> **BioRAG-Hybrid** — *Classical–neural biomedical retrieval-augmented question answering*

The image text gives the following design:

- **Problem:** biomedical search must find relevant evidence and compress it into supported answers.
- **Proposed solution:** hybrid lexical–semantic retrieval, HMM evidence selection, and RNN summarization.
- **Stated innovation:** make classical probabilistic components visible inside a modern RAG stack.
- **Benchmark anchor:** BioASQ.
- **Methods named:** n-grams, BM25/TF-IDF, CBOW/Skip-gram, HMM/Viterbi, RNN, BioASQ.
- **Outputs named:** exact answers, evidence snippets, and paragraph summaries.
- **Metrics named:** MRR, Recall@k, F1, ROUGE, and supported-claim/factuality analysis.
- **Execution phases named:** corpus parser, lexical retriever, BioASQ loader; embeddings; HMM/Viterbi; RNN summarizer; benchmark scoring; later negation/conflict analysis and stronger reranking.

## Refined problem statement

**Research question:** *On BioASQ Task B-style biomedical questions, does a transparent hybrid retriever that combines lexical matching with a biomedical/distributional semantic signal, followed by explicit HMM/Viterbi sentence evidence selection, improve evidence retrieval and supported extractive answering over lexical-only and semantic-only baselines?*

This wording preserves the images’ core objective while making the first experiment measurable. The final RNN summarizer remains a planned extension; the Evaluation-1 milestone uses an extractive answer stage so that evidence quality can be tested before adding neural generation.

## NLP task decomposition

1. **Information retrieval:** rank biomedical abstracts or article snippets for a natural-language question.
2. **Passage/sentence selection:** identify a coherent set of evidence sentences instead of passing every retrieved sentence to a generator.
3. **Question answering:** produce a short answer for yes/no, factoid, or list questions.
4. **Multi-document summarization:** compress supporting evidence into an ideal-answer-style paragraph for summary questions.
5. **Faithfulness analysis:** check whether answer claims have support in selected evidence.

## Inputs and outputs

| Element | Defined project interpretation |
|---|---|
| Input | English biomedical question, linked BioASQ documents/snippets, and optional article titles/abstracts |
| Intermediate outputs | Normalized text, document ranking, sentence-level evidence states, evidence scores |
| Final output | Exact/short answer where applicable, selected evidence snippets, and a concise supported summary |
| Users | Biomedical literature researchers, students, and information-access systems; not a clinical diagnosis tool |
| Success | Better evidence recall/ranking and more supported summaries under the same corpus and evaluation split |

## Motivation and real-world relevance

Biomedical literature is large, terminology-heavy, and continuously updated. A useful system must retrieve evidence, expose the path from question to evidence, and avoid treating a fluent answer as proof of correctness. BioASQ is a strong anchor because its Task B records combine questions, relevant articles/snippets, exact answers, and ideal paragraph answers. The BioASQ-QA descriptor states that the resource supports information retrieval, passage retrieval, question answering, summarization, paraphrasing, and textual entailment [D1].

## Existing approaches and shortcomings

- **Lexical retrieval** such as BM25 is fast and interpretable, but can miss synonyms, abbreviations, and paraphrases.
- **Dense/semantic retrieval** can bridge wording differences, but can over-retrieve semantically related yet unsupported text and is sensitive to corpus/domain mismatch.
- **Vanilla RAG** concatenates retrieved text into a model context. MIRAGE shows that corpus and retriever choices strongly affect medical QA, and that more context can eventually add noise [P1].
- **Modern filtering RAG** such as RAG² improves retrieval quality with rationale queries, balanced corpora, and a learned filter, but adds model training and inference components [P2].
- **BioASQ annotation** gives relevant materials and ideal answers, but gold materials can vary across experts: the data descriptor reports low snippet/document agreement while ideal-answer agreement is higher [D1].

The specific design opportunity is therefore not “we used a different model.” It is an **inspectable comparison of ranking signals and evidence-sequence decisions** on a benchmark that exposes both supporting snippets and summaries.

## Proposed solution and expected contribution

### Evaluation-1 implementation

1. Load and validate BioASQ-shaped JSON.
2. Normalize biomedical text without destroying abbreviations or negation markers.
3. Build a BM25-style lexical index.
4. Add a deterministic corpus-derived character n-gram distributional signal as a no-download semantic proxy.
5. Fuse the two signals with fixed, documented weights.
6. Run a three-state HMM/Viterbi selector over candidate sentences (`SUPPORT`, `CONTEXT`, `NOISE`).
7. Produce an extractive evidence-supported answer.
8. Compute recall@k, MRR, exact-match/token-F1 sanity checks, and ROUGE-L.

### Planned final extension

- Replace the proxy semantic representation with a licensed biomedical encoder such as MedCPT or a compact Sentence-Transformers model.
- Train a small RNN encoder–decoder or GRU summarizer using retrieved/supporting sentences and ideal answers.
- Add an ablation matrix and manual supported-claim/negation/conflict analysis.

## Feasibility

The narrowed project is feasible for a student research project if the team treats the RNN as a second milestone rather than the first dependency. The retrieval and evidence modules can run on CPU for a small/medium local corpus. Full PubMed-scale indexing and large neural models require substantial disk, RAM/GPU, and licensing checks; those should be a later scale experiment, not a prerequisite for Evaluation 1.

## Technical challenges, dependencies, and risks

| Risk | Effect | Mitigation |
|---|---|---|
| BioASQ access/version changes | Counts and splits may differ | Record Zenodo/challenge version and download date; report official counts separately from local counts |
| Biomedical synonyms and abbreviations | Lexical recall loss | Add normalization/character n-grams first; later evaluate MedCPT/BioBERT |
| Similar but unsupported passages | Hallucinated or weakly supported answers | HMM evidence selection, snippet-level recall, and supported-claim review |
| Conflicting or negated evidence | Incorrect answer polarity | Preserve negation tokens; add conflict/negation error analysis in final phase |
| Gold evidence disagreement | No single “correct” retrieval path | Report document and snippet metrics with caution; use ideal-answer metrics and manual examples |
| RNN training data/cost | Weak generation before evidence is stable | Use extractive baseline first; train RNN only after split and evidence labels are fixed |
| Data leakage | Inflated retrieval/answer scores | Split by question ID before fitting statistics; never use test ideal answers during retrieval/tuning |
| Clinical overinterpretation | Unsafe use of outputs | Position the artifact as research literature QA, not a diagnostic or treatment recommender |

## Information missing from the uploaded material

1. Student names, roll numbers, institution, course/instructor, and number of Team 4 members.
2. Whether the team should use BioASQ11 or a newer release, and whether participant-area access is available.
3. Whether the required target is all four BioASQ question types or only summary QA.
4. The intended semantic encoder, RNN architecture, training split, and hardware budget.
5. A separate article-text corpus compatible with the downloaded BioASQ document references; the public BioASQ11 JSON itself contains URLs/snippets rather than embedded abstracts.
6. Any measured official-data baseline results, a fixed definition of “supported claim,” and whether manual expert evaluation is available.

These are recorded as placeholders/assumptions rather than silently filled in.
