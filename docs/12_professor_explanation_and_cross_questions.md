# BioRAG-Hybrid: professor explanation and cross-question guide

**Verified on 13 October 2026.** Based on current source code, dataset inspection, saved results, the local MedRAG paper PDF, and the official BioASQ-QA paper. Older workspace notes describe earlier milestones; use this guide for the current implementation boundary. No application code was changed for this guide.

## 1. Start with this explanation

> Our project, BioRAG-Hybrid, is a biomedical literature question-answering prototype. We use BioASQ11 for expert-written questions and reference evidence, and a separately fetched PubMed collection for searchable article text. We combine BM25 word matching with character-trigram similarity, retrieve the top three articles, and apply a three-state HMM with Viterbi decoding to select candidate evidence sentences. The current answer is up to two extracted sentences with source identifiers, not neural text generation. Our methodological base is the MedRAG/MIRAGE paper. Our additional experiment is an explicit, inspectable sentence-selection layer. The pipeline works, but improved quality over baselines still needs controlled held-out evaluation.

**Three distinctions to remember:**

1. **Implemented is not the same as proven better.** We have implemented the architecture, not established superiority.
2. **Subword similarity is not learned semantics.** No BioBERT, MedCPT, CBOW, Skip-gram, RNN, or LLM is running in this prototype.
3. **Saved full-data results are not held-out test results.** A split file exists, but the runner does not use it.

## 2. Dataset: what we actually use

### A. BioASQ11 question and annotation dataset

File: `data/official/training11b.json`.

| Property | Verified local value |
|---|---:|
| Questions | 4,719 |
| Factoid questions: seek an entity/name | 1,417 |
| List questions: seek several entities | 901 |
| Summary questions: seek an explanation | 1,130 |
| Yes/no questions | 1,271 |
| Document-reference occurrences | 42,608 |
| Gold snippets | 56,795 |
| Records with nonempty ideal-answer fields | 4,719 |
| Records with nonempty exact-answer fields | 3,589 |
| Duplicate question IDs | 0 |
| Embedded article texts | 0 |

The dataset contains English biomedical questions created by experts. It is not a patient-record dataset and was not created by our team. Our work is ingestion, corpus alignment, retrieval, sentence selection, and evaluation infrastructure.

The 2023 dataset paper reports **4,721** questions. Our downloaded release contains **4,719**. These are different reported artifacts/counts; do not invent an explanation for the two-record difference. Cite the local filename and checksum for our experiments.

### B. Separate PubMed title/abstract corpus

Files: `data/official/pubmed_abstracts.json` and `data/official/pubmed_abstracts.metadata.json`.

- We extracted **40,221 unique PMIDs** from the BioASQ document-reference fields.
- The NCBI PubMed E-utilities ingestion retained **40,160 article records**, leaving **61 requested IDs unrepresented**. The count alone does not establish why those IDs were not retained.
- **43 records have an empty abstract `text` field.** The retriever skips empty-text records when returning results. Therefore, say “40,160 article records,” not “40,160 complete abstracts.” Their titles still enter the current index statistics.
- Each record stores `id`, `title`, `text`, and `source`.
- Gold snippets and ideal answers are **not** used as searchable article text.

**Important scope limitation:** this is a fixed, BioASQ-reference-derived PubMed subset, not all PubMed. The corpus membership was selected using benchmark references across the question collection. Retrieval searches the same supplied corpus for each query, not only that query's gold articles, but the collection is still relevance-enriched. It is not an unbiased open-PubMed benchmark.

### C. How the fields are used

| Field/resource | Meaning | Current use |
|---|---|---|
| `body` | Natural-language question | Query for retrieval and sentence overlap |
| `type` | Factoid/list/summary/yesno | Retained in output; does not yet select a different answering algorithm |
| `documents` | Expert-associated article URLs | PMID corpus acquisition and gold retrieval labels |
| `snippets` | Expert-selected evidence passages | Loaded; source IDs contribute to gold document IDs; snippet scoring is pending |
| `exact_answer` | Short answer or entity-list variants | Loaded/flattened, but not predicted/scored by the runner |
| `ideal_answer` | Expert-written explanatory reference | Used after prediction for token F1 and ROUGE-L |
| PubMed `title` + `text` | Independent article content | Retrieval index input |
| PubMed `text` | Abstract content | Sentence extraction input |

Ideal answers are available across question types, not only `summary` questions. Evaluating ideal-answer overlap across types can be informative, but it must be labeled and broken down by type; it is not a substitute for exact-answer evaluation.

### D. Split and reproducibility

`data/official/question_split.json` stores **3,712 train / 506 validation / 501 test** question IDs. The builder uses a deterministic hash of seed **17** and the question ID, with approximate 80/10/10 thresholds. It is not an exact-size, type-stratified, document-disjoint, or temporal split.

**The current runner does not consume this manifest.** The saved full run processes all 4,719 records. Also, the current prototype has no learned parameters being trained on the nominal training partition.

The BioASQ release metadata is documented as CC BY 2.5. Keep attribution. Do not assume that this license also grants unrestricted redistribution rights for every independently fetched PubMed abstract.

## 3. The two pipelines

### Pipeline A: offline ingestion and preparation

```text
BioASQ11 JSON
    -> inspect schema, counts, duplicate IDs, checksum
    -> retain question/type/gold evidence/reference answers
    -> extract and deduplicate article PMIDs
    -> fetch PubMed titles and abstracts through NCBI
    -> align document IDs and save corpus + checksum metadata
    -> write question-ID split manifest [not yet applied by runner]

At runner startup:
    corpus titles + abstracts
    -> normalize and tokenize
    -> BM25 statistics and character-trigram count vectors in memory
```

The ingestion tool supports batching, retries, throttling, and a partial checkpoint. Retrieval uses local files afterwards; it is not a live PubMed API call for each question. Indexes are rebuilt per run rather than saved as a production search index.

### Pipeline B: online question to output

```text
Question
    -> normalize/tokenize
    -> BM25 scores + character-trigram cosine scores
    -> min-max normalize each score list
    -> 0.60 * BM25_normalized + 0.40 * trigram_normalized
    -> top 3 positive-scoring documents
    -> split each abstract into sentences
    -> question/sentence token overlap -> LOW/MID/HIGH observations
    -> fixed HMM + Viterbi decoding, reset per document
    -> retain SUPPORT/CONTEXT candidates above minimum overlap
    -> remove exact duplicate sentences
    -> take first 2 candidates in document-rank/sentence order
    -> extracted text + PMID + sentence position + scores/state
    -> compare predictions with gold labels and ideal answers
```

**Gold answers are not an inference input.** The runner stores the reference summary alongside predictions for evaluation, but retrieval and HMM selection use the question and corpus content.

## 4. What each algorithm is doing

### Normalization and tokenization

- Whitespace is collapsed and selected punctuation variants are normalized.
- Text is lowercased for matching.
- Tokens preserve biomedical-looking forms such as hyphenated terms.
- A small stopword list is removed for matching.
- Negation words are not deliberately removed by a separate negation module; this is a limitation for biomedical polarity questions.
- Sentence splitting is a lightweight regular expression, not a biomedical sentence tokenizer.

### BM25 branch

BM25 is the lexical baseline. It rewards query terms that occur in a document, gives more weight to rarer terms, and applies document-length normalization. The implementation uses:

```text
k1 = 1.5
b  = 0.75
```

Why it is useful: biomedical questions often contain exact gene, disease, protein, or drug names, so lexical matching is strong and interpretable.

Weakness: it can miss paraphrases, synonyms, spelling variants, and concepts expressed with different words.

### Character 3-gram branch

For every token, the implementation creates boundary-aware character trigrams. It compares the query and document trigram count vectors using cosine similarity. This can help with shared word fragments, morphology, and small spelling variants.

It is called a semantic **proxy** in the documentation, but the most precise description is **character 3-gram orthographic/subword similarity**. It is not a trained distributional embedding and cannot reliably understand synonyms or entailment. It is also not equivalent to the CBOW/Skip-gram idea shown in the original proposal.

### Fusion

The raw BM25 and trigram values are separately min-max scaled over the corpus for the current question. Then:

```text
hybrid = 0.60 * normalized_BM25 + 0.40 * normalized_trigram
```

The top three positive-scoring documents are returned. The 0.60/0.40 weights are prototype configuration values, not learned or tuned through the held-out test split.

### HMM/Viterbi evidence selection

For each sentence, the emission signal is:

```text
number of distinct question terms also in sentence
---------------------------------------------------
number of distinct question terms
```

The overlap is mapped to:

- `HIGH` if it is at least 0.18;
- `MID` if it is at least 0.06 and below 0.18;
- `LOW` otherwise.

The fixed states are:

- `SUPPORT`: likely answer-bearing sentence;
- `CONTEXT`: related sentence that may help explain the answer;
- `NOISE`: likely irrelevant sentence.

Viterbi chooses the most probable state sequence using hand-specified start, transition, and emission probabilities in log space. This is sequence-level selection: neighboring sentence labels influence each other, rather than treating every sentence independently. The sequence resets for each document.

**What it does not do:** it does not perform biomedical entailment, fact verification, contradiction detection, negation handling, or learned HMM estimation. `SUPPORT` means “candidate support according to overlap and the fixed model,” not “clinically verified truth.”

### Answer stage

The prototype returns at most two verbatim selected sentences and their source document IDs. This is deliberately extractive: it reduces hallucination and makes it possible to inspect the source. It also means the output may be incomplete, repetitive, or not fluent.

`short_answer` is currently `null` for every prediction. Therefore, we cannot claim that the current system answers factoid, list, or yes/no questions according to official BioASQ exact-answer scoring.

## 5. What is implemented versus planned

### Implemented and verified

- BioASQ-shaped loader and answer-field flattening.
- Official BioASQ11 inspection and local checksum record.
- PubMed title/abstract acquisition aligned to BioASQ PMIDs.
- Biomedical-safe basic normalization, tokenization, and sentence splitting.
- In-memory BM25 retrieval.
- Character 3-gram cosine branch.
- Min-max weighted fusion.
- Top-3 document retrieval.
- Fixed three-state HMM and log-space Viterbi decoding.
- Verbatim evidence extraction with source IDs, state, overlap, and position.
- Retrieval metrics and generic overlap metrics.
- 11 unit tests, rerun successfully.
- End-to-end six-question synthetic fixture, rerun successfully.

### Not implemented or not yet defensible as a final experiment

- MedCPT, BioBERT, SPECTER, a real dense encoder, CBOW, or Skip-gram.
- RNN/GRU encoder-decoder generation.
- LLM generation or standard generative RAG.
- Exact-answer prediction and official factoid/list/yes-no scoring.
- Gold-snippet sentence recall/precision evaluation.
- Split-aware runner using train/validation/test.
- Parameter tuning on validation followed by one held-out test evaluation.
- BM25-only versus trigram-only versus fused ablation on the official split.
- HMM versus top-overlap sentence ablation on the official split.
- Manual expert evaluation for support, contradiction, negation, completeness, and readability.
- Persistent retrieval indexes; indexes are rebuilt in memory for each run.

## 6. Results: what may and may not be said

### Verified local execution

Command used:

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp/omnirush \
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Result: **11 tests passed**.

The synthetic six-question fixture has 13 documents and produced:

| Run | Records | Corpus | Recall@3 | MRR@3 | Token F1 | ROUGE-L F1 |
|---|---:|---:|---:|---:|---:|---:|
| Synthetic smoke fixture | 6 | 13 | 1.000000 | 1.000000 | 0.443306 | 0.429417 |
| Official smoke artifact | 20 | 40,160 | 0.293591 | 0.950000 | 0.246352 | 0.180096 |
| Official all-record artifact | 4,719 | 40,160 | 0.436163 | 0.855725 | 0.251309 | 0.198357 |

The first row is a software smoke test, not a BioASQ performance claim. The 20-record run is exploratory. The 4,719-record run is an exploratory all-record aggregate, not a held-out benchmark result, because the current runner does not apply `question_split.json`.

### Metric interpretation

- **Recall@3:** in the current helper, the fraction of all gold document IDs found in the first three returned documents. If a question has many gold documents, finding one relevant document may still yield low recall.
- **MRR@3:** reciprocal rank of the first returned gold document. A high MRR with lower recall is possible: the first relevant article can be ranked first while other relevant articles are missed.
- **Token F1:** token overlap between extracted text and ideal answer. It is not factual correctness.
- **ROUGE-L F1:** longest-common-subsequence overlap. It is not entailment or clinical safety.

Use this wording:

> “The implementation executed end to end and the official-data run is exploratory. We are not presenting the current aggregate as a final held-out BioASQ score because split-aware and question-type-specific evaluation is still required.”

Do **not** say:

- “Our accuracy is 85.57%.” That number is MRR, not accuracy.
- “Our semantic model understands biomedical meaning.” It is a trigram proxy.
- “The HMM proves the answer is supported.” It only selects candidates.
- “We reproduced MedRAG.” We did not.
- “The RNN generates the summary.” It is planned, not implemented.
- “The full BioASQ test set was evaluated.” The local file is `training11b.json` and the current run is all-record exploratory.

## 7. Base paper: MedRAG/MIRAGE, explained simply

### Citation and purpose

**Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang (2024), “Benchmarking Retrieval-Augmented Generation for Medicine,” Findings of ACL 2024, pp. 6233–6251.** The repository includes the paper PDF at `research papers/2024.findings-acl.372.pdf`.

The paper asks: *Which combinations of biomedical corpus, retriever, and language model work best for medical RAG?*

### What the paper built

- **MIRAGE:** 7,663 multi-choice questions from five medical QA datasets.
- **MedRAG:** a toolkit for comparing corpora, retrievers, and LLM backbones.
- Corpora: PubMed, StatPearls, medical textbooks, Wikipedia, and their combined MedCorp.
- Retrievers: BM25, Contriever, SPECTER, MedCPT, and reciprocal-rank fusion (RRF).
- Backbones: GPT-4, GPT-3.5, Mixtral, Llama-2, Meditron, and PMC-LLaMA.
- Default experiment: retrieve snippets, prepend them to the question, and ask the LLM to answer the multi-choice question.
- Metric: multi-choice accuracy.

### Main findings to remember

- Retrieval can improve medical QA, but gains depend strongly on corpus, retriever, task, and model.
- The paper reports improvement of roughly 1–18% over its chain-of-thought baselines across LLMs/settings; the abstract summarizes this as up to 18%.
- For GPT-3.5, the paper reports average MIRAGE accuracy of 60.69 without retrieval and 71.57 with MedCorp plus RRF-4.
- BM25 and MedCPT are strong individual choices; fusion can help but is not automatically better in every corpus.
- Too little context can miss useful information; too much can introduce noise. The paper studies retrieved-snippet count and “lost in the middle.”

### What we adopt

1. Keep BM25 as an interpretable lexical baseline.
2. Compare lexical and non-lexical signals rather than treating one retriever as universally best.
3. Keep corpus, retrieval, and answer evaluation as separate stages.
4. Use a small top-k and study context/evidence quality.
5. Report ablations instead of only one final number.

### What we do not reproduce

- We do not use MIRAGE's five-dataset 7,663-question setup.
- We do not use their 23.9M-document PubMed corpus, 54.2M-snippet MedCorp, or 1.8 trillion prompt-token scale.
- We do not use their dense retrievers or LLM backbones.
- We do not use multi-choice answer-option prediction as our main task.
- We do not claim their accuracy numbers as our results.

### Our defensible difference

MedRAG primarily retrieves snippets and prepends them to an LLM for multi-choice generation. Our current prototype targets the richer BioASQ Task B structure and adds an explicit sentence-level evidence stage:

```text
hybrid document ranking
    -> sentence overlap emissions
    -> fixed SUPPORT/CONTEXT/NOISE HMM
    -> Viterbi sequence decoding
    -> verbatim evidence with PMIDs
```

This makes evidence decisions inspectable and low-cost. The contribution is currently a **testable design difference/hypothesis**, not a demonstrated state-of-the-art result. The required proof is an ablation comparing top-overlap selection with HMM/Viterbi on the same split, corpus, and metrics.

## 8. Why BioASQ is our dataset foundation

The BioASQ-QA paper is **Krithara et al. (2023), “BioASQ-QA: A manually curated corpus for Biomedical Question Answering,” Scientific Data, 10, 170.** It is a data descriptor rather than a model paper.

It is a good fit because BioASQ Task B contains the outputs our system is designed around:

- relevant articles;
- relevant snippets;
- exact answers for short-answer types;
- an ideal paragraph answer for explanation/summarization;
- four question types: factoid, list, yes/no, and summary.

The paper also explains that experts can follow different evidence paths. Document/snippet agreement can be low while ideal-answer agreement is higher. Therefore, a system should not be treated as wrong solely because it selected a different but genuinely supporting article; this motivates manual support review and metrics beyond exact document overlap.

## 9. Difference from the base paper in one table

| Dimension | MedRAG/MIRAGE | BioRAG-Hybrid current prototype |
|---|---|---|
| Main task | Multi-choice medical QA | BioASQ Task B open biomedical QA/evidence |
| Dataset | 7,663 questions from 5 datasets; BioASQ-Y/N subset | 4,719 local BioASQ11 training records |
| Corpus | PubMed, StatPearls, textbooks, Wikipedia, MedCorp | 40,160 local PubMed title/abstract records aligned to BioASQ references |
| Lexical retrieval | BM25 | BM25 with `k1=1.5`, `b=0.75` |
| Semantic retrieval | Contriever/SPECTER/MedCPT | Character 3-gram cosine proxy; learned encoder planned |
| Fusion | RRF variants | Min-max score fusion: 0.60 lexical + 0.40 trigram |
| Context selection | Retrieved snippets supplied to LLM | Sentence overlap plus fixed HMM/Viterbi |
| Generator | Six LLM backbones | No LLM; max two verbatim sentences |
| Main metric | Multi-choice accuracy | Recall@3/MRR plus preliminary token F1/ROUGE-L; type-specific scoring pending |
| Scale | Large, expensive benchmark | CPU-runnable, transparent prototype |
| Claim | Broad component recommendations | Small, inspectable evidence-selection hypothesis |

## 10. Cross-questions and strong answers

### Dataset and data quality

**Q1. Why did you choose BioASQ?**

**Answer:** Because it matches our complete task: biomedical questions, relevant articles, supporting snippets, exact answers for short-answer types, and ideal paragraph answers. Many medical benchmarks are only multi-choice; BioASQ lets us evaluate retrieval, evidence, and answer summarization separately.

**Q2. What exactly is in one BioASQ record?**

**Answer:** A question ID, body, type, document references, snippets, exact answer where applicable, and an ideal answer. In the local file the documents are URLs, not embedded abstract text, so we had to build a separate searchable corpus.

**Q3. Why do you say 4,719 when the paper says 4,721?**

**Answer:** The paper reports the cumulative size at publication time, while our downloaded BioASQ11 `training11b.json` release has 4,719 records. We report the exact release, checksum, and local inspection result rather than mixing version counts.

**Q4. Is BioASQ a training dataset or a test dataset?**

**Answer:** The file we have is a BioASQ11 training release. We created an internal question-ID train/validation/test manifest for controlled experiments, but the current runner has not applied it. Therefore, current full-data numbers are exploratory, not official held-out scores.

**Q5. Is the PubMed corpus the full PubMed database?**

**Answer:** No. It is a 40,160-record title/abstract subset retrieved from the 40,221 unique PubMed IDs referenced by this BioASQ release. It is sufficient for our aligned prototype but not equivalent to the 23.9M-document PubMed collection in MedRAG or all PubMed.

**Q6. Did you use the gold snippets for retrieval?**

**Answer:** No. Gold snippets are retained as evaluation labels. The searchable text comes from separate NCBI PubMed title/abstract retrieval. Indexing gold snippets would create direct evidence leakage.

**Q7. Why is there a separate corpus if BioASQ has document links?**

**Answer:** A URL identifies an article but is not the article text. Retrieval requires searchable text. We resolve the linked PMIDs through NCBI, save titles/abstracts locally, and retain provenance and checksums.

**Q8. What are the limitations of this dataset?**

**Answer:** It is English and expert-curated, focused on biomedical literature, and not representative of lay questions, other languages, or clinical notes. The annotation is not exhaustive: different experts can select different valid documents/snippets. The release and PubMed content can also change over time.

### Retrieval and hybrid method

**Q9. Why BM25?**

**Answer:** BM25 is a strong, fast, interpretable lexical baseline. Biomedical questions frequently contain exact names such as genes, diseases, drugs, and proteins. Its weakness is that exact term matching misses paraphrases and synonyms.

**Q10. What do you mean by semantic retrieval?**

**Answer:** In the current prototype, strictly speaking, it is not learned semantic retrieval. It is character 3-gram cosine similarity, an orthographic subword proxy. A real biomedical encoder such as MedCPT is planned for a later controlled comparison.

**Q11. Why use character 3-grams?**

**Answer:** They are deterministic, CPU-friendly, and can partially capture morphological or spelling similarity, which is useful for biomedical terms. They are transparent and require no model download. We do not claim that they understand meaning or entailment.

**Q12. Why are the weights 60% and 40%?**

**Answer:** They are fixed prototype configuration values: 0.60 BM25 and 0.40 subword similarity. They are not learned from the test set. In the final experiment, we should tune weights only on validation data and report a sensitivity/ablation table.

**Q13. Why min-max normalization before fusion?**

**Answer:** BM25 and cosine similarity have different numerical scales. Min-max normalization puts each signal on a comparable 0-to-1 range for the current query before weighted combination. The limitation is that it is query-dependent and can be sensitive to outliers.

**Q14. Why top three documents?**

**Answer:** This is a low-context prototype choice: it keeps evidence inspection manageable and reduces noise. It is not claimed to be optimal. Top-k must be evaluated as an ablation because the base paper also shows that too little or too much context can hurt.

**Q15. Is the system truly hybrid?**

**Answer:** Yes, in the engineering sense that two independently computed retrieval signals are combined. But the semantic branch is currently a character-trigram proxy, so the honest name is lexical-plus-subword hybrid. We should reserve “lexical-plus-biomedical-semantic hybrid” for the version with MedCPT or another trained encoder.

### HMM and evidence

**Q16. Why use an HMM after document retrieval?**

**Answer:** Document relevance does not guarantee that every sentence is useful. The HMM treats sentence selection as a sequence problem with SUPPORT, CONTEXT, and NOISE states. Viterbi uses transitions between states, so the decision is inspectable and can prefer a coherent sequence rather than isolated high-overlap sentences.

**Q17. Were the HMM parameters learned?**

**Answer:** No. The current start, transition, emission probabilities, and thresholds are hand-specified for a transparent prototype. No EM or labeled HMM training was performed.

**Q18. Does SUPPORT mean the sentence entails the answer?**

**Answer:** No. It means the sentence was assigned a candidate support state based on overlap emissions and fixed transitions. It does not certify entailment, factual correctness, or clinical safety. We need a manual support/contradiction review or a trained verifier for that claim.

**Q19. Why not simply select the highest-overlap sentence?**

**Answer:** That is an important baseline and is supported by the implementation through the `overlap` selector. The HMM is justified only if it improves evidence recall, support rate, coherence, or summary quality under the same corpus and split. We have not yet completed that official ablation.

**Q20. How does the HMM handle negation?**

**Answer:** It does not explicitly handle negation. A sentence containing the same terms as the question can still be selected even if it says “not,” “ไม่มี,” or expresses the opposite relation. Negation and contradiction analysis is a known gap and a required next experiment.

**Q21. Why call the output extractive?**

**Answer:** The output copies up to two sentences from retrieved abstracts verbatim rather than generating new text. This makes source inspection easy and lowers hallucination risk, but it can be incomplete or poorly phrased.

**Q22. Is the RNN implemented?**

**Answer:** No. RNN/GRU encoder-decoder generation is planned after retrieval and evidence evaluation are stable. The current output is extractive, and `short_answer` is null.

### Evaluation and results

**Q23. What does Recall@3 measure?**

**Answer:** In our current helper, it measures the fraction of a question’s gold document IDs that appear in the first three ranked documents. It measures coverage, not whether the generated answer is correct. We must state this definition because some retrieval papers use “recall@k” as any-relevant-hit recall.

**Q24. Why is MRR 0.8557 but Recall@3 only 0.4362?**

**Answer:** They measure different things. MRR rewards the position of the first relevant document; recall measures coverage of all gold documents. The system often retrieves one gold article near the top but not the complete gold set.

**Q25. Why are you using ROUGE-L?**

**Answer:** It gives a basic sequence-overlap sanity check against the ideal answer. It is not sufficient for biomedical correctness. BioASQ ideal-answer evaluation also values information recall, precision, repetition, and readability, so manual or expert review remains necessary.

**Q26. Why did you not report exact match/F1 for all questions?**

**Answer:** The current answer stage produces a summary-like extracted string and does not produce a type-specific short answer. Factoid, list, and yes/no need dedicated prediction and normalization. Reporting one generic metric over all types would be misleading.

**Q27. Can you call the 0.855725 number accuracy?**

**Answer:** No. It is MRR@3. The current official all-record artifact has MRR@3 of 0.855725 and Recall@3 of 0.436163. Neither is answer accuracy.

**Q28. Why are the full-data results not final?**

**Answer:** The saved run evaluates all 4,719 questions together and does not consume the saved split manifest. Also, the recall definition and summary metrics need a final type-aware protocol. We should first run split-aware baselines and ablations, then evaluate once on held-out test.

**Q29. Does a perfect synthetic Recall@3 prove the system works?**

**Answer:** No. The six-question fixture is designed as a smoke test with 13 documents. It proves the loader, retrieval, HMM, extraction, and metrics execute end to end. It is not representative of official BioASQ performance.

**Q30. What ablations will prove your claim?**

**Answer:** On the same fixed corpus and split: BM25-only, trigram-only, fused retrieval; top-overlap versus HMM sentence selection; top-k values; fusion weights; and, if resources permit, a real biomedical encoder such as MedCPT. Report document retrieval, gold-snippet evidence, type-specific answers, summary quality, and manual support/contradiction errors.

### Base paper and novelty

**Q31. What is the base paper?**

**Answer:** For methodology, Xiong et al. (2024), “Benchmarking Retrieval-Augmented Generation for Medicine,” which introduced MIRAGE and MedRAG and systematically compared medical corpora, retrievers, fusion, and LLMs. For the data contract, Krithara et al. (2023), “BioASQ-QA,” defines the questions, evidence, exact answers, and ideal answers.

**Q32. What did MedRAG do that you did not?**

**Answer:** MedRAG evaluated 7,663 multi-choice questions with multiple corpora, four retriever families, rank fusion, and six LLMs at very large scale. Our CPU prototype is much smaller and focuses on BioASQ’s open evidence/summary structure. We do not reproduce their corpus, retrievers, LLM generation, or accuracy results.

**Q33. What exactly is your difference?**

**Answer:** We add a transparent sentence-level sequence selector after hybrid document retrieval: question overlap becomes HMM emissions, and Viterbi labels sentences as SUPPORT, CONTEXT, or NOISE before verbatim extraction. The difference is implemented; its benefit still requires an ablation.

**Q34. Is HMM/Viterbi itself novel?**

**Answer:** No. HMMs and Viterbi are established methods. Our project contribution is their application as an inspectable low-cost evidence-selection baseline within this BioASQ hybrid retrieval workflow, not invention of the algorithm.

**Q35. Can you claim state of the art?**

**Answer:** No. We should claim a reproducible prototype and a testable evidence-selection hypothesis. State-of-the-art claims require controlled comparison with strong biomedical dense retrievers, appropriate BioASQ scoring, and held-out evaluation.

### Safety and deployment

**Q36. Can a doctor use this for diagnosis?**

**Answer:** No. It is a research prototype for retrieving and presenting literature evidence. It has no clinical validation, may miss articles, may mishandle negation or conflicting evidence, and does not guarantee entailment. It must not be presented as a clinical decision-support product.

**Q37. What does “grounded” mean here?**

**Answer:** In the current implementation it means that the output is copied from a retrieved article sentence and accompanied by a PMID. It does not mean that the sentence was verified to entail the question or that it is clinically correct.

**Q38. What is the main failure mode?**

**Answer:** The system can retrieve a document containing the right vocabulary but the wrong claim, then select a high-overlap sentence that is context, negated, incomplete, or contradictory. This is why evidence support, negation, contradiction, and manual review are explicit next steps.

## 11. One-minute technical walkthrough

Use this if the professor asks you to explain the complete flow quickly:

> “First, we inspect and validate BioASQ11. It gives us 4,719 expert biomedical questions with document links, snippets, exact answers, and ideal answers, but not article text. We extract the linked PMIDs and fetch a separate local PubMed title/abstract corpus through NCBI; gold snippets are never indexed as corpus text. At query time, we normalize the question, calculate BM25 and character 3-gram cosine scores for every corpus article, min-max normalize both lists, and combine them with 0.60 lexical and 0.40 subword weights. We keep the top three documents. We split each abstract into sentences and calculate question-term overlap. A fixed three-state HMM—SUPPORT, CONTEXT, and NOISE—uses those observations and Viterbi decoding to choose candidate evidence. Finally, we return up to two verbatim sentences with PMIDs and calculate retrieval/overlap sanity metrics. The current system is extractive; the RNN and real biomedical dense encoder are future work. Current official aggregates are exploratory because the runner has not yet applied the saved split.”

## 12. Claims to memorize exactly

- “Our official local file is BioASQ11 `training11b.json`, with 4,719 records.”
- “The 2023 paper reports 4,721; those are different release/count contexts.”
- “The BioASQ file contains references and snippets, but no embedded article abstracts.”
- “We fetched 40,221 PMIDs and retained 40,160 PubMed records; 61 requested IDs were unavailable to the build.”
- “The current semantic branch is a character 3-gram similarity proxy, not MedCPT or BioBERT.”
- “The HMM is fixed and hand-parameterized; SUPPORT is a candidate state, not entailment.”
- “The current answer is up to two verbatim sentences; the RNN is not implemented.”
- “The saved split is 3,712/506/501, but the current runner does not use it.”
- “Our all-record numbers are exploratory and must not be called held-out BioASQ performance.”
- “Our implemented difference from MedRAG is transparent sentence-sequence evidence selection after hybrid retrieval; superiority remains to be tested.”

## 13. Recommended next experiments before a final claim

1. Make the runner consume `question_split.json` and save split labels in every output.
2. Decide whether the corpus is fixed for all splits; document any article overlap across question splits.
3. Run BM25-only, trigram-only, and fused retrieval with validation-selected weights.
4. Run top-overlap versus HMM/Viterbi evidence selection.
5. Report Recall@1/@3/@5 and MRR with an explicit any-hit versus full-set definition.
6. Add gold-snippet sentence/evidence metrics.
7. Add exact-match/F1 and yes/no macro-F1 for the appropriate question types; add list-item scoring.
8. Compute ROUGE-L only for summary questions, then separately provide an explicitly labeled exploratory all-type analysis if desired.
9. Manually review a stratified sample for support, unsupported claims, negation, contradiction, completeness, and readability.
10. Add MedCPT or another licensed biomedical encoder only after the lexical/proxy ablations are stable.
11. Train an RNN/GRU only after leakage-safe summary targets and evidence evaluation are stable.

## 14. Evidence files and commands

| Purpose | Path/command |
|---|---|
| Project overview | `README.md` |
| Full current status | `docs/11_project_status_full.md` |
| Dataset methodology | `docs/03_dataset_methodology.md` |
| Pipeline mapping | `docs/10_professor_ready_pipelines.md` |
| Base-paper review | `docs/02_literature_review.md` |
| Base-paper PDF | `research papers/2024.findings-acl.372.pdf` |
| Local BioASQ inspection | `artifacts/results/bioasq11_inspection.json` |
| PubMed provenance | `data/official/pubmed_abstracts.metadata.json` |
| Official split manifest | `data/official/question_split.json` |
| Exploratory full results | `artifacts/results/official_notebook_results.json` |
| Tests | `PYTHONPATH=src python3 -m unittest discover -s tests -v` |
| Fixture demo | `PYTHONPATH=src python3 -m biorag.run_pipeline --data data/sample/bioasq_sample.json` |
| Reinspect BioASQ | `PYTHONPATH=src python3 tools/inspect_bioasq.py data/official/training11b.json` |
