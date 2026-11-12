# E. Dataset decision and proposed methodology

## Candidate comparison

| Candidate | Verified properties | Fit | Decision |
|---|---|---|---|
| **BioASQ-QA** | English expert questions; article/snippet links; exact answers and ideal paragraph answers; JSON; 4,721 questions reported in the 2023 descriptor; continuously extended | Directly matches the images’ BioASQ anchor, retrieval, evidence, QA, and summarization outputs | **Primary recommendation** |
| **MIRAGE** | 7,663 medical questions from five datasets; contains BioASQ-Y/N 618-question subset; multi-choice zero-shot RAG benchmark | Excellent for a later controlled RAG comparison, but it removes supporting contexts and targets multi-choice accuracy | Secondary comparator, not primary dataset |
| **PubMedQA*** | MIRAGE paper’s 500-question multi-choice literature-QA subset | Useful for a small retrieval experiment but does not preserve the full BioASQ exact/ideal/sentence evidence structure | Optional sanity comparator |

## Primary dataset slide facts

- **Name:** BioASQ-QA / BioASQ Task B corpus.
- **Source:** BioASQ participants’ data portal and the 2023 Zenodo record, linked from the Scientific Data paper.
- **Official URL:** https://doi.org/10.5281/zenodo.7655130; portal https://participants-area.bioasq.org/datasets/
- **Type/format:** English biomedical QA records in JSON, combining structured and unstructured fields.
- **Officially reported volume:** 4,721 questions in the 2023 descriptor at publication time. The downloaded BioASQ11 `training11b.json` was inspected locally and contains **4,719 records**; this version-specific count is not substituted for the paper’s reported cumulative count.
- **Question types:** yes/no, factoid, list, summary.
- **Targets:** relevant articles/snippets, exact answer for short-answer types where applicable, and ideal paragraph answer.
- **Domain:** biomedical research literature, mainly MEDLINE/PubMed abstracts in the current challenge design.
- **Splits:** the BioASQ11 training file is a release, not an agreed project train/validation/test split. Proposed leakage-safe project split: question-ID/group split of released training records into 80% train, 10% validation, 10% internal test; use official challenge test only if access and labels are available.
- **License/access:** the verified Zenodo BioASQ11 record is open access and labels its dataset **CC BY 2.5**; retain attribution and the release metadata, and follow the current dataset/challenge terms. The record contains `training11b.json`; do not redistribute restricted participant-area material.
- **Why it fits:** it supplies both retrieval evidence and summary targets, exactly matching the project’s multi-output design.

## Access and inspection procedure

1. Download the selected release from the official Zenodo record or authorized BioASQ participant portal.
2. Save the release identifier/date in `data/README.md` and do not commit private/restricted data.
3. Inspect JSON keys, question types, missing article text, duplicate IDs, URL validity, and snippet offsets.
4. Count questions/documents/snippets/tokens locally and record class/type distribution.
5. Split by question ID before fitting vocabulary, BM25 statistics, embeddings, thresholds, or summarizer parameters.
6. Keep the official test labels untouched; tune only on training/validation data.

## Proposed end-to-end method

### Phase 1: transparent baseline

`question → normalization → BM25 → top-k documents → first/highest-overlap sentence → answer/summary`

### Phase 2: BioRAG-Hybrid milestone

`question → normalization → BM25 + distributional n-gram representation → weighted hybrid rank → candidate sentences → HMM/Viterbi evidence sequence → extractive supported answer`

### Phase 3: planned neural extension

`selected evidence → RNN/GRU encoder–decoder → paragraph summary → evidence-support check`

### Method justification

- **BM25/TF-IDF:** interpretable lexical anchor and strong low-cost baseline.
- **Subword/distributional representation:** captures spelling variants and related biomedical forms in a small local prototype; a biomedical encoder is the planned upgrade.
- **HMM/Viterbi:** makes the sequence decision inspectable and encourages coherent supporting sentences instead of isolated similarity peaks.
- **Extractive first:** prevents a fluent generator from obscuring retrieval errors during Evaluation 1.
- **RNN later:** honours the original project idea only after target summaries, evidence alignment, and evaluation are stable.

## Evaluation plan

| Level | Metric | Purpose |
|---|---|---|
| Document retrieval | Recall@1/@3/@5, MRR | Are relevant articles found early? |
| Snippet/sentence retrieval | gold-snippet recall, evidence precision | Is the selected evidence actually useful? |
| Short answer | exact match, token F1; macro-F1 for yes/no | Does the answer match the target type? |
| Summary | ROUGE-L plus manual information precision/recall/readability sample | Is the summary concise and supported? |
| Reliability | supported-claim rate, unsupported claim count, negation/conflict error count | Does retrieval translate into grounded output? |

## Data quality and bias considerations

- The corpus is English and expert-curated, so results do not automatically transfer to other languages, lay users, or clinical notes.
- Supporting snippets reflect annotation decisions and may not be exhaustive; disagreement is documented by the data paper.
- MEDLINE/PubMed coverage and annual vocabulary changes create temporal drift.
- Question types are not necessarily balanced; report per-type results rather than only a global average.
- Use document URLs/IDs carefully; article text may be absent or updated.
