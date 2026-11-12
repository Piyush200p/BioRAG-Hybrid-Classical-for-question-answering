# Dataset access and local-data policy

The repository contains the synthetic fixture at `sample/bioasq_sample.json` and, when the official build has been run locally, generated PubMed abstract artifacts under `official/`. The generated corpus is a local research artifact and should be kept out of version control if repository size limits apply.

## Recommended source

- BioASQ participants’ data portal: https://participants-area.bioasq.org/datasets/
- Zenodo BioASQ-QA record: https://doi.org/10.5281/zenodo.7655130
- Zenodo API metadata: https://zenodo.org/api/records/7655130

The verified Zenodo record is **BioASQ11**, open access, and identifies the dataset licence as **CC BY 2.5**. The record exposes `training11b.json`. The 2023 Scientific Data descriptor reports 4,721 questions at publication time and notes that the challenge is continuously extended. Treat that as an official reported size, not the local size of a future download.

## Suggested download/inspection procedure

1. Download `training11b.json` from the official record or use the authorized participant portal.
2. Save the version, checksum, date, and licence/attribution in a local experiment log.
3. Keep the file outside version control if it is large; configure the CLI with its local path.
4. Run the loader:

   ```bash
   PYTHONPATH=src python3 -c "from biorag.data import load_questions; q=load_questions('/path/to/training11b.json'); print(len(q))"
   ```

5. Inspect missing article text, duplicate question IDs, question-type counts, document/snippet counts, and token lengths.
6. Split by question ID before fitting BM25 statistics, n-gram vectors, thresholds, or a summarizer.

## Build the official searchable corpus

BioASQ document fields are PubMed URLs, not embedded abstracts. The repository now includes a reproducible NCBI E-utilities ingestion tool:

```bash
python3 tools/build_official_corpus.py \
  --questions data/official/training11b.json \
  --output data/official/pubmed_abstracts.json \
  --split-output data/official/question_split.json
```

The tool extracts PubMed IDs, fetches titles/abstracts in batches with retry and checkpoint support, writes source/checksum metadata, and creates a stable question-ID split. Gold snippets are never inserted into the searchable corpus. NCBI requests should be used responsibly; providing `--email` is recommended.

The loader accepts a JSON list or a wrapper with `questions`, `data`, or `items`. It preserves question type, document references, snippets, exact answers, and ideal answers without downloading remote article URLs. For retrieval, supply a separate versioned article-text corpus; never use the gold snippets as a substitute corpus for evaluation.
