# Final verification and evaluation readiness

## Artifact checks performed

- The PowerPoint was generated as an editable `.pptx` with **16 slides**.
- LibreOffice successfully opened and exported the deck to PDF for visual review.
- The workflow and architecture PNG exports were opened and visually reviewed at full image size and in the slide render.
- Both `.drawio` files parsed as valid XML.
- Diagram labels use the same BioASQ, BM25, semantic-proxy, HMM/Viterbi, extractive-output, and planned RNN/GRU vocabulary as the implementation and documents.
- The simple workflow has separate lexical and semantic branches into fusion.
- Dataset/storage nodes are represented as cylinders; future neural generation is labelled planned.
- The prototype executed end to end on the included six-record fixture.
- The final test run passed all **11 tests**.

## Evidence boundaries

| Claim | Status |
|---|---|
| Topic/title extracted from uploaded images | Verified |
| Three paper metadata/venues/DOIs | Verified through official sources |
| BioASQ11 Zenodo licence = CC BY 2.5 | Verified through official API metadata |
| BioASQ11 JSON downloaded and inspected locally | Verified: 4,719 records; no embedded article text |
| Official BioASQ retrieval performance | **Not claimed** |
| RNN/GRU trained | **Not claimed** |
| Fixture pipeline executed | Verified |
| Unit tests passed | Verified |

## What still needs team-specific completion before submission

1. Replace student names/roll numbers and course/instructor placeholders.
2. Confirm whether BioASQ11 or a newer release is the evaluation target.
3. Obtain/version a searchable article-text corpus corresponding to document references.
4. Finalize the split and run the ablation matrix on non-leaky evidence.
5. Decide hardware/model/licence details for the neural extension.

## Ready-to-demo commands

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m biorag.run_pipeline \
  --data data/sample/bioasq_sample.json \
  --output artifacts/results/demo_results.json
```

The observed aggregate fixture metrics are: Recall@3 1.0; MRR 1.0; summary token F1 0.443306; ROUGE-L F1 0.429417. Use them only as sanity evidence of execution.
