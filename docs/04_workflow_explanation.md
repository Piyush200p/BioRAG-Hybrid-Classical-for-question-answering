# D. Workflow and technical architecture explanation

## Simple flowchart

The presentation-ready simple flow is in:

- `artifacts/diagrams/simple_workflow.drawio` — editable diagrams.net XML.
- `artifacts/diagrams/simple_workflow.svg` — vector export for slides.
- `artifacts/diagrams/simple_workflow.png` — high-resolution raster export.
- `artifacts/diagrams/simple_workflow.mmd` — Mermaid secondary source.

```mermaid
flowchart LR
  A[(BioASQ-QA JSON)] --> B[Inspect + validate]
  B --> C[Normalize + split by question ID]
  C --> D[BM25 / TF-IDF]
  C --> E[Distributional n-gram representation]
  D --> F[Weighted hybrid rank]
  E --> F
  F --> G[Candidate documents + sentences]
  G --> H[HMM / Viterbi evidence sequence]
  H --> I[Extractive supported answer]
  I --> J[Recall@k · MRR · F1 · ROUGE-L]
  J --> K[Error + support analysis]
```

## Detailed technical architecture

```mermaid
flowchart TB
  subgraph DATA[Data layer]
    Q[BioASQ question + type]
    C[(Article/abstract corpus)]
    G[(Gold documents, snippets, exact/ideal answers)]
  end
  subgraph PREP[Preparation]
    V[Schema validation + missing-value checks]
    N[Regex normalization + biomedical tokenization]
    S[Question-ID train/validation/test split]
    I1[Inverted index + document statistics]
    I2[Character n-gram/distributional vectors]
  end
  subgraph RET[Hybrid retrieval]
    L[BM25 lexical score]
    M[Semantic-proxy cosine score]
    F[Weighted score fusion]
    R[Top-k ranked documents]
  end
  subgraph EVID[Evidence selection]
    T[Sentence segmentation]
    H[3-state HMM emissions]
    VIT[Viterbi decoding]
    E[SUPPORT / CONTEXT / NOISE evidence]
  end
  subgraph QA[Answering and evaluation]
    X[Extractive supported answer]
    RNN[Planned GRU/RNN summarizer]
    MTR[Recall@k · MRR · EM/F1 · ROUGE-L]
    ERR[Error, negation, conflict, support review]
  end
  Q --> V --> N --> S
  C --> V
  N --> I1
  N --> I2
  Q --> L
  Q --> M
  I1 --> L
  I2 --> M
  L --> F
  M --> F
  F --> R --> T --> H --> VIT --> E --> X
  E --> RNN
  X --> MTR
  RNN -. planned .-> MTR
  G --> MTR
  MTR --> ERR
```

## Stage contract

| Stage | Input | Operation/tool | Output/dependency |
|---|---|---|---|
| Data acquisition | BioASQ JSON/release metadata | Official Zenodo/participant portal access | Versioned question records; no private data in Git |
| Inspection | Raw records | Schema validation, duplicate/missing checks | Clean records and local counts |
| Preprocessing | Question/document/snippet text | Whitespace, punctuation, abbreviation-safe tokenization | Tokens/sentences; preserves IDs and negation terms |
| Split | Question records | Question-ID split before fitting statistics | Train/validation/internal test |
| Representation | Training corpus | BM25 statistics and n-gram vectors | Inverted index/vector store |
| Retrieval | Query + index | BM25, distributional similarity, weighted fusion | Top-k ranked documents with component scores |
| Evidence | Top-k documents | Sentence segmentation; HMM emissions and transitions; Viterbi | Ordered support/context sentences |
| Answer | Selected evidence + question type | Extractive answer now; RNN/GRU planned | Exact/short answer or supported summary |
| Evaluation | Prediction + gold fields | Retrieval, answer, summary, support/error metrics | Per-type results and failure cases |

## Implementation boundary for Evaluation 1

The code in `src/biorag` implements data loading, normalization, hybrid retrieval, HMM/Viterbi selection, extractive output, and sanity metrics. The RNN/GRU block is shown as a dashed planned module in the architecture because no trained model or official local dataset was available at workspace creation. This is an honest representation of the milestone.
