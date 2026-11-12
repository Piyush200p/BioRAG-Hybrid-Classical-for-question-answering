# B. Verified literature review

**Verification basis:** official ACL Anthology/Nature pages, Crossref metadata, BioASQ official pages, and the authors’ code repositories were checked on 7 October 2026 in this workspace. The three selected works are complementary: one systematic medical-RAG benchmark/toolkit, one reliability-oriented medical RAG method, and the benchmark/data descriptor that defines the project’s target evidence and summary outputs.

## Paper 1 — Benchmarking Retrieval-Augmented Generation for Medicine

**Citation and status**

- **Authors:** Guangzhi Xiong, Qiao Jin, Zhiyong Lu, Aidong Zhang.
- **Year/venue:** 2024, *Findings of the Association for Computational Linguistics: ACL 2024*, pp. 6233–6251.
- **Status/quality:** published peer-reviewed ACL Findings proceedings paper; official ACL Anthology record.
- **DOI:** [10.18653/v1/2024.findings-acl.372](https://doi.org/10.18653/v1/2024.findings-acl.372)
- **Official paper:** https://aclanthology.org/2024.findings-acl.372/
- **Code/toolkit:** https://github.com/gzxiong/MedRAG

### Problem and research question

Medical RAG has flexible choices of corpus, retriever, and LLM, but there was no systematic guidance on how these components interact. The paper asks which combinations work across medical QA settings and whether retrieval can improve model accuracy over chain-of-thought without retrieval.

### Methodology and architecture

The authors introduce:

1. **MIRAGE**, a benchmark of 7,663 questions from five medical QA datasets: MMLU-Med, MedQA-US, MedMCQA, PubMedQA*, and BioASQ-Y/N.
2. **MedRAG**, a toolkit combining document corpora, retrievers, and LLM backbones.
3. Four corpora: PubMed, StatPearls, medical textbooks, and Wikipedia, plus a combined MedCorp collection.
4. BM25, Contriever, SPECTER, and biomedical MedCPT retrievers; reciprocal-rank fusion (RRF) combines ranked lists.
5. Six LLM backbones: GPT-4, GPT-3.5, Mixtral, Llama2, MEDITRON, and PMC-LLaMA.
6. A zero-shot, multi-choice, question-only retrieval setting in which answer options are not used for retrieval.

The basic RAG implementation prepends retrieved snippets to the question and uses deterministic temperature-0 generation. It is a useful comparator for the images’ hybrid lexical–semantic idea because it explicitly compares BM25 with dense/domain retrievers and fuses them.

### Data and setup

MIRAGE has 7,663 questions. The paper’s table reports: MMLU-Med 1,089; MedQA-US 1,273; MedMCQA 4,183; PubMedQA* 500; BioASQ-Y/N 618. MedRAG’s reported corpora include 23.9M PubMed documents/snippets, 9.3k StatPearls documents/301.2k snippets, 18 textbook documents/125.8k snippets, 6.5M Wikipedia documents/29.9M snippets, and a 54.2M-snippet MedCorp union. It retrieves 32 snippets by default in the main comparison.

### Metrics, baselines, and results

- Main metric: accuracy on the multi-choice MIRAGE tasks, with standard deviation.
- Baseline: chain-of-thought without retrieval.
- The paper reports that MedRAG improves six LLMs by 1–18% relative to their CoT baselines; GPT-3.5 rises from 60.69 average accuracy to 71.57 with MedCorp + RRF-4, and GPT-4 reaches 79.97 with MedRAG compared with 73.44 for CoT.
- On MedCorp with GPT-3.5, RRF-4 is reported at 71.57 average accuracy; BM25 is 70.05, Contriever 68.61, SPECTER 64.63, MedCPT 70.06, and RRF-2 70.61.
- The authors report a log-linear relation between performance and retrieved-snippet count up to a point, followed by noise; they also observe a “lost-in-the-middle” effect.

### Contributions

- A reproducible, large-scale comparison of medical RAG components.
- Evidence that corpus choice and retriever/corpus fit matter as much as the LLM.
- Practical guidance: PubMed is robust for literature QA; BM25 and MedCPT are strong; RRF can improve robustness but must be used carefully.
- Open-source benchmark/toolkit references that make a student ablation defensible.

### Limitations

**Explicitly acknowledged by the authors:** the study mainly evaluates vanilla RAG rather than newer active/iterative RAG; additional corpora such as PMC and trustworthy FAQs are not covered; retrieval evaluation is possible only for PubMedQA* and BioASQ-Y/N because other datasets lack supporting-document labels; cross-encoder reranking is left for future work; the evaluation uses multi-choice QA and does not fully evaluate generated rationales.

**Reasonable methodological inferences:** 1.8 trillion prompt tokens and 54.2M snippets are far beyond a student CPU budget; multi-choice accuracy does not directly test paragraph-summary faithfulness; results can be sensitive to model/API versions and corpus snapshots; RRF does not itself guarantee sentence-level evidence coherence.

### What BioRAG-Hybrid should learn or avoid

- **Adapt:** compare BM25, a semantic retriever, and rank fusion; use PubMed/BioASQ-style evidence; report retrieval and answer metrics separately; use a small top-k because excessive context can hurt.
- **Avoid copying:** do not claim the full MIRAGE scale, GPT-4 performance, or a massive corpus if we only run on a small local subset; do not evaluate only exact answer accuracy when our target includes support and summaries.
- **Our extension:** add a visible sentence-sequence evidence layer and a constrained extractive summary evaluation; later compare HMM selection against top-score sentence selection.

## Paper 2 — Rationale-Guided Retrieval Augmented Generation for Medical Question Answering

**Citation and status**

- **Authors:** Jiwoong Sohn, Yein Park, Chanwoong Yoon, Sihyeon Park, Hyeon Hwang, Mujeen Sung, Hyunjae Kim, Jaewoo Kang.
- **Year/venue:** 2025, *Proceedings of NAACL 2025: Human Language Technologies, Volume 1: Long Papers*, pp. 12739–12753.
- **Status/quality:** published peer-reviewed NAACL long paper; official ACL Anthology record.
- **DOI:** [10.18653/v1/2025.naacl-long.635](https://doi.org/10.18653/v1/2025.naacl-long.635)
- **Official paper:** https://aclanthology.org/2025.naacl-long.635/
- **Code:** https://github.com/dmis-lab/RAG2

### Problem and research question

The paper targets three medical RAG failure modes: irrelevant context can distract an LLM, a raw medical question may be a poor retrieval query, and retrievers may favour dominant corpora. It asks whether rationale-based querying, balanced multi-corpus retrieval, and a learned evidence filter improve reliability and accuracy.

### Methodology and model architecture

RAG² has four stages:

1. **Rationale query:** the backbone LLM generates a step-by-step rationale; that rationale, rather than only the original question, is used as the retrieval query.
2. **Balanced retrieval:** retrieve equal numbers of snippets from four corpora—PubMed, PMC, clinical practice guidelines, and textbooks—then rerank with MedCPT.
3. **Rationale-guided filtering:** train a Flan-T5-large filter. A question–snippet pair is labelled using the change in the backbone’s perplexity when the snippet is added; helpful snippets are retained.
4. **Answer generation:** the backbone produces the final answer using filtered evidence.

The filter is intentionally smaller than the answer LLM. The paper discusses a 770M-parameter Flan-T5 filter and a 24GB RTX 3090 feasibility target, while its appendix reports the actual filter training run on one 80GB H100 for 40 epochs, learning rate 3e-5, and per-device batch size 16. The released code notes that the exact biomedical corpora and trained filter checkpoint are not redistributed, and that some scripts require environment-specific path/device changes.

### Data and training

Experiments use MedQA, MedMCQA, and MMLU-Med. Reported counts are MedQA 10,178/1,272/1,273 train/validation/test; MedMCQA 182,822/4,183/6,150; MMLU-Med has 1,089 test questions and no training split in the paper’s table. The filter is trained using MedQA and MedMCQA training data. The paper’s feasibility discussion mentions one RTX 3090 24GB GPU; the appendix reports the actual filter training run on one H100 80GB GPU for 40 epochs, learning rate 3e-5, and per-device batch size 16.

### Metrics, baselines, and results

- Main metric: multi-choice accuracy.
- Baselines include no-retrieval LLMs, MedCPT, MedCPT plus rationale, MedRAG, query2doc, Adaptive-RAG, and InstructRAG-ICL.
- On Llama-3-8B-Instruct, RAG² reports 64.6 MedQA, 59.4 MedMCQA, 74.8 MMLU-Med, average 66.3, compared with 60.7 average for MedRAG.
- On Meerkat-7B, RAG² reports average 72.4; on GPT-4o, average 86.9.
- The paper reports average improvements of 6.1% for Llama-3-8B-Instruct, 3.8% for Meerkat-7B, and 0.9% for GPT-4o over the corresponding base models.
- Ablations support the rationale query and filtering stages; a case study shows a distracting tension-pneumothorax snippet causing an incorrect clinical choice, which filtering removes.

### Contributions

- A reliability-oriented RAG design that treats evidence utility as more than similarity.
- An efficient, small learned filter with perplexity-derived labels.
- A balanced retrieval policy that reduces large-corpus dominance.
- Released retrieval/filtering code and a representative data artifact.

### Limitations

**Explicitly acknowledged by the authors:** only the biomedical domain is tested; only one Flan-T5 size is used; an incorrect rationale can produce distracting retrieval; snippets are labelled independently, missing the combined effect of multiple snippets; Flan-T5 filters one snippet at a time due to context limits.

**Computational cost:** the appendix reports a 116.7M-snippet retrieval corpus with a 564.2GB index for the RAG² collection, in addition to H100 filter training. This is substantially beyond the Evaluation-1 CPU prototype.

**Reasonable methodological inferences:** rationale generation adds latency and depends on the backbone’s reasoning quality; perplexity is only a proxy for answer correctness; the four-corpus setup is expensive to reproduce; most experiments remain closed-book multi-choice rather than BioASQ paragraph summarization; a learned filter is less transparent than a state sequence whose decisions can be inspected.

### What BioRAG-Hybrid should learn or avoid

- **Adapt:** separate retrieval relevance from evidence usefulness; include an ablation for filtering/selection; record the retrieved snippets that cause errors; treat corpus imbalance as a measurable issue.
- **Avoid copying:** do not implement a Flan-T5 filter before we have a leakage-safe evidence split; do not use generated rationales as ground truth; do not imply a local HMM has the same learned filtering capability.
- **Our extension:** test whether a simpler HMM/Viterbi selector can provide a transparent, lower-cost evidence coherence baseline on BioASQ’s paragraph-answer setting.

## Paper 3 — BioASQ-QA: A manually curated corpus for Biomedical Question Answering

**Citation and status**

- **Authors:** Anastasia Krithara, Anastasios Nentidis, Konstantinos Bougiatiotis, Georgios Paliouras.
- **Year/venue:** 2023, *Scientific Data*, volume 10, article 170.
- **Status/quality:** published open-access peer-reviewed Nature/Springer Scientific Data descriptor; published 27 March 2023.
- **DOI:** [10.1038/s41597-023-02068-4](https://doi.org/10.1038/s41597-023-02068-4)
- **Paper:** https://www.nature.com/articles/s41597-023-02068-4
- **Zenodo data record:** https://doi.org/10.5281/zenodo.7655130
- **Official BioASQ portal:** https://participants-area.bioasq.org/datasets/

### Problem and motivation

The descriptor documents a benchmark designed around biomedical experts’ information needs. It is not a model paper; its contribution is the corpus, annotation process, data format, and validation needed to evaluate biomedical retrieval, QA, and summarization.

### Data model and task structure

BioASQ Task B questions are in English and can be yes/no, factoid, list, or summary questions. Each question can include relevant articles, text snippets, concepts/triples in historical versions, an exact answer for short-answer types, and an ideal paragraph answer. The descriptor reports 4,721 questions and notes that the corpus continues to grow as the challenge runs.

The published data are JSON. The source literature is primarily MEDLINE/PubMed abstracts; the descriptor explains that BioASQ restricted current challenge material to abstracts after early experiments with full-text PMC because PMC represented a small fraction of the overall literature.

### Annotation and evaluation

Experts formulate questions, select relevant concepts, retrieve articles, select articles, mark consecutive supporting snippets, revise queries if needed, write an exact answer where appropriate, and write an ideal answer. System answers are evaluated using automatic exact-answer comparisons where possible and expert judgements of ideal-answer information recall, information precision, repetition, and readability on a 1–5 scale.

### Reported validation and limitations

- The descriptor reports that about 21 experts contributed over the dataset’s creation history.
- The paper reports approximately 500 expert-created questions per year and 4,721 questions at the time of publication.
- Inter-annotator agreement for documents/snippets can be low (the paper reports mean snippet/document F1 below 50% in the early agreement study), while ideal-answer agreement is substantially higher.
- The data evolve with annual MeSH and literature changes; therefore a retrieved count must identify a release/version.

### What BioRAG-Hybrid should learn or avoid

- **Adapt:** use both supporting snippets and ideal answers; report retrieval and summary results separately; use question type; preserve article/document identifiers.
- **Avoid:** treat one expert’s selected snippet list as the only possible valid evidence path; use challenge test answers during tuning; describe the official cumulative count as the local downloaded count.
- **Our extension:** quantify how much a transparent HMM sentence selector changes evidence recall and ROUGE-L relative to the top-ranked sentence baseline.

## Comparative table

| Paper and year | Venue and quality | Problem | Methodology | Dataset | Metrics | Main results | Strengths | Limitations | Relevance | Ideas to implement |
|---|---|---|---|---|---|---|---|---|---|---|
| Xiong et al. (2024) | Findings ACL 2024; published ACL proceedings | Which medical RAG components work best? | MIRAGE; MedRAG; BM25/dense retrievers; RRF; multiple corpora/LLMs | 7,663 questions; five datasets; BioASQ-Y/N included | Accuracy; scaling/position analyses | Up to 18% improvement over CoT; GPT-3.5 MedCorp+RRF-4 average 71.57 | Broad component ablation; open toolkit | Vanilla RAG; high compute; limited retrieval labels; multi-choice focus | Direct comparator for lexical + semantic retrieval | BM25 baseline, semantic comparator, fusion, top-k ablation |
| Sohn et al. (2025) | NAACL 2025 long paper; published ACL proceedings | Irrelevant context, weak queries, corpus bias | Rationale query; balanced multi-corpus retrieval; MedCPT reranking; Flan-T5 filter | MedQA, MedMCQA, MMLU-Med | Accuracy; ablations | RAG² average 66.3 on Llama-3-8B vs 60.7 MedRAG | Reliability-oriented; explicit failure case; code | Extra latency/cost; rationale/perplexity proxy; single-snippet filter | Strong recent alternative for evidence utility | Error analysis, filtering ablation, corpus-bias discussion |
| Krithara et al. (2023) | Scientific Data; peer-reviewed OA journal | Provide realistic biomedical QA data and evidence | Expert annotation of articles, snippets, exact and ideal answers | 4,721 reported at publication; JSON; English; four types | Expert 1–5 answer scores; EM/IR measures in challenge | Supports retrieval, QA, and summarization; high ideal-answer utility | Exact target benchmark and output schema | Evolving release; evidence agreement can be low; no model | Defines our data, labels, and evaluation target | BioASQ loader, type-aware scoring, leakage-safe split |

## Why these three are the defensible set

They cover three different layers that the project actually needs: **MedRAG/MIRAGE** gives a strong recent experimental reference for corpus/retriever choices; **RAG²** gives a recent reliability-oriented approach and a concrete failure mode for evidence filtering; **BioASQ-QA** provides the dataset semantics and the exact/ideal answer structure shown in the images. Selecting three similarly named RAG papers would leave the team unable to defend the dataset, annotation quality, or summary metrics.

## Research gap and proposed contribution

### Established facts from the papers

- Medical RAG is sensitive to corpus, retriever, and context size [P1].
- Similarity alone does not ensure helpful evidence; filtering and query formulation can help [P2].
- BioASQ includes human-selected supporting material and paragraph answers, but evidence paths can disagree [D1].

### Gap inferred by comparing them

The selected works do not directly provide a small, transparent ablation on BioASQ that compares lexical/semantic ranking with a sequence-level evidence selector and evaluates the resulting supported summary. This is an **inferred gap**, not a claim that no other paper has ever done it.

### Testable contribution

Build a reproducible BioASQ Task B prototype and measure:

1. BM25-only vs distributional-only vs weighted hybrid retrieval.
2. Top-sentence selection vs HMM/Viterbi selection.
3. Document Recall@k/MRR, snippet recall, exact answer EM/F1 where applicable, ROUGE-L, and a manually reviewed supported-claim sample.

The hypothesis is that hybrid ranking plus sequence selection can improve evidence coverage/coherence at low compute. It remains **untested** on official BioASQ until the dataset is obtained and the experiment is run.
