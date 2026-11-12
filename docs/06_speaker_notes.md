# Speaker notes and paper defense preparation

## Paper 1 — Xiong et al. (2024)

**One-minute explanation:** The authors isolate the three major RAG choices—corpus, retriever, and LLM—inside a common toolkit. They build MIRAGE with 7,663 medical questions and compare BM25, dense retrievers, RRF, and multiple corpora/backbones. Their central lesson is that RAG is not one model: the corpus and retriever must match the question type, and more retrieved context can eventually become noise.

**Method details to defend:** PubMed, StatPearls, textbooks, and Wikipedia are chunked; BM25/Contriever/SPECTER/MedCPT retrieve; RRF fuses lists; snippets are prepended to the LLM; temperature is 0; CoT without RAG is the baseline.

**Limitation to volunteer:** The benchmark is mainly multi-choice, while our project includes paragraph summaries and evidence support. It is also too large to reproduce fully in a student CPU project.

## Paper 2 — Sohn et al. (2025)

**One-minute explanation:** RAG² treats retrieved evidence as potentially harmful. It uses an LLM rationale as the retrieval query, balances retrieval across four biomedical sources, reranks with MedCPT, and uses a small Flan-T5 filter trained from perplexity differences to keep helpful snippets. It improves several closed-book medical QA baselines, but its rationale and filtering stages add cost.

**Method details to defend:** A useful snippet is not defined only by similarity; the paper measures how the snippet changes the model’s rationale perplexity. The released code does not ship all corpora or the exact trained filter checkpoint, so it is not a one-command reproduction.

**Limitation to volunteer:** Their per-snippet filter may miss interactions among multiple evidence sentences. That motivates our sequence-level HMM experiment, while acknowledging that HMM emissions are a simpler proxy.

## Paper 3 — Krithara et al. (2023)

**One-minute explanation:** This is the data descriptor behind our benchmark. Experts create questions, retrieve articles, select consecutive snippets, formulate exact answers for short-answer types, and write ideal paragraph answers. The data therefore support retrieval, QA, and summarization in one schema.

**Method details to defend:** BioASQ has yes/no, factoid, list, and summary questions. Ideal answers are judged on information recall, information precision, repetition, and readability. Evidence lists may disagree across experts even when ideal answers agree, so retrieval should not be interpreted as a single absolute path.

## How to answer “why HMM?”

An HMM is appropriate as an interpretable sequence model when sentence labels are locally dependent: a supporting sentence is more likely to be followed by another supporting/context sentence than by a random distractor. Viterbi gives an explicit best state path. It is not claimed to be better than a learned filter until the ablation is run.

## How to answer “why not use an LLM directly?”

Direct generation can be fluent but can hide retrieval and evidence errors. The first milestone isolates retrieval and evidence selection so that any later RNN/LLM result can be attributed to a defined evidence set. This is also easier to run and reproduce without API keys.
