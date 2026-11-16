# Pretrained embedding caches

Generate caches in Colab/Kaggle or another machine with a compatible PyTorch
GPU runtime. Copy the `.npz` and adjacent `.npz.json` metadata file into this
directory (or pass their paths to the evaluator). The evaluator does not load
models or access the network; it needs NumPy only.

Each cache contains:

- `ids`: unique question IDs or PubMed IDs as Unicode strings.
- `vectors`: a float32 matrix with rows in the same order as `ids`.
- Adjacent metadata: model key/revision, input checksum, pooling method,
  normalization, max token length, vector dimension, and cache checksum.

BioBERT uses masked mean pooling over `dmis-lab/biobert-base-cased-v1.2` token
states. This is a transparent embedding baseline, not a retrieval-fine-tuned
BioBERT checkpoint. MedCPT follows its model cards: the query encoder and
article encoder are distinct, and both use the `[CLS]` vector. MedCPT articles
are passed as title/abstract pairs. Documents are truncated at 512 tokens and
questions at 64 by default.

## Generate caches

Install PyTorch for the selected notebook runtime, plus Transformers and
NumPy, then run from the repository root:

```bash
pip install "transformers>=4.40,<5" numpy
# Install the PyTorch build recommended for the active Colab/Kaggle runtime.

python tools/encode_embeddings.py --kind documents --model biobert \
  --input data/official/pubmed_abstracts.json \
  --output data/official/embeddings/biobert_documents.npz
python tools/encode_embeddings.py --kind questions --model biobert \
  --input data/official/training11b.json \
  --output data/official/embeddings/biobert_queries.npz
python tools/encode_embeddings.py --kind documents --model medcpt-article \
  --input data/official/pubmed_abstracts.json \
  --output data/official/embeddings/medcpt_documents.npz
python tools/encode_embeddings.py --kind questions --model medcpt-query \
  --input data/official/training11b.json \
  --output data/official/embeddings/medcpt_queries.npz
```

Use a smaller `--batch-size` if the GPU runs out of memory. Model weights are
pretrained and frozen; this tool performs inference only.

## Kaggle browser workflow

1. Create a Kaggle Dataset and upload `artifacts/colab/biorag_colab_bundle.zip`
   (or the equivalent Kaggle bundle in `artifacts/kaggle/`).
2. Create a Kaggle Notebook, attach the dataset with **Add Input**, select two
   T4 GPUs if Kaggle offers `GPU T4 x2`, and turn **Internet** on so Hugging Face can provide the
   pretrained model weights.
3. Import and run `notebooks/04_kaggle_embedding_generation.ipynb` from top to
   bottom. It reads the bundle from `/kaggle/input` and writes caches under
   `/kaggle/working`.
4. Save a notebook version, download `biorag_embedding_caches.zip` from the
   notebook's output files, and extract it into this directory.

The Kaggle notebook checks that two GPUs are visible and uses PyTorch
DataParallel to split each batch across them. Set its batch size to 16 by
default (8 examples per GPU); lower it to 8 if memory is tight. It uses the
same model encoders and cache format as the Colab notebook. Kaggle attached
inputs are read-only; generated caches are written to the notebook output area.

The encoder flushes a resumable partial vector file and progress manifest every
512 input records. If an encoding cell is interrupted while the Kaggle runtime
is still alive, rerun that same cell; it resumes at the last saved batch. Once
each model finishes, the notebook creates a model-specific ZIP with a clickable
download link, followed by one ZIP containing all four caches. Click each
model checkpoint link to keep a local copy before starting the next model.
Kaggle does not push notebook files directly into a local Windows folder; the
link uses the browser's download mechanism. Also save a Kaggle notebook version
to retain its `/kaggle/working` outputs after the session ends.

## Evaluate

From a CPU environment with the project requirements installed:

```bash
python tools/evaluate_retrieval.py \
  --biobert-docs data/official/embeddings/biobert_documents.npz \
  --biobert-queries data/official/embeddings/biobert_queries.npz \
  --medcpt-docs data/official/embeddings/medcpt_documents.npz \
  --medcpt-queries data/official/embeddings/medcpt_queries.npz \
  --output artifacts/results/neural_retrieval_evaluation.json
```

The evaluator reports BM25, BioBERT, and MedCPT independently; searches a
validation-only grid for a separate BM25+BioBERT and BM25+MedCPT dense weight;
then scores the selected configurations on the held-out test split. The
primary tuning objective is mean MRR@10, with Recall@10 as a tie breaker and a
preference for the lower dense weight on an exact tie. It validates question
and corpus checksums plus exact document-ID alignment before scoring.

Keep generated caches outside version control unless their size and model/data
terms have been reviewed. Retain the adjacent metadata file with every cache.
