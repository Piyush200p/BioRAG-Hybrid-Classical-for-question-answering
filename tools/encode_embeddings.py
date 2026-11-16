"""Generate BioBERT or MedCPT embeddings in a GPU notebook/runtime.

Examples:
  python tools/encode_embeddings.py --kind documents --model medcpt-article --input data/official/pubmed_abstracts.json --output data/official/embeddings/medcpt_documents.npz
  python tools/encode_embeddings.py --kind questions --model medcpt-query --input data/official/training11b.json --output data/official/embeddings/medcpt_queries.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biorag.data import load_questions  # noqa: E402


MODELS = {
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "medcpt-query": "ncbi/MedCPT-Query-Encoder",
    "medcpt-article": "ncbi/MedCPT-Article-Encoder",
}


def records(path: Path, kind: str, model_key: str) -> tuple[list[str], list[str] | list[tuple[str, str]]]:
    if kind == "questions":
        questions = load_questions(path)
        return [item.question_id for item in questions], [item.body for item in questions]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Document input must be a JSON array of {id,title,text} records")
    ids, texts = [], []
    for item in raw:
        if not isinstance(item, dict) or "id" not in item:
            raise ValueError("Every document record must have an id")
        ids.append(str(item["id"]))
        title, abstract = str(item.get("title", "")), str(item.get("text", ""))
        texts.append((title, abstract) if model_key == "medcpt-article" else (title + " " + abstract).strip())
    if len(set(ids)) != len(ids):
        raise ValueError("Input contains duplicate document IDs")
    return ids, texts


def encode(input_path: Path, output_path: Path, kind: str, model_key: str, batch_size: int, max_length: int, device: str | None, multi_gpu: bool = False, checkpoint_every: int = 512) -> None:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Embedding generation requires torch and transformers; install them in Colab/Kaggle.") from error
    if not output_path.name.endswith(".npz"):
        raise ValueError("--output must end with .npz")
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    if model_key.startswith("medcpt") and kind == "questions" and model_key != "medcpt-query":
        raise ValueError("MedCPT question embeddings require --model medcpt-query")
    if model_key.startswith("medcpt") and kind == "documents" and model_key != "medcpt-article":
        raise ValueError("MedCPT document embeddings require --model medcpt-article")
    ids, texts = records(input_path, kind, model_key)
    if not ids:
        raise ValueError("No records to encode")
    model_name = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if target_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU. Enable the Kaggle GPU accelerator and restart the session.")
    base_model.to(target_device).eval()
    if multi_gpu:
        gpu_count = torch.cuda.device_count()
        if not target_device.startswith("cuda") or gpu_count < 2:
            raise RuntimeError(f"--multi-gpu requires at least two visible CUDA GPUs; found {gpu_count}.")
        model = torch.nn.DataParallel(base_model, device_ids=list(range(gpu_count)))
        print(f"Using {gpu_count} GPUs: {[torch.cuda.get_device_name(i) for i in range(gpu_count)]}", file=sys.stderr)
    else:
        model = base_model
    outputs = []
    pooling = "cls" if model_key.startswith("medcpt") else "masked_mean"
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    id_hash = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial.npy")
    checkpoint_path = output_path.with_suffix(output_path.suffix + ".checkpoint.json")
    dimension = int(base_model.config.hidden_size)
    checkpoint_identity = {
        "model_key": model_key, "model_name": model_name,
        "model_revision": getattr(base_model.config, "_commit_hash", None),
        "kind": kind, "pooling": pooling, "max_length": max_length,
        "input_sha256": input_hash, "ids_sha256": id_hash,
        "count": len(ids), "dimension": dimension,
    }
    if partial_path.exists() != checkpoint_path.exists():
        raise ValueError(f"Incomplete checkpoint pair: remove or restore both {partial_path} and {checkpoint_path}")
    if checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        for key, value in checkpoint_identity.items():
            if saved.get(key) != value:
                raise ValueError(f"Checkpoint mismatch for {key}; use a clean output path or the matching inputs/model settings")
        start_index = int(saved.get("next_index", 0))
        if not 0 <= start_index <= len(ids):
            raise ValueError("Checkpoint next_index is outside the input range")
        vectors_memmap = np.lib.format.open_memmap(partial_path, mode="r+", dtype=np.float32, shape=(len(ids), dimension))
        print(f"Resuming {model_key} from item {start_index}/{len(ids)}", file=sys.stderr)
    else:
        start_index = 0
        vectors_memmap = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=(len(ids), dimension))

    last_saved = start_index

    def save_progress(next_index: int) -> None:
        vectors_memmap.flush()
        state = {**checkpoint_identity, "next_index": next_index, "checkpoint_every": checkpoint_every}
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary_path.replace(checkpoint_path)
        print(f"Checkpointed {model_key}: {next_index}/{len(ids)}", file=sys.stderr)

    if not checkpoint_path.exists():
        save_progress(start_index)

    with torch.inference_mode():
        for start in range(start_index, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(target_device) for key, value in encoded.items()}
            # Tuple output keeps gather compatible with torch.nn.DataParallel.
            hidden = model(**encoded, return_dict=False)[0]
            if pooling == "cls":
                vectors = hidden[:, 0, :]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=1)
            end = start + len(batch)
            vectors_memmap[start:end] = vectors.cpu().numpy().astype(np.float32)
            if end - last_saved >= checkpoint_every or end == len(texts):
                save_progress(end)
                last_saved = end
    vectors_memmap.flush()
    np.savez(output_path, ids=np.asarray(ids, dtype=np.str_), vectors=np.asarray(vectors_memmap))
    metadata = {
        "model_key": model_key, "model_name": model_name,
        "model_revision": getattr(base_model.config, "_commit_hash", None),
        "kind": kind, "pooling": pooling, "normalized": True,
        "max_length": max_length, "dimension": dimension,
        "count": len(ids), "input_path": str(input_path),
        "input_sha256": input_hash,
        "cache_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "format": "NumPy NPZ: ids (unicode), vectors (float32)",
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    del vectors_memmap
    partial_path.unlink(missing_ok=True)
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["documents", "questions"], required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, help="Defaults: 512 documents; 64 queries")
    parser.add_argument("--device", help="Defaults to CUDA when available, otherwise CPU")
    parser.add_argument("--multi-gpu", action="store_true", help="Use all visible CUDA GPUs for data-parallel inference")
    parser.add_argument("--checkpoint-every", type=int, default=512, help="Flush a resumable checkpoint every N records")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    max_length = args.max_length or (512 if args.kind == "documents" else 64)
    if max_length < 2:
        parser.error("--max-length must be at least 2")
    encode(args.input, args.output, args.kind, args.model, args.batch_size, max_length, args.device, args.multi_gpu, args.checkpoint_every)


if __name__ == "__main__":
    main()
