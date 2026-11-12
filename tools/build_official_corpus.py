"""Build a searchable PubMed abstract corpus aligned to BioASQ document URLs.

This tool deliberately fetches article metadata/abstracts from NCBI E-utilities;
BioASQ gold snippets are retained only as evaluation labels and are never used as
the searchable document text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


PUBMED_ID = re.compile(r"/(\d+)/?$")


def load_questions(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("questions", raw) if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError("Expected a BioASQ list or questions wrapper")
    return records


def extract_pmids(records: list[dict]) -> list[str]:
    ids: set[str] = set()
    for record in records:
        for value in record.get("documents", []) if isinstance(record, dict) else []:
            match = PUBMED_ID.search(str(value))
            if match:
                ids.add(match.group(1))
    return sorted(ids, key=int)


def _text(element: ET.Element | None) -> str:
    return "".join(element.itertext()).strip() if element is not None else ""


def parse_pubmed(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    output: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article.find(".//MedlineCitation/PMID"))
        title = _text(article.find(".//ArticleTitle"))
        abstract_parts: list[str] = []
        for node in article.findall(".//Abstract/AbstractText"):
            label = node.attrib.get("Label", "").strip()
            value = _text(node)
            if value:
                abstract_parts.append(f"{label}: {value}" if label else value)
        abstract = " ".join(abstract_parts)
        if pmid and (title or abstract):
            output.append({"id": pmid, "title": title, "text": abstract, "source": "PubMed"})
    return output


def fetch_batch(pmids: list[str], email: str | None, tool: str) -> list[dict]:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if email:
        params["email"] = email
    if tool:
        params["tool"] = tool
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    encoded = urllib.parse.urlencode(params).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        method="POST",
        headers={"User-Agent": "BioRAG-Hybrid/0.1 academic research", "Content-Type": "application/x-www-form-urlencoded"},
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return parse_pubmed(response.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == 3:
                break
            delay = 2 ** attempt
            print(f"Fetch retry {attempt + 1}/3 after {error}; sleeping {delay}s", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError(f"PubMed batch failed after retries: {last_error}")


def write_split(records: list[dict], path: Path, seed: int) -> None:
    # Stable hash split: reproducible without depending on input ordering.
    buckets = {"train": [], "validation": [], "test": []}
    for record in records:
        digest = hashlib.sha256(f"{seed}:{record.get('id', '')}".encode()).digest()[0] / 255
        split = "train" if digest < 0.8 else "validation" if digest < 0.9 else "test"
        buckets[split].append(record.get("id", ""))
    path.write_text(json.dumps(buckets, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON corpus output")
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.34, help="Seconds between NCBI requests")
    parser.add_argument("--email", help="Recommended by NCBI for responsible use")
    parser.add_argument("--tool", default="biorag_hybrid")
    parser.add_argument("--limit", type=int, help="Development limit; omit for all IDs")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    questions = load_questions(args.questions)
    pmids = extract_pmids(questions)
    if args.limit:
        pmids = pmids[: args.limit]
    print(f"BioASQ questions: {len(questions)}; PubMed IDs to fetch: {len(pmids)}", file=sys.stderr)

    checkpoint = args.output.with_suffix(".partial.json")
    documents: dict[str, dict] = {}
    if checkpoint.exists():
        documents = {item["id"]: item for item in json.loads(checkpoint.read_text(encoding="utf-8"))}
        pmids = [pmid for pmid in pmids if pmid not in documents]
        print(f"Resuming from checkpoint: {len(documents)} documents already fetched", file=sys.stderr)
    for start in range(0, len(pmids), args.batch_size):
        batch = pmids[start : start + args.batch_size]
        for document in fetch_batch(batch, args.email, args.tool):
            documents[document["id"]] = document
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps(sorted(documents.values(), key=lambda item: int(item["id"]))), encoding="utf-8")
        done = min(start + len(batch), len(pmids))
        print(f"Fetched {done}/{len(pmids)}; abstracts retained: {len(documents)}", file=sys.stderr)
        if done < len(pmids):
            time.sleep(max(0.0, args.sleep))

    result = sorted(documents.values(), key=lambda item: int(item["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    write_split(questions, args.split_output, args.seed)
    metadata = {
        "question_file": str(args.questions),
        "question_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "requested_pubmed_ids": len(pmids),
        "retrieved_documents": len(result),
        "corpus_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "source": "NCBI PubMed E-utilities efetch",
        "gold_snippets_indexed": False,
        "split_seed": args.seed,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
