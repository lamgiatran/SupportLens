"""Compare raw-dialogue API requests with offline prepared retrieval on dev only."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.api import create_app
from supportlens.data import examples
from supportlens.retrieval import BM25Chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("data/processed/docs_qa"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/minilm/tokenizer.json"))
    parser.add_argument("--raw-validation", type=Path, default=Path("data/raw/multidoc2dial/multidoc2dial_dial_validation.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/docs_qa/api_audit.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.bundle / "queries.dev.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = {}
    for domain in sorted({r["domain"] for r in rows}):
        group = sorted((r for r in rows if r["domain"] == domain and r["valid_input"]), key=lambda r: r["query_id"])
        if not group:
            continue
        candidates = [max(group, key=lambda r: r["original_tokens"]), min(group, key=lambda r: r["original_tokens"])] + group
        added = 0
        for row in candidates:
            if row["query_id"] not in selected:
                selected[row["query_id"]] = row
                added += 1
            if added == 5:
                break
    raw = {r["query_id"]: r for r in examples(args.raw_validation) if r["query_id"] in selected}
    if set(raw) != set(selected):
        raise ValueError("dev query IDs do not match the raw validation source")
    retriever = BM25Chunks.from_bundle(args.bundle, args.tokenizer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    details = []
    with TestClient(create_app(retriever, log_path=args.out.with_suffix(".requests.jsonl"))) as client:
        for qid, row in selected.items():
            example = raw[qid]
            query = retriever.prepare_query(example["question"], example["history"])
            if query.text != row["query_text"]:
                raise ValueError("API and prepared query policy differ")
            expected = retriever.search(row["query_text"], k=5)
            response = client.post("/v1/assist", json={"mode": "docs_qa", "question": example["question"],
                                                     "history": example["history"]})
            response.raise_for_status()
            value = response.json()
            if ([e["evidence_id"] for e in value["evidence"]] != [e.evidence_id for e in expected]
                    or any(e["unit"] != "chunk" for e in value["evidence"])
                    or value["answer"] is not None or "history_not_used" in value["reason"]):
                raise ValueError("API response differs from the offline retriever")
            details.append({"query_id": qid, "domain": row["domain"], "truncated": query.truncated,
                            "used_tokens": query.used_tokens, "request_id": value["request_id"]})
    report = {"status": "pass", "scope": "in-process ASGI raw-dialogue smoke; dev only; no generation",
              "artifact_version": retriever.version, "cases": len(details),
              "domains": dict(Counter(r["domain"] for r in details)),
              "truncated_history_cases": sum(r["truncated"] for r in details),
              "all_offline_api_evidence_match": True, "samples": details}
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}))


if __name__ == "__main__":
    main()
