"""Recompute counts from the archived dev responses and claim annotations; no inference."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(rows, method):
    counts = Counter()
    fully_supported = direct = 0
    for row in rows:
        record = row[method]
        counts[record["status"]] += 1
        if record["status"] != "answer":
            continue
        labels = [unit["label"] for unit in record["review_units"]]
        if not labels or set(labels) - {"supported", "wrong_source", "insufficient", "uncited"}:
            raise ValueError(f'Missing or invalid review: {row["query_id"]}')
        supported = all(label == "supported" for label in labels)
        fully_supported += supported
        direct += supported and record["addresses_request"]
    n, a = len(rows), counts["answer"]
    return {"queries": n, "states": dict(counts), "answers": a, "fully_supported": fully_supported,
            "supported_and_responsive": direct, "coverage": a/n,
            "fully_supported_per_answer": fully_supported/a if a else None,
            "supported_and_responsive_per_query": direct/n}


def calculate(data):
    out = {}
    blocked = set(data["prior_50_dialogues"])
    for stage, methods in [("discovery", ("v2",)), ("pilot", ("v2", "quote")),
                           ("fresh", ("v2", "selector"))]:
        rows = data[stage]
        dialogues = {row["dialogue_id"] for row in rows}
        if stage != "discovery":
            if len(dialogues) != len(rows) or dialogues & blocked:
                raise ValueError(f"contaminated sample: {stage}")
            blocked.update(dialogues)
        elif not dialogues <= blocked:
            raise ValueError("discovery rows outside the original 50")
        out[stage] = {method: summarize(rows, method) for method in methods}
    return out


def restore_case(row, chunks):
    sources = {}
    for alias, ref in row["sources"].items():
        body = chunks[ref["chunk"]][:ref["visible_chars"]]
        if hashlib.sha256(body.encode()).hexdigest() != ref["sha256"]:
            raise ValueError(f"Source prefix differs: {ref['chunk']}")
        sources[alias] = body
    case = {**row, "sources": sources}
    sel = row.get("selector", {})
    if sel.get("status") == "answer":
        case["selector"] = {**sel, "answer": sources[sel["source"]][sel["start"]:sel["end"]]}
    return case


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audit", nargs="?", type=Path,
                   default=Path("runs/docs_qa/claim_review_audit.json"))
    p.add_argument("--show", metavar="QUERY_ID", help="Print the archived answer(s) with exact visible sources")
    p.add_argument("--chunks", type=Path, default=Path("data/processed/docs_qa/chunks.jsonl"))
    args = p.parse_args()
    data = json.loads(args.audit.read_text(encoding="utf-8"))
    report = calculate(data)
    if report != data["counts"]:
        raise SystemExit("Stored counts differ from annotations.")
    if args.show:
        row = next((r for stage in ("discovery", "pilot", "fresh") for r in data[stage]
                    if r["query_id"] == args.show), None)
        if row is None:
            p.error("query ID outside the audit")
        chunks = {r["chunk_id"]: r["text"] for r in map(json.loads, args.chunks.read_text(encoding="utf-8").splitlines())}
        print(json.dumps(restore_case(row, chunks), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(report, indent=2))
