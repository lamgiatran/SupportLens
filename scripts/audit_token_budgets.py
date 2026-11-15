"""Measure query truncation/chunk fit on a labelled train-only smoke sample."""

import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.data import documents, examples
from supportlens.text import JsonTokenizer, build_query, chunk_document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--documents", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("runs/token_budget_audit.json"))
    args = parser.parse_args()
    tokenizer = JsonTokenizer("data/tokenizers/minilm/tokenizer.json")
    queries = [build_query(x["question"], x["history"], tokenizer)
               for x in islice(examples("data/raw/multidoc2dial/multidoc2dial_dial_train.json"), args.queries)]
    chunks = [c for d in islice(documents("data/raw/multidoc2dial/multidoc2dial_doc.json"), args.documents)
              for c in chunk_document(d.domain, d.doc_id, d.title, d.text, tokenizer)]
    assert queries and chunks
    assert all(q.used_tokens <= 256 for q in queries)
    assert all(c.tokens <= 220 for c in chunks)
    report = {"sample": "first_n_train_queries_and_documents_not_a_benchmark",
              "queries": len(queries), "query_truncated": sum(q.truncated for q in queries),
              "query_truncation_rate": sum(q.truncated for q in queries)/len(queries),
              "current_question_truncated": sum(q.current_question_truncated for q in queries),
              "max_query_tokens": max(q.used_tokens for q in queries),
              "chunks": len(chunks), "max_chunk_tokens": max(c.tokens for c in chunks)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
