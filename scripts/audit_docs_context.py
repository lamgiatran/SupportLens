"""Audit gold support after top-3/600-Qwen-token packing. No LLM generation."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.docs_eval import load_dev, load_dev_grounding, visible_gold_hit, write_report
from supportlens.generation import make_prompt
from supportlens.retrieval import BM25Chunks
from supportlens.text import JsonTokenizer


def run(bundle, tokenizer, qwen_tokenizer, raw_dir, predictions, out):
    manifest, queries, qrels = load_dev(bundle)
    raw_docs, raw = load_dev_grounding(raw_dir, manifest, queries)
    base = BM25Chunks.from_bundle(bundle, tokenizer)
    chunks = {c.chunk_id: c for c in base.chunks}
    qwen = JsonTokenizer(qwen_tokenizer)
    rows = [json.loads(line) for line in Path(predictions).read_text(encoding="utf-8").splitlines() if line]
    if any(r.get("split") != "dev" or r["query_id"] not in raw for r in rows):
        raise ValueError("require dev-only retrieval predictions")
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    report = {"split": "dev", "evidence_budget_qwen_tokens": 600, "max_passages": 3,
              "qwen_tokenizer_sha256": qwen.fingerprint, "methods": {},
              "heldout_retrieval_run": False, "generation_run": False,
              "definition": ">=50% of any one gold span survives in an evidence body; titles excluded; not all-claims support or answer accuracy"}
    for method, values in groups.items():
        if len(values) != len(queries) or {r["query_id"] for r in values} != {q["query_id"] for q in queries}:
            raise ValueError("require each dev query exactly once per method")
        top3, packed, truncated, overflow = 0, 0, 0, 0
        for row in values:
            gold = set(qrels[row["query_id"]]["strict"])
            if set(row["gold_ids"]) != gold:
                raise ValueError("gold does not match the frozen qrels")
            selected = [chunks[key] for key in row["retrieved"][:3]]
            example = raw[row["query_id"]]
            top3 += int(bool(gold & {c.chunk_id for c in selected}))
            try:
                prompt = make_prompt(example["question"].strip(),
                    [{"evidence_id": c.chunk_id, "text": c.text} for c in selected], qwen)
            except ValueError as error:
                if "exceeds num_ctx" not in str(error):
                    raise
                overflow += 1
                continue
            packed += int(visible_gold_hit(prompt, chunks, example["domain"], example["gold_references"], raw_docs))
            truncated += int(any(prompt.visible_characters.get(c.chunk_id, 0) < len(c.text) for c in selected))
        n = len(values)
        report["methods"][method] = {"queries": n, "strict_hit_at_3": top3 / n,
            "packed_gold_support": packed / n, "gold_support_lost_by_packing": top3 - packed,
            "queries_with_any_evidence_trimmed_or_omitted": truncated, "prompt_overflow_rejected": overflow}
    write_report(out, report)
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("data/processed/docs_qa"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/minilm/tokenizer.json"))
    p.add_argument("--qwen-tokenizer", type=Path, default=Path("data/tokenizers/qwen/tokenizer.json"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/multidoc2dial"))
    p.add_argument("--predictions", type=Path, default=Path("runs/docs_qa/retrieval_predictions.dev.jsonl"))
    p.add_argument("--out", type=Path, default=Path("runs/docs_qa/context_audit.json"))
    a = p.parse_args()
    run(a.bundle, a.tokenizer, a.qwen_tokenizer, a.raw_dir, a.predictions, a.out)
