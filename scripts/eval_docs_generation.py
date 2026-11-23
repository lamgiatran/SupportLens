"""Dev-only Docs-QA generation check: R1 retrieval -> dev gate -> local Ollama answer.

Measures what the API would publish on a fixed, hash-ordered dev sample:
answer/fallback rates, whether cited passages are strict-gold passages, and TTFT/latency.
Citation-vs-qrels is a proxy, not answer correctness; the predictions file is for manual review.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.docs_eval import load_dev, write_report
from supportlens.docs_gate import DocsGate
from supportlens.docs_generator import GeneratorConfig, OllamaGenerator
from supportlens.retrieval import BM25Chunks
from supportlens.text import JsonTokenizer


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("data/processed/docs_qa"))
    parser.add_argument("--retrieval-tokenizer", type=Path, default=Path("data/tokenizers/minilm/tokenizer.json"))
    parser.add_argument("--generator-tokenizer", type=Path, default=Path("data/tokenizers/qwen/tokenizer.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--gate", type=Path, default=Path("artifacts/docs_gates/R1.json"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--deadline", type=float, default=None)
    parser.add_argument("--model", default=None, help="override the Ollama model, e.g. qwen2.5:3b")
    parser.add_argument("--prompt-policy", default=None, choices=["v1", "v2"])
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/docs_qa/generation_dev.json"))
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")

    config = GeneratorConfig.load(args.config, deadline_seconds=args.deadline, model=args.model,
                                  prompt_policy=args.prompt_policy, num_predict=args.num_predict)
    tag = f"{config.model.replace(':', '-')}_{config.prompt_policy}_n{config.num_predict}"
    if args.out == parser.get_default("out"):
        # One tracked metrics file per configuration so variants on the same dev sample can be compared.
        args.out = args.out.with_name(f"generation_dev_{tag}_metrics.json")
    generator = OllamaGenerator(config, JsonTokenizer(args.generator_tokenizer))
    generator.start()
    report = {"split": "dev", "generator": generator.snapshot(), "config": config.__dict__,
              "metric_scope": "dev sample; citation-vs-strict-qrels is a proxy, not answer correctness"}
    if not generator.ready:
        report["status"] = "unavailable"
        write_report(args.out, report)
        print(json.dumps(report, ensure_ascii=True))
        return 2

    retriever = BM25Chunks.from_bundle(args.bundle, args.retrieval_tokenizer)
    gate = None
    if args.gate.exists():
        gate = DocsGate(**json.loads(args.gate.read_text(encoding="utf-8"))["gate"])
        if gate.artifact_version != retriever.version:
            raise SystemExit("gate artifact does not match the R1 bundle")
    _, queries, qrels = load_dev(args.bundle)
    # Hash order spreads the sample across dialogues instead of taking the first few.
    queries = sorted((q for q in queries if q["valid_input"]),
                     key=lambda q: hashlib.sha256(q["query_id"].encode()).hexdigest())[:args.limit]

    rows = []
    predictions = args.out.with_name(f"generation_predictions_{tag}.dev.jsonl")  # not committed
    with predictions.open("w", encoding="utf-8", newline="\n") as handle:
        for q in queries:
            evidence, score = retriever.retrieve(q["query_text"], k=5)
            gold = set(qrels[q["query_id"]]["strict"])
            shown = [e.evidence_id for e in evidence[:config.evidence_top_k]]
            row = {"query_id": q["query_id"], "question": q["question"], "retrieval_score": score,
                   "gold_in_prompt": bool(gold & set(shown)), "prompt_evidence_ids": shown}
            if not evidence or (gate is not None and not gate.accepts(score)):
                row.update(status="retrieval_handoff", decision="handoff")
            else:
                result = generator.generate(q["question"], evidence, use_cache=False)
                decision, reason = result.decision()
                row.update(status=result.status, decision=decision, reason=reason, answer=result.answer,
                           citations=result.citations, invalid_citations=result.invalid_citations,
                           cited_gold=bool(gold & set(result.citations)),
                           citation_precision=(len(gold & set(result.citations)) / len(result.citations)
                                               if result.citations else None),
                           ttft_ms=result.ttft_ms, generate_ms=result.generate_ms,
                           error_type=result.error_type)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    generated = [r for r in rows if r["status"] != "retrieval_handoff"]
    answers = [r for r in generated if r["status"] == "answer"]
    with_gold = [r for r in answers if r["gold_in_prompt"]]
    ttft = [r["ttft_ms"] for r in generated if r.get("ttft_ms") is not None]
    total = [r["generate_ms"] for r in generated]
    report.update({
        "status": "complete", "retrieval_artifact": retriever.version,
        "gate": None if gate is None else {"method": gate.method, "threshold": gate.threshold},
        "queries": len(rows), "generated": len(generated),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "decision_counts": dict(Counter(r["decision"] for r in rows)),
        "answer_rate_of_generated": len(answers) / len(generated) if generated else None,
        "answers_citing_strict_gold": sum(r["cited_gold"] for r in answers) / len(answers) if answers else None,
        "answers_citing_gold_when_gold_in_prompt": (sum(r["cited_gold"] for r in with_gold) / len(with_gold)
                                                    if with_gold else None),
        "mean_citation_precision_vs_strict_gold": (float(np.mean([r["citation_precision"] for r in answers]))
                                                   if answers else None),
        "ttft_ms": {"p50": percentile(ttft, 50), "p95": percentile(ttft, 95)},
        "generate_ms": {"p50": percentile(total, 50), "p95": percentile(total, 95)},
        "predictions": predictions.as_posix(),
    })
    write_report(args.out, report)
    print(json.dumps({k: report[k] for k in ("status", "queries", "status_counts", "answer_rate_of_generated",
                                              "answers_citing_strict_gold", "ttft_ms", "generate_ms")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
