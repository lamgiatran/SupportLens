"""Compare R6 to the recorded R1/R3 rankings on every dev turn and calibrate its own gate."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.docs_eval import (aggregate_metrics, load_dev, load_dev_grounding, paired_dialogue_bootstrap,
                                  query_metrics, visible_gold_hit, write_report)
from supportlens.docs_gate import select_coverage_gate
from supportlens.docs_generator import GeneratorConfig, OllamaGenerator
from supportlens.query_memory import QueryMemory, _key
from supportlens.passage_ranker import PassageRanker
from supportlens.retrieval import BM25Chunks
from supportlens.text import JsonTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--baseline", type=Path, default=Path("runs/docs_qa/retrieval_predictions.dev.jsonl"))
    p.add_argument("--baseline-metrics", type=Path, default=Path("runs/docs_qa/retrieval_metrics.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ranker", type=Path, help="evaluate a frozen R7 JSON instead of fitting R6")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bundle = args.data / "processed/docs_qa"
    manifest, queries, qrels = load_dev(bundle)
    docs, examples = load_dev_grounding(args.data/"raw/multidoc2dial", manifest, queries)
    baseline = args.baseline.read_bytes()
    expected = json.loads(args.baseline_metrics.read_bytes())["predictions_sha256"]
    if hashlib.sha256(baseline).hexdigest() != expected:
        raise ValueError("expected the archived R1/R2/R3 dev predictions")
    predictions = {m: {} for m in ("R1", "R3")}
    for row in map(json.loads, baseline.splitlines()):
        if row["method"] in predictions:
            predictions[row["method"]][row["query_id"]] = row["retrieved"]
    if any(by_id.keys() != qrels.keys() for by_id in predictions.values()):
        raise ValueError("baseline dev IDs differ")
    base = BM25Chunks.from_bundle(bundle, args.data / "tokenizers/minilm/tokenizer.json")
    started = perf_counter()
    memory = PassageRanker.load(base, args.ranker) if args.ranker else QueryMemory.from_bundle(base, bundle)
    method = memory.method
    chunk_map = {c.chunk_id: c for c in base.chunks}
    fit_seconds = perf_counter() - started
    gen = OllamaGenerator(GeneratorConfig.load("configs/default.json"),
                          JsonTokenizer(args.data/"tokenizers/qwen/tokenizer.json"))
    methods = ("R1", "R3", method)
    scores, timings, rows = [], [], []
    try:
        for q in queries:
            qid = q["query_id"]
            start = perf_counter()
            evidence, score = memory.retrieve(q["query_text"], 10)
            timings.append(1000 * (perf_counter()-start))
            scores.append(score)
            r = {"query_id": qid, "dialogue": q["dialogue_id"], "domain": q["domain"]}
            if not args.ranker:
                r["seen_current"] = _key(q["question"]) in memory.seen_questions
            for m in methods:
                keys = [e.evidence_id for e in evidence] if m == method else predictions[m][qid]
                sm = query_metrics(keys, qrels[qid]["strict"])
                if sm is None:
                    raise ValueError("expected a strict positive for every recorded dev query")
                packed = gen.build_prompt(q["question"], [{"evidence_id": key, "text": chunk_map[key].text}
                                                          for key in keys[:3]])
                vis = visible_gold_hit(packed, chunk_map, q["domain"], examples[qid]["gold_references"], docs)
                r[m] = {**sm, "visible": float(vis), "lenient_hit_at_3":
                        float(bool(set(keys[:3]).intersection(qrels[qid]["lenient"])))}
            r["top10"], r["score"] = [e.evidence_id for e in evidence], score
            rows.append(r)
    finally:
        gen.client.close()
    errors = [not r[method]["hit_at_3"] for r in rows]
    scored = [(s, err) for s, err in zip(scores, errors) if s is not None]
    gate = select_coverage_gate([s for s, _ in scored], [e for _, e in scored], method=method, artifact_version=memory.version, split="dev",
                                score_name=memory.score_name, target_coverage=0.70)
    accepted = [err for s, err in scored if gate.accepts(s)]
    gate_report = {"gate": asdict(gate), "queries": len(rows), "accepted": len(accepted),
                   "coverage": len(accepted)/len(rows), "risk_gold_missing_top3_given_accept": sum(accepted)/len(accepted)}
    write_report(args.out/(method+".json"), gate_report)
    def table(sample):
        return {m: aggregate_metrics([r[m] for r in sample]) for m in methods}
    file = args.out/"predictions.dev.jsonl"
    with file.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    report = {"split": "dev", "version": memory.version,
              "baseline_sha256": expected, "predictions_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
              "tokenizer": gen.tokenizer.fingerprint, "dev_queries": len(rows), "dialogues": len({r["dialogue"] for r in rows}),
              "methods": table(rows), "by_domain": {d: table([r for r in rows if r["domain"] == d])
                                                      for d in sorted({r["domain"] for r in rows})},
              f"paired_{method.lower()}_minus_r3": {k: paired_dialogue_bootstrap([r["dialogue"] for r in rows],
                      [r[method][k] for r in rows], [r["R3"][k] for r in rows]) for k in ("hit_at_3", "visible")},
              f"{method.lower()}_retrieval_top10_p95_ms": float(np.percentile(timings, 95)), "gate": gate_report}
    if args.ranker:
        report["model_sha256"] = hashlib.sha256(args.ranker.read_bytes()).hexdigest()
    else:
        report.update(training_queries=len(memory.rows), train_dialogues=len({r[1] for r in memory.rows}),
                      training_seconds=fit_seconds, unseen_current_question=table([r for r in rows if not r["seen_current"]]))
    write_report(args.out/("passage_ranker_metrics.json" if args.ranker else "query_memory_metrics.json"), report)
    print(json.dumps({m: report["methods"][m]["hit_at_3"] for m in methods}))


if __name__ == "__main__":
    main()
