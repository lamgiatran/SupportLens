"""Run the local Docs-QA API with R3 and R7, using the same checked v2 generator."""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval_claim_citations import digest, one_per_dialogue
from supportlens.api import create_app
from supportlens.dense import DenseChunks, DenseEncoder, DenseEncoderConfig, HybridChunks, load_corpus_cache
from supportlens.docs_eval import load_dev, load_dev_grounding, prompt_support, write_report
from supportlens.docs_gate import DocsGate
from supportlens.docs_generator import GeneratorConfig, OllamaGenerator
from supportlens.passage_ranker import PassageRanker
from supportlens.retrieval import BM25Chunks
from supportlens.text import JsonTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("artifacts/docs_ranker/model.json"))
    p.add_argument("--gate", type=Path, default=Path("artifacts/docs_gates/R7.json"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data, bundle = args.data, args.data/"processed/docs_qa"
    raw = data/"raw/multidoc2dial"
    manifest, queries, _ = load_dev(bundle, raw_validation=raw/"multidoc2dial_dial_validation.json")
    prev = json.loads(Path("runs/docs_qa/context_answers_audit.json").read_bytes())
    excluded = set(prev["excluded_dialogues"]) | {k.rsplit(":", 1)[0] for k in prev["ids"]}
    sample = one_per_dialogue(queries, excluded, 32)
    docs, examples = load_dev_grounding(raw, manifest, sample)
    base = BM25Chunks.from_bundle(bundle, data/"tokenizers/minilm/tokenizer.json")
    chunks = {c.chunk_id: c for c in base.chunks}
    encoder = DenseEncoder(DenseEncoderConfig.load("configs/docs_encoder.json"), local_files_only=True)
    matrix, _ = load_corpus_cache(base, encoder, data/"processed/docs_dense")
    registry = {"R3": HybridChunks(base, DenseChunks(base, encoder, matrix)), "R7": PassageRanker.load(base, args.model)}
    gates = {m: DocsGate(**json.loads(path.read_bytes())["gate"]) for m, path in
             [("R3", Path("artifacts/docs_gates/R3.json")), ("R7", args.gate)]}
    config = replace(GeneratorConfig.load("configs/default.json"), prompt_policy="v2_checked", cache_size=0)
    gen = OllamaGenerator(config, JsonTokenizer(data/"tokenizers/qwen/tokenizer.json"), journal_dir=args.out/"live")
    api = create_app(retrievers=registry, docs_gates=gates, generator=gen, log_path=None)
    audit = {"split": "dev", "arms": list(registry), "ids": [q["query_id"] for q in sample],
             "excluded_dialogues": sorted(excluded), "bundle_sha256": digest((bundle/"manifest.json").read_bytes()),
             "generator": gen.version, "model": config.model, "rows": []}
    write_report(args.out/"selection.json", {k: v for k, v in audit.items() if k != "rows"})
    try:
        with TestClient(api) as client:
            if not client.get("/health").json()["docs_generator_ready"] or gen.model_digest != prev["model_digest"]:
                raise RuntimeError("expected the pinned local GGUF")
            audit["model_digest"] = gen.model_digest
            for i, q in enumerate(sample):
                row = {"id": q["query_id"], "domain": q["domain"]}
                for m in (("R3", "R7") if i % 2 == 0 else ("R7", "R3")):
                    response = client.post("/v1/assist", json={"question": q["question"], "history": q["history"],
                                                               "mode": "docs_qa", "retriever": m})
                    response.raise_for_status()
                    result = response.json()
                    if ("not_ready" in result["reason"] or "not_loaded" in result["reason"] or
                            any(x in result["reason"] for x in ("timeout", "generator_error", "unavailable"))):
                        raise RuntimeError(result["reason"])
                    prompt = gen.build_prompt(q["question"], result["evidence"])
                    _, hit = prompt_support(prompt, result["citations"], chunks, examples[q["query_id"]], docs)
                    row[m] = {"status": result["decision"], "reason": result["reason"], "answer": result["answer"] or "",
                              "sources": {a: {"id": k, "chars": prompt.visible_characters[k],
                                          "sha256": digest(chunks[k].text[:prompt.visible_characters[k]].encode())}
                                          for a, k in prompt.citation_map.items()},
                              "cites_visible_gold": hit, "total_ms": result["latency"]["total_ms"],
                              "first_display_ms": result["latency"]["ttft_ms"]}
                audit["rows"].append(row)
    finally:
        gen.client.close()
    audit["metrics"] = {m: {"decisions": dict(Counter(r[m]["status"] for r in audit["rows"])),
                            "answers_citing_visible_gold": sum(r[m]["cites_visible_gold"] for r in audit["rows"]),
                            "request_p95_ms": float(np.percentile([r[m]["total_ms"] for r in audit["rows"]], 95))}
                        for m in registry}
    write_report(args.out/"served_answers_audit.json", audit)
    print(json.dumps(audit["metrics"]))


if __name__ == "__main__":
    main()
