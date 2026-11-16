"""Evaluate R1/R2/R3 on DEV only; report hit@3 and hit@5 on identical queries."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.dense import DenseEncoder, DenseEncoderConfig, DenseChunks, HybridChunks, load_corpus_cache, validate_vectors
from supportlens.docs_eval import load_dev, query_metrics, aggregate_metrics, paired_dialogue_bootstrap, write_report
from supportlens.retrieval import BM25Chunks, words


def run(bundle, tokenizer, config, cache, out, *, split="dev", limit=None, local_files_only=True, device="cpu"):
    if split != "dev":
        raise ValueError("model selection is dev-only; held-out has no entry point here")
    manifest, queries, qrels = load_dev(bundle)
    queries = [q for q in queries if q["valid_input"]]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        queries = queries[:limit]
    base = BM25Chunks.from_bundle(bundle, tokenizer)
    encoder = DenseEncoder(DenseEncoderConfig.load(config), device=device, local_files_only=local_files_only)
    matrix, cache_manifest = load_corpus_cache(base, encoder, cache)
    dense, output = DenseChunks(base, encoder, matrix), Path(out)
    hybrid = HybridChunks(base, dense)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder.warmup(base.tokenizer)
    # These are query vectors, not corpus vectors. Their identity includes every packed query.
    query_identity = hashlib.sha256(json.dumps([(q["query_id"], q["query_text"]) for q in queries],
                                                ensure_ascii=False).encode()).hexdigest()
    vector_path = output.with_name(f"dev_queries-{encoder.config.fingerprint[:12]}-{query_identity[:12]}.npz")
    vector_manifest = vector_path.with_suffix(".json")
    encoded_seconds, replay = 0.0, vector_path.exists()
    if replay:
        meta = json.loads(vector_manifest.read_text(encoding="utf-8"))
        if (meta["split"] != "dev" or meta["encoder_fingerprint"] != encoder.config.fingerprint
                or meta["npz_sha256"] != hashlib.sha256(vector_path.read_bytes()).hexdigest()):
            raise ValueError("query vector cache bytes changed")
        with np.load(vector_path, allow_pickle=False) as values:
            vectors, ids = values["embeddings"], values["query_ids"].tolist()
        if ids != [q["query_id"] for q in queries] or meta["query_identity"] != query_identity:
            raise ValueError("query vector row identity mismatch")
    else:
        started = perf_counter()
        vectors = encoder.encode([q["query_text"] for q in queries])
        encoded_seconds = perf_counter() - started
        np.savez_compressed(vector_path, embeddings=vectors, query_ids=np.asarray([q["query_id"] for q in queries]))
        write_report(vector_manifest, {"split": "dev", "query_identity": query_identity,
                     "encoder_fingerprint": encoder.config.fingerprint, "artifact_version": dense.version,
                     "npz_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
                     "encode_seconds": encoded_seconds})
    validate_vectors(vectors, len(queries), encoder.config.dimension)
    all_values = {m: {r: [] for r in ("strict", "lenient")} for m in ("R1", "R2", "R3")}
    timings = {m: [] for m in all_values}
    records_path = output.with_name("retrieval_predictions.dev.jsonl")
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row, vector in zip(queries, vectors):
            for method, retriever in [("R1", base), ("R2", dense), ("R3", hybrid)]:
                started = perf_counter()
                if method == "R1":
                    evidence = base.search(row["query_text"], k=10)
                    score = evidence[0].score / max(1, len(words(row["query_text"])))
                elif method == "R2":
                    evidence = dense.search_encoded(vector, k=10)
                    score = evidence[0].score
                else:
                    evidence = hybrid.search_encoded(row["query_text"], vector, k=10)
                    score = hybrid.confidence_encoded(evidence, vector)
                timings[method].append((perf_counter() - started) * 1000)
                ids = [e.evidence_id for e in evidence]
                labels = qrels[row["query_id"]]
                for relevance in all_values[method]:
                    all_values[method][relevance].append(query_metrics(ids, labels[relevance]))
                handle.write(json.dumps({"split": "dev", "method": method, "query_id": row["query_id"],
                    "dialogue_id": row["dialogue_id"], "artifact_version": retriever.version,
                    "score_name": retriever.score_name, "gate_score": score, "retrieved": ids,
                    "gold_ids": labels["strict"], "nterms": len(words(row["query_text"])),
                    "raw_top1": evidence[0].score, "margin": evidence[0].score - evidence[1].score,
                    **({"top1_per_term": score} if method == "R1" else {})}, separators=(",", ":")) + "\n")
    methods = {}
    for method, retriever in [("R1", base), ("R2", dense), ("R3", hybrid)]:
        methods[method] = {"artifact_version": retriever.version,
            "metrics": {r: aggregate_metrics(v) for r, v in all_values[method].items()},
            "lookup_ms_preencoded_query": {"p50": float(np.percentile(timings[method], 50)),
                                          "p95": float(np.percentile(timings[method], 95))}}
    comparisons = {}
    for method in ("R2", "R3"):
        for metric in ("hit_at_3", "hit_at_5", "mrr_at_10"):
            valid = [i for i in range(len(queries)) if all_values[method]["strict"][i] is not None
                     and all_values["R1"]["strict"][i] is not None]
            comparisons[f"{method}_minus_R1_{metric}"] = paired_dialogue_bootstrap(
                [queries[i]["dialogue_id"] for i in valid],
                [all_values[method]["strict"][i][metric] for i in valid],
                [all_values["R1"]["strict"][i][metric] for i in valid])
    report = {"split": "dev", "scope": "smoke subset" if limit else "full dev",
        "queries": len(queries), "history_turns": base.history_turns, "methods": methods,
        "comparisons": comparisons, "corpus_cache_id": cache_manifest["cache_id"],
        "query_vectors": {"file": vector_path.name, "replayed": replay, "encode_seconds_this_run": encoded_seconds},
        "cold_encoder_load_and_warmup_ms": encoder.warmup_ms,
        "latency_caveat": "shared development machine; lookup excludes query encode; no UI/generation latency claim",
        "hit3_caveat": "candidate coverage only; the 600-Qwen-token prompt may trim the third passage",
        "heldout_retrieval_run": False, "generation_run": False, "training_run": False,
        "input_hashes": {"queries_dev": manifest["files_sha256"]["queries.dev.jsonl"],
                         "qrels_dev": manifest["files_sha256"]["qrels.dev.jsonl"]},
        "predictions_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest()}
    write_report(output, report)
    print(json.dumps({"split": "dev", "queries": len(queries), "methods": methods}), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("data/processed/docs_qa"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/minilm/tokenizer.json"))
    p.add_argument("--encoder-config", type=Path, default=Path("configs/docs_encoder.json"))
    p.add_argument("--cache", type=Path, default=Path("data/processed/docs_dense"))
    p.add_argument("--out", type=Path, default=Path("runs/docs_qa/retrieval_metrics.json"))
    p.add_argument("--split", choices=["dev"], default="dev")
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-download", action="store_true")
    a = p.parse_args()
    run(a.bundle, a.tokenizer, a.encoder_config, a.cache, a.out, split=a.split, limit=a.limit,
        local_files_only=not a.allow_download, device=a.device)
