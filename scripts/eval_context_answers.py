"""Paired, dialogue-disjoint dev evaluation of the local conversational generator."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval_claim_citations import digest, one_per_dialogue
from supportlens.docs_eval import load_dev, load_dev_grounding, prompt_support, write_report
from supportlens.docs_gate import DocsGate
from supportlens.docs_generator import GeneratorConfig, OllamaGenerator
from supportlens.retrieval import BM25Chunks
from supportlens.text import JsonTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=48)
    args = p.parse_args()
    if args.count < 1:
        p.error("count must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    bundle = args.data / "processed/docs_qa"
    raw = args.data / "raw/multidoc2dial"
    manifest, queries, _ = load_dev(bundle, raw_validation=raw / "multidoc2dial_dial_validation.json")
    previous = json.loads(Path("runs/docs_qa/claim_review_audit.json").read_text(encoding="utf-8"))
    excluded = set(previous["prior_50_dialogues"])
    excluded.update(r["dialogue_id"] for stage in ("pilot", "fresh") for r in previous[stage])
    sample = one_per_dialogue(queries, excluded, args.count)
    docs, examples = load_dev_grounding(raw, manifest, sample)
    retriever = BM25Chunks.from_bundle(bundle, args.data / "tokenizers/minilm/tokenizer.json")
    chunks = {c.chunk_id: c for c in retriever.chunks}
    gate = DocsGate(**json.loads(Path("artifacts/docs_gates/R1.json").read_text())["gate"])
    if retriever.version != gate.artifact_version:
        raise ValueError("retrieval gate mismatch")
    tok = JsonTokenizer(args.data / "tokenizers/qwen/tokenizer.json")
    cfg = GeneratorConfig.load("configs/default.json")
    models = {v: OllamaGenerator(replace(cfg, prompt_policy=v), tok, journal_dir=args.out / "live")
              for v in ("v2", "v3")}
    try:
        for gen in models.values():
            gen.start()
            if not gen.ready:
                raise RuntimeError(gen.snapshot())
        digest_ref = json.loads(Path("runs/docs_qa/generation_dev_qwen2.5-3b_v2_n128_metrics.json").read_text()
                                )["generator"]["model_digest"]
        if any(g.model_digest != digest_ref for g in models.values()):
            raise ValueError("expected the same archived local model digest for both arms")
        audit = {"split": "dev", "primary": "all claims supported by their citations AND addresses the request / all",
                 "selection": "one_per_dialogue/citation-pair-v1", "excluded_dialogues": sorted(excluded),
                 "ids": [q["query_id"] for q in sample], "model": cfg.model, "model_digest": digest_ref,
                 "generator_ids": {v: g.version for v, g in models.items()}, "retriever": retriever.version,
                 "gate_threshold": gate.threshold, "bundle_sha256": digest((bundle/"manifest.json").read_bytes()),
                 "tokenizer_sha256": tok.fingerprint, "rows": []}
        write_report(args.out / "selection.json", {k: v for k, v in audit.items() if k != "rows"})
        for i, q in enumerate(sample):
            ev, score = retriever.retrieve(q["query_text"], k=5)
            row = {"id": q["query_id"], "domain": q["domain"], "sources": {}}
            if ev and gate.accepts(score):
                for v in (("v2", "v3") if i % 2 == 0 else ("v3", "v2")):
                    gen = models[v]
                    prompt = gen.build_prompt(q["question"], ev, history=q["history"])
                    sources = {alias: {"id": key, "chars": prompt.visible_characters[key],
                               "sha256": digest(chunks[key].text[:prompt.visible_characters[key]].encode())}
                               for alias, key in prompt.citation_map.items()}
                    if row["sources"] and sources != row["sources"]:
                        raise ValueError("unequal evidence or aliases in paired comparison")
                    row["sources"] = sources
                    res = gen.generate(q["question"], ev, history=q["history"], use_cache=False,
                                       retrieval_version=retriever.version)
                    if res.status in {"timeout", "error", "cancelled"}:
                        raise RuntimeError((q["query_id"], v, res.status, res.error_type))
                    _, hit = prompt_support(prompt, res.citations, chunks, examples[q["query_id"]], docs)
                    row[v] = {"status": res.status, "answer": res.answer if res.status == "answer" else "",
                              "cites_visible_gold": hit, "generate_ms": round(res.generate_ms, 2),
                              "first_display_ms": round(res.ttft_ms, 2) if res.ttft_ms is not None else None}
            else:
                row["v2"] = row["v3"] = {"status": "retrieval_handoff", "answer": "", "cites_visible_gold": False}
            audit["rows"].append(row)
        write_report(args.out / "context_answers_audit.json", audit)
        print(f"Completed {len(sample)} paired dev dialogues.")
    finally:
        for gen in models.values():
            gen.client.close()


if __name__ == "__main__":
    main()
