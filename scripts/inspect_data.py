"""Read raw JSON and report measured counts, references and split overlap."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.data import documents, examples, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/multidoc2dial"))
    parser.add_argument("--out", type=Path, default=Path("runs/data_inspection.json"))
    parser.add_argument("--export-documents", type=Path, default=Path("data/processed/documents.jsonl"))
    args = parser.parse_args()
    docs = list(documents(args.raw_dir / "multidoc2dial_doc.json"))
    stats = {"documents": len(docs), "domains": dict(Counter(d.domain for d in docs)), "splits": {}}
    ids = {}
    for split in ("train", "validation"):
        file = args.raw_dir / f"multidoc2dial_dial_{split}.json"
        raw = json.loads(file.read_text(encoding="utf-8"))["dial_data"]
        ids[split] = {d["dial_id"] for group in raw.values() for d in group}
        rows = list(examples(file))
        stats["splits"][split] = {"dialogues": len(ids[split]), "query_target_pairs": len(rows),
                                  "queries_with_references": sum(bool(x["gold_references"]) for x in rows)}
    stats["train_validation_dialogue_overlap"] = len(ids["train"] & ids["validation"])
    if stats["train_validation_dialogue_overlap"]:
        raise ValueError("train/validation dialogue IDs overlap; resolve before training")
    write_jsonl(args.export_documents, docs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
