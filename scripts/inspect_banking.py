"""Source audit only; no model or OOS threshold is fitted here."""

import csv
import json
from pathlib import Path


def main():
    base = Path("data/raw/banking77")
    rows = {s: list(csv.DictReader((base / (s + ".csv")).open(encoding="utf-8"))) for s in ("train", "test")}
    report = {"banking77": {s: {"samples": len(v), "labels": len({r["category"] for r in v})}
                             for s, v in rows.items()}, "banking77_oos": {}}
    train_text = {r["text"].strip().casefold() for r in rows["train"]}
    test_text = {r["text"].strip().casefold() for r in rows["test"]}
    report["banking77"]["exact_normalized_train_test_overlap"] = len(train_text & test_text)
    root = Path("data/raw/banking77_oos")
    for group in ("", "id-oos", "ood-oos"):
        for split in ("train", "valid", "test"):
            folder = root / group / split
            if not (folder / "seq.in").exists():
                continue
            text = (folder / "seq.in").read_text(encoding="utf-8").splitlines()
            labels = (folder / "label").read_text(encoding="utf-8").splitlines()
            assert len(text) == len(labels)
            item = {"samples": len(text), "labels": len(set(labels))}
            if (folder / "label_original").exists():
                original = (folder / "label_original").read_text(encoding="utf-8").splitlines()
                assert len(original) == len(text)
                item["original_intents"] = len(set(original))
            report["banking77_oos"][(group or "in_scope") + "/" + split] = item
    out = Path("runs/banking_inspection.json")
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
