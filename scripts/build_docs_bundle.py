"""Build token-bounded passages plus strict/lenient qrels on raw MultiDoc2Dial."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.docs_pipeline import build_docs_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/multidoc2dial"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/minilm/tokenizer.json"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/docs_qa"))
    parser.add_argument("--config", type=Path, default=Path("configs/docs_qa.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_docs_bundle(args.raw_dir, args.tokenizer, args.out, config=config)
    print(json.dumps({k: v for k, v in report.items() if k != "files_sha256"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
