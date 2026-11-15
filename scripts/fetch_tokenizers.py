"""Download tokenizer assets only, using revisions and SHA-256 from config."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.download import download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-generator", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path("configs/tokenizers.json").read_text(encoding="utf-8"))
    for key in (["minilm", "qwen"] if args.include_generator else ["minilm"]):
        item = config[key]
        record = download(item["url"], Path("data/tokenizers") / item["path"], item["sha256"])
        print(json.dumps({"tokenizer": key, "sha256": record["sha256"]}))


if __name__ == "__main__":
    main()
