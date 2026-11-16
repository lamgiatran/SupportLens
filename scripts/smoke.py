"""python scripts/smoke.py artifacts/handoff50/draft [--entrypoint module:fn]"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.smoke import run_package


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--entrypoint")
    args = parser.parse_args()
    try:
        result = run_package(args.package, entrypoint=args.entrypoint)
    except Exception as error:
        print(json.dumps({"status": "fail", "error_type": type(error).__name__,
                          "message": str(error)}, ensure_ascii=True))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
