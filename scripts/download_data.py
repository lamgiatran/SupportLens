"""Run from the repo root. Download data only; never execute remote loaders."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.download import download, extract_multidoc, sha256, verify_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["multidoc2dial", "banking77", "banking77_oos"], default="multidoc2dial")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--sources", type=Path, default=Path("configs/sources.json"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--preloaded", action="store_true", help="Inventory local bytes; label acquisition honestly")
    args = parser.parse_args()
    root = args.raw_dir
    manifest_path = root / args.dataset / "manifest.json"
    config = json.loads(args.sources.read_text(encoding="utf-8"))[args.dataset]
    if args.verify_only:
        verify_manifest(root, json.loads(manifest_path.read_text(encoding="utf-8")))
        print(json.dumps({"status": "verified", "manifest": str(manifest_path)}))
        return
    previous = {}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_manifest(root, old)
        previous = {x["path"]: x["sha256"] for x in old["files"]}
    if args.dataset == "multidoc2dial":
        archive = root / config["path"]
        if not args.preloaded:
            download(config["url"], archive, expected_sha256=config["sha256"])
        if sha256(archive) != config["sha256"]:
            raise ValueError("archive differs from the pinned release; review source before updating checksum")
        outputs = extract_multidoc(archive, root / args.dataset)
        files = [{"path": config["path"], "url": config["url"]}] + [
            {"path": p.relative_to(root).as_posix(), "url": config["url"], "archive_member": p.name}
            for p in outputs]
    else:
        files = config["files"]
        if not args.preloaded:
            for source in files:
                download(source["url"], root / source["path"], expected_sha256=previous.get(source["path"]))
    value = write_manifest(root, files, manifest_path, "preloaded" if args.preloaded else "downloaded")
    print(json.dumps({"status": "ready", "dataset": args.dataset, "files": len(value["files"]),
                      "manifest": str(manifest_path)}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": type(error).__name__, "message": str(error)}, ensure_ascii=True))
        raise SystemExit(2)
