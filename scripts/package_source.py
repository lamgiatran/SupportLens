"""Package tracked source + a portable Git bundle; never include raw data/venv."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def git(*args):
    return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-banking-cache", action="store_true",
                        help="include the verified, untracked NPZ in this handoff ZIP only")
    parser.add_argument("--include-docs-bundle", action="store_true",
                        help="include the verified passage/qrels bundle and its small tokenizer")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if git("status", "--porcelain").strip():
        raise ValueError("commit source changes before packaging")
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tracked = [x.decode("utf-8") for x in git("ls-files", "-z").split(b"\0") if x]
    oversized = [name for name in tracked if (root / name).stat().st_size > 10 * 1024 * 1024]
    if oversized:
        raise ValueError(f"tracked files exceed the 10 MiB source-package budget: {oversized}")
    forbidden = [name for name in tracked if name.startswith(("data/raw/", "data/tokenizers/", ".venv/"))
                 or name == ".env" or Path(name).suffix in {".npz", ".npy", ".safetensors", ".gguf", ".pt", ".pth"}]
    if forbidden:
        raise ValueError(f"large/private runtime files must not be tracked in the source package: {forbidden}")
    shared_assets = []
    def add_shared(relative: str, expected: str):
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"runtime asset bytes do not match the manifest: {relative}")
        if relative not in tracked:
            shared_assets.append({"path": relative, "sha256": actual, "bytes": path.stat().st_size})
    if args.include_banking_cache:
        cache = root / "data/processed/handoff50/embeddings.npz"
        manifest_path = cache.with_name("embedding_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        add_shared(cache.relative_to(root).as_posix(), manifest["npz_sha256"])
    if args.include_docs_bundle:
        docs = root / "data/processed/docs_qa"
        manifest_path = docs / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_names = {"chunks.jsonl", "splits.json", "qrels_audit.jsonl", "qrels_issues.jsonl"} | {
            f"{kind}.{role}.jsonl" for role in ("train", "dev", "heldout") for kind in ("queries", "qrels")}
        if set(manifest["files_sha256"]) != expected_names:
            raise ValueError("unexpected Docs-QA asset list")
        for name, expected in manifest["files_sha256"].items():
            add_shared("data/processed/docs_qa/" + name, expected)
        add_shared("data/tokenizers/minilm/tokenizer.json", manifest["tokenizer_sha256"])
        add_shared("data/processed/docs_qa/manifest.json", hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    bundle = destination.with_suffix(".bundle")
    subprocess.run(["git", "bundle", "create", str(bundle), "--all"], cwd=root, check=True)
    commit = git("rev-parse", "HEAD").decode().strip()
    tags = git("tag", "--points-at", "HEAD").decode().splitlines()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in tracked:
            archive.write(root / name, "SupportLens/" + name)
        for asset in shared_assets:
            archive.write(root / asset["path"], "SupportLens/" + asset["path"])
        archive.write(bundle, "SupportLens.history.bundle")
        archive.writestr("PACKAGE.json", json.dumps({"commit": commit, "tags": tags,
                                                    "source_files": len(tracked),
                                                    "shared_assets": shared_assets}, indent=2))
        archive.writestr("OPEN_FIRST.txt", "Source: SupportLens/README.md\n"
                          "Docs-QA run steps: SupportLens/docs/RUN_DOCS_QA.md\n"
                          "Team tasks: SupportLens/docs/TEAM_HANDOFF.md and SupportLens/issues/\n"
                          "Optional Git history/tag restore:\n"
                          "git clone SupportLens.history.bundle SupportLens-with-history\n"
                          "No remote GitHub repository has been created.\n"
                          "Raw data, model weights and virtualenv are excluded.\n"
                          "Explicit runtime assets (when included) are listed in PACKAGE.json.\n"
                          "After cloning the bundle, copy runtime assets from the extracted SupportLens folder.\n")
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert not any("/.venv/" in n or "/data/raw/" in n for n in names)
        assert all("/data/tokenizers/" not in n or n == "SupportLens/data/tokenizers/minilm/tokenizer.json" for n in names)
    print(json.dumps({"archive": str(destination), "files": len(names),
                      "commit": commit, "tags": tags, "bytes": destination.stat().st_size}, ensure_ascii=True))


if __name__ == "__main__":
    main()
