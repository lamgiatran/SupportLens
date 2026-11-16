"""Encode the shared passage corpus once. No query labels or fine-tuning."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.dense import DenseEncoder, DenseEncoderConfig, build_corpus_cache
from supportlens.retrieval import BM25Chunks


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("data/processed/docs_qa"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/minilm/tokenizer.json"))
    p.add_argument("--encoder-config", type=Path, default=Path("configs/docs_encoder.json"))
    p.add_argument("--out", type=Path, default=Path("data/processed/docs_dense"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-download", action="store_true", help="allow downloading pinned model weights if absent")
    a = p.parse_args()
    base = BM25Chunks.from_bundle(a.bundle, a.tokenizer)
    encoder = DenseEncoder(DenseEncoderConfig.load(a.encoder_config), device=a.device,
                           local_files_only=not a.allow_download)
    reused = (a.out / "embedding_manifest.json").exists()
    # A shipped corpus cache is not shipped model weights. This command also
    # prepares/verifies the live encoder even when no re-encoding is necessary.
    encoder.warmup(base.tokenizer)
    manifest = build_corpus_cache(base, encoder, a.out, batch_size=a.batch_size)
    print(json.dumps({"cache_reused": reused, "encoder_warmup_ms_this_run": encoder.warmup_ms,
                      "manifest": manifest}))
