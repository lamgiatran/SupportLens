"""Frozen R2 and RRF R3. Corpus embeddings are independent of history policy."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np

from .contracts import Evidence
from .retrieval import BM25Chunks, deduplicate_chunks, rrf


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class DenseEncoderConfig:
    model_id: str
    revision: str
    max_seq_length: int = 256
    dimension: int = 384
    normalize_embeddings: bool = True
    text_policy: str = "strip_supplied_text_v1"

    def __post_init__(self):
        if (len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision)
                or self.max_seq_length < 3 or self.dimension < 1 or not self.normalize_embeddings
                or self.text_policy != "strip_supplied_text_v1"):
            raise ValueError("pin an encoder commit, normalized vectors and explicit text policy")

    @property
    def fingerprint(self):
        return fingerprint(asdict(self))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class DenseEncoder:
    """One encode() for Docs cache + serving, independent of Cau's Banking code.

    This MiniLM baseline adds no separate query/document instruction prompt.
    """
    def __init__(self, config: DenseEncoderConfig, *, device="cpu", backend=None,
                 local_files_only=True):
        self.config, self.device = config, device
        self._backend, self.local_files_only = backend, local_files_only
        self._load_lock = Lock()
        self.ready = False
        self.warmup_ms = None

    def _load(self):
        with self._load_lock:
            if self._backend is None:
                from sentence_transformers import SentenceTransformer
                self._backend = SentenceTransformer(self.config.model_id, revision=self.config.revision,
                    device=self.device, trust_remote_code=False, local_files_only=self.local_files_only)
            self._backend.max_seq_length = self.config.max_seq_length
            return self._backend

    def encode(self, texts, *, batch_size=32):
        if isinstance(texts, str) or batch_size < 1:
            raise ValueError("pass a sequence of strings and positive batch size")
        values = list(texts)
        if any(not isinstance(t, str) or not t.strip() for t in values):
            raise ValueError("encoder inputs must be non-blank strings")
        if not values:
            return np.empty((0, self.config.dimension), dtype=np.float32)
        matrix = np.asarray(self._load().encode([t.strip() for t in values], batch_size=batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
        validate_vectors(matrix, len(values), self.config.dimension)
        return matrix

    def warmup(self, tokenizer=None):
        if self.ready:
            return self.warmup_ms
        started = perf_counter()
        self.encode(["SupportLens encoder warmup"])
        if tokenizer is not None and hasattr(self._backend, "tokenizer"):
            for value in ["A current question?", "[2] address\nprevious reply", "Unusual punctuation: $20."]:
                actual = self._backend.tokenizer(value, add_special_tokens=True)["input_ids"]
                expected = tokenizer.backend.encode(value, add_special_tokens=True).ids
                if actual != expected:
                    raise ValueError("live encoder tokenizer differs from passage/query tokenizer")
        self.warmup_ms = (perf_counter() - started) * 1000
        self.ready = True
        return self.warmup_ms


def validate_vectors(matrix, rows, dimension):
    matrix = np.asarray(matrix)
    if (matrix.dtype != np.float32 or matrix.shape != (rows, dimension)
            or not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4)):
        raise ValueError("finite L2-normalized float32 embeddings with aligned dimensions required")


def corpus_identity(base, config):
    # Only encoder + corpus bytes/tokenizer, never query history or dev labels.
    return {"chunks": fingerprint([asdict(c) for c in base.chunks]),
            "encoder": asdict(config), "tokenizer_sha256": base.tokenizer.fingerprint}


def build_corpus_cache(base, encoder, destination, *, batch_size=32):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    identity = corpus_identity(base, encoder.config)
    manifest_path = destination / "embedding_manifest.json"
    if manifest_path.exists():
        # A mismatching existing cache must be rebuilt at an explicit new path.
        return load_corpus_cache(base, encoder, destination)[1]
    encoder.warmup(base.tokenizer)
    started = perf_counter()
    matrix = encoder.encode([c.text for c in base.chunks], batch_size=batch_size)
    validate_vectors(matrix, len(base.chunks), encoder.config.dimension)
    cache_path = destination / "corpus.npz"
    np.savez_compressed(cache_path, embeddings=matrix,
                        chunk_ids=np.asarray([c.chunk_id for c in base.chunks], dtype=str))
    manifest = {"identity": identity, "cache_id": fingerprint(identity),
                "npz_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                "rows": len(base.chunks), "dimension": encoder.config.dimension,
                "encode_seconds": perf_counter() - started, "warmup_ms": encoder.warmup_ms,
                "training_run": False, "contains": "corpus only; no query vectors or labels"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def load_corpus_cache(base, encoder, directory):
    directory = Path(directory)
    manifest = json.loads((directory / "embedding_manifest.json").read_text(encoding="utf-8"))
    identity = corpus_identity(base, encoder.config)
    if manifest["identity"] != identity or manifest["cache_id"] != fingerprint(identity):
        raise ValueError("dense cache encoder/corpus identity mismatch")
    payload = directory / "corpus.npz"
    if hashlib.sha256(payload.read_bytes()).hexdigest() != manifest["npz_sha256"]:
        raise ValueError("dense cache bytes changed")
    with np.load(payload, allow_pickle=False) as cache:
        ids, matrix = cache["chunk_ids"], cache["embeddings"]
    if ids.tolist() != [c.chunk_id for c in base.chunks]:
        raise ValueError("dense cache chunk row order mismatch")
    validate_vectors(matrix, len(base.chunks), encoder.config.dimension)
    if hashlib.sha256(matrix.tobytes()).hexdigest() != manifest["matrix_sha256"]:
        raise ValueError("dense matrix digest mismatch")
    matrix.setflags(write=False)
    return matrix, manifest


class DenseChunks:
    method, score_name = "R2", "dense_top1_cosine_v1"

    def __init__(self, base: BM25Chunks, encoder: DenseEncoder, matrix):
        validate_vectors(matrix, len(base.chunks), encoder.config.dimension)
        if (base.query_max_tokens > encoder.config.max_seq_length
                or any(c.tokens > encoder.config.max_seq_length for c in base.chunks)):
            raise ValueError("query/chunk exceeds encoder budget")
        self.base, self.encoder, self.matrix = base, encoder, matrix
        self.chunks, self.tokenizer = base.chunks, base.tokenizer
        self.query_max_tokens, self.history_turns = base.query_max_tokens, base.history_turns
        self.by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self.version = "R2-chunk-" + fingerprint({"base": base.version, "encoder": encoder.config.fingerprint,
            "matrix": hashlib.sha256(matrix.tobytes()).hexdigest()})[:16]

    def prepare_query(self, question, history):
        return self.base.prepare_query(question, history)

    def scores_encoded(self, vector):
        validate_vectors(np.asarray(vector)[None, :], 1, self.encoder.config.dimension)
        return self.matrix @ vector

    def search_encoded(self, vector, k=5):
        scores = self.scores_encoded(vector)
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id))
        selected = deduplicate_chunks([(self.chunks[i], float(scores[i])) for i in order], k, self.base.dedup_overlap)
        return [Evidence(evidence_id=c.chunk_id, doc_id=c.doc_id, domain=c.domain, title=c.title,
                         text=c.text, score=s, unit="chunk") for c, s in selected]

    def search(self, query_text, k=5):
        return self.search_encoded(self.encoder.encode([query_text])[0], k)

    def retrieve(self, query_text, k=5):
        evidence = self.search(query_text, k)
        return evidence, evidence[0].score if evidence else None


class HybridChunks:
    method, score_name = "R3", "max_dense_cosine_of_hybrid_top3_v1"

    def __init__(self, base: BM25Chunks, dense: DenseChunks, *, depth=50, rrf_k=60):
        if base.version != dense.base.version or depth < 10 or rrf_k < 1:
            raise ValueError("hybrid needs identical chunks/query policy and valid RRF settings")
        self.base, self.dense, self.depth, self.rrf_k = base, dense, depth, rrf_k
        self.chunks, self.tokenizer = base.chunks, base.tokenizer
        self.version = "R3-chunk-" + fingerprint({"bm25": base.version, "dense": dense.version,
            "depth": depth, "rrf_k": rrf_k, "dedup_overlap": base.dedup_overlap})[:16]

    def prepare_query(self, question, history):
        return self.base.prepare_query(question, history)

    def search_encoded(self, query_text, vector, k=5):
        if not 1 <= k <= self.depth:
            raise ValueError("hybrid k must be within candidate depth")
        left, right = self.base.search(query_text, self.depth), self.dense.search_encoded(vector, self.depth)
        rankings = [[e.evidence_id for e in values] for values in (left, right)]
        merged = rrf(rankings, k=self.rrf_k)
        selected = deduplicate_chunks([(self.chunks[self.dense.by_id[key]], score) for key, score in merged],
                                       k, self.base.dedup_overlap)
        return [Evidence(evidence_id=c.chunk_id, doc_id=c.doc_id, domain=c.domain, title=c.title,
                         text=c.text, score=s, unit="chunk") for c, s in selected]

    def confidence_encoded(self, evidence, vector):
        # Confidence is dense support for the actual top-3 hybrid results, never RRF magnitude.
        scores = self.dense.scores_encoded(vector)
        return max((float(scores[self.dense.by_id[e.evidence_id]]) for e in evidence[:3]), default=None)

    def retrieve(self, query_text, k=5):
        vector = self.dense.encoder.encode([query_text])[0]
        evidence = self.search_encoded(query_text, vector, k)
        return evidence, self.confidence_encoded(evidence, vector)

    def search(self, query_text, k=5):
        return self.retrieve(query_text, k)[0]
