"""R6 retrieves published passages via similar, grounded questions from public train."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import Evidence
from .retrieval import deduplicate_chunks, rrf, words


def _key(text):
    return " ".join(words(text))


def _weighted_query(text):
    return text.split("\n", 1)[0] + "\n" + text


def training_bundle(base, directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    if manifest["artifact_version"] != base.version:
        raise ValueError("training data and base retriever differ")

    def verified(name):
        payload = (directory / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["files_sha256"][name]:
            raise ValueError(f"training bundle checksum mismatch: {name}")
        return payload

    splits = json.loads(verified("splits.json"))["dialogues"]
    queries, qrels = (list(map(json.loads, verified(name).splitlines()))
                      for name in ("queries.train.jsonl", "qrels.train.jsonl"))
    return queries, qrels, splits


class QueryMemory:
    method = "R6"
    score_name = "max_neighbor_cosine_supporting_top3_v1"

    def __init__(self, base, queries, qrels, splits):
        train = set(splits["train"])
        if train.intersection(splits["dev"]) or train.intersection(splits["heldout"]):
            raise ValueError("training dialogues overlap evaluation dialogues")
        self.base, self.tokenizer = base, base.tokenizer
        self.chunks = {c.chunk_id: c for c in base.chunks}
        gold = {r["query_id"]: tuple(sorted(set(r["strict"]))) for r in qrels}
        keys = [q["query_id"] for q in queries]
        if len(gold) != len(qrels) or len(set(keys)) != len(keys) or set(keys) != gold.keys():
            raise ValueError("duplicate or missing query/qrels ID")
        texts, self.rows = [], []
        self.seen_questions = set()
        for q in queries:
            qid = q["query_id"]
            if q["dialogue_id"] not in train or qid.rsplit(":", 1)[0] != q["dialogue_id"]:
                raise ValueError("non-training dialogue in query memory")
            if any(c not in self.chunks for c in gold[qid]):
                raise ValueError("training label points outside the corpus")
            if q["valid_input"] and gold[qid] and _key(q["query_text"]):
                texts.append(q["query_text"])
                self.rows.append((qid, q["dialogue_id"], _key(texts[-1]), gold[qid]))
                self.seen_questions.add(_key(q["question"]))
        if not texts:
            raise ValueError("empty grounded training memory")
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32,
                                          token_pattern=r"(?u)\b\w+\b")
        self.matrix = self.vectorizer.fit_transform(map(_weighted_query, texts)).T.tocsr()
        identity = [base.version, sklearn.__version__, "query-memory-v1/2x-current/12-dialogues/rrf10/depth50",
                    list(zip(self.rows, texts))]
        self.version = "R6-chunk-" + hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:16]

    @classmethod
    def from_bundle(cls, base, directory):
        return cls(base, *training_bundle(base, directory))

    def prepare_query(self, question, history):
        return self.base.prepare_query(question, history)

    def neighbors(self, query):
        vec = self.vectorizer.transform([_weighted_query(query)])
        sim = (vec @ self.matrix).tocoo()
        query_key = _key(query)
        used, neighbors = set(), []
        for i, score in sorted(zip(sim.col, sim.data), key=lambda x: (-x[1], self.rows[x[0]][0])):
            _, dialogue, key, chunks = self.rows[i]
            if key != query_key and dialogue not in used:
                neighbors.append((float(score), chunks))
                used.add(dialogue)
                if len(neighbors) == 12:
                    break
        return neighbors

    def retrieve(self, query_text, k=5):
        if not 1 <= k <= 50:
            raise ValueError("R6 candidate depth is 50")
        votes, confidence = defaultdict(float), defaultdict(float)
        for cosine, chunks in self.neighbors(query_text):
            for key in chunks:
                votes[key] += cosine ** 2 / len(chunks)
                confidence[key] = max(confidence[key], cosine)
        learned = sorted(votes, key=lambda key: (-votes[key], key))
        bm25 = [e.evidence_id for e in self.base.search(query_text, 50)]
        ranked = [(self.chunks[key], score) for key, score in rrf([bm25, learned], k=10)]
        selected = deduplicate_chunks(ranked, k, self.base.dedup_overlap)
        evidence = [Evidence(evidence_id=c.chunk_id, doc_id=c.doc_id, domain=c.domain, title=c.title,
                             text=c.text, score=s, unit="chunk") for c, s in selected]
        gate_score = max((confidence[e.evidence_id] for e in evidence[:3]), default=0.0)
        return evidence, gate_score
