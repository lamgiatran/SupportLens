"""Train a passage reranker on public train. Serving requires only NumPy and the BM25 corpus."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import Evidence
from .retrieval import deduplicate_chunks


FEATURES = ("score", "relative", "doc_relative", "before", "after")
VIEWS = ("full", "current", "previous", "older")
SCHEMA = "passage-ranker-v1/36-per-view-and-adjacent"


class PassageRanker:
    method, score_name = "R7", "top1_boosted_margin_v1"

    def __init__(self, base, model=None):
        self.base, self.tokenizer = base, base.tokenizer
        self.keys = np.array([c.chunk_id for c in base.chunks])
        groups = defaultdict(list)
        for i, c in enumerate(base.chunks):
            groups[c.domain, c.doc_id].append((c.start, i))
        self.doc = np.empty(len(self.keys), dtype=int)
        self.before = np.arange(len(self.keys))
        self.after = self.before.copy()
        for d, group in enumerate(groups.values()):
            indices = [i for _, i in sorted(group)]
            self.doc[indices] = d
            self.before[indices[1:]] = indices[:-1]
            self.after[indices[:-1]] = indices[1:]
        self.groups = len(groups)
        self.model = model
        if model is not None:
            if (model["schema"], model["base"], model["features"]) != (SCHEMA, base.version, self.feature_names()):
                raise ValueError("ranker and BM25 corpus or feature version differ")
            tree_data = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            self.version = "R7-chunk-" + hashlib.sha256(tree_data).hexdigest()[:16]
            self.trees = [tuple(np.array(t[k]) for k in ("left", "right", "feature", "threshold", "value"))
                          for t in model["trees"]]

    @staticmethod
    def feature_names():
        return [v + "_" + f for v in VIEWS for f in FEATURES] + ["current_length", "history_length"]

    @classmethod
    def load(cls, base, path):
        return cls(base, json.loads(Path(path).read_bytes()))

    def prepare_query(self, question, history):
        return self.base.prepare_query(question, history)

    def features(self, query):
        turns = query.split("\n", 2)
        current = turns[0]
        previous, older = (turns[1:] + ["", ""])[:2]
        columns, pool = [], []
        scores = {}
        for view in (query, current, previous, older):
            if view not in scores:
                scores[view] = self.base.scores(view)
            s = scores[view]
            peak = max(1.0, float(s.max()))
            relative = s/peak
            doc = np.zeros(self.groups)
            np.maximum.at(doc, self.doc, relative)
            columns.extend([s, relative, doc[self.doc], relative[self.before], relative[self.after]])
            if view and np.any(s > 0):
                order = np.lexsort((self.keys, -s))[:36]
                pool.extend(order[s[order] > 0])
        if not pool:
            return np.array([], dtype=int), np.empty((0, len(self.feature_names())), dtype=np.float32)
        ids = np.unique(np.concatenate([pool, self.before[pool], self.after[pool]]))
        x = [c[ids] for c in columns]
        x.extend(np.full(len(ids), len(v.split())) for v in (current, previous+" "+older))
        return ids, np.column_stack(x).astype(np.float32)

    def decision_function(self, x):
        scores = np.zeros(len(x))
        rows = np.arange(len(x))
        for left, right, feature, threshold, value in self.trees:
            nodes = np.zeros(len(x), dtype=int)
            for _ in range(3):
                active = rows[left[nodes] != -1]
                n = nodes[active]
                nodes[active] = np.where(x[active, feature[n]] <= threshold[n], left[n], right[n])
            scores += value[nodes]
        return scores * self.model["learning_rate"]

    def retrieve(self, query_text, k=5):
        ids, x = self.features(query_text)
        if not len(ids):
            return [], None
        scores = self.decision_function(x)
        order = np.lexsort((self.keys[ids], -scores))
        chosen = deduplicate_chunks([(self.base.chunks[ids[i]], float(scores[i])) for i in order], k,
                                    self.base.dedup_overlap)
        evidence = [Evidence(evidence_id=c.chunk_id, doc_id=c.doc_id, domain=c.domain, title=c.title,
                             text=c.text, score=s, unit="chunk") for c, s in chosen]
        return evidence, evidence[0].score
