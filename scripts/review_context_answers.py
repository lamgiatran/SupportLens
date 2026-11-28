"""Export a blind reviewer worksheet, or score completed judgements without Ollama."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.docs_eval import load_dev, paired_dialogue_bootstrap

VERSIONS = ("v2", "v3")
LABELS = {"yes", "no"}
FIELDS = ["case", "side", "question", "past_turns", "sources", "answer", "supported", "responsive", "note"]


def ordering(query_id, salt, versions=VERSIONS):
    bit = sha256((salt + "/" + query_id).encode()).digest()[0] & 1
    return versions[::-1] if bit else versions


def load_audit(path):
    raw = path.read_bytes()
    audit = json.loads(raw)
    ids = [r["id"] for r in audit["rows"]]
    dialogues = {q.rsplit(":", 1)[0] for q in ids}
    if (audit["split"] != "dev" or ids != audit["ids"] or len(dialogues) != len(ids)
            or dialogues.intersection(audit["excluded_dialogues"])):
        raise ValueError("incomplete, repeated or previously reviewed dialogues")
    return audit, sha256(raw).hexdigest()


def export(audit, data, dest, salt):
    bundle = data / "processed/docs_qa"
    if sha256((bundle/"manifest.json").read_bytes()).hexdigest() != audit["bundle_sha256"]:
        raise ValueError("unexpected data bundle")
    _, queries, _ = load_dev(bundle, raw_validation=data / "raw/multidoc2dial/multidoc2dial_dial_validation.json")
    queries = {q["query_id"]: q for q in queries}
    chunks = {r["chunk_id"]: r["text"] for r in map(json.loads, (bundle / "chunks.jsonl").read_bytes().splitlines())}
    with dest.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in audit["rows"]:
            q = queries[r["id"]]
            for side, v in zip(("A", "B"), ordering(r["id"], salt, audit.get("arms", VERSIONS))):
                if r[v]["status"] == "answer":
                    bodies = {}
                    for alias, ref in r[v].get("sources", r.get("sources", {})).items():
                        body = chunks[ref["id"]][:ref["chars"]]
                        if sha256(body.encode()).hexdigest() != ref["sha256"]:
                            raise ValueError("evidence text changed")
                        bodies[alias] = body
                    writer.writerow(dict(case=r["id"], side=side, question=q["question"],
                                         past_turns=json.dumps(q["history"][-2:], ensure_ascii=False),
                                         sources=json.dumps(bodies, ensure_ascii=False), answer=r[v]["answer"]))


def score(audit, labels, salt):
    versions = audit.get("arms", VERSIONS)
    expected = {(r["id"], side) for r in audit["rows"]
                for side, v in zip(("A", "B"), ordering(r["id"], salt, versions)) if r[v]["status"] == "answer"}
    if set(labels) != expected:
        raise ValueError("missing or unexpected answer grades")
    rows = audit["rows"]
    decisions = {v: [] for v in versions}
    result = {}
    for v in versions:
        supported = 0
        statuses = Counter(r[v]["status"] for r in rows)
        for r in rows:
            ok = False
            if r[v]["status"] == "answer":
                side = "AB"[ordering(r["id"], salt, versions).index(v)]
                s, direct = labels[r["id"], side]
                if s not in LABELS or direct not in LABELS:
                    raise ValueError("give yes/no for both labels on every published answer")
                supported += s == "yes"
                ok = s == direct == "yes"
            decisions[v].append(int(ok))
        display = [r[v]["first_display_ms"] for r in rows if r[v]["status"] == "answer"]
        result[v] = {"status": dict(statuses), "all_supported": supported, "both": sum(decisions[v]),
                     "cites_visible_gold": sum(r[v]["cites_visible_gold"] for r in rows),
                     "first_display_p95_ms_among_answers": float(np.percentile(display, 95)) if display else None}
    left, right = versions
    pairs = list(zip(decisions[left], decisions[right]))
    wins, losses = pairs.count((0, 1)), pairs.count((1, 0))
    n = wins + losses
    result["paired"] = {"queries": len(rows), "wins": wins, "losses": losses,
                        "exact_two_sided_p": min(1, 2 * sum(comb(n, j) for j in range(min(wins, losses)+1))/2**n),
                        "bootstrap": paired_dialogue_bootstrap([r["id"].rsplit(":", 1)[0] for r in rows],
                                                               decisions[right], decisions[left])}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, default=Path("runs/docs_qa/context_answers_audit.json"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--salt", default="context-review-2411")
    opts = p.add_mutually_exclusive_group(required=True)
    opts.add_argument("--export", type=Path, help="fresh CSV with hidden system labels and verified source text")
    opts.add_argument("--score", type=Path, help="completed CSV or the published compact review JSON")
    p.add_argument("--save-labels", type=Path, help="optional compact record for the scored CSV")
    args = p.parse_args()
    audit, fingerprint = load_audit(args.audit)
    if args.export:
        export(audit, args.data, args.export, args.salt)
        return
    if args.score.suffix.lower() == ".json":
        packet = json.loads(args.score.read_bytes())
        if (packet["audit_sha256"], packet["salt"]) != (fingerprint, args.salt):
            raise ValueError("review belongs to a different run or blind order")
        grades = packet["grades"]
    else:
        with args.score.open(encoding="utf-8-sig", newline="") as f:
            grades = [{k: r[k] for k in ("case", "side", "supported", "responsive", "note")} for r in csv.DictReader(f)]
        expected_answers = {(r["id"], side): r[v]["answer"] for r in audit["rows"]
                            for side, v in zip(("A", "B"), ordering(r["id"], args.salt, audit.get("arms", VERSIONS)))
                            if r[v]["status"] == "answer"}
        with args.score.open(encoding="utf-8-sig", newline="") as f:
            if any(expected_answers.get((r["case"], r["side"])) != r["answer"] for r in csv.DictReader(f)):
                raise ValueError("graded answers differ from the recorded run")
    labels = {(r["case"], r["side"]): (r["supported"], r["responsive"]) for r in grades}
    if len(labels) != len(grades):
        raise ValueError("duplicate review row")
    result = score(audit, labels, args.salt)
    if args.save_labels:
        with args.save_labels.open("x", encoding="utf-8", newline="\n") as f:
            json.dump({"audit_sha256": fingerprint, "salt": args.salt, "grades": grades}, f, indent=2)
            f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
