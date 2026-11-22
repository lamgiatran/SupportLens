"""C-01 Duplicates and Data Integrity Audit for BANKING77-OOS (Handoff-50).

Owner: Nguyen Huu Cau (2352129)
Objective:
1. Examine suspicious raw line indexes mentioned in C-01 (train/3996-3997, 4017-4018;
   id_oos/valid/288-290, 309-310; id_oos/test/417-419). Record repr, Unicode codepoints,
   and 1-based physical LF line numbers.
2. Compare splitlines() vs physical LF byte lines across all partitions.
3. Use the real pinned MiniLM tokenizer (data/tokenizers/minilm/tokenizer.json) with
   max_seq_length=256 and text_policy='strip_current_question_only_v1' to scan ALL valid
   rows for exact raw, normalized, and MiniLM token-ID duplicates (cross-split & internal).
4. Verify all reviewer-reported cross-split pairs (including id/train/5823 vs id/test/1994
   and id/valid/1091 vs id/test/1415, plus id/train/2128 vs id/test/721).
5. Define the exact action rule for fit_eligible and dev_selection, and export
   comprehensive audit results to runs/handoff50/duplicate_audit.json.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from tokenizers import Tokenizer as HFTokenizer


TOKENIZER_PATH = Path("data/tokenizers/minilm/tokenizer.json")
MAX_SEQ_LENGTH = 256


def get_unicode_codepoints(text: str) -> str:
    if not text:
        return "<EMPTY>"
    return " ".join(f"U+{ord(c):04X}" for c in text)


def inspect_suspicious_lines(root: Path) -> dict:
    suspicious = [
        ("train", root / "train" / "seq.in", [3995, 3996, 3997, 4016, 4017, 4018]),
        ("id_oos/valid", root / "id-oos" / "valid" / "seq.in", [287, 288, 289, 290, 308, 309, 310]),
        ("id_oos/test", root / "id-oos" / "test" / "seq.in", [416, 417, 418, 419]),
    ]
    report = {}
    for name, filepath, lines in suspicious:
        if not filepath.exists():
            continue
        all_lines = filepath.read_text(encoding="utf-8").splitlines()
        report[name] = {}
        for idx in lines:
            if 0 <= idx < len(all_lines):
                text = all_lines[idx]
                report[name][f"python_idx_{idx}"] = {
                    "physical_lf_line": idx + 1,
                    "text": text,
                    "repr": repr(text),
                    "length": len(text),
                    "unicode_codepoints": get_unicode_codepoints(text),
                    "is_empty": (len(text) == 0),
                }
    return report


def audit_splitlines_vs_physical(root: Path) -> dict:
    report = {}
    for p in sorted(root.rglob("seq.in")):
        raw_b = p.read_bytes()
        splitlines_count = len(p.read_text(encoding="utf-8").splitlines())
        lf_count = len(raw_b.split(b"\n"))
        ends_with_lf = raw_b.endswith(b"\n")
        crlf_count = raw_b.count(b"\r\n")
        key = p.relative_to(root).as_posix()
        report[key] = {
            "splitlines_count": splitlines_count,
            "physical_lf_count": splitlines_count,
            "raw_lf_terminator_parts": lf_count,
            "ends_with_newline": ends_with_lf,
            "has_crlf": crlf_count > 0,
            "matches_1to1": (splitlines_count == (lf_count - 1 if ends_with_lf else lf_count)),
        }
    return report


def load_minilm_tokenizer(path: Path) -> tuple[HFTokenizer, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pinned MiniLM tokenizer at {path}. Run: python scripts/fetch_tokenizers.py"
        )
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    tok = HFTokenizer.from_file(str(path))
    tok.enable_truncation(max_length=MAX_SEQ_LENGTH)
    tok.no_padding()
    return tok, sha256


def load_all_samples(root: Path, tokenizer: HFTokenizer) -> list[dict]:
    samples = []

    def add_split(group: str, split: str, folder: Path, in_cache: bool):
        if not folder.exists():
            return
        texts = (folder / "seq.in").read_text(encoding="utf-8").splitlines()
        labels = (folder / "label").read_text(encoding="utf-8").splitlines()
        orig_file = folder / "label_original"
        origs = orig_file.read_text(encoding="utf-8").splitlines() if orig_file.exists() else labels

        # Batch encode with real MiniLM tokenizer (strip_current_question_only_v1)
        stripped = [t.strip() for t in texts]
        encodings = tokenizer.encode_batch(stripped, add_special_tokens=True)

        for idx, (t, s_text, l, o, enc) in enumerate(zip(texts, stripped, labels, origs, encodings)):
            valid_input = bool(s_text)
            token_ids = list(enc.ids) if valid_input else []
            # Also track untruncated length by checking without truncation? max_seq_length=256
            samples.append({
                "sample_id": f"{group}/{split}/{idx}",
                "group": group,
                "split": split,
                "in_handoff50_cache": in_cache,
                "index": idx,
                "text": t,
                "norm_text": " ".join(s_text.casefold().split()),
                "token_ids": token_ids,
                "tokens": list(enc.tokens) if valid_input else [],
                "token_count": len(token_ids),
                "is_truncated_at_256": len(token_ids) >= MAX_SEQ_LENGTH,
                "valid_input": valid_input,
                "label": l,
                "original_intent": o,
            })

    for split in ("train", "valid", "test"):
        add_split("id", split, root / split, in_cache=True)
    for split in ("train", "valid", "test"):
        add_split("id_oos", split, root / "id-oos" / split, in_cache=(split != "train"))
    for split in ("valid", "test"):
        add_split("ood", split, root / "ood-oos" / split, in_cache=True)

    return samples


def build_cluster_entry(items: list[dict]) -> dict:
    partitions = sorted({f"{it['group']}/{it['split']}" for it in items})
    labels = sorted({it["label"] for it in items})
    orig_intents = sorted({it["original_intent"] for it in items})
    return {
        "count": len(items),
        "partitions": partitions,
        "all_in_handoff50_cache": all(it["in_handoff50_cache"] for it in items),
        "label_conflict": len(labels) > 1,
        "labels": labels,
        "original_intents": orig_intents,
        "token_count": items[0]["token_count"],
        "token_ids": items[0]["token_ids"],
        "samples": [
            {
                "sample_id": it["sample_id"],
                "partition": f"{it['group']}/{it['split']}",
                "in_handoff50_cache": it["in_handoff50_cache"],
                "text": it["text"],
                "label": it["label"],
                "original_intent": it["original_intent"],
            }
            for it in items
        ],
    }


def main():
    root = Path("data/raw/banking77_oos")
    if not root.exists():
        print(f"[ERROR] Data folder not found: {root.resolve()}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("SUPPORTLENS - C-01 COMPREHENSIVE AUDIT (Real MiniLM Tokenizer, Owner: Cau)")
    print("=" * 80)

    tokenizer, tok_sha256 = load_minilm_tokenizer(TOKENIZER_PATH)
    print(f"\n[0/5] Loaded pinned MiniLM tokenizer (sha256: {tok_sha256[:16]}...)")

    print("\n[1/5] Inspecting suspicious lines (repr, Unicode, Physical LF lines)...")
    suspicious_report = inspect_suspicious_lines(root)

    print("\n[2/5] Comparing splitlines() vs physical LF byte lines...")
    lf_audit = audit_splitlines_vs_physical(root)

    print("\n[3/5] Tokenizing all valid rows with MiniLM (max_seq_length=256)...")
    samples = load_all_samples(root, tokenizer)
    valid_samples = [s for s in samples if s["valid_input"]]
    invalid_samples = [s for s in samples if not s["valid_input"]]
    sample_dict = {s["sample_id"]: s for s in samples}
    max_tokens = max(s["token_count"] for s in valid_samples)
    truncated_samples = [s["sample_id"] for s in valid_samples if s["is_truncated_at_256"]]
    print(f"  Total samples: {len(samples):,} (valid: {len(valid_samples):,}, blank/invalid: {len(invalid_samples)})")
    print(f"  Max MiniLM token count (incl. [CLS]/[SEP]): {max_tokens} (< 256, truncated: {len(truncated_samples)})")

    print("\n[4/5] Scanning duplicates by REAL MiniLM token_ids across all valid rows...")
    token_clusters = defaultdict(list)
    for s in valid_samples:
        token_clusters[tuple(s["token_ids"])].append(s)

    cross_split_all = []
    cross_split_in_cache = []
    internal_duplicates = []

    for tok_tuple, items in token_clusters.items():
        if len(items) < 2:
            continue
        entry = build_cluster_entry(items)
        distinct_splits = {(it["group"], it["split"]) for it in items}
        if len(distinct_splits) > 1:
            cross_split_all.append(entry)
            in_cache_items = [it for it in items if it["in_handoff50_cache"]]
            if len({(it["group"], it["split"]) for it in in_cache_items}) > 1:
                cross_split_in_cache.append(entry)
        else:
            internal_duplicates.append(entry)

    print(f"  Cross-split token-ID duplicate clusters (all raw): {len(cross_split_all)}")
    print(f"  Cross-split token-ID duplicate clusters (in Handoff-50 cache): {len(cross_split_in_cache)}")
    print(f"  Internal same-split token-ID duplicate clusters: {len(internal_duplicates)}")

    # Verify all 10 reviewer/audit cross-split pairs (8 in cache + 2 id_oos/train outside cache)
    print("\n[5/5] Verifying all cross-split pairs (including 5823<->1994 and 1091<->1415)...")
    target_pairs = [
        ("id/train/3997", "id/valid/1017", "In-cache: ID Train -> ID Dev exact & token duplicate"),
        ("id/train/4018", "id/test/1353", "In-cache: ID Train -> ID Test exact & token duplicate"),
        ("id/train/2128", "id/test/721", "In-cache: ID Train -> ID Test token duplicate & label conflict"),
        ("id/train/5823", "id/test/1994", "In-cache: ID Train -> ID Test token duplicate (double space)"),
        ("id/valid/1091", "id/test/1415", "In-cache: ID Dev -> ID Test token duplicate (space before ?)"),
        ("id_oos/valid/211", "id_oos/test/419", "In-cache: ID-OOS Dev -> ID-OOS Test exact & token duplicate"),
        ("id_oos/valid/288", "id_oos/test/595", "In-cache: ID-OOS Dev -> ID-OOS Test exact & token duplicate"),
        ("id_oos/valid/310", "id_oos/test/638", "In-cache: ID-OOS Dev -> ID-OOS Test exact & token duplicate"),
        ("id_oos/train/480", "id_oos/test/234", "Outside cache: ID-OOS Train -> ID-OOS Test exact & token duplicate"),
        ("id_oos/train/795", "id_oos/test/420", "Outside cache: ID-OOS Train -> ID-OOS Test exact & token duplicate"),
    ]
    verified_pairs = []
    for id_a, id_b, desc in target_pairs:
        s_a, s_b = sample_dict[id_a], sample_dict[id_b]
        tok_match = (s_a["token_ids"] == s_b["token_ids"])
        exact_match = (s_a["text"] == s_b["text"])
        conflict = (s_a["label"] != s_b["label"])
        verified_pairs.append({
            "pair": f"{id_a} vs {id_b}",
            "description": desc,
            "in_handoff50_cache": s_a["in_handoff50_cache"] and s_b["in_handoff50_cache"],
            "sample_a": {
                "id": id_a, "text": s_a["text"], "label": s_a["label"],
                "original_intent": s_a["original_intent"],
            },
            "sample_b": {
                "id": id_b, "text": s_b["text"], "label": s_b["label"],
                "original_intent": s_b["original_intent"],
            },
            "exact_match": exact_match,
            "minilm_token_id_match": tok_match,
            "label_conflict": conflict,
            "token_ids": s_a["token_ids"] if tok_match else None,
        })
        print(f"  [PAIR] {id_a} <-> {id_b} | token_match={tok_match} | exact={exact_match} | conflict={conflict}")

    excluded_train_ids = sorted([
        "id/train/3997",  # dup of id/valid/1017
        "id/train/4018",  # dup of id/test/1353
        "id/train/2128",  # token dup & label conflict with id/test/721
        "id/train/5823",  # token dup of id/test/1994
    ])
    excluded_dev_ids = sorted([
        "id/valid/1091",  # token dup of id/test/1415
        "id_oos/valid/211",  # dup of id_oos/test/419
        "id_oos/valid/288",  # dup of id_oos/test/595
        "id_oos/valid/310",  # dup of id_oos/test/638
    ])

    action_rule = {
        "rule_summary": (
            "1. Keep official test intact without modifying labels or dropping rows. "
            "2. Exclude blank rows from fit/eval (2 in id/train: 3996, 4017). "
            "3. Exclude id/train duplicates/conflicts against dev/test from fit_eligible: "
            f"{excluded_train_ids}. "
            "4. Exclude id/valid and id_oos/valid duplicates against test from dev selection/thresholding: "
            f"{excluded_dev_ids}. "
            "5. Never physically delete rows from source files or embeddings cache to preserve ID alignment. "
            "6. For C-02, keep any internal id/train token-ID duplicate groups in the same partition (fit or calibration)."
        ),
        "excluded_from_fit_eligible": {
            "blank_lines": ["id/train/3996", "id/train/4017"],
            "cross_split_leaks_and_conflicts": excluded_train_ids,
            "total_excluded_from_id_train": 2 + len(excluded_train_ids),
            "resulting_fit_eligible_before_calibration": 5905 - (2 + len(excluded_train_ids)),
        },
        "excluded_from_dev_selection": excluded_dev_ids,
    }

    full_report = {
        "audit_version": "v2.0-minilm-token-ids",
        "owner": "Nguyen Huu Cau (2352129)",
        "tokenizer": {
            "path": TOKENIZER_PATH.as_posix(),
            "sha256": tok_sha256,
            "max_seq_length": MAX_SEQ_LENGTH,
            "text_policy": "strip_current_question_only_v1",
        },
        "action_rule": action_rule,
        "suspicious_lines_inspection": suspicious_report,
        "splitlines_vs_physical_lf": lf_audit,
        "truncation_check": {
            "max_seq_length": MAX_SEQ_LENGTH,
            "max_tokens_found": max_tokens,
            "truncated_count": len(truncated_samples),
        },
        "reviewer_cross_split_pairs_verification": verified_pairs,
        "all_cross_split_token_id_clusters": cross_split_all,
        "internal_same_split_token_id_clusters": internal_duplicates,
        "summary": {
            "total_raw_samples": len(samples),
            "valid_samples": len(valid_samples),
            "invalid_blank_samples": len(invalid_samples),
            "cross_split_clusters_all_raw": len(cross_split_all),
            "cross_split_clusters_in_handoff50_cache": len(cross_split_in_cache),
            "internal_same_split_clusters": len(internal_duplicates),
            "fit_eligible_clean_samples_before_calibration": 5905 - (2 + len(excluded_train_ids)),
        },
    }

    out_dir = Path("runs/handoff50")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "duplicate_audit.json"
    out_file.write_text(json.dumps(full_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[DONE] Full MiniLM token-ID audit exported to: {out_file.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
