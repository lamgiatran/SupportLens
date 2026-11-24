"""Package B: Train 50-class Logistic Regression on fit_eligible, select C & H1 gate on dev_eligible.

Owner: Nguyen Huu Cau (2352129)
Rules:
- Train ONLY on fit_eligible (5,308 ID train rows after C-01/C-02 exclusions).
- Select C in [0.1, 1.0, 10.0, 100.0] ONLY on dev_eligible ID rows (excludes id/valid/1091).
- Select initial H1 threshold ONLY on dev_eligible (ID + ID-OOS + OOD, excludes 4 dev dups).
- Never touch calibration (591 rows) or test (4,076 rows) during Package B training/selection.
"""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.banking import EncoderConfig
from supportlens.banking_h1 import load_pipeline
from supportlens.evaluation import risk_coverage, select_gate


C_GRID = [0.1, 1.0, 10.0, 100.0]
ARTIFACT_VERSION = "handoff50-lr-h1-v1"


def compute_aurc(rc_points: list[dict]) -> float:
    covs = [p["coverage"] for p in rc_points if p["risk"] is not None]
    risks = [p["risk"] for p in rc_points if p["risk"] is not None]
    if not covs:
        return 0.0
    covs = [0.0] + covs
    risks = [risks[0]] + risks
    return float(np.trapezoid(risks, covs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/processed/handoff50/embeddings.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/handoff50/embedding_manifest.json"))
    parser.add_argument("--encoder-config", type=Path, default=Path("configs/banking_encoder.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/banking77_oos"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/handoff50/h1"))
    parser.add_argument("--metrics-out", type=Path, default=Path("runs/handoff50/h1_dev_metrics.json"))
    parser.add_argument("--target-risk", type=float, default=0.10)
    args = parser.parse_args()

    enc_cfg = EncoderConfig.load(args.encoder_config)
    emb_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if enc_cfg.fingerprint != emb_manifest["encoder_fingerprint"]:
        raise ValueError("Encoder config fingerprint does not match embedding_manifest.json")

    with np.load(args.cache, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        sample_ids = data["sample_id"]
        labels = data["label"]
        splits = data["split"]
        groups = data["group"]
        fit_mask = data["fit_eligible"].astype(bool)
        cal_mask = data["calibration"].astype(bool)
        dev_elig_mask = data["dev_eligible"].astype(bool)

    assert not (fit_mask & cal_mask).any(), "fit and calibration must be disjoint"
    assert not (fit_mask & dev_elig_mask).any(), "fit and dev_eligible must be disjoint"

    X_fit = embeddings[fit_mask]
    y_fit = labels[fit_mask]

    dev_id_mask = dev_elig_mask & (splits == "dev") & (groups == "id")
    X_dev_id = embeddings[dev_id_mask]
    y_dev_id = labels[dev_id_mask]

    X_dev_all = embeddings[dev_elig_mask]
    y_dev_all = labels[dev_elig_mask]
    g_dev_all = groups[dev_elig_mask]
    ids_dev_all = sample_ids[dev_elig_mask]

    classes = sorted(set(y_fit.tolist()))
    assert len(classes) == 50, f"Expected 50 classes in fit, got {len(classes)}"

    # 1. Grid search C on dev_eligible (ID)
    grid_results = []
    best_model = None
    best_key = None
    for c_val in C_GRID:
        clf = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
        clf.fit(X_fit, y_fit)
        assert list(clf.classes_) == classes

        probs_dev_id = clf.predict_proba(X_dev_id)
        preds_dev_id = clf.classes_[np.argmax(probs_dev_id, axis=1)]
        acc = float(accuracy_score(y_dev_id, preds_dev_id))
        macro_f1 = float(f1_score(y_dev_id, preds_dev_id, average="macro"))
        nll = float(log_loss(y_dev_id, probs_dev_id, labels=classes))

        entry = {
            "C": c_val,
            "dev_id_accuracy": acc,
            "dev_id_macro_f1": macro_f1,
            "dev_id_nll": nll,
        }
        grid_results.append(entry)
        # Select by highest macro-F1, then accuracy, then lowest NLL
        key = (round(macro_f1, 12), round(acc, 12), -nll)
        if best_key is None or key > best_key:
            best_key = key
            best_model = (c_val, clf, entry)

    selected_c, head, best_entry = best_key_result = best_model

    # 2. Compute H1 (max softmax probability) & Risk-Coverage on full dev_eligible (ID + ID-OOS + OOD)
    probs_dev_all = head.predict_proba(X_dev_all)
    scores_dev_all = np.max(probs_dev_all, axis=1).tolist()
    preds_dev_all = head.classes_[np.argmax(probs_dev_all, axis=1)]
    # Error = True if OOS sample (should not be accepted) OR wrong ID intent
    errors_dev_all = [
        (grp != "id") or (pred != true_lbl)
        for grp, pred, true_lbl in zip(g_dev_all, preds_dev_all, y_dev_all)
    ]

    rc_points = risk_coverage(scores_dev_all, errors_dev_all)
    aurc = compute_aurc(rc_points)
    gate = select_gate(
        scores_dev_all,
        errors_dev_all,
        method="H1",
        artifact_version=ARTIFACT_VERSION,
        split="dev",
        target_risk=args.target_risk,
    )

    accepted_mask = np.array([gate.accepts(s) for s in scores_dev_all])
    dev_id_sub = g_dev_all == "id"
    dev_oos_sub = g_dev_all != "id"
    coverage_all = float(accepted_mask.mean())
    risk_all = float(np.mean([e for e, a in zip(errors_dev_all, accepted_mask) if a])) if accepted_mask.any() else 0.0
    oos_false_accept_rate = float(accepted_mask[dev_oos_sub].mean())
    id_coverage = float(accepted_mask[dev_id_sub].mean())
    id_selective_acc = (
        float(np.mean([p == y for p, y, a in zip(preds_dev_all[dev_id_sub], y_dev_all[dev_id_sub], accepted_mask[dev_id_sub]) if a]))
        if accepted_mask[dev_id_sub].any()
        else 0.0
    )

    # 3. Save artifact directory
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(head, args.artifact_dir / "head.joblib")
    (args.artifact_dir / "encoder_config.json").write_text(
        args.encoder_config.read_text(encoding="utf-8"), encoding="utf-8"
    )
    artifact_manifest = {
        "model_scope": "handoff50",
        "artifact_version": ARTIFACT_VERSION,
        "factory": "supportlens.banking_h1:load_pipeline",
        "method": "H1",
        "selected_C": selected_c,
        "threshold": gate.threshold,
        "target_risk": args.target_risk,
        "fitted_split": "dev",
        "encoder_fingerprint": enc_cfg.fingerprint,
        "npz_sha256": emb_manifest["npz_sha256"],
        "fit_rows": int(fit_mask.sum()),
        "dev_id_rows": int(dev_id_mask.sum()),
        "dev_eligible_rows": int(dev_elig_mask.sum()),
        "classes": classes,
    }
    (args.artifact_dir / "manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8"
    )

    # 4. Generate 3 real smoke samples from dev_eligible (2 accepted ID + 1 handoff OOS/low-conf)
    # Load raw texts for dev_eligible
    raw_texts = {}
    for grp, subdir in [("id", ""), ("id_oos", "id-oos"), ("ood", "ood-oos")]:
        seq_lines = (args.raw / subdir / "valid" / "seq.in").read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(seq_lines):
            raw_texts[f"{grp}/valid/{i}"] = line

    # Pick 2 high-confidence correct ID samples (from different intents) and 1 handoff sample
    id_correct_indices = [
        i for i, (grp, pred, y, sc, acc) in enumerate(
            zip(g_dev_all, preds_dev_all, y_dev_all, scores_dev_all, accepted_mask)
        )
        if grp == "id" and pred == y and acc
    ]
    id_correct_indices.sort(key=lambda i: -scores_dev_all[i])
    sample_idx_1 = id_correct_indices[0]
    intent_1 = y_dev_all[sample_idx_1]
    sample_idx_2 = next(i for i in id_correct_indices[1:] if y_dev_all[i] != intent_1)

    handoff_indices = [
        i for i, (grp, acc) in enumerate(zip(g_dev_all, accepted_mask))
        if grp != "id" and not acc
    ]
    sample_idx_3 = handoff_indices[0] if handoff_indices else int(np.argmin(scores_dev_all))

    # Verify via actual load_pipeline() so expected outputs match live encode() 100%
    pipeline = load_pipeline(artifact_dir=args.artifact_dir, local_files_only=True)
    smoke_lines = []
    for idx in (sample_idx_1, sample_idx_2, sample_idx_3):
        sid = str(ids_dev_all[idx])
        question = raw_texts[sid]
        pred = pipeline.predict(question, method="H1")
        record = {
            "sample_id": sid,
            "input": {"question": question},
            "expected": {
                "predicted_intent": pred.predicted_intent,
                "score": pred.score,
                "decision": pred.decision,
                "reason": pred.reason,
                "artifact_version": pred.artifact_version,
                "score_method": "H1",
            },
        }
        smoke_lines.append(json.dumps(record, ensure_ascii=False))
    (args.artifact_dir / "smoke_samples.jsonl").write_text(
        "\n".join(smoke_lines) + "\n", encoding="utf-8"
    )

    # 5. Save dev metrics report
    metrics_report = {
        "owner": "Nguyen Huu Cau (2352129)",
        "package": "B",
        "artifact_version": ARTIFACT_VERSION,
        "fit_rows": int(fit_mask.sum()),
        "calibration_rows_untouched": int(cal_mask.sum()),
        "dev_id_rows": int(dev_id_mask.sum()),
        "dev_eligible_rows": int(dev_elig_mask.sum()),
        "c_grid_results": grid_results,
        "selected_C": selected_c,
        "selected_dev_id_metrics": best_entry,
        "h1_dev_eligible_gate": {
            "method": "H1",
            "target_risk": args.target_risk,
            "threshold": gate.threshold,
            "aurc": aurc,
            "coverage_all": coverage_all,
            "risk_all": risk_all,
            "id_coverage": id_coverage,
            "id_selective_accuracy": id_selective_acc,
            "oos_false_acceptance_rate": oos_false_accept_rate,
        },
        "smoke_samples_count": len(smoke_lines),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics_report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics_report, indent=2))
    print(f"\n[DONE] Saved artifact to {args.artifact_dir} and metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
