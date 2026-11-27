"""Package C: Fit H2 Temperature Scaling on calibration; evaluate H1-H4 on dev_eligible;
build artifacts/handoff50/selected/ and 2-sample smoke_samples.jsonl (1 accept, 1 handoff).

Owner: Nguyen Huu Cau (2352129)
Rules:
- Keep frozen C=10.0 LR head from artifacts/handoff50/h1/head.joblib.
- H2: Fit scalar temperature T > 0 by minimizing NLL ONLY on calibration (591 rows).
- H3: Reference bank uses ONLY fit_eligible (5,308 rows), NEVER calibration/dev/test/OOS.
- H4: Energy score = logsumexp(logits) from the frozen LR head.
- Evaluate & compare H1-H4 ONLY on dev_eligible (2,230 rows). Never touch test.
"""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp, softmax
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.banking import EncoderConfig
from supportlens.banking_c import load_pipeline
from supportlens.evaluation import risk_coverage, select_gate


ARTIFACT_VERSION = "handoff50-pkgc-h1-v1"
H3_KNN_K = 10


def compute_ece(probs: np.ndarray, y_true: np.ndarray, classes: list[str], n_bins: int = 15) -> float:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[y] for y in y_true])
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_idx).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confidences >= lo) & (confidences <= hi) if b == 0 else (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def compute_aurc(rc_points: list[dict]) -> float:
    covs = [p["coverage"] for p in rc_points if p["risk"] is not None]
    risks = [p["risk"] for p in rc_points if p["risk"] is not None]
    if not covs:
        return 0.0
    covs = [0.0] + covs
    risks = [risks[0]] + risks
    return float(np.trapezoid(risks, covs))


def fit_temperature(logits_cal: np.ndarray, y_cal: np.ndarray, classes: list[str]) -> float:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[y] for y in y_cal])

    def nll_loss(t: float) -> float:
        lsm = log_softmax(logits_cal / t, axis=1)
        return float(-np.mean(lsm[np.arange(len(y_idx)), y_idx]))

    res = minimize_scalar(nll_loss, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def evaluate_at_coverages(
    scores: np.ndarray,
    preds: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    target_coverages: list[float],
) -> list[dict]:
    n = len(scores)
    is_oos = groups != "id"
    errors = np.array([(oos or (p != y)) for oos, p, y in zip(is_oos, preds, y_true)], dtype=bool)
    rc_points = risk_coverage(scores.tolist(), errors.tolist())

    rows = []
    for target_cov in target_coverages:
        valid_pts = [p for p in rc_points if p["threshold"] is not None]
        ge_pts = [p for p in valid_pts if p["coverage"] >= target_cov]
        pt = min(ge_pts, key=lambda p: p["coverage"]) if ge_pts else valid_pts[-1]
        thr = pt["threshold"]
        acc_mask = scores >= thr
        n_acc = int(acc_mask.sum())
        n_err = int((acc_mask & errors).sum())
        n_id_err = int((acc_mask & (~is_oos) & errors).sum())
        n_oos_err = int((acc_mask & is_oos).sum())
        rows.append({
            "target_coverage": target_cov,
            "actual_coverage": n_acc / n,
            "threshold": float(thr),
            "accepted": n_acc,
            "errors": n_err,
            "risk": (n_err / n_acc) if n_acc else 0.0,
            "id_errors": n_id_err,
            "oos_false_accepts": n_oos_err,
            "oos_false_accept_rate": float(n_oos_err / is_oos.sum()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/processed/handoff50/embeddings.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/handoff50/embedding_manifest.json"))
    parser.add_argument("--encoder-config", type=Path, default=Path("configs/banking_encoder.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/banking77_oos"))
    parser.add_argument("--h1-dir", type=Path, default=Path("artifacts/handoff50/h1"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/handoff50/selected"))
    parser.add_argument("--out", type=Path, default=Path("runs/handoff50/h1_h4_dev_metrics.json"))
    parser.add_argument("--target-risk", type=float, default=0.10)
    args = parser.parse_args()

    enc_cfg = EncoderConfig.load(args.encoder_config)
    emb_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    head = joblib.load(args.h1_dir / "head.joblib")
    h1_manifest = json.loads((args.h1_dir / "manifest.json").read_text(encoding="utf-8"))
    classes = list(head.classes_)
    assert classes == h1_manifest["classes"]

    with np.load(args.cache, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        sample_ids = data["sample_id"]
        labels = data["label"]
        groups = data["group"]
        fit_mask = data["fit_eligible"].astype(bool)
        cal_mask = data["calibration"].astype(bool)
        dev_elig_mask = data["dev_eligible"].astype(bool)

    assert not (fit_mask & cal_mask).any()
    assert not (fit_mask & dev_elig_mask).any()
    assert not (cal_mask & dev_elig_mask).any()

    X_fit = embeddings[fit_mask]
    X_cal = embeddings[cal_mask]
    y_cal = labels[cal_mask]
    X_dev = embeddings[dev_elig_mask]
    y_dev = labels[dev_elig_mask]
    g_dev = groups[dev_elig_mask]
    ids_dev = sample_ids[dev_elig_mask]
    dev_id_sub = g_dev == "id"

    logits_cal = head.decision_function(X_cal)
    logits_dev = head.decision_function(X_dev)
    preds_dev = head.classes_[np.argmax(logits_dev, axis=1)]

    # H1 & H2
    probs_h1_dev = softmax(logits_dev, axis=1)
    T_opt = fit_temperature(logits_cal, y_cal, classes)
    probs_h2_cal_before = softmax(logits_cal, axis=1)
    probs_h2_cal_after = softmax(logits_cal / T_opt, axis=1)
    probs_h2_dev = softmax(logits_dev / T_opt, axis=1)

    cal_metrics = {
        "T_opt": T_opt,
        "cal_rows": int(cal_mask.sum()),
        "cal_nll_before_T1": float(log_loss(y_cal, probs_h2_cal_before, labels=classes)),
        "cal_nll_after_T": float(log_loss(y_cal, probs_h2_cal_after, labels=classes)),
        "cal_ece_before_T1": compute_ece(probs_h2_cal_before, y_cal, classes),
        "cal_ece_after_T": compute_ece(probs_h2_cal_after, y_cal, classes),
        "dev_id_nll_H1": float(log_loss(y_dev[dev_id_sub], probs_h1_dev[dev_id_sub], labels=classes)),
        "dev_id_nll_H2": float(log_loss(y_dev[dev_id_sub], probs_h2_dev[dev_id_sub], labels=classes)),
        "dev_id_ece_H1": compute_ece(probs_h1_dev[dev_id_sub], y_dev[dev_id_sub], classes),
        "dev_id_ece_H2": compute_ece(probs_h2_dev[dev_id_sub], y_dev[dev_id_sub], classes),
    }

    # H3 (fit_eligible bank ONLY) & H4 (Energy)
    sims = X_dev @ X_fit.T
    sorted_sims = np.sort(sims, axis=1)[:, ::-1]
    scores_h3_k1 = sorted_sims[:, 0]
    scores_h3_k5 = sorted_sims[:, :5].mean(axis=1)
    scores_h3_k10 = sorted_sims[:, :H3_KNN_K].mean(axis=1)
    scores_h4 = logsumexp(logits_dev, axis=1)

    all_methods = {
        "H1": np.max(probs_h1_dev, axis=1),
        "H2": np.max(probs_h2_dev, axis=1),
        "H3_k1": scores_h3_k1,
        "H3_k5": scores_h3_k5,
        "H3": scores_h3_k10,
        "H4": scores_h4,
    }

    is_oos = g_dev != "id"
    errors_dev = [(oos or (p != y)) for oos, p, y in zip(is_oos, preds_dev, y_dev)]
    coverages = [0.40, 0.50, 0.60, 0.70, 0.80]

    comparison = {}
    selected_thresholds = {}
    for name, sc in all_methods.items():
        rc = risk_coverage(sc.tolist(), errors_dev)
        aurc = compute_aurc(rc)
        gate = select_gate(
            sc.tolist(), errors_dev, method=name, artifact_version=ARTIFACT_VERSION, split="dev", target_risk=args.target_risk
        )
        acc_at_target = sc >= gate.threshold if gate.threshold is not None else np.zeros(len(sc), dtype=bool)
        n_acc = int(acc_at_target.sum())
        n_err = int((acc_at_target & np.array(errors_dev)).sum())
        n_oos_err = int((acc_at_target & is_oos).sum())
        if name in {"H1", "H2", "H3", "H4"}:
            selected_thresholds[name] = float(gate.threshold)
        comparison[name] = {
            "aurc": aurc,
            "at_target_risk_10pct": {
                "threshold": gate.threshold,
                "coverage": n_acc / len(sc),
                "accepted": n_acc,
                "errors": n_err,
                "risk": (n_err / n_acc) if n_acc else 0.0,
                "oos_false_accepts": n_oos_err,
                "oos_false_accept_rate": float(n_oos_err / is_oos.sum()),
            },
            "matched_coverage": evaluate_at_coverages(sc, preds_dev, y_dev, g_dev, coverages),
        }

    # Pick winner among H1..H4 by lowest AURC (and highest coverage at target risk)
    best_method = min(["H1", "H2", "H3", "H4"], key=lambda m: comparison[m]["aurc"])

    # Build artifacts/handoff50/selected/
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(head, args.artifact_dir / "head.joblib")
    # Save float16 fit_bank compressed (~3.5MB) so H3 works in UI without violating .gitignore (*.npy)
    joblib.dump(X_fit.astype(np.float16), args.artifact_dir / "fit_bank.joblib", compress=3)
    (args.artifact_dir / "encoder_config.json").write_text(
        args.encoder_config.read_text(encoding="utf-8"), encoding="utf-8"
    )
    artifact_manifest = {
        "model_scope": "handoff50",
        "artifact_version": ARTIFACT_VERSION,
        "factory": "supportlens.banking_c:load_pipeline",
        "selected_method": best_method,
        "selected_threshold": selected_thresholds[best_method],
        "thresholds": selected_thresholds,
        "h2_temperature": T_opt,
        "h3_knn_k": H3_KNN_K,
        "selected_C": h1_manifest["selected_C"],
        "target_risk": args.target_risk,
        "fitted_split": "dev",
        "encoder_fingerprint": enc_cfg.fingerprint,
        "npz_sha256": emb_manifest["npz_sha256"],
        "fit_rows": int(fit_mask.sum()),
        "calibration_rows": int(cal_mask.sum()),
        "dev_eligible_rows": int(dev_elig_mask.sum()),
        "classes": classes,
    }
    (args.artifact_dir / "manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8"
    )

    # Generate 2 smoke samples (1 accepted, 1 handoff) for Tuan's app check
    raw_texts = {}
    for grp, subdir in [("id", ""), ("id_oos", "id-oos"), ("ood", "ood-oos")]:
        seq_lines = (args.raw / subdir / "valid" / "seq.in").read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(seq_lines):
            raw_texts[f"{grp}/valid/{i}"] = line

    sc_best = all_methods[best_method]
    thr_best = selected_thresholds[best_method]
    acc_idx = max(
        [i for i, (g, p, y, s) in enumerate(zip(g_dev, preds_dev, y_dev, sc_best)) if g == "id" and p == y and s >= thr_best],
        key=lambda i: sc_best[i],
    )
    handoff_idx = next(
        i for i, (g, s) in enumerate(zip(g_dev, sc_best)) if g != "id" and s < thr_best
    )

    pipeline = load_pipeline(artifact_dir=args.artifact_dir, local_files_only=True)
    smoke_lines = []
    for idx in (acc_idx, handoff_idx):
        sid = str(ids_dev[idx])
        question = raw_texts[sid]
        pred = pipeline.predict(question, method=best_method)
        smoke_lines.append(json.dumps({
            "sample_id": sid,
            "input": {"question": question},
            "expected": {
                "predicted_intent": pred.predicted_intent,
                "score": pred.score,
                "decision": pred.decision,
                "reason": pred.reason,
                "artifact_version": pred.artifact_version,
                "score_method": best_method,
            },
        }, ensure_ascii=False))
    (args.artifact_dir / "smoke_samples.jsonl").write_text("\n".join(smoke_lines) + "\n", encoding="utf-8")

    report = {
        "owner": "Nguyen Huu Cau (2352129)",
        "package": "C",
        "artifact_version": ARTIFACT_VERSION,
        "selected_method": best_method,
        "selected_threshold": selected_thresholds[best_method],
        "dev_eligible_rows": int(dev_elig_mask.sum()),
        "dev_id_rows": int(dev_id_sub.sum()),
        "dev_oos_rows": int(is_oos.sum()),
        "fit_bank_rows": int(fit_mask.sum()),
        "h2_calibration": cal_metrics,
        "methods": comparison,
        "smoke_samples_count": len(smoke_lines),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Selected method: {best_method} | threshold: {selected_thresholds[best_method]:.6f}")
    print(f"Saved artifacts to {args.artifact_dir} and metrics to {args.out}")


if __name__ == "__main__":
    main()
