"""
Evaluation metrics — Section 3.8.

Computes per-split and per-degradation accuracy, macro precision/recall/F1,
confusion matrices, robustness score, and efficiency stats.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 43,
) -> dict:
    """
    y_true, y_pred: 1D int arrays of equal length.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class_prec, per_class_rec, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return {
        "accuracy": acc,
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "per_class_precision": per_class_prec.tolist(),
        "per_class_recall": per_class_rec.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def predict_labels(model, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    probs = model.predict(X, batch_size=batch_size, verbose=0)
    return np.argmax(probs, axis=1)


def robustness_score(acc_clean: float, acc_degraded_mean: float) -> float:
    """
    Section 3.8(iii): ratio of mean degraded accuracy to clean accuracy.
    """
    if acc_clean == 0:
        return 0.0
    return float(acc_degraded_mean / acc_clean)


def evaluate_with_degradations(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    degradation_grid: list[tuple[str, list]],
    batch_size: int = 64,
) -> dict:
    """
    Evaluate on clean test set and on each degradation/severity combination.

    degradation_grid: list of (degradation_name, [severity_values])
      e.g. [("fog", ["light","moderate","heavy"]), ("low_light", [0.7,0.5,0.3]), ...]
    Returns dict with clean metrics + per-condition metrics + robustness score.
    """
    from src.augment.degradations import apply_degradation_by_name

    # Clean
    y_pred_clean = predict_labels(model, X_test, batch_size=batch_size)
    clean_metrics = evaluate_classifier(y_test, y_pred_clean)
    acc_clean = clean_metrics["accuracy"]

    per_condition: dict[str, dict] = {}
    degraded_accs: list[float] = []

    for deg_name, severities in degradation_grid:
        for sev in severities:
            # Apply degradation to entire test set
            X_deg = np.stack(
                [apply_degradation_by_name(x, deg_name, sev) for x in X_test],
                axis=0,
            ).astype(np.float32)
            y_pred = predict_labels(model, X_deg, batch_size=batch_size)
            m = evaluate_classifier(y_test, y_pred)
            key = f"{deg_name}/{sev}"
            per_condition[key] = m
            degraded_accs.append(m["accuracy"])

    mean_degraded = float(np.mean(degraded_accs)) if degraded_accs else 0.0
    r = robustness_score(acc_clean, mean_degraded)

    return {
        "clean": clean_metrics,
        "per_condition": per_condition,
        "mean_degraded_accuracy": mean_degraded,
        "robustness_score": r,
    }
