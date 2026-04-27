from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)


def preds_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "qwk": float(qwk),
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def compute_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    loss = F.cross_entropy(logits, labels, reduction="mean")
    return float(loss.item())


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    score = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(score)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    return float(ece)


def compute_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    max_error = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.any(in_bin):
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            max_error = max(max_error, np.abs(acc_in_bin - conf_in_bin))

    return float(max_error)


def compute_error_detection_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    errors = (predictions != labels).astype(np.int32)
    uncertainty_score = 1.0 - confidences

    if len(np.unique(errors)) < 2:
        return 0.5

    return float(roc_auc_score(errors, uncertainty_score))


def summarize_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> Dict[str, float]:
    probs_t = probs_from_logits(logits)
    preds_t = preds_from_logits(logits)

    probs = probs_t.detach().cpu().numpy()
    preds = preds_t.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    cls_metrics = compute_classification_metrics(y_true, preds)
    nll = compute_nll(logits, labels)
    brier = compute_brier_score(probs, y_true, num_classes=probs.shape[1])
    ece = compute_ece(probs, y_true, n_bins=n_bins)
    mce = compute_mce(probs, y_true, n_bins=n_bins)
    error_auroc = compute_error_detection_auroc(probs, y_true)

    return {
        **cls_metrics,
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "mce": mce,
        "error_detection_auroc": error_auroc,
    }