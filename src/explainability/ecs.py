import torch

from src.explainability.retina_mask import batch_retina_overlap_score


def minmax_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_min = torch.min(x)
    x_max = torch.max(x)
    return (x - x_min) / (x_max - x_min + eps)


def compute_confidence(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: shape (B, C)
    returns: shape (B,)
    """
    return torch.max(probs, dim=1).values


def compute_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    probs: shape (B, C)
    returns entropy per sample: shape (B,)
    """
    return -torch.sum(probs * torch.log(probs + eps), dim=1)


def compute_cam_focus_score(cam_batch: torch.Tensor) -> torch.Tensor:
    """
    cam_batch: shape (B, H, W), values in [0,1]

    Focus score = mean activation of top 20% CAM pixels
    returns: shape (B,)
    """
    B = cam_batch.shape[0]
    flat = cam_batch.view(B, -1)

    k = max(1, int(0.2 * flat.shape[1]))
    topk_vals, _ = torch.topk(flat, k=k, dim=1)

    focus_score = topk_vals.mean(dim=1)
    return focus_score


def compute_ecs(
    probs: torch.Tensor,
    cam_batch: torch.Tensor,
    retina_mask_batch: torch.Tensor,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.1,
) -> torch.Tensor:
    """
    Updated ECS using retina overlap.

    ECS =
        alpha * normalized_confidence
      + beta  * normalized_retina_overlap
      + gamma * normalized_focus_score
      + delta * (1 - normalized_entropy)

    Args:
        probs: tensor (B, C)
        cam_batch: tensor (B, H, W)
        retina_mask_batch: tensor (B, H, W)

    Returns:
        ecs: tensor (B,)
    """
    confidence = compute_confidence(probs)
    entropy = compute_entropy(probs)
    focus_score = compute_cam_focus_score(cam_batch)
    retina_overlap = batch_retina_overlap_score(cam_batch, retina_mask_batch)

    confidence_n = minmax_normalize(confidence)
    entropy_n = minmax_normalize(entropy)
    focus_n = minmax_normalize(focus_score)
    retina_n = minmax_normalize(retina_overlap)

    ecs = (
        alpha * confidence_n
        + beta * retina_n
        + gamma * focus_n
        + delta * (1.0 - entropy_n)
    )

    return ecs


def selective_metrics(
    ecs_scores: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Computes coverage and selective accuracy.
    """
    selected = ecs_scores >= threshold

    coverage = selected.float().mean().item()

    if selected.sum() == 0:
        selective_acc = 0.0
    else:
        selective_acc = (preds[selected] == labels[selected]).float().mean().item()

    return {
        "coverage": coverage,
        "selective_accuracy": selective_acc,
        "num_selected": int(selected.sum().item()),
        "num_total": int(len(labels)),
    }