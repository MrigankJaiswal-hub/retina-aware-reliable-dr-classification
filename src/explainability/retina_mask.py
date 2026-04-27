import cv2
import numpy as np
import torch


def create_retina_mask_from_rgb(image_rgb: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Create a rough binary retina mask from an RGB fundus image.

    Args:
        image_rgb: uint8 RGB image of shape (H, W, 3)
        threshold: grayscale threshold to separate retina from black background

    Returns:
        mask: float32 numpy array of shape (H, W), values in {0,1}
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Input image_rgb is empty.")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Basic threshold for non-black region
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Keep the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_mask = (labels == largest_idx).astype(np.uint8) * 255
    else:
        largest_mask = mask

    # Smooth / refine mask
    kernel = np.ones((7, 7), np.uint8)
    largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_CLOSE, kernel)
    largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel)

    mask_float = (largest_mask > 0).astype(np.float32)
    return mask_float


def resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize binary mask to target size.
    """
    resized = cv2.resize(mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    resized = (resized > 0.5).astype(np.float32)
    return resized


def retina_overlap_score(cam: np.ndarray, retina_mask: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute how much CAM energy lies inside retina region.

    Args:
        cam: Grad-CAM map of shape (H, W), values in [0,1]
        retina_mask: binary mask of shape (H, W), values in {0,1}

    Returns:
        score in [0,1]
    """
    if cam.shape != retina_mask.shape:
        raise ValueError(f"Shape mismatch: cam {cam.shape}, retina_mask {retina_mask.shape}")

    cam = cam.astype(np.float32)
    retina_mask = retina_mask.astype(np.float32)

    total_energy = np.sum(cam) + eps
    inside_energy = np.sum(cam * retina_mask)

    score = inside_energy / total_energy
    return float(score)


def batch_retina_overlap_score(cam_batch: torch.Tensor, retina_mask_batch: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Torch batch version.

    Args:
        cam_batch: tensor (B, H, W)
        retina_mask_batch: tensor (B, H, W)

    Returns:
        scores: tensor (B,)
    """
    if cam_batch.shape != retina_mask_batch.shape:
        raise ValueError(
            f"Shape mismatch: cam_batch {cam_batch.shape}, retina_mask_batch {retina_mask_batch.shape}"
        )

    total_energy = cam_batch.view(cam_batch.size(0), -1).sum(dim=1) + eps
    inside_energy = (cam_batch * retina_mask_batch).view(cam_batch.size(0), -1).sum(dim=1)

    return inside_energy / total_energy