import torch
import numpy as np

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def compute_caes(probs, cam):
    """
    probs: [B, C]
    cam: [B, H, W]
    """

    confidence = torch.max(probs, dim=1).values
    entropy = compute_entropy(probs)

    # CAM focus score (mean activation)
    cam_score = cam.view(cam.size(0), -1).mean(dim=1)

    # Normalize everything
    confidence = normalize(confidence)
    entropy = normalize(entropy)
    cam_score = normalize(cam_score)

    # Final CAES score
    caes = confidence * cam_score * (1 - entropy)

    return caes