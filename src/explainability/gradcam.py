import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Generic Grad-CAM implementation for torchvision classification models.

    Usage:
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(input_tensor, class_idx=None)

    Returns:
        cam: numpy array of shape (H, W) normalized to [0, 1]
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = None
        self.backward_handle = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()
        if self.backward_handle is not None:
            self.backward_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM for a single image tensor.

        Args:
            input_tensor: shape (1, C, H, W)
            class_idx: target class index. If None, uses predicted class.

        Returns:
            cam: numpy array (H, W), normalized to [0,1]
        """
        self.model.zero_grad()

        output = self.model(input_tensor)  # shape (1, num_classes)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        score = output[:, class_idx]
        score.backward(retain_graph=True)

        activations = self.activations  # (1, C, h, w)
        gradients = self.gradients      # (1, C, h, w)

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.astype(np.float32)


def get_target_layer(model: torch.nn.Module, model_name: str):
    """
    Returns the target layer for Grad-CAM based on model type.
    """
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        return model.features[-1]
    elif model_name == "efficientnet_b3":
        return model.features[-1]
    elif model_name == "resnet50":
        return model.layer4[-1]
    elif model_name == "densenet121":
        return model.features
    else:
        raise ValueError(f"Unsupported model_name for Grad-CAM: {model_name}")


def denormalize_image(
    image_tensor: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Convert normalized tensor image to uint8 RGB image.
    Input: tensor (C, H, W)
    Output: numpy RGB image (H, W, 3)
    """
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return image


def overlay_cam_on_image(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay heatmap on RGB image.
    Returns RGB uint8 image.
    """
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay


def save_cam_image(
    image_tensor: torch.Tensor,
    cam: np.ndarray,
    save_path: str,
    alpha: float = 0.4,
):
    """
    Save Grad-CAM overlay image to disk.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image_rgb = denormalize_image(image_tensor)
    overlay = overlay_cam_on_image(image_rgb, cam, alpha=alpha)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))