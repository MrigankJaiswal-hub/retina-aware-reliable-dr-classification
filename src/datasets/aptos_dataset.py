import os
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def crop_black_borders(image: np.ndarray, threshold: int = 7) -> np.ndarray:
    """
    Crop black borders from retinal fundus images.
    Args:
        image: RGB image as numpy array of shape (H, W, 3)
        threshold: Pixel intensity threshold for non-black region
    Returns:
        Cropped RGB image
    """
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold

    if not np.any(mask):
        return image

    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return image[y_min:y_max + 1, x_min:x_max + 1]


def apply_clahe_rgb(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE enhancement in LAB color space.
    Args:
        image: RGB image as numpy array
    Returns:
        Enhanced RGB image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return enhanced


class AptosDataset(Dataset):
    """
    APTOS dataset loader.

    Expected CSV columns:
    - id_code
    - diagnosis (optional for inference/test CSV)

    Example:
        id_code,diagnosis
        000c1434d8d7,2
        001639a390f0,4
    """

    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        transform: Optional[Callable] = None,
        image_ext: str = ".png",
        crop_black: bool = True,
        use_clahe: bool = False,
        return_path: bool = False,
    ) -> None:
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.image_ext = image_ext
        self.crop_black = crop_black
        self.use_clahe = use_clahe
        self.return_path = return_path

        if "id_code" not in self.df.columns:
            raise ValueError("CSV must contain 'id_code' column.")

        self.has_labels = "diagnosis" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_id: str) -> np.ndarray:
        img_path = os.path.join(self.image_dir, f"{image_id}{self.image_ext}")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.crop_black:
            image = crop_black_borders(image)

        if self.use_clahe:
            image = apply_clahe_rgb(image)

        return image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_id = str(row["id_code"])
        image = self._load_image(image_id)

        pil_image = Image.fromarray(image)

        if self.transform is not None:
            image_tensor = self.transform(pil_image)
        else:
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        sample: Dict[str, Any] = {
            "image": image_tensor,
            "id": image_id,
        }

        if self.has_labels:
            sample["label"] = torch.tensor(int(row["diagnosis"]), dtype=torch.long)

        if self.return_path:
            sample["path"] = os.path.join(self.image_dir, f"{image_id}{self.image_ext}")

        return sample