import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from src.datasets.aptos_dataset import crop_black_borders
from src.datasets.transforms import get_val_transforms
from src.evaluation.calibration import apply_temperature
from src.explainability.ecs import compute_ecs
from src.explainability.gradcam import GradCAM, get_target_layer, overlay_cam_on_image
from src.explainability.retina_mask import create_retina_mask_from_rgb
from src.models.model_factory import build_model


CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


st.set_page_config(
    page_title="Reliable DR Classification",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_model(checkpoint_path, model_name="efficientnet_b0"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        model_name=model_name,
        num_classes=5,
        pretrained=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


def load_temperature(temperature_path):
    if temperature_path is None or temperature_path.strip() == "":
        return None

    if not os.path.exists(temperature_path):
        return None

    temp_data = torch.load(temperature_path, map_location="cpu")
    return temp_data["temperature"]


def preprocess_image(uploaded_image, img_size=224, crop_black=True):
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)

    if crop_black:
        image_np = crop_black_borders(image_np)

    image_pil = Image.fromarray(image_np)

    transform = get_val_transforms(img_size)
    image_tensor = transform(image_pil).unsqueeze(0)

    return image_np, image_tensor


def compute_single_prediction(model, image_tensor, device, temperature=None):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor).cpu()

    raw_probs = F.softmax(logits, dim=1)

    if temperature is not None:
        calibrated_logits = apply_temperature(logits, temperature)
        probs = F.softmax(calibrated_logits, dim=1)
    else:
        calibrated_logits = logits
        probs = raw_probs

    pred_idx = int(torch.argmax(probs, dim=1).item())
    confidence = float(torch.max(probs, dim=1).values.item())

    entropy = float((-torch.sum(probs * torch.log(probs + 1e-8), dim=1)).item())

    return {
        "logits": logits,
        "calibrated_logits": calibrated_logits,
        "probs": probs,
        "pred_idx": pred_idx,
        "confidence": confidence,
        "entropy": entropy,
    }


def generate_gradcam(model, image_tensor, device, model_name, class_idx):
    image_tensor = image_tensor.to(device)

    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    cam = gradcam.generate(image_tensor, class_idx=class_idx)
    gradcam.remove_hooks()

    return cam


def resize_original_for_overlay(image_np, img_size=224):
    resized = cv2.resize(image_np, (img_size, img_size))
    return resized


def compute_retina_aware_ecs(probs, cam, retina_mask):
    cam_tensor = torch.from_numpy(cam).float().unsqueeze(0)
    mask_tensor = torch.from_numpy(retina_mask).float().unsqueeze(0)

    ecs = compute_ecs(
        probs=probs,
        cam_batch=cam_tensor,
        retina_mask_batch=mask_tensor,
        alpha=0.4,
        beta=0.3,
        gamma=0.2,
        delta=0.1,
    )

    return float(ecs.item())


def main():
    st.title("🧠 Retina-Aware Reliable Diabetic Retinopathy Classification")
    st.markdown(
        """
        This demo classifies diabetic retinopathy severity and estimates whether the prediction is reliable using:

        **EfficientNet-B0 + Temperature Scaling + Grad-CAM + Retina-aware ECS**
        """
    )

    with st.sidebar:
        st.header("Model Settings")

        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="outputs/aptos_exp1/best_model.pt",
        )

        temperature_path = st.text_input(
            "Temperature File",
            value="outputs/aptos_exp1/temperature_scaler.pt",
        )

        model_name = st.selectbox(
            "Model",
            ["efficientnet_b0", "efficientnet_b3", "resnet50", "densenet121"],
            index=0,
        )

        img_size = st.selectbox(
            "Image Size",
            [224, 256, 300],
            index=0,
        )

        ecs_threshold = st.slider(
            "ECS Reliability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

        crop_black = st.checkbox("Crop black borders", value=True)

    uploaded_file = st.file_uploader(
        "Upload a retinal fundus image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is None:
        st.info("Upload a fundus image to run the demo.")
        return

    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found: {checkpoint_path}")
        return

    model, device = load_model(checkpoint_path, model_name)
    temperature = load_temperature(temperature_path)

    original_np, image_tensor = preprocess_image(
        uploaded_file,
        img_size=img_size,
        crop_black=crop_black,
    )

    result = compute_single_prediction(
        model=model,
        image_tensor=image_tensor,
        device=device,
        temperature=temperature,
    )

    cam = generate_gradcam(
        model=model,
        image_tensor=image_tensor,
        device=device,
        model_name=model_name,
        class_idx=result["pred_idx"],
    )

    original_resized = resize_original_for_overlay(original_np, img_size=img_size)
    overlay = overlay_cam_on_image(original_resized, cam, alpha=0.4)

    retina_mask = create_retina_mask_from_rgb(original_resized)
    ecs_score = compute_retina_aware_ecs(
        probs=result["probs"],
        cam=cam,
        retina_mask=retina_mask,
    )

    decision = "ACCEPT" if ecs_score >= ecs_threshold else "REJECT / REFER TO EXPERT"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Fundus Image")
        st.image(original_np, use_container_width=True)

    with col2:
        st.subheader("Grad-CAM Explanation")
        st.image(overlay, use_container_width=True)

    st.divider()

    col3, col4, col5, col6 = st.columns(4)

    with col3:
        st.metric("Prediction", CLASS_NAMES[result["pred_idx"]])

    with col4:
        st.metric("Confidence", f"{result['confidence']:.4f}")

    with col5:
        st.metric("Entropy", f"{result['entropy']:.4f}")

    with col6:
        st.metric("ECS Score", f"{ecs_score:.4f}")

    if decision == "ACCEPT":
        st.success(f"✅ Decision: {decision}")
    else:
        st.warning(f"⚠️ Decision: {decision}")

    st.subheader("Class Probabilities")

    probs_np = result["probs"].squeeze(0).detach().cpu().numpy()

    prob_df = pd.DataFrame({
        "Class": [CLASS_NAMES[i] for i in range(5)],
        "Probability": probs_np,
    })

    st.bar_chart(prob_df.set_index("Class"))

    st.subheader("Reliability Interpretation")

    st.markdown(
        f"""
        - The model predicts **{CLASS_NAMES[result['pred_idx']]}**.
        - Calibrated confidence is **{result['confidence']:.4f}**.
        - Entropy is **{result['entropy']:.4f}**.
        - Retina-aware ECS is **{ecs_score:.4f}**.
        - With threshold **{ecs_threshold:.2f}**, the system decision is: **{decision}**.
        """
    )


if __name__ == "__main__":
    main()