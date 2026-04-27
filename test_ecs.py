import argparse
import csv
import json
import os
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.aptos_dataset import AptosDataset
from src.datasets.transforms import get_val_transforms
from src.evaluation.calibration import apply_temperature
from src.evaluation.metrics import summarize_from_logits
from src.explainability.ecs import compute_ecs, selective_metrics
from src.explainability.gradcam import (
    GradCAM,
    denormalize_image,
    get_target_layer,
    save_cam_image,
)
from src.explainability.retina_mask import create_retina_mask_from_rgb
from src.models.model_factory import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model with retina-aware ECS on test set")

    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--temperature_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/ecs_exp1")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--num_classes", type=int, default=5)

    parser.add_argument("--ecs_threshold", type=float, default=0.5)

    parser.add_argument("--save_gradcam", action="store_true")
    parser.add_argument("--num_gradcam_images", type=int, default=25)

    return parser.parse_args()


@torch.no_grad()
def collect_logits_and_labels(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []
    all_ids = []
    all_images = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        ids = batch["id"]

        logits = model(images)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_ids.extend(ids)
        all_images.append(batch["image"].cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    images = torch.cat(all_images, dim=0)

    return logits, labels, all_ids, images


def generate_cam_batch(
    model,
    dataset,
    device,
    model_name: str,
    calibrated_logits: torch.Tensor = None,
) -> torch.Tensor:
    """
    Generate one CAM per sample.
    Returns tensor of shape (N, H, W).
    """
    model.eval()
    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    cams = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)

        if calibrated_logits is not None:
            class_idx = int(torch.argmax(calibrated_logits[idx]).item())
        else:
            with torch.no_grad():
                pred_logits = model(image)
                class_idx = int(torch.argmax(pred_logits, dim=1).item())

        cam = gradcam.generate(image, class_idx=class_idx)
        cam_tensor = torch.from_numpy(cam).float()
        cams.append(cam_tensor)

    gradcam.remove_hooks()
    return torch.stack(cams, dim=0)


def generate_retina_mask_batch_from_dataset(dataset) -> torch.Tensor:
    """
    Create retina masks for all dataset samples.
    Returns tensor of shape (N, H, W).
    """
    retina_masks = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image_tensor = sample["image"]  # (C, H, W)

        image_rgb = denormalize_image(image_tensor)
        retina_mask = create_retina_mask_from_rgb(image_rgb)
        retina_mask_t = torch.from_numpy(retina_mask).float()

        retina_masks.append(retina_mask_t)

    return torch.stack(retina_masks, dim=0)


def save_prediction_ecs_csv(
    ids: List[str],
    labels: torch.Tensor,
    raw_logits: torch.Tensor,
    calibrated_logits: torch.Tensor,
    ecs_scores: torch.Tensor,
    threshold: float,
    save_path: str,
):
    raw_probs = F.softmax(raw_logits, dim=1)
    raw_preds = torch.argmax(raw_probs, dim=1)
    raw_confs = torch.max(raw_probs, dim=1).values

    cal_probs = F.softmax(calibrated_logits, dim=1)
    cal_preds = torch.argmax(cal_probs, dim=1)
    cal_confs = torch.max(cal_probs, dim=1).values

    selected = ecs_scores >= threshold

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "true_label",
            "pred_label_raw",
            "confidence_raw",
            "pred_label_calibrated",
            "confidence_calibrated",
            "ecs_score",
            "selected_by_ecs",
            "correct_raw",
            "correct_calibrated",
        ])

        for i in range(len(ids)):
            writer.writerow([
                ids[i],
                int(labels[i].item()),
                int(raw_preds[i].item()),
                float(raw_confs[i].item()),
                int(cal_preds[i].item()),
                float(cal_confs[i].item()),
                float(ecs_scores[i].item()),
                int(selected[i].item()),
                int(raw_preds[i].item() == labels[i].item()),
                int(cal_preds[i].item() == labels[i].item()),
            ])


def confidence_selective_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
):
    preds = torch.argmax(probs, dim=1)
    confs = torch.max(probs, dim=1).values
    selected = confs >= threshold

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


def save_gradcam_examples(
    model,
    dataset,
    device,
    model_name,
    output_dir,
    calibrated_logits=None,
    ecs_scores=None,
    num_images=25,
):
    model.eval()
    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    gradcam_dir = os.path.join(output_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    limit = min(num_images, len(dataset))

    for idx in range(limit):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        label = int(sample["label"].item())
        image_id = sample["id"]

        if calibrated_logits is not None:
            pred = int(torch.argmax(calibrated_logits[idx]).item())
        else:
            with torch.no_grad():
                pred = int(torch.argmax(model(image), dim=1).item())

        cam = gradcam.generate(image, class_idx=pred)

        ecs_suffix = ""
        if ecs_scores is not None:
            ecs_suffix = f"_ecs{ecs_scores[idx].item():.3f}"

        filename = f"{idx:03d}_{image_id}_true{label}_pred{pred}{ecs_suffix}.png"
        save_path = os.path.join(gradcam_dir, filename)
        save_cam_image(sample["image"], cam, save_path)

    gradcam.remove_hooks()
    print(f"Saved Grad-CAM images to: {gradcam_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = AptosDataset(
        csv_file=args.test_csv,
        image_dir=args.image_dir,
        transform=get_val_transforms(args.img_size),
        crop_black=True,
        use_clahe=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=False,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Raw logits
    raw_logits, labels, ids, _ = collect_logits_and_labels(model, test_loader, device)

    print("\n📊 RAW TEST METRICS")
    raw_metrics = summarize_from_logits(raw_logits, labels)
    for key, value in raw_metrics.items():
        print(f"{key}: {value:.4f}")

    with open(os.path.join(args.output_dir, "raw_metrics.json"), "w") as f:
        json.dump(raw_metrics, f, indent=2)

    # Calibrated logits
    calibrated_logits = raw_logits.clone()
    calibrated_metrics = None
    temperature = None

    if args.temperature_file is not None:
        temp_data = torch.load(args.temperature_file, map_location=device)
        temperature = temp_data["temperature"]
        calibrated_logits = apply_temperature(raw_logits, temperature)

        print("\n🔥 CALIBRATED TEST METRICS")
        calibrated_metrics = summarize_from_logits(calibrated_logits, labels)
        for key, value in calibrated_metrics.items():
            print(f"{key}: {value:.4f}")

        with open(os.path.join(args.output_dir, "calibrated_metrics.json"), "w") as f:
            json.dump(calibrated_metrics, f, indent=2)

    # Probs and preds from calibrated logits
    calibrated_probs = F.softmax(calibrated_logits, dim=1)
    calibrated_preds = torch.argmax(calibrated_probs, dim=1)

    # Generate CAMs
    print("\nGenerating Grad-CAM maps for ECS...")
    cam_batch = generate_cam_batch(
        model=model,
        dataset=test_dataset,
        device=device,
        model_name=args.model_name,
        calibrated_logits=calibrated_logits,
    )

    # Generate retina masks
    print("Generating retina masks...")
    retina_mask_batch = generate_retina_mask_batch_from_dataset(test_dataset)

    # Compute retina-aware ECS
    ecs_scores = compute_ecs(
        probs=calibrated_probs,
        cam_batch=cam_batch,
        retina_mask_batch=retina_mask_batch,
        alpha=0.4,
        beta=0.3,
        gamma=0.2,
        delta=0.1,
    )

    ecs_result = selective_metrics(
        ecs_scores=ecs_scores,
        preds=calibrated_preds,
        labels=labels,
        threshold=args.ecs_threshold,
    )

    conf_result = confidence_selective_metrics(
        probs=calibrated_probs,
        labels=labels,
        threshold=args.ecs_threshold,
    )

    print("\n🧠 ECS SELECTIVE METRICS")
    for key, value in ecs_result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\n📌 CONFIDENCE-ONLY SELECTIVE METRICS")
    for key, value in conf_result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    ecs_summary = {
        "ecs_threshold": args.ecs_threshold,
        "temperature": temperature,
        "ecs_selective_metrics": ecs_result,
        "confidence_selective_metrics": conf_result,
        "ecs_score_mean": float(ecs_scores.mean().item()),
        "ecs_score_std": float(ecs_scores.std().item()),
        "ecs_score_min": float(ecs_scores.min().item()),
        "ecs_score_max": float(ecs_scores.max().item()),
    }

    with open(os.path.join(args.output_dir, "ecs_metrics.json"), "w") as f:
        json.dump(ecs_summary, f, indent=2)

    csv_path = os.path.join(args.output_dir, "test_predictions_ecs.csv")
    save_prediction_ecs_csv(
        ids=ids,
        labels=labels,
        raw_logits=raw_logits,
        calibrated_logits=calibrated_logits,
        ecs_scores=ecs_scores,
        threshold=args.ecs_threshold,
        save_path=csv_path,
    )
    print(f"\nSaved ECS prediction CSV to: {csv_path}")

    if args.save_gradcam:
        save_gradcam_examples(
            model=model,
            dataset=test_dataset,
            device=device,
            model_name=args.model_name,
            output_dir=args.output_dir,
            calibrated_logits=calibrated_logits,
            ecs_scores=ecs_scores,
            num_images=args.num_gradcam_images,
        )


if __name__ == "__main__":
    main()