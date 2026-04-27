import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.aptos_dataset import AptosDataset
from src.datasets.transforms import get_val_transforms
from src.evaluation.calibration import apply_temperature
from src.evaluation.metrics import summarize_from_logits
from src.explainability.gradcam import GradCAM, get_target_layer, save_cam_image
from src.models.model_factory import build_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--temperature_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/exp1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--save_gradcam", action="store_true")
    parser.add_argument("--num_gradcam_images", type=int, default=25)

    return parser.parse_args()


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []
    all_ids = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        ids = batch["id"]

        logits = model(images)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_ids.extend(ids)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return logits, labels, all_ids


def save_prediction_csv(
    ids,
    labels,
    logits,
    save_path,
    calibrated_logits=None,
):
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    confs = torch.max(probs, dim=1).values

    calibrated_probs = None
    calibrated_preds = None
    calibrated_confs = None

    if calibrated_logits is not None:
        calibrated_probs = F.softmax(calibrated_logits, dim=1)
        calibrated_preds = torch.argmax(calibrated_probs, dim=1)
        calibrated_confs = torch.max(calibrated_probs, dim=1).values

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "id",
            "true_label",
            "pred_label_raw",
            "confidence_raw",
        ]

        if calibrated_logits is not None:
            header += [
                "pred_label_calibrated",
                "confidence_calibrated",
            ]

        writer.writerow(header)

        for i in range(len(ids)):
            row = [
                ids[i],
                int(labels[i].item()),
                int(preds[i].item()),
                float(confs[i].item()),
            ]

            if calibrated_logits is not None:
                row += [
                    int(calibrated_preds[i].item()),
                    float(calibrated_confs[i].item()),
                ]

            writer.writerow(row)


def generate_gradcam_examples(
    model,
    dataset,
    device,
    model_name,
    output_dir,
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

        output = model(image)
        pred = int(torch.argmax(output, dim=1).item())

        cam = gradcam.generate(image, class_idx=pred)

        filename = f"{idx:03d}_{image_id}_true{label}_pred{pred}.png"
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

    logits, labels, ids = collect_logits(model, test_loader, device)

    # Raw metrics
    raw_metrics = summarize_from_logits(logits, labels)

    print("\n📊 RAW TEST METRICS")
    for key, value in raw_metrics.items():
        print(f"{key}: {value:.4f}")

    with open(os.path.join(args.output_dir, "raw_metrics.json"), "w") as f:
        json.dump(raw_metrics, f, indent=2)

    calibrated_logits = None

    # Calibrated metrics
    if args.temperature_file is not None:
        temp_data = torch.load(args.temperature_file, map_location=device)
        temperature = temp_data["temperature"]

        calibrated_logits = apply_temperature(logits, temperature)
        calibrated_metrics = summarize_from_logits(calibrated_logits, labels)

        print("\n🔥 CALIBRATED TEST METRICS")
        for key, value in calibrated_metrics.items():
            print(f"{key}: {value:.4f}")

        with open(os.path.join(args.output_dir, "calibrated_metrics.json"), "w") as f:
            json.dump(calibrated_metrics, f, indent=2)

    # Save prediction CSV
    pred_csv_path = os.path.join(args.output_dir, "test_predictions.csv")
    save_prediction_csv(
        ids=ids,
        labels=labels,
        logits=logits,
        calibrated_logits=calibrated_logits,
        save_path=pred_csv_path,
    )
    print(f"Saved prediction CSV to: {pred_csv_path}")

    # Save Grad-CAM examples
    if args.save_gradcam:
        # Use raw model for explanation
        generate_gradcam_examples(
            model=model,
            dataset=test_dataset,
            device=device,
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_images=args.num_gradcam_images,
        )


if __name__ == "__main__":
    main()