import torch
import argparse
import json
import os
from torch.utils.data import DataLoader

from src.datasets.aptos_dataset import AptosDataset
from src.datasets.transforms import get_val_transforms
from src.models.model_factory import build_model
from src.evaluation.metrics import summarize_from_logits


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output_dir", default="results/ensemble")

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AptosDataset(
        csv_file=args.test_csv,
        image_dir=args.image_dir,
        transform=get_val_transforms(224),
        crop_black=True,
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    models = []
    for ckpt_path in args.checkpoints:
        model = build_model("efficientnet_b0", num_classes=5, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    all_logits = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        logits_sum = 0
        for model in models:
            logits_sum += model(images)

        logits_avg = logits_sum / len(models)

        all_logits.append(logits_avg.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    metrics = summarize_from_logits(logits, labels)

    print("\n🚀 ENSEMBLE RESULTS")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    with open(os.path.join(args.output_dir, "ensemble_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main() 