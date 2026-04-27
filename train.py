import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.datasets.aptos_dataset import AptosDataset
from src.datasets.transforms import get_train_transforms, get_val_transforms
from src.evaluation.calibration import fit_temperature_scaler
from src.models.model_factory import build_model
from src.training.trainer import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_weighted_sampler(dataset: AptosDataset):
    labels = dataset.df["diagnosis"].values
    class_sample_count = np.array([(labels == t).sum() for t in np.unique(labels)])
    class_weights = 1.0 / class_sample_count
    sample_weights = np.array([class_weights[label] for label in labels], dtype=np.float64)

    sample_weights = torch.from_numpy(sample_weights).double()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def get_class_weights(dataset: AptosDataset, num_classes: int = 5):
    labels = dataset.df["diagnosis"].values
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    return torch.tensor(weights, dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Train APTOS DR classification model")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="outputs/exp1")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_clahe", action="store_true")
    parser.add_argument("--crop_black", action="store_true")
    parser.add_argument("--use_sampler", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--calibrate", action="store_true")

    parser.add_argument("--monitor_metric", type=str, default="qwk")
    parser.add_argument("--monitor_mode", type=str, default="max")
    parser.add_argument("--patience", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tfms = get_train_transforms(img_size=args.img_size)
    val_tfms = get_val_transforms(img_size=args.img_size)

    train_dataset = AptosDataset(
        csv_file=args.train_csv,
        image_dir=args.image_dir,
        transform=train_tfms,
        crop_black=args.crop_black,
        use_clahe=args.use_clahe,
    )

    val_dataset = AptosDataset(
        csv_file=args.val_csv,
        image_dir=args.image_dir,
        transform=val_tfms,
        crop_black=args.crop_black,
        use_clahe=args.use_clahe,
    )

    sampler = None
    shuffle = True
    if args.use_sampler:
        sampler = create_weighted_sampler(train_dataset)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = build_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    if args.use_class_weights:
        class_weights = get_class_weights(train_dataset, num_classes=args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using class-weighted CrossEntropyLoss")
        print("Class weights:", class_weights.detach().cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=args.output_dir,
        scheduler=scheduler,
        monitor_metric=args.monitor_metric,
        monitor_mode=args.monitor_mode,
        early_stopping_patience=args.patience,
        n_bins=15,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
    )

    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Training completed. Outputs saved to: {args.output_dir}")

    if args.calibrate:
        print("\nRunning temperature scaling on validation set...")
        collected = trainer.collect_logits(val_loader)
        logits = collected["logits"]
        labels = collected["labels"]

        scaler, result = fit_temperature_scaler(
            logits=logits,
            labels=labels,
            device=str(device),
        )

        print("Calibration results:")
        for key, value in result.items():
            print(f"  {key}: {value:.6f}")

        temp_path = os.path.join(args.output_dir, "temperature_scaler.pt")
        torch.save(
            {
                "temperature": scaler.get_temperature(),
                "result": result,
            },
            temp_path,
        )
        print(f"Saved temperature scaler to: {temp_path}")


if __name__ == "__main__":
    main()