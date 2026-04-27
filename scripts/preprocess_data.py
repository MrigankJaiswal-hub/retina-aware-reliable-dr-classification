import argparse
import os
from pathlib import Path

import pandas as pd


def verify_dataset(raw_dir: str):
    raw_dir = Path(raw_dir)

    train_csv = raw_dir / "train.csv"
    test_csv = raw_dir / "test.csv"
    train_images = raw_dir / "train_images"
    test_images = raw_dir / "test_images"

    required = [train_csv, test_csv, train_images, test_images]

    for item in required:
        if not item.exists():
            raise FileNotFoundError(f"Missing: {item}")

    df = pd.read_csv(train_csv)

    if "id_code" not in df.columns or "diagnosis" not in df.columns:
        raise ValueError("train.csv must contain id_code and diagnosis columns.")

    print("Dataset verification complete.")
    print(f"Training samples: {len(df)}")
    print("Class distribution:")
    print(df["diagnosis"].value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    args = parser.parse_args()

    verify_dataset(args.raw_dir)


if __name__ == "__main__":
    main()