import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    input_csv: str,
    output_dir: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    df = pd.read_csv(input_csv)

    if "id_code" not in df.columns or "diagnosis" not in df.columns:
        raise ValueError("Input CSV must contain 'id_code' and 'diagnosis' columns.")

    os.makedirs(output_dir, exist_ok=True)

    # First split: train vs temp
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=df["diagnosis"],
        random_state=random_state,
    )

    # Second split: val vs test from temp
    relative_test_size = test_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["diagnosis"],
        random_state=random_state,
    )

    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")
    test_path = os.path.join(output_dir, "test_split.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Splits created successfully:")
    print(f"  Train: {train_path} ({len(train_df)} samples)")
    print(f"  Val:   {val_path} ({len(val_df)} samples)")
    print(f"  Test:  {test_path} ({len(test_df)} samples)")

    print("\nClass distribution:")
    print("Train:")
    print(train_df["diagnosis"].value_counts(normalize=True).sort_index())
    print("\nVal:")
    print(val_df["diagnosis"].value_counts(normalize=True).sort_index())
    print("\nTest:")
    print(test_df["diagnosis"].value_counts(normalize=True).sort_index())


def parse_args():
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits for APTOS.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to original train.csv")
    parser.add_argument("--output_dir", type=str, default="data/splits", help="Directory to save split CSV files")
    parser.add_argument("--train_size", type=float, default=0.70)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_splits(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
    )