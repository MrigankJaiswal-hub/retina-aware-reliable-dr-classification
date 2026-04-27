import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_ecs_result(result_dir: str):
    metrics_path = os.path.join(result_dir, "ecs_metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Missing file: {metrics_path}")

    with open(metrics_path, "r") as f:
        data = json.load(f)

    row = {
        "threshold": data["ecs_threshold"],
        "ecs_coverage": data["ecs_selective_metrics"]["coverage"],
        "ecs_selective_accuracy": data["ecs_selective_metrics"]["selective_accuracy"],
        "ecs_num_selected": data["ecs_selective_metrics"]["num_selected"],
        "conf_coverage": data["confidence_selective_metrics"]["coverage"],
        "conf_selective_accuracy": data["confidence_selective_metrics"]["selective_accuracy"],
        "conf_num_selected": data["confidence_selective_metrics"]["num_selected"],
        "ecs_score_mean": data["ecs_score_mean"],
        "ecs_score_std": data["ecs_score_std"],
        "ecs_score_min": data["ecs_score_min"],
        "ecs_score_max": data["ecs_score_max"],
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="Plot ECS threshold sweep results")
    parser.add_argument(
        "--result_dirs",
        nargs="+",
        required=True,
        help="List of result directories, each containing ecs_metrics.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ecs_sweep_summary",
        help="Directory to save summary CSV and plots",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for result_dir in args.result_dirs:
        rows.append(load_ecs_result(result_dir))

    df = pd.DataFrame(rows)
    df = df.sort_values("threshold").reset_index(drop=True)

    # Add risk columns
    df["ecs_risk"] = 1.0 - df["ecs_selective_accuracy"]
    df["conf_risk"] = 1.0 - df["conf_selective_accuracy"]

    csv_path = os.path.join(args.output_dir, "ecs_threshold_sweep_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to: {csv_path}")
    print("\nSummary:")
    print(df)

    # Plot 1: Selective Accuracy vs Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["ecs_selective_accuracy"], marker="o", label="Retina-aware ECS")
    plt.plot(df["threshold"], df["conf_selective_accuracy"], marker="o", label="Confidence-only")
    plt.xlabel("Threshold")
    plt.ylabel("Selective Accuracy")
    plt.title("Selective Accuracy vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "selective_accuracy_vs_threshold.png"), dpi=300)
    plt.close()

    # Plot 2: Coverage vs Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["ecs_coverage"], marker="o", label="Retina-aware ECS")
    plt.plot(df["threshold"], df["conf_coverage"], marker="o", label="Confidence-only")
    plt.xlabel("Threshold")
    plt.ylabel("Coverage")
    plt.title("Coverage vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "coverage_vs_threshold.png"), dpi=300)
    plt.close()

    # Plot 3: Risk-Coverage Curve
    plt.figure(figsize=(8, 5))
    plt.plot(df["ecs_coverage"], df["ecs_risk"], marker="o", label="Retina-aware ECS")
    plt.plot(df["conf_coverage"], df["conf_risk"], marker="o", label="Confidence-only")
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - Selective Accuracy)")
    plt.title("Risk-Coverage Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "risk_coverage_curve.png"), dpi=300)
    plt.close()

    print(f"\nSaved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()