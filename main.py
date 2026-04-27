# main.py

import argparse
import os
import subprocess
import sys


def run_command(cmd):
    print(f"\n🚀 Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("❌ Error occurred.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Reliable DR Classification Main Pipeline"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "train",
            "test",
            "ecs",
            "ensemble",
            "sweep"
        ],
        help="Pipeline mode"
    )

    args = parser.parse_args()

    if args.mode == "train":
        cmd = """
python train.py \
--train_csv data/splits/train_split.csv \
--val_csv data/splits/val_split.csv \
--image_dir data/raw/train_images \
--output_dir outputs/aptos_exp1 \
--model_name efficientnet_b0 \
--img_size 224 \
--batch_size 8 \
--epochs 10 \
--lr 1e-4 \
--weight_decay 1e-4 \
--num_workers 0 \
--seed 42 \
--pretrained \
--crop_black \
--use_class_weights \
--calibrate
"""
        run_command(cmd)

    elif args.mode == "test":
        cmd = """
python test.py \
--test_csv data/splits/test_split.csv \
--image_dir data/raw/train_images \
--checkpoint outputs/aptos_exp1/best_model.pt \
--temperature_file outputs/aptos_exp1/temperature_scaler.pt \
--output_dir results/exp1
"""
        run_command(cmd)

    elif args.mode == "ecs":
        cmd = """
python test_ecs.py \
--test_csv data/splits/test_split.csv \
--image_dir data/raw/train_images \
--checkpoint outputs/aptos_exp1/best_model.pt \
--temperature_file outputs/aptos_exp1/temperature_scaler.pt \
--output_dir results/ecs_exp1 \
--save_gradcam \
--num_gradcam_images 10 \
--ecs_threshold 0.5
"""
        run_command(cmd)

    elif args.mode == "ensemble":
        cmd = """
python ensemble.py \
--test_csv data/splits/test_split.csv \
--image_dir data/raw/train_images \
--checkpoints \
outputs/aptos_exp1/best_model.pt \
outputs/aptos_seed123/best_model.pt \
outputs/aptos_seed777/best_model.pt
"""
        run_command(cmd)

    elif args.mode == "sweep":
        cmd = """
python plot_ecs_sweep.py \
--result_dirs \
results/ecs_retina_t03 \
results/ecs_retina_t04 \
results/ecs_retina_t05 \
results/ecs_retina_t06 \
results/ecs_retina_t07 \
--output_dir results/ecs_retina_sweep_summary
"""
        run_command(cmd)


if __name__ == "__main__":
    main()