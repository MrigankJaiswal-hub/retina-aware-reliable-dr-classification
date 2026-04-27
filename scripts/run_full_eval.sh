#!/bin/bash

python test.py \
  --test_csv data/splits/test_split.csv \
  --image_dir data/raw/train_images \
  --checkpoint outputs/aptos_exp1/best_model.pt \
  --temperature_file outputs/aptos_exp1/temperature_scaler.pt \
  --output_dir results/exp1_gradcam \
  --save_gradcam \
  --num_gradcam_images 25

python test_ecs.py \
  --test_csv data/splits/test_split.csv \
  --image_dir data/raw/train_images \
  --checkpoint outputs/aptos_exp1/best_model.pt \
  --temperature_file outputs/aptos_exp1/temperature_scaler.pt \
  --output_dir results/ecs_retina_exp1 \
  --save_gradcam \
  --num_gradcam_images 25 \
  --ecs_threshold 0.5