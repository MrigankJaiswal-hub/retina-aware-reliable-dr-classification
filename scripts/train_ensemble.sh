#!/bin/bash

for SEED in 42 123 777
do
  python train.py \
    --train_csv data/splits/train_split.csv \
    --val_csv data/splits/val_split.csv \
    --image_dir data/raw/train_images \
    --output_dir outputs/aptos_seed${SEED} \
    --model_name efficientnet_b0 \
    --img_size 224 \
    --batch_size 8 \
    --epochs 10 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_workers 0 \
    --seed ${SEED} \
    --pretrained \
    --crop_black \
    --use_class_weights \
    --calibrate
done