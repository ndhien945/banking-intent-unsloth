#!/bin/bash

# Dừng script ngay lập tức nếu có bất kỳ lệnh nào bị lỗi
set -e

export CUDA_VISIBLE_DEVICES=0

echo -e "\n1. Preprocessing..."
python scripts/preprocess_data.py --num_train_samples 5390 --num_test_samples 1540

echo -e "\n2. Fine-tuning..."
python scripts/train.py

echo -e "\nDone"