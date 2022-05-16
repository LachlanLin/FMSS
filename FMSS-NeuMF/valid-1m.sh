#!/bin/bash
# This script is used to select hyperparameters on ML1M.

for lr in '0.0001' '0.0005' '0.001' '0.005'; do
  CUDA_VISIBLE_DEVICES=1 python3 main.py --path ./ML1M/ --train_data ML1M_copy1_train --test_data ML1M_copy1_valid --n 6040 --m 3952 --lr ${lr} >./valid_results/ML1M-lr-${lr}.txt
done
