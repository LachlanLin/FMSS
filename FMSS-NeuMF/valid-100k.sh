#!/bin/bash
# This script is used to select hyperparameters on ML100K.

for lr in '0.0001' '0.0005' '0.001' '0.005'; do
  CUDA_VISIBLE_DEVICES=1 python3 main.py --path ./ML100K/ --train_data ML100K_copy1_train --test_data ML100K_copy1_valid --n 943 --m 1683 --lr ${lr} >./valid_results/ML100K-lr-${lr}.txt
done
