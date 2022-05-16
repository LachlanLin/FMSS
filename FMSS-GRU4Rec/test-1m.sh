#!/bin/bash
# This script is used to test our FMSS on ML1M.

runpython() {
  CUDA_VISIBLE_DEVICES=0 python3 main.py --lr 0.1 --path ./ML1M/ --train_data train_5.csv --test_data test_5.csv --rho $2 --c $3 >./results/ML1M_$1_rho_$2_c_$3.txt
}

for copy in 'run1' 'run2' 'run3'; do
  rho='0'
  c='0'
  runpython ${copy} ${rho} ${c}
  rho='3'
  for c in '0' '1' '2'; do
    runpython ${copy} ${rho} ${c}
  done
  c='3'
  for rho in '0' '1' '2'; do
    runpython ${copy} ${rho} ${c}
  done
  rho='3'
  runpython ${copy} ${rho} ${c}
done
