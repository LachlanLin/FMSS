#!/bin/bash
# This script is used to test our FMSS on ML100K.

runpython() {
  CUDA_VISIBLE_DEVICES=1 python3 main.py --path ./ML100K/ --train_data ML100K_$1_train --test_data ML100K_$1_test --n 943 --m 1683 --rho $2 --c $3 --lr 0.0005 >./results/ML100K_$1_rho_$2_c_$3.txt
}

for copy in 'copy1' 'copy2' 'copy3' 'copy4' 'copy5'; do
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
