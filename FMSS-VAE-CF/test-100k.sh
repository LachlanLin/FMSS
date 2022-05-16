#!/bin/bash

runpython() {
  CUDA_VISIBLE_DEVICES=0 python3 main.py --path ./ML100K/ --train_data ML100K-$1-train --test_data ML100K-$1-test --n 943 --m 1683 --c $2 >./results/ML100K-$1-c-$2.txt
}

for copy in 'copy1' 'copy2' 'copy3'; do
  for c in '0' '1' '2' '3'; do
    runpython ${copy} ${c}
  done
done
