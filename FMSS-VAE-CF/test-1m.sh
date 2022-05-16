#!/bin/bash

runpython() {
  CUDA_VISIBLE_DEVICES=0 python3 main.py --path ./ML1M/ --train_data ML1M-$1-train --test_data ML1M-$1-test --n 6040 --m 3952 --c $2 >./results/ML1M-$1-c-$2.txt
}

for copy in 'copy1' 'copy2' 'copy3'; do
  for c in '0' '1' '2' '3'; do
    runpython ${copy} ${c}
  done
done
