#!/bin/bash
# This script is used to test our FMSS on ML100K.

runjava() {
  java -Xmx40960m Main -lambda 0.01 -gamma 0.06 -xi 0.99 -d 20 -fnTrainingData ./ML100K/ML100K_$1_train -fnTestData ./ML100K/ML100K_$1_test -rho $2 -c $3 -n 943 -m 1682 -num_iterations 80 >./results/ML100K_$1_rho_$2_c_$3.txt
}

for copy in 'copy1' 'copy2' 'copy3' 'copy4' 'copy5'; do
  rho='0'
  c='0'
  runjava ${copy} ${rho} ${c}
  rho='3'
  for c in '0' '1' '2'; do
    runjava ${copy} ${rho} ${c}
  done
  c='3'
  for rho in '0' '1' '2'; do
    runjava ${copy} ${rho} ${c}
  done
  rho='3'
  runjava ${copy} ${rho} ${c}
done
