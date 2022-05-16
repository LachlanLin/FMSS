#!/bin/bash
# This script is used to select hyperparameters on ML100K.

for gamma in '0.01' '0.02' '0.04' '0.06' '0.08' '0.1'; do
    for xi in '0.9' '0.99' '1.0'; do
      for lambda in '0.001' '0.01' '0.1'; do
      java -Xmx40960m Main -lambda ${lambda} -gamma ${gamma} -xi ${xi} -d 20 -fnTrainingData ./ML100K/ML100K_copy1_train -fnTestData ./ML100K/ML100K_copy1_valid -rho 0 -c 0 -n 943 -m 1682 -num_iterations 100 >./valid_results/ML100K_gamma_${gamma}_xi_${xi}_lambda_${lambda}.txt
    done
  done
done
