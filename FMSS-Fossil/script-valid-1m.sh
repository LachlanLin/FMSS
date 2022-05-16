#!/bin/bash
# This script is used to select hyperparameters on ML1M

for lambda in '0.001' '0.01' '0.1'
do
for L in '1' '2' '3'
do
for gamma in '0.01' '0.1' '1.0'
do
for xi in  '0.9' '1.0'
do
java -Xmx40960m Main -lambda ${lambda} -gamma ${gamma} -xi ${xi} -d 20 -L ${L} -fnTrainData ./ML1M/train_5.csv -fnTestData ./ML1M/valid_5.csv -rho 0 -c 0 -n 6040 -m 3416 -num_iterations 2000 -topK 5 > ./valid_results/ML1M_lambda_${lambda}_L_${L}_gamma_${gamma}_xi_${xi}.txt
done
done
done
done
