#!/bin/bash
# This script is used to select hyperparameters on ML100K

for lambda in '0.001' '0.01' '0.1'
do
for L in '1' '2' '3'
do
for gamma in '0.01' '0.1' '1.0'
do
for xi in  '0.9' '1.0'
do
java -Xmx40960m Main -lambda ${lambda} -gamma ${gamma} -xi ${xi} -d 20 -L ${L} -fnTrainData ./ML100K/train_5.csv -fnTestData ./ML100K/valid_5.csv -rho 0 -c 0 -n 943 -m 1349 -num_iterations 2000 -topK 5 > ./valid_results/ML100K_lambda_${lambda}_L_${L}_gamma_${gamma}_xi_${xi}.txt
done
done
done
done
